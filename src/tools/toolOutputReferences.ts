/**
 * Tool output reference registry.
 *
 * When enabled via `RunConfig.toolOutputReferences.enabled`, ToolNode
 * stores each successful tool output under a stable key
 * (`tool<idx>turn<turn>`) where `idx` is the tool's position within a
 * ToolNode batch and `turn` is the batch index within the run
 * (incremented once per ToolNode invocation).
 *
 * Subsequent tool calls can pipe a previous output into their args by
 * embedding `{{tool<idx>turn<turn>}}` inside any string argument;
 * {@link ToolOutputReferenceRegistry.resolve} walks the args and
 * substitutes the placeholders immediately before invocation.
 *
 * The registry stores the *raw, untruncated* tool output so a later
 * `{{…}}` substitution pipes the full payload into the next tool —
 * even when the LLM only saw a head+tail-truncated preview in
 * `ToolMessage.content`. Outputs are stored without any annotation
 * (the `_ref` key or the `[ref: ...]` prefix seen by the LLM is
 * strictly a UX signal attached to `ToolMessage.content`). Keeping the
 * registry pristine means downstream bash/jq piping receives the
 * complete, verbatim output with no injected fields.
 */

import { ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolMessageRefMetadata } from '@/types/messages';
import {
  calculateMaxTotalToolOutputSize,
  HARD_MAX_TOOL_RESULT_CHARS,
  HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE,
} from '@/utils/truncation';

/**
 * Non-global matcher for a single `{{tool<i>turn<n>}}` placeholder.
 * Exported for consumers that want to detect references (e.g., syntax
 * highlighting, docs). The stateful `g` variant lives inside the
 * registry so nobody trips on `lastIndex`.
 */
export const TOOL_OUTPUT_REF_PATTERN = /\{\{(tool\d+turn\d+)\}\}/;

/** Object key used when a parsed-object output has `_ref` injected. */
export const TOOL_OUTPUT_REF_KEY = '_ref';

/**
 * Object key used to carry unresolved reference warnings on a parsed-
 * object output. Using a dedicated field instead of a trailing text
 * line keeps the annotated `ToolMessage.content` parseable as JSON for
 * downstream consumers that rely on the object shape.
 */
export const TOOL_OUTPUT_UNRESOLVED_KEY = '_unresolved_refs';

/** Single-line prefix prepended to non-object tool outputs so the LLM sees the reference key. */
export function buildReferencePrefix(key: string): string {
  return `[ref: ${key}]`;
}

/** Stable registry key for a tool output. */
export function buildReferenceKey(toolIndex: number, turn: number): string {
  return `tool${toolIndex}turn${turn}`;
}

export type ToolOutputReferenceRegistryOptions = {
  /** Maximum characters stored per registered output. */
  maxOutputSize?: number;
  /** Maximum total characters retained across all registered outputs. */
  maxTotalSize?: number;
  /**
   * Upper bound on the number of concurrently-tracked runs. When
   * exceeded, the oldest run bucket is evicted (FIFO). Defaults to 32.
   */
  maxActiveRuns?: number;
};

/**
 * Result of resolving placeholders in tool args.
 */
export type ResolveResult<T> = {
  /** Arguments with placeholders replaced. Same shape as the input. */
  resolved: T;
  /** Reference keys that were referenced but had no stored value. */
  unresolved: string[];
};

/**
 * Read-only view over a frozen registry snapshot. Returned by
 * {@link ToolOutputReferenceRegistry.snapshot} for callers that need
 * to resolve placeholders against the registry state at a specific
 * point in time, ignoring any subsequent registrations.
 */
export interface ToolOutputResolveView {
  resolve<T>(args: T): ResolveResult<T>;
}

/**
 * Pre-resolved arg map keyed by `toolCallId`. Used by the mixed
 * direct+event dispatch path to feed event calls' resolved args
 * (captured pre-batch) into the dispatcher without re-resolving
 * against the now-stale live registry.
 */
export type PreResolvedArgsMap = Map<
  string,
  { resolved: Record<string, unknown>; unresolved: string[] }
>;

/**
 * Per-call sink for resolved args, keyed by `toolCallId`. Threaded
 * as a per-batch local map so concurrent `ToolNode.run()` calls do
 * not race on shared sink state.
 */
export type ResolvedArgsByCallId = Map<string, Record<string, unknown>>;

const EMPTY_ENTRIES: ReadonlyMap<string, string> = new Map<string, string>();

/**
 * Per-run state bucket held inside the registry. Each distinct
 * `run_id` gets its own bucket so overlapping concurrent runs on a
 * shared registry cannot leak outputs, turn counters, or warn-memos
 * into one another.
 */
class RunStateBucket {
  entries: Map<string, string> = new Map();
  totalSize: number = 0;
  turnCounter: number = 0;
  warnedNonStringTools: Set<string> = new Set();
}

/**
 * Anonymous (`run_id` absent) bucket key. Anonymous batches are
 * treated as fresh runs on every invocation — see `nextTurn`.
 */
const ANON_RUN_KEY = '\0anon';

/**
 * Default upper bound on the number of concurrently-tracked runs per
 * registry. When exceeded, the oldest run's bucket (by insertion
 * order) is evicted. Keeps memory bounded when a ToolNode is reused
 * across many runs without explicit `releaseRun` calls.
 */
const DEFAULT_MAX_ACTIVE_RUNS = 32;

/**
 * Ordered map of reference-key → stored output, partitioned by run so
 * concurrent / interleaved runs sharing one registry cannot leak
 * outputs between each other.
 *
 * Each public method takes a `runId` which selects the run's bucket.
 * Hosts typically get one registry per run via `Graph`, in which
 * case only a single bucket is ever populated; the partitioning
 * exists so the registry also behaves correctly when a single
 * instance is reused directly.
 */
export class ToolOutputReferenceRegistry {
  private runStates: Map<string, RunStateBucket> = new Map();
  private readonly maxOutputSize: number;
  private readonly maxTotalSize: number;
  private readonly maxActiveRuns: number;
  /**
   * Local stateful matcher used only by `replaceInString`. Kept
   * off-module so callers of the exported `TOOL_OUTPUT_REF_PATTERN`
   * never see a stale `lastIndex`.
   */
  private static readonly PLACEHOLDER_MATCHER = /\{\{(tool\d+turn\d+)\}\}/g;

  constructor(options: ToolOutputReferenceRegistryOptions = {}) {
    /**
     * Per-output default is the same ~400 KB budget as the standard
     * tool-result truncation (`HARD_MAX_TOOL_RESULT_CHARS`). This
     * keeps a single `{{…}}` substitution at a size that is safe to
     * pass through typical shell `ARG_MAX` limits and matches what
     * the LLM would otherwise have seen. Hosts that want larger per-
     * output payloads (API consumers, long JSON streams) can raise
     * the cap explicitly up to the 5 MB total budget.
     */
    const perOutput =
      options.maxOutputSize != null && options.maxOutputSize > 0
        ? options.maxOutputSize
        : HARD_MAX_TOOL_RESULT_CHARS;
    /**
     * Clamp a caller-supplied `maxTotalSize` to
     * `HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE` (5 MB) so the documented
     * absolute cap is enforced regardless of host config —
     * `calculateMaxTotalToolOutputSize` already applies the same
     * upper bound on its computed default, but the user-provided
     * branch was bypassing it.
     */
    const totalRaw =
      options.maxTotalSize != null && options.maxTotalSize > 0
        ? Math.min(options.maxTotalSize, HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE)
        : calculateMaxTotalToolOutputSize(perOutput);
    this.maxTotalSize = totalRaw;
    /**
     * The per-output cap can never exceed the per-run aggregate cap:
     * if a single entry were allowed to be larger than `maxTotalSize`,
     * the eviction loop would either blow the cap (to keep the entry)
     * or self-evict a just-stored value. Clamping here turns
     * `maxTotalSize` into a hard upper bound on *any* state the
     * registry retains per run.
     */
    this.maxOutputSize = Math.min(perOutput, totalRaw);
    this.maxActiveRuns =
      options.maxActiveRuns != null && options.maxActiveRuns > 0
        ? options.maxActiveRuns
        : DEFAULT_MAX_ACTIVE_RUNS;
  }

  private keyFor(runId: string | undefined): string {
    return runId ?? ANON_RUN_KEY;
  }

  private getOrCreate(runId: string | undefined): RunStateBucket {
    const key = this.keyFor(runId);
    let state = this.runStates.get(key);
    if (state == null) {
      state = new RunStateBucket();
      this.runStates.set(key, state);
      if (this.runStates.size > this.maxActiveRuns) {
        const oldest = this.runStates.keys().next().value;
        if (oldest != null && oldest !== key) {
          this.runStates.delete(oldest);
        }
      }
    }
    return state;
  }

  /** Registers (or replaces) the output stored under `key` for `runId`. */
  set(runId: string | undefined, key: string, value: string): void {
    const bucket = this.getOrCreate(runId);
    const clipped =
      value.length > this.maxOutputSize
        ? value.slice(0, this.maxOutputSize)
        : value;
    const existing = bucket.entries.get(key);
    if (existing != null) {
      bucket.totalSize -= existing.length;
      bucket.entries.delete(key);
    }
    bucket.entries.set(key, clipped);
    bucket.totalSize += clipped.length;
    this.evictWithinBucket(bucket);
  }

  /** Returns the stored value for `key` in `runId`'s bucket, or `undefined`. */
  get(runId: string | undefined, key: string): string | undefined {
    return this.runStates.get(this.keyFor(runId))?.entries.get(key);
  }

  /**
   * Returns `true` when `key` is currently stored in `runId`'s bucket.
   * Used by {@link annotateMessagesForLLM} to gate transient annotation
   * on whether the registry still owns the referenced output (a stale
   * `_refKey` from a prior run silently no-ops here).
   */
  has(runId: string | undefined, key: string): boolean {
    return this.runStates.get(this.keyFor(runId))?.entries.has(key) ?? false;
  }

  /** Total number of registered outputs across every run bucket. */
  get size(): number {
    let n = 0;
    for (const bucket of this.runStates.values()) {
      n += bucket.entries.size;
    }
    return n;
  }

  /** Maximum characters retained per output (post-clip). */
  get perOutputLimit(): number {
    return this.maxOutputSize;
  }

  /** Maximum total characters retained *per run*. */
  get totalLimit(): number {
    return this.maxTotalSize;
  }

  /** Drops every run's state. */
  clear(): void {
    this.runStates.clear();
  }

  /**
   * Explicitly release `runId`'s state. Safe to call when a run has
   * finished. Hosts sharing one registry across runs should call this
   * to reclaim memory deterministically; otherwise LRU eviction kicks
   * in when `maxActiveRuns` runs accumulate.
   */
  releaseRun(runId: string | undefined): void {
    this.runStates.delete(this.keyFor(runId));
  }

  /**
   * Claims the next batch turn synchronously from `runId`'s bucket.
   *
   * Must be called once at the start of each ToolNode batch before
   * any `await`, so concurrent invocations within the same run see
   * distinct turn values (reads are effectively atomic by JS's
   * single-threaded execution of the sync prefix).
   *
   * If `runId` is missing the anonymous bucket is dropped and a
   * fresh one created so each anonymous call behaves as its own run.
   */
  nextTurn(runId: string | undefined): number {
    if (runId == null) {
      this.runStates.delete(ANON_RUN_KEY);
    }
    const bucket = this.getOrCreate(runId);
    return bucket.turnCounter++;
  }

  /**
   * Records that `toolName` has been warned about in `runId` (returns
   * `true` on the first call per run, `false` after). Used by
   * ToolNode to emit one log line per offending tool per run when a
   * `ToolMessage.content` isn't a string.
   */
  claimWarnOnce(runId: string | undefined, toolName: string): boolean {
    const bucket = this.getOrCreate(runId);
    if (bucket.warnedNonStringTools.has(toolName)) {
      return false;
    }
    bucket.warnedNonStringTools.add(toolName);
    return true;
  }

  /**
   * Walks `args` and replaces every `{{tool<i>turn<n>}}` placeholder in
   * string values with the stored output *from `runId`'s bucket*. Non-
   * string values and object keys are left untouched. Unresolved
   * references are left in-place and reported so the caller can
   * surface them to the LLM. When no placeholder appears anywhere in
   * the serialized args, the original input is returned without
   * walking the tree.
   */
  resolve<T>(runId: string | undefined, args: T): ResolveResult<T> {
    if (!hasAnyPlaceholder(args)) {
      return { resolved: args, unresolved: [] };
    }
    const bucket = this.runStates.get(this.keyFor(runId));
    return this.resolveAgainst(bucket?.entries ?? EMPTY_ENTRIES, args);
  }

  /**
   * Captures a frozen snapshot of `runId`'s current entries and
   * returns a view that resolves placeholders against *only* that
   * snapshot. The snapshot is decoupled from the live registry, so
   * subsequent `set()` calls (for example, same-turn direct outputs
   * registering while an event branch is still in flight) are
   * invisible to the snapshot's `resolve`. Used by the mixed
   * direct+event dispatch path to preserve same-turn isolation when
   * a `PreToolUse` hook rewrites event args after directs have
   * completed.
   */
  snapshot(runId: string | undefined): ToolOutputResolveView {
    const bucket = this.runStates.get(this.keyFor(runId));
    const entries: ReadonlyMap<string, string> = bucket
      ? new Map(bucket.entries)
      : EMPTY_ENTRIES;
    return {
      resolve: <T>(args: T): ResolveResult<T> =>
        this.resolveAgainst(entries, args),
    };
  }

  private resolveAgainst<T>(
    entries: ReadonlyMap<string, string>,
    args: T
  ): ResolveResult<T> {
    if (!hasAnyPlaceholder(args)) {
      return { resolved: args, unresolved: [] };
    }
    const unresolved = new Set<string>();
    const resolved = this.transform(entries, args, unresolved) as T;
    return { resolved, unresolved: Array.from(unresolved) };
  }

  private transform(
    entries: ReadonlyMap<string, string>,
    value: unknown,
    unresolved: Set<string>
  ): unknown {
    if (typeof value === 'string') {
      return this.replaceInString(entries, value, unresolved);
    }
    if (Array.isArray(value)) {
      return value.map((item) => this.transform(entries, item, unresolved));
    }
    if (value !== null && typeof value === 'object') {
      const source = value as Record<string, unknown>;
      const next: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(source)) {
        next[key] = this.transform(entries, item, unresolved);
      }
      return next;
    }
    return value;
  }

  private replaceInString(
    entries: ReadonlyMap<string, string>,
    input: string,
    unresolved: Set<string>
  ): string {
    if (input.indexOf('{{tool') === -1) {
      return input;
    }
    return input.replace(
      ToolOutputReferenceRegistry.PLACEHOLDER_MATCHER,
      (match, key: string) => {
        const stored = entries.get(key);
        if (stored == null) {
          unresolved.add(key);
          return match;
        }
        return stored;
      }
    );
  }

  private evictWithinBucket(bucket: RunStateBucket): void {
    if (bucket.totalSize <= this.maxTotalSize) {
      return;
    }
    for (const key of bucket.entries.keys()) {
      if (bucket.totalSize <= this.maxTotalSize) {
        return;
      }
      const entry = bucket.entries.get(key);
      if (entry == null) {
        continue;
      }
      bucket.totalSize -= entry.length;
      bucket.entries.delete(key);
    }
  }
}

/**
 * Cheap pre-check: returns true if any string value in `args` contains
 * the `{{tool` substring. Lets `resolve()` skip the deep tree walk (and
 * its object allocations) for the common case of plain args.
 */
function hasAnyPlaceholder(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.indexOf('{{tool') !== -1;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      if (hasAnyPlaceholder(item)) {
        return true;
      }
    }
    return false;
  }
  if (value !== null && typeof value === 'object') {
    for (const item of Object.values(value as Record<string, unknown>)) {
      if (hasAnyPlaceholder(item)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

/**
 * Annotates `content` with a reference key and/or unresolved-ref
 * warnings so the LLM sees both alongside the tool output.
 *
 * Behavior:
 *  - If `content` parses as a plain (non-array, non-null) JSON object
 *    and the object does not already have a conflicting `_ref` key,
 *    the reference key and (when present) `_unresolved_refs` array
 *    are injected as object fields, preserving JSON validity for
 *    downstream consumers that parse the output.
 *  - Otherwise (string output, JSON array/primitive, parse failure,
 *    or `_ref` collision), a `[ref: <key>]\n` prefix line is
 *    prepended and unresolved refs are appended as a trailing
 *    `[unresolved refs: …]` line.
 *
 * The annotated string is what the LLM sees as `ToolMessage.content`.
 * The *original* (un-annotated) value is what gets stored in the
 * registry, so downstream piping remains pristine.
 *
 * @param content     Raw (post-truncation) tool output.
 * @param key         Reference key for this output, or undefined when
 *                    there is nothing to register (errors etc.).
 * @param unresolved  Reference keys that failed to resolve during
 *                    argument substitution. Surfaced so the LLM can
 *                    self-correct its next tool call.
 */
export function annotateToolOutputWithReference(
  content: string,
  key: string | undefined,
  unresolved: string[] = []
): string {
  const hasRefKey = key != null;
  const hasUnresolved = unresolved.length > 0;
  if (!hasRefKey && !hasUnresolved) {
    return content;
  }
  const trimmed = content.trimStart();
  if (trimmed.startsWith('{')) {
    const annotated = tryInjectRefIntoJsonObject(content, key, unresolved);
    if (annotated != null) {
      return annotated;
    }
  }
  const prefix = hasRefKey ? `${buildReferencePrefix(key!)}\n` : '';
  const trailer = hasUnresolved
    ? `\n[unresolved refs: ${unresolved.join(', ')}]`
    : '';
  return `${prefix}${content}${trailer}`;
}

function tryInjectRefIntoJsonObject(
  content: string,
  key: string | undefined,
  unresolved: string[]
): string | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    return null;
  }

  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return null;
  }

  const obj = parsed as Record<string, unknown>;
  const injectingRef = key != null;
  const injectingUnresolved = unresolved.length > 0;

  /**
   * Reject the JSON-injection path (fall back to prefix form) when
   * either of our keys collides with real payload data:
   *  - `_ref` collision: existing value is non-null and differs from
   *    the key we're about to inject.
   *  - `_unresolved_refs` collision: existing value is non-null and
   *    is not a deep-equal match for the array we'd inject.
   * This keeps us from silently overwriting legitimate tool output.
   */
  if (
    injectingRef &&
    TOOL_OUTPUT_REF_KEY in obj &&
    obj[TOOL_OUTPUT_REF_KEY] !== key &&
    obj[TOOL_OUTPUT_REF_KEY] != null
  ) {
    return null;
  }
  if (
    injectingUnresolved &&
    TOOL_OUTPUT_UNRESOLVED_KEY in obj &&
    obj[TOOL_OUTPUT_UNRESOLVED_KEY] != null &&
    !arraysShallowEqual(obj[TOOL_OUTPUT_UNRESOLVED_KEY], unresolved)
  ) {
    return null;
  }

  /**
   * Only strip the framework-owned key we're actually injecting —
   * leave everything else (including a pre-existing `_ref` on the
   * unresolved-only path, or a pre-existing `_unresolved_refs` on a
   * plain-annotation path) untouched so we annotate rather than
   * mutate downstream payload data. Our injected keys land first in
   * the serialized JSON so the LLM sees them before the body.
   */
  const omitKeys = new Set<string>();
  if (injectingRef) omitKeys.add(TOOL_OUTPUT_REF_KEY);
  if (injectingUnresolved) omitKeys.add(TOOL_OUTPUT_UNRESOLVED_KEY);
  const rest: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (!omitKeys.has(k)) {
      rest[k] = v;
    }
  }
  const injected: Record<string, unknown> = {};
  if (injectingRef) {
    injected[TOOL_OUTPUT_REF_KEY] = key;
  }
  if (injectingUnresolved) {
    injected[TOOL_OUTPUT_UNRESOLVED_KEY] = unresolved;
  }
  Object.assign(injected, rest);

  const pretty = /^\{\s*\n/.test(content);
  return pretty ? JSON.stringify(injected, null, 2) : JSON.stringify(injected);
}

function arraysShallowEqual(a: unknown, b: readonly string[]): boolean {
  if (!Array.isArray(a) || a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Lazy projection that, given a registry and a runId, returns a new
 * `messages` array where each `ToolMessage` carrying a live `_refKey`
 * (and/or unresolved-ref metadata) has its `content` annotated for the
 * LLM. The original input array and its messages are never mutated.
 *
 * Annotation is gated on registry presence: a stale `_refKey` from a
 * prior run (e.g. one that survived in persisted history) silently
 * no-ops, exactly matching the "refs are run-scoped" mental model.
 * `_unresolvedRefs` is always meaningful and is not gated.
 *
 * When neither a registry nor any ToolMessage in the batch carries
 * relevant metadata, the input array is returned unchanged (reference-
 * equal) so the common case is allocation-free.
 */
export function annotateMessagesForLLM(
  messages: BaseMessage[],
  registry: ToolOutputReferenceRegistry | undefined,
  runId: string | undefined
): BaseMessage[] {
  if (registry == null) return messages;

  const out = messages.map((m) => {
    if (m._getType() !== 'tool') return m;
    const meta = m.additional_kwargs as
      | (ToolMessageRefMetadata & Record<string, unknown>)
      | undefined;
    const refKey = meta?._refKey;
    const unresolved = meta?._unresolvedRefs ?? [];
    if (refKey == null && unresolved.length === 0) return m;

    const liveRef =
      refKey != null && registry.has(runId, refKey) ? refKey : undefined;

    if (liveRef == null && unresolved.length === 0) return m;

    const tm = m as ToolMessage;
    const cleanedKwargs = stripRefMetadata(meta);
    if (typeof tm.content === 'string') {
      const annotated = annotateToolOutputWithReference(
        tm.content,
        liveRef,
        unresolved
      );
      return new ToolMessage({
        id: tm.id,
        name: tm.name,
        status: tm.status,
        artifact: tm.artifact,
        tool_call_id: tm.tool_call_id,
        response_metadata: tm.response_metadata,
        additional_kwargs: cleanedKwargs,
        content: annotated,
      });
    }
    if (Array.isArray(tm.content) && unresolved.length > 0) {
      const warningBlock = {
        type: 'text' as const,
        text: `[unresolved refs: ${unresolved.join(', ')}]`,
      };
      return new ToolMessage({
        id: tm.id,
        name: tm.name,
        status: tm.status,
        artifact: tm.artifact,
        tool_call_id: tm.tool_call_id,
        response_metadata: tm.response_metadata,
        additional_kwargs: cleanedKwargs,
        content: [
          warningBlock,
          ...tm.content,
        ] as unknown as ToolMessage['content'],
      });
    }
    return m;
  });

  for (let i = 0; i < out.length; i++) {
    if (out[i] !== messages[i]) return out;
  }
  return messages;
}

/**
 * Returns a copy of `meta` with the framework-owned ref keys stripped.
 * Used by {@link annotateMessagesForLLM} so the projected ToolMessage
 * does not round-trip the metadata back into state with annotation
 * already applied. Returns `undefined` when no other keys remain.
 */
function stripRefMetadata(
  meta: (ToolMessageRefMetadata & Record<string, unknown>) | undefined
): Record<string, unknown> | undefined {
  if (meta == null) return undefined;
  const rest: Record<string, unknown> = {};
  let hasOther = false;
  for (const [k, v] of Object.entries(meta)) {
    if (k === '_refKey' || k === '_unresolvedRefs') continue;
    rest[k] = v;
    hasOther = true;
  }
  return hasOther ? rest : undefined;
}
