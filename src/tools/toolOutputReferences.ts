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

import {
  calculateMaxTotalToolOutputSize,
  HARD_MAX_TOOL_RESULT_CHARS,
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
 * Ordered map of reference-key → stored output with FIFO eviction when
 * the aggregate size exceeds `maxTotalSize`.
 *
 * A single shared registry lives on the ToolNode for the duration of a
 * run; it is not persisted across runs and is cleared when the graph's
 * heavy state is cleared.
 */
export class ToolOutputReferenceRegistry {
  private entries: Map<string, string> = new Map();
  private totalSize: number = 0;
  private readonly maxOutputSize: number;
  private readonly maxTotalSize: number;
  /**
   * Monotonic batch counter shared across every ToolNode that holds
   * this registry. Multi-agent graphs use one registry per run so
   * turns are globally unique across agents — `tool0turn3` from agent
   * B does not collide with `tool0turn3` from agent A.
   */
  private turnCounter: number = 0;
  /**
   * Last observed `run_id`. Drives the reset when we cross into a
   * new run. Undefined when the feature is invoked without a
   * `run_id`; in that case every call is treated as a fresh run.
   */
  private lastRunId: string | undefined;
  /**
   * Per-run memo of tool names we've already logged a non-string-
   * content warning for. Cleared on every run change so each run
   * emits at most one line per offending tool.
   */
  private warnedNonStringTools: Set<string> = new Set();
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
    const totalRaw =
      options.maxTotalSize != null && options.maxTotalSize > 0
        ? options.maxTotalSize
        : calculateMaxTotalToolOutputSize(perOutput);
    this.maxTotalSize = totalRaw;
    /**
     * The per-output cap can never exceed the aggregate cap: if a
     * single entry were allowed to be larger than `maxTotalSize`, the
     * eviction loop would either blow the cap (to keep the entry) or
     * self-evict a just-stored value. Clamping here turns
     * `maxTotalSize` into a hard upper bound on *any* state the
     * registry retains.
     */
    this.maxOutputSize = Math.min(perOutput, totalRaw);
  }

  /** Registers (or replaces) the output stored under `key`. */
  set(key: string, value: string): void {
    const clipped =
      value.length > this.maxOutputSize
        ? value.slice(0, this.maxOutputSize)
        : value;

    const existing = this.entries.get(key);
    if (existing != null) {
      this.totalSize -= existing.length;
      this.entries.delete(key);
    }

    this.entries.set(key, clipped);
    this.totalSize += clipped.length;
    this.evictUntilWithinLimit();
  }

  /** Returns the stored value for `key`, or `undefined` if unknown. */
  get(key: string): string | undefined {
    return this.entries.get(key);
  }

  /** Current number of registered outputs. */
  get size(): number {
    return this.entries.size;
  }

  /** Maximum characters retained per output (post-clip). */
  get perOutputLimit(): number {
    return this.maxOutputSize;
  }

  /** Maximum total characters retained across the registry. */
  get totalLimit(): number {
    return this.maxTotalSize;
  }

  /** Drops all registered outputs. */
  clear(): void {
    this.entries.clear();
    this.totalSize = 0;
  }

  /**
   * Claims the next batch turn synchronously.
   *
   * Must be called once at the start of each ToolNode batch before
   * any `await`, so concurrent invocations within the same run see
   * distinct turn values (reads are effectively atomic by JS's
   * single-threaded execution of the sync prefix).
   *
   * If `runId` differs from the last-seen value — or is missing —
   * the registry, warn-set, and counter are cleared before claiming
   * turn 0. Missing `runId` is treated as "always a new run" so
   * anonymous callers never leak state across invocations.
   */
  nextTurn(runId: string | undefined): number {
    if (runId == null || runId !== this.lastRunId) {
      this.entries.clear();
      this.totalSize = 0;
      this.turnCounter = 0;
      this.warnedNonStringTools.clear();
      this.lastRunId = runId;
    }
    return this.turnCounter++;
  }

  /**
   * Records that `toolName` has been warned about (returns `true`
   * on the first call per run, `false` after). Used by ToolNode to
   * emit one log line per offending tool per run when a
   * `ToolMessage.content` isn't a string.
   */
  claimWarnOnce(toolName: string): boolean {
    if (this.warnedNonStringTools.has(toolName)) {
      return false;
    }
    this.warnedNonStringTools.add(toolName);
    return true;
  }

  /**
   * Walks `args` and replaces every `{{tool<i>turn<n>}}` placeholder in
   * string values with the stored output. Non-string values and object
   * keys are left untouched. Unresolved references are left in-place and
   * reported so the caller can surface them to the LLM. When no
   * placeholder appears anywhere in the serialized args, the original
   * input is returned without walking the tree.
   */
  resolve<T>(args: T): ResolveResult<T> {
    if (!hasAnyPlaceholder(args)) {
      return { resolved: args, unresolved: [] };
    }
    const unresolved = new Set<string>();
    const resolved = this.transform(args, unresolved) as T;
    return { resolved, unresolved: Array.from(unresolved) };
  }

  private transform(value: unknown, unresolved: Set<string>): unknown {
    if (typeof value === 'string') {
      return this.replaceInString(value, unresolved);
    }
    if (Array.isArray(value)) {
      return value.map((item) => this.transform(item, unresolved));
    }
    if (value !== null && typeof value === 'object') {
      const source = value as Record<string, unknown>;
      const next: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(source)) {
        next[key] = this.transform(item, unresolved);
      }
      return next;
    }
    return value;
  }

  private replaceInString(input: string, unresolved: Set<string>): string {
    if (input.indexOf('{{tool') === -1) {
      return input;
    }
    return input.replace(
      ToolOutputReferenceRegistry.PLACEHOLDER_MATCHER,
      (match, key: string) => {
        const stored = this.get(key);
        if (stored == null) {
          unresolved.add(key);
          return match;
        }
        return stored;
      }
    );
  }

  private evictUntilWithinLimit(): void {
    if (this.totalSize <= this.maxTotalSize) {
      return;
    }
    for (const key of this.entries.keys()) {
      if (this.totalSize <= this.maxTotalSize) {
        return;
      }
      const entry = this.entries.get(key);
      if (entry == null) {
        continue;
      }
      this.totalSize -= entry.length;
      this.entries.delete(key);
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
