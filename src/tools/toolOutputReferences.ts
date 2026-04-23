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
    const perOutput =
      options.maxOutputSize != null && options.maxOutputSize > 0
        ? options.maxOutputSize
        : HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE;
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
 * Attempts to annotate `content` with its reference key so the LLM sees
 * the key alongside the output.
 *
 * Behavior:
 *  - If `content` parses as a plain (non-array, non-null) JSON object
 *    and does not already have a conflicting `_ref` key, the key is
 *    injected and the object re-serialized (compact or pretty,
 *    matching the original layout).
 *  - Otherwise (string output, JSON array/primitive, parse failure, or
 *    `_ref` collision), a `[ref: <key>]\n` prefix line is prepended.
 *
 * The annotated string is what the LLM sees as `ToolMessage.content`.
 * The *original* (un-annotated) value is what gets stored in the
 * registry, so downstream piping remains pristine.
 */
export function annotateToolOutputWithReference(
  content: string,
  key: string
): string {
  const prefix = buildReferencePrefix(key);
  const trimmed = content.trimStart();
  if (trimmed.startsWith('{')) {
    const annotated = tryInjectRefIntoJsonObject(content, key);
    if (annotated != null) {
      return annotated;
    }
  }
  return `${prefix}\n${content}`;
}

function tryInjectRefIntoJsonObject(
  content: string,
  key: string
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
  if (
    TOOL_OUTPUT_REF_KEY in obj &&
    obj[TOOL_OUTPUT_REF_KEY] !== key &&
    obj[TOOL_OUTPUT_REF_KEY] != null
  ) {
    return null;
  }

  /**
   * Drop any pre-existing `_ref` (the guard above already rejected
   * real conflicts; what reaches here is either a null or a matching
   * value) so we can place our injected key *first* in the serialized
   * JSON. Leading-key placement means the LLM sees the reference
   * identifier before it reads the payload, which matters for large
   * objects where the tail might be truncated by downstream consumers.
   */
  const { [TOOL_OUTPUT_REF_KEY]: _existing, ...rest } = obj;
  void _existing;
  const injected: Record<string, unknown> = {
    [TOOL_OUTPUT_REF_KEY]: key,
    ...rest,
  };

  const pretty = /^\{\s*\n/.test(content);
  return pretty ? JSON.stringify(injected, null, 2) : JSON.stringify(injected);
}
