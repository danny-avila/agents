// src/llm/streamLimits.ts
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { getStreamedToolCallSeal } from '@/tools/streamedToolCallSeals';

/**
 * Circuit breakers for pathological model streams.
 *
 * A malformed generation can stream a single tool call's arguments for many
 * minutes at the provider's full token rate while the arguments never become
 * executable (observed live: one 149,923-char SQL argument streamed for 26
 * minutes before the 64k output-token ceiling finally ended the run). These
 * guards fail fast instead: when a limit trips, the stream handler throws
 * `StreamLimitExceededError` out of the run's `streamEvents` loop, which
 * tears down the in-flight provider request where it stands (see the
 * mid-flight halt notes in `Run.processStream`: leaving the loop cancels the
 * reader and langgraph aborts the model call).
 */

/** Default cap on a single streamed tool call's cumulative argument bytes (64 KiB). */
export const DEFAULT_MAX_TOOL_CALL_ARG_BYTES = 65_536;

/** Limits normalized by {@link resolveStreamLimits}; `0` uniformly means disabled. */
export interface ResolvedStreamLimits {
  maxToolCallArgBytes: number;
  maxDeltaEventsPerTurn: number;
}

/** Cumulative streamed argument bytes for one in-flight tool call. */
export interface StreamedToolCallArgTally {
  bytes: number;
  name?: string;
}

/**
 * The graph-owned state the guards read and write. Structural on purpose:
 * handler-level tests stub graphs with plain objects, and the guards lazily
 * create the tally maps so partial stubs need no extra setup.
 */
export interface StreamLimitState {
  streamLimits?: ResolvedStreamLimits;
  streamedToolCallArgTallies?: Map<string, StreamedToolCallArgTally>;
  streamDeltaEventCounts?: Map<string, number>;
}

function resolveLimit(value: number | undefined, fallback: number): number {
  if (value == null || Number.isNaN(value)) {
    return fallback;
  }
  if (!Number.isFinite(value)) {
    return 0;
  }
  const whole = Math.floor(value);
  return whole > 0 ? whole : 0;
}

/**
 * Normalizes host-supplied limits once at graph construction. `undefined`
 * applies the default for each guard (the tool-argument byte cap is ON by
 * default, the per-turn event cap is opt-in), `0` and negative values
 * disable a guard, `Infinity` means "no limit" and also disables, and `NaN`
 * falls back to the default.
 */
export function resolveStreamLimits(
  limits?: t.StreamLimits
): ResolvedStreamLimits {
  return {
    maxToolCallArgBytes: resolveLimit(
      limits?.maxToolCallArgBytes,
      DEFAULT_MAX_TOOL_CALL_ARG_BYTES
    ),
    maxDeltaEventsPerTurn: resolveLimit(limits?.maxDeltaEventsPerTurn, 0),
  };
}

const DEFAULT_RESOLVED_LIMITS: ResolvedStreamLimits = resolveStreamLimits();

export type StreamLimitKind = 'tool_call_args' | 'delta_events';

function buildLimitMessage(
  kind: StreamLimitKind,
  limit: number,
  toolName?: string
): string {
  if (kind === 'tool_call_args') {
    const named =
      toolName != null && toolName !== '' ? ` (tool call: ${toolName})` : '';
    return (
      `Streamed tool call arguments exceeded the ${limit}-byte safety limit${named}. ` +
      'The generation was aborted mid-stream: arguments growing this large without completing ' +
      'usually indicate a runaway or malformed tool call. Raise \'maxToolCallArgBytes\' if your ' +
      'tools legitimately need larger arguments.'
    );
  }
  return (
    `Model stream exceeded the ${limit}-event safety limit for a single generation turn. ` +
    'The generation was aborted mid-stream: this usually indicates a looping or duplicated ' +
    'provider stream. Raise \'maxDeltaEventsPerTurn\' if legitimate generations need more stream events.'
  );
}

/**
 * Raised when a {@link t.StreamLimits} guard trips. Thrown from inside the
 * run's `streamEvents` loop, so the in-flight provider request is torn down
 * and `processStream` rejects with this error.
 */
export class StreamLimitExceededError extends Error {
  readonly kind: StreamLimitKind;
  readonly limit: number;
  readonly observed: number;
  readonly toolName?: string;

  constructor({
    kind,
    limit,
    observed,
    toolName,
  }: {
    kind: StreamLimitKind;
    limit: number;
    observed: number;
    toolName?: string;
  }) {
    super(buildLimitMessage(kind, limit, toolName));
    this.name = 'StreamLimitExceededError';
    this.kind = kind;
    this.limit = limit;
    this.observed = observed;
    this.toolName = toolName;
  }
}

/**
 * Identity of one model generation, derived from langgraph's node-execution
 * metadata. Deliberately NOT `Graph.getStepKey()`: the step key forks within
 * a single generation on reasoning transitions (`'reasoning'` /
 * `post-reasoning-<n>` suffixes in `getKeyList`) and on mid-turn server-tool
 * results (`invokedToolIds` count), which would hand a fresh budget to each
 * segment. One agent-node execution is one superstep, so
 * `checkpoint_ns + node + step` stays stable for the whole generation and
 * distinguishes parallel agents in the same superstep.
 */
export function resolveGenerationKey(
  metadata: Record<string, unknown> | undefined
): string {
  if (metadata == null) {
    return '';
  }
  const checkpointNs = metadata.langgraph_checkpoint_ns ?? '';
  const node = metadata.langgraph_node ?? '';
  const step = metadata.langgraph_step ?? '';
  return `${checkpointNs}|${node}|${step}`;
}

/**
 * True when this chunk restates an already-streamed argument string instead
 * of extending it. The OpenAI Responses adapter attaches a `single` seal to
 * its `response.function_call_arguments.done` chunk, whose `args` is the
 * complete argument string all over again; summing it would double-count
 * every legitimate call. Adapters that seal without restating are unaffected:
 * Bedrock Converse seals with an empty-args chunk (skipped before this
 * check) and Google seals atomic single-chunk calls with `kind: 'all'`.
 */
function isSealRestatement(
  seal: ReturnType<typeof getStreamedToolCallSeal>,
  chunk: ToolCallChunk
): boolean {
  if (seal == null || seal.kind !== 'single') {
    return false;
  }
  if (seal.index != null) {
    return seal.index === chunk.index;
  }
  return seal.id != null && seal.id === chunk.id;
}

/**
 * Accumulates the UTF-8 byte size of streamed tool-call argument chunks per
 * in-flight tool call and throws once a single call's cumulative bytes
 * exceed `maxToolCallArgBytes`. Runs once per streamed chunk event, before
 * complete tool calls are dispatched or eagerly executed and before chunks
 * are recorded, so a tripped limit stops the run without dispatching the
 * offending call.
 *
 * Calls are keyed by generation and chunk `index`, falling back to the
 * chunk `id` and then to a shared bucket when a provider identifies chunks
 * by neither. The shared-bucket fallback is deliberately conservative: such
 * chunks cannot be attributed to distinct calls anywhere in the pipeline,
 * and a runaway stream is exactly the case that must still be bounded.
 */
export function enforceStreamedToolCallArgLimit({
  graph,
  metadata,
  toolCallChunks,
  responseMetadata,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  toolCallChunks: ToolCallChunk[];
  responseMetadata?: Record<string, unknown>;
}): void {
  const limit = (graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS)
    .maxToolCallArgBytes;
  if (limit <= 0) {
    return;
  }
  const tallies = (graph.streamedToolCallArgTallies ??= new Map());
  const generationKey = resolveGenerationKey(metadata);
  const seal = getStreamedToolCallSeal(responseMetadata);
  for (const chunk of toolCallChunks) {
    const key = `${generationKey}:${chunk.index ?? chunk.id ?? 0}`;
    let tally = tallies.get(key);
    if (tally == null) {
      tally = { bytes: 0 };
      tallies.set(key, tally);
    }
    if (tally.name == null && chunk.name != null && chunk.name !== '') {
      tally.name = chunk.name;
    }
    const args = chunk.args;
    if (typeof args !== 'string' || args === '') {
      continue;
    }
    const argBytes = Buffer.byteLength(args, 'utf8');
    tally.bytes = isSealRestatement(seal, chunk)
      ? argBytes
      : tally.bytes + argBytes;
    if (tally.bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: tally.bytes,
        toolName: tally.name,
      });
    }
  }
}

/**
 * Counts streamed chunk events per model generation and throws once a single
 * generation exceeds `maxDeltaEventsPerTurn`. Opt-in defense in depth for
 * pathologies a byte cap cannot see, such as a provider stream looping on
 * empty chunks. Zero cost while disabled.
 */
export function enforceStreamDeltaEventLimit({
  graph,
  metadata,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
}): void {
  const limit = (graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS)
    .maxDeltaEventsPerTurn;
  if (limit <= 0) {
    return;
  }
  const counts = (graph.streamDeltaEventCounts ??= new Map());
  const key = resolveGenerationKey(metadata);
  const next = (counts.get(key) ?? 0) + 1;
  if (next > limit) {
    throw new StreamLimitExceededError({
      kind: 'delta_events',
      limit,
      observed: next,
    });
  }
  counts.set(key, next);
}
