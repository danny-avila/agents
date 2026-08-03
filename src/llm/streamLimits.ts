// src/llm/streamLimits.ts
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { getStreamedToolCallSeal } from '@/tools/streamedToolCallSeals';
import { Constants } from '@/common';

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
 *
 * The `INVOKED_MODEL` stamp scopes attempts within one node execution:
 * `tryFallbackProviders` (and summarization's fallback path) stamps it into
 * each fallback's config metadata, so a fallback's chunks key separately
 * from the failed primary's even when the decoupled `streamEvents` reader
 * drains the primary's buffered chunks late. Those late chunks land in the
 * primary's own bucket instead of polluting the fallback's.
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
  const invokedModel = metadata[Constants.INVOKED_MODEL] ?? '';
  return `${checkpointNs}|${node}|${step}|${invokedModel}`;
}

/**
 * Event-metadata marker for a chunk the SDK re-dispatches inline after
 * transforming it (`attemptInvoke`'s OpenRouter final-reasoning replay). The
 * original wire chunk still reaches the handler through `streamEvents` and
 * is counted there, so the re-dispatch must not consume a second
 * event-budget slot.
 */
export const STREAM_LIMIT_REDISPATCH_KEY = 'lc_stream_limit_redispatch';

/**
 * True when the event's `single` seal marks this chunk's own call as
 * complete. On the OpenAI Responses adapter the sealing chunk RESTATES the
 * full argument string (`response.function_call_arguments.done`), so summing
 * it would double-count every legitimate call; on Bedrock Converse the
 * sealing chunk carries empty args. Either way the call is finished: its
 * tally is replaced by the sealing chunk's own bytes, checked, and released.
 */
function sealsChunk(
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

function chunkToolName(chunk: ToolCallChunk): string | undefined {
  return chunk.name != null && chunk.name !== '' ? chunk.name : undefined;
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
 * chunk `id` and then to the chunk's position within the event's batch when
 * a provider identifies chunks by neither (Google emits complete parallel
 * calls with optional ids and no index). A `kind: 'all'` arrival seal marks
 * every chunk in the event as its own complete call, so those are checked
 * standalone and never share a budget; a matching `kind: 'single'` seal
 * replaces the call's tally with the sealing chunk's own bytes (the OpenAI
 * Responses done-chunk restates the full argument string) and releases it.
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
  for (let i = 0; i < toolCallChunks.length; i++) {
    const chunk = toolCallChunks[i];
    const args = chunk.args;
    const hasArgs = typeof args === 'string' && args !== '';
    if (seal?.kind === 'all') {
      if (!hasArgs) {
        continue;
      }
      const bytes = Buffer.byteLength(args, 'utf8');
      if (bytes > limit) {
        throw new StreamLimitExceededError({
          kind: 'tool_call_args',
          limit,
          observed: bytes,
          toolName: chunkToolName(chunk),
        });
      }
      continue;
    }
    const key = `${generationKey}:${chunk.index ?? chunk.id ?? `#${i}`}`;
    const sealed = sealsChunk(seal, chunk);
    let tally = tallies.get(key);
    if (!hasArgs) {
      if (tally == null) {
        if (!sealed) {
          tallies.set(key, { bytes: 0, name: chunkToolName(chunk) });
        }
      } else if (sealed) {
        tallies.delete(key);
      } else if (tally.name == null) {
        tally.name = chunkToolName(chunk);
      }
      continue;
    }
    if (tally == null) {
      tally = { bytes: 0 };
      tallies.set(key, tally);
    }
    if (tally.name == null) {
      tally.name = chunkToolName(chunk);
    }
    const argBytes = Buffer.byteLength(args, 'utf8');
    tally.bytes = sealed ? argBytes : tally.bytes + argBytes;
    if (tally.bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: tally.bytes,
        toolName: tally.name ?? chunkToolName(chunk),
      });
    }
    if (sealed) {
      tallies.delete(key);
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
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
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

/**
 * Releases one generation's tallies and event count. Called at the start of
 * every model attempt (`attemptInvoke`): primary, fallback, and retry
 * attempts within one node share the same langgraph metadata, so without
 * this a fallback that re-streams a tool call from scratch would be charged
 * the failed primary's partial bytes and could falsely trip the limit. The
 * decoupled `streamEvents` reader can lag an attempt boundary by whatever it
 * has buffered, so attribution is best-effort by design; the reset shrinks a
 * mischarge from "the whole failed attempt" to that small in-flight tail.
 * No-ops on two size checks while nothing was counted.
 */
export function resetStreamLimitTallies({
  graph,
  metadata,
}: {
  graph: StreamLimitState | undefined;
  metadata: Record<string, unknown> | undefined;
}): void {
  if (graph == null) {
    return;
  }
  const tallies = graph.streamedToolCallArgTallies;
  const counts = graph.streamDeltaEventCounts;
  const hasTallies = tallies != null && tallies.size > 0;
  const hasCounts = counts != null && counts.size > 0;
  if (!hasTallies && !hasCounts) {
    return;
  }
  const generationKey = resolveGenerationKey(metadata);
  if (hasTallies) {
    const prefix = `${generationKey}:`;
    for (const key of tallies.keys()) {
      if (key.startsWith(prefix)) {
        tallies.delete(key);
      }
    }
  }
  if (hasCounts) {
    counts.delete(generationKey);
  }
}
