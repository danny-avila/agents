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
  /** Per-tool overrides of the byte cap, keyed by model-facing tool name. */
  maxToolCallArgBytesByTool?: Readonly<Record<string, number>>;
  maxDeltaEventsPerTurn: number;
}

/** Cumulative streamed argument bytes for one in-flight tool call. */
export interface StreamedToolCallArgTally {
  bytes: number;
  name?: string;
  /** Secondary key this tally is also registered under — the batch position
   * for id-only chunks, or the id for chunks carrying both identifiers — so
   * a later delta that drops an identifier still shares the budget; both
   * entries are released together. */
  aliasKey?: string;
  /**
   * True when the previous chunk ended on an unpaired UTF-16 high surrogate.
   * Counting each half of a split surrogate pair alone yields 3 bytes per
   * half (the replacement-character encoding) versus 4 for the pair, so the
   * next chunk starting with the low surrogate reconciles by subtracting 2.
   */
  pendingHighSurrogate?: boolean;
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
  /** Per-chunk-object, per-generation charge balance: producer visits
   * increment, consumer (handler echo) visits decrement, and a visit only
   * charges when the other side has not pre-charged the same emission.
   * Count-balancing rather than a lifetime set, because a streaming model
   * may mutate and re-yield the same chunk object; scoped by generation so
   * parallel generations sharing one reused object cannot cancel each
   * other's charges. */
  streamLimitChargeCredits?: WeakMap<object, Map<string, number>>;
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
  const maxToolCallArgBytes = resolveLimit(
    limits?.maxToolCallArgBytes,
    DEFAULT_MAX_TOOL_CALL_ARG_BYTES
  );
  let maxToolCallArgBytesByTool: Record<string, number> | undefined;
  if (limits?.maxToolCallArgBytesByTool != null) {
    for (const [name, value] of Object.entries(
      limits.maxToolCallArgBytesByTool
    )) {
      /** An unusable entry (NaN) falls back to the global cap by omission,
       * matching how the global field treats NaN. */
      if (name === '' || Number.isNaN(value)) {
        continue;
      }
      /** Prototype-free: bracket-assigning a `__proto__` tool name into a
       * plain object invokes the prototype setter instead of creating an own
       * property, silently dropping that tool's configured override. */
      maxToolCallArgBytesByTool ??= Object.create(null) as Record<
        string,
        number
      >;
      maxToolCallArgBytesByTool[name] = resolveLimit(
        value,
        maxToolCallArgBytes
      );
    }
  }
  return {
    maxToolCallArgBytes,
    ...(maxToolCallArgBytesByTool != null && { maxToolCallArgBytesByTool }),
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
      'usually indicate a runaway or malformed tool call. Raise \'maxToolCallArgBytes\' (or the ' +
      'tool\'s \'maxToolCallArgBytesByTool\' entry) if your tools legitimately need larger arguments.'
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
 * The attempt stamp scopes attempts within one node execution:
 * `attemptInvoke` is the single funnel for primary, fallback, and
 * summarization model calls and stamps {@link STREAM_LIMIT_ATTEMPT_KEY}
 * with a unique sequence number into each attempt's callback metadata. A
 * fallback's chunks therefore key separately from the failed primary's,
 * even for two fallbacks configured with the same provider and model name,
 * and even when the decoupled `streamEvents` reader drains a failed
 * attempt's buffered chunks late. Those late chunks land in their own
 * attempt's bucket instead of polluting the next one's.
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
  const attempt = metadata[STREAM_LIMIT_ATTEMPT_KEY] ?? '';
  return `${checkpointNs}|${node}|${step}|${attempt}`;
}

/**
 * Metadata key carrying the unique per-model-attempt sequence number that
 * `attemptInvoke` stamps into every attempt's callback metadata. Part of
 * the generation key so budgets never alias across attempts.
 */
export const STREAM_LIMIT_ATTEMPT_KEY = 'lc_stream_limit_attempt';

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

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
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
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  const globalLimit = resolved.maxToolCallArgBytes;
  const byTool = resolved.maxToolCallArgBytesByTool;
  if (globalLimit <= 0 && byTool == null) {
    return;
  }
  /** An inline re-dispatch of a transformed chunk (OpenRouter final-reasoning
   * replay) duplicates tool-call chunks the original `streamEvents` event
   * already charged — the original always survives the content-specific
   * skips when it carries tool calls, so counting the marked copy would
   * double every legitimate argument byte. */
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
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
      const sealedName = chunkToolName(chunk);
      const limit =
        byTool != null && sealedName != null && Object.hasOwn(byTool, sealedName)
          ? byTool[sealedName]
          : globalLimit;
      if (limit > 0 && bytes > limit) {
        throw new StreamLimitExceededError({
          kind: 'tool_call_args',
          limit,
          observed: bytes,
          toolName: sealedName,
        });
      }
      continue;
    }
    const key = `${generationKey}:${chunk.index ?? chunk.id ?? `#${i}`}`;
    const sealed = sealsChunk(seal, chunk);
    let tally = tallies.get(key);
    /** Secondary key this call is also reachable under, so a later delta
     * that drops an identifier still lands on the same tally: the
     * batch-position fallback for id-only chunks (an index-less adapter can
     * put the id only on the first chunk), or the id itself for chunks
     * carrying both identifiers (an adapter can drop the index later). */
    let chunkAliasKey: string | undefined;
    if (chunk.id != null) {
      chunkAliasKey =
        chunk.index == null
          ? `${generationKey}:#${i}`
          : `${generationKey}:${chunk.id}`;
    }
    const registerAliasKey = (target: StreamedToolCallArgTally): void => {
      if (chunkAliasKey == null) {
        return;
      }
      const currentOwner = tallies.get(chunkAliasKey);
      if (currentOwner === target) {
        return;
      }
      /** Parallel index-less calls all land on batch position #i, so the
       * newest live call takes the alias over and the previous owner is
       * disowned — its seal must not delete an alias it no longer holds. */
      if (currentOwner?.aliasKey === chunkAliasKey) {
        currentOwner.aliasKey = undefined;
      }
      tallies.set(chunkAliasKey, target);
      target.aliasKey = chunkAliasKey;
    };
    const releaseTally = (target: StreamedToolCallArgTally): void => {
      tallies.delete(key);
      if (
        target.aliasKey != null &&
        tallies.get(target.aliasKey) === target
      ) {
        tallies.delete(target.aliasKey);
      }
    };
    if (!hasArgs) {
      if (tally == null) {
        if (!sealed) {
          tally = { bytes: 0, name: chunkToolName(chunk) };
          tallies.set(key, tally);
          registerAliasKey(tally);
        }
        continue;
      }
      /** A sealing chunk is about to release this tally; taking the position
       * alias here would steal it from a still-live parallel call. */
      if (!sealed) {
        registerAliasKey(tally);
      }
      if (tally.name == null) {
        const lateName = chunkToolName(chunk);
        if (lateName != null) {
          /** Bytes tallied before the name arrived were only held against the
           * global cap; a newly applicable per-tool override must re-judge
           * them, including on a sealing chunk about to release the tally. */
          tally.name = lateName;
          const limit =
            byTool != null && Object.hasOwn(byTool, lateName)
              ? byTool[lateName]
              : globalLimit;
          if (limit > 0 && tally.bytes > limit) {
            throw new StreamLimitExceededError({
              kind: 'tool_call_args',
              limit,
              observed: tally.bytes,
              toolName: lateName,
            });
          }
        }
      }
      if (sealed) {
        releaseTally(tally);
      }
      continue;
    }
    if (tally == null) {
      tally = { bytes: 0 };
      tallies.set(key, tally);
    }
    if (!sealed) {
      registerAliasKey(tally);
    }
    if (tally.name == null) {
      tally.name = chunkToolName(chunk);
    }
    const argBytes = Buffer.byteLength(args, 'utf8');
    if (sealed) {
      tally.bytes = argBytes;
    } else {
      const reconcilesSplitPair =
        tally.pendingHighSurrogate === true &&
        isLowSurrogate(args.charCodeAt(0));
      tally.bytes += reconcilesSplitPair ? argBytes - 2 : argBytes;
    }
    tally.pendingHighSurrogate =
      !sealed && isHighSurrogate(args.charCodeAt(args.length - 1));
    const toolName = tally.name ?? chunkToolName(chunk);
    const limit =
      byTool != null && toolName != null && Object.hasOwn(byTool, toolName)
        ? byTool[toolName]
        : globalLimit;
    if (limit > 0 && tally.bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: tally.bytes,
        toolName,
      });
    }
    if (sealed) {
      releaseTally(tally);
    }
  }
}

/** Structural subset of a complete parsed tool call. */
interface CompleteToolCallLike {
  name?: string;
  args?: unknown;
}

/**
 * Standalone byte check for complete parsed tool calls that arrive without a
 * raw chunk representation: a streaming custom or OpenAI-compatible
 * `ChatModel` can yield fully parsed `tool_calls` with empty
 * `tool_call_chunks`, which would otherwise dispatch without consuming any
 * byte budget. Complete calls are self-contained, so each is judged
 * standalone without tallying.
 */
export function enforceCompleteToolCallArgLimit({
  graph,
  metadata,
  toolCalls,
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  toolCalls: CompleteToolCallLike[];
}): void {
  const resolved = graph.streamLimits ?? DEFAULT_RESOLVED_LIMITS;
  const globalLimit = resolved.maxToolCallArgBytes;
  const byTool = resolved.maxToolCallArgBytesByTool;
  if (globalLimit <= 0 && byTool == null) {
    return;
  }
  if (metadata?.[STREAM_LIMIT_REDISPATCH_KEY] === true) {
    return;
  }
  for (const toolCall of toolCalls) {
    const name =
      toolCall.name != null && toolCall.name !== '' ? toolCall.name : undefined;
    const limit =
      byTool != null && name != null && Object.hasOwn(byTool, name)
        ? byTool[name]
        : globalLimit;
    if (limit <= 0) {
      continue;
    }
    const args = toolCall.args;
    let serialized: string;
    if (typeof args === 'string') {
      serialized = args;
    } else if (args == null) {
      serialized = '';
    } else {
      serialized = JSON.stringify(args) ?? '';
    }
    const bytes = Buffer.byteLength(serialized, 'utf8');
    if (bytes > limit) {
      throw new StreamLimitExceededError({
        kind: 'tool_call_args',
        limit,
        observed: bytes,
        toolName: name,
      });
    }
  }
}

/**
 * Claims accounting ownership of one EMISSION of a wire chunk. LangChain can
 * hand the same chunk object to the decoupled `streamEvents` handler
 * (`consumer`) and to the dispatch loop (`producer`) in either order, and a
 * streaming model may mutate and re-yield the same object across emissions —
 * so dedup is a signed credit balance per object rather than a lifetime set.
 * Each producer visit adds a credit, each consumer visit removes one, and a
 * visit charges only when the other side has not already charged that
 * emission (positive balance = producer ahead, negative = consumer ahead).
 * Paths where only one side ever observes the chunk (summarization, local
 * replay-skip) charge every visit, since their balance never crosses zero
 * the other way. Balances are scoped by generation identity so parallel
 * generations sharing one reused chunk object cannot cancel each other's
 * charges. Non-object chunks cannot be identity-tracked and are always
 * claimable.
 */
export function claimStreamLimitCharge(
  graph: StreamLimitState,
  chunk: unknown,
  side: 'producer' | 'consumer',
  metadata: Record<string, unknown> | undefined
): boolean {
  if (typeof chunk !== 'object' || chunk == null) {
    return true;
  }
  const credits = (graph.streamLimitChargeCredits ??= new WeakMap());
  let byGeneration = credits.get(chunk);
  if (byGeneration == null) {
    byGeneration = new Map();
    credits.set(chunk, byGeneration);
  }
  const generationKey = resolveGenerationKey(metadata);
  const balance = byGeneration.get(generationKey) ?? 0;
  if (side === 'producer') {
    byGeneration.set(generationKey, balance + 1);
    return balance >= 0;
  }
  byGeneration.set(generationKey, balance - 1);
  return balance <= 0;
}

/**
 * Synchronous producer-side accounting for wire chunks that would otherwise
 * be judged only when the decoupled `streamEvents` reader catches up — or,
 * on replay-skipped and summarization chunks, not at all. A lagging reader
 * would let an oversized complete call return to LangGraph and reach
 * `ToolNode` before the queued handler throws; charging in the producer
 * loop keeps the breaker ahead of graph progression. Claim-based, so
 * whichever of this path and the handler echo sees the chunk object first
 * charges it and the other skips.
 */
export function enforceStreamLimitsForWireChunk({
  graph,
  metadata,
  chunk,
  side = 'producer',
}: {
  graph: StreamLimitState;
  metadata: Record<string, unknown> | undefined;
  chunk: {
    tool_call_chunks?: ToolCallChunk[];
    tool_calls?: CompleteToolCallLike[];
    response_metadata?: Record<string, unknown>;
  };
  /** Claim side for the credit balance. The local dispatch branch charges
   * as `consumer` because its handler-handled and replay-skipped emissions
   * of one reused object ALTERNATE — mixed sides would pair them as
   * producer/echo and swallow a charge. */
  side?: 'producer' | 'consumer';
}): void {
  if (!claimStreamLimitCharge(graph, chunk, side, metadata)) {
    return;
  }
  enforceStreamDeltaEventLimit({ graph, metadata });
  if (chunk.tool_call_chunks != null && chunk.tool_call_chunks.length > 0) {
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: chunk.tool_call_chunks,
      responseMetadata: chunk.response_metadata,
    });
  }
  /** Judged whenever parsed calls are present, not only when raw chunks are
   * absent: an adapter can pair an empty or partial raw chunk with a
   * complete parsed call, and the standalone check is stateless so the
   * common both-present case is not double-tallied. */
  if (chunk.tool_calls != null && chunk.tool_calls.length > 0) {
    enforceCompleteToolCallArgLimit({ graph, metadata, toolCalls: chunk.tool_calls });
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
