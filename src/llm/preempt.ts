// src/llm/preempt.ts
import type { AIMessageChunk } from '@langchain/core/messages';
import {
  ContentTypes,
  DEFAULT_MAX_SEALS,
  DEFAULT_PREEMPT_RESTART_GRACE_MS,
} from '@/common';
import { isReasoningContentBlock } from '@/messages/core';

/**
 * Normalizes a host-supplied seal budget.
 *
 * Read in two places that interpret it differently — a numeric comparison in
 * the seal gate and an addition into the recursion limit — so a value like
 * `1.5` would permit two seals while reserving fractional headroom, `NaN`
 * would poison the recursion limit outright, and `Infinity` would remove both
 * bounds at once. Normalizing once keeps the two readings in agreement.
 *
 * `0` is honored as a deliberate "never seal"; anything not finite falls back
 * to the default rather than silently disabling the feature.
 */
export function resolveMaxSeals(maxSeals: number | undefined): number {
  if (maxSeals == null || !Number.isFinite(maxSeals)) {
    return DEFAULT_MAX_SEALS;
  }
  const whole = Math.floor(maxSeals);
  return whole > 0 ? whole : 0;
}

/**
 * True when the accumulated content carries at least one non-whitespace text
 * block. `thinking` / `reasoning` blocks deliberately do not count: they are
 * stripped or signed on several providers and cannot stand in for the visible
 * assistant turn a sealed sequence needs.
 */
function hasNonEmptyTextContent(content: AIMessageChunk['content']): boolean {
  if (typeof content === 'string') {
    return content.trim() !== '';
  }
  for (const block of content) {
    if (block.type !== ContentTypes.TEXT) {
      continue;
    }
    const text = block[ContentTypes.TEXT];
    if (typeof text === 'string' && text.trim() !== '') {
      return true;
    }
  }
  return false;
}

/**
 * True while a Gemini server-side tool call is still awaiting its response.
 *
 * Google's server-side tools (Search, URL context) never populate
 * `tool_calls` or `tool_call_chunks` — those are derived from `functionCall`
 * parts only, while a server-side invocation arrives as a `toolCall` CONTENT
 * block and its result as a later `toolResponse` block. The tool-call gates
 * cannot see it, so sealing between the two would replay a model turn holding
 * an unanswered server-side call.
 *
 * Counted rather than id-matched: Google answers calls in order, and counting
 * stays correct when a part omits its id.
 */
function hasOpenGoogleServerToolCall(
  content: AIMessageChunk['content']
): boolean {
  if (typeof content === 'string') {
    return false;
  }
  let calls = 0;
  let responses = 0;
  for (const block of content) {
    if (block.type === 'toolCall') {
      calls += 1;
    } else if (block.type === 'toolResponse') {
      responses += 1;
    }
  }
  return calls > responses;
}

/**
 * Ids of Anthropic server-side tool calls whose paired result block is already
 * on the accumulated content. Those calls are answered and cannot be orphaned
 * by a seal.
 */
function settledServerToolCallIds(
  content: AIMessageChunk['content']
): Set<string> {
  const settled = new Set<string>();
  if (typeof content === 'string') {
    return settled;
  }
  for (const block of content) {
    if (block.type !== 'web_search_tool_result') {
      continue;
    }
    const id = block.tool_use_id;
    if (typeof id === 'string') {
      settled.add(id);
    }
  }
  return settled;
}

/** Tool calls still awaiting an answer, ignoring ones already settled. */
function countOpenToolCalls(
  calls: ReadonlyArray<{ id?: string }> | undefined,
  settled: Set<string>
): number {
  if (calls == null || calls.length === 0) {
    return 0;
  }
  if (settled.size === 0) {
    return calls.length;
  }
  let open = 0;
  for (const call of calls) {
    if (call.id == null || !settled.has(call.id)) {
      open += 1;
    }
  }
  return open;
}

/**
 * Cooperative mid-generation seal gate. Returns true ONLY when sealing here
 * yields a message sequence valid on EVERY supported provider:
 *  - non-whitespace TEXT content, so the FIRST injected user turn is preceded
 *    by a non-empty assistant turn — no empty-content 400s. Note this says
 *    nothing about adjacency AMONG several injected turns: a boundary that
 *    drains two steers emits two consecutive user messages, which strict
 *    providers reject. That is normalized at the provider-facing hop by
 *    `coalesceAdjacentUserTurns`, not here;
 *  - no tool call in flight, so no `tool_use` can be orphaned AND no eagerly
 *    prestarted execution can be stripped out from under the model.
 *
 * Anthropic's server-side tools need no check of their own. Every
 * `server_tool_use` content block also emits a `tool_call_chunk`
 * (`_makeMessageChunkFromAnthropicEvent`), and `concat` keeps that chunk on
 * the accumulated message for the remainder of the turn, so the tool-call
 * gates below already cover it. The practical consequence is worth stating
 * plainly: once a turn starts a web search it is no longer preemptible, and
 * a queued message waits for the ordinary tool boundary instead.
 *
 * Nothing is stripped and nothing is repaired: when the accumulated shape is
 * not already safe the stream simply runs on, and whatever the host queued
 * lands at the next tool boundary instead. Chunks accumulate monotonically
 * through `concat`, so an unsafe shape can never be observed at a seal point.
 */
export function canSealPreempt(chunk: AIMessageChunk | undefined): boolean {
  if (chunk == null) {
    return false;
  }
  /**
   * Anthropic's server-side tools run inside the provider, so unlike a client
   * tool they never reach `ToolNode` and never produce a `PostToolBatch`
   * boundary. `concat` also keeps their `server_tool_use` chunk on the
   * accumulated message for the rest of the turn. Together that means a naive
   * tool-call gate makes a turn permanently unsealable the moment a web
   * search starts, with no later boundary to drain into — the queued message
   * waits for the whole turn to finish rather than for the next tool step.
   *
   * So calls whose paired result block has already landed are treated as
   * settled: they cannot be orphaned, because the provider has already
   * answered them.
   */
  const settled = settledServerToolCallIds(chunk.content);
  if (countOpenToolCalls(chunk.tool_calls, settled) > 0) {
    return false;
  }
  if (countOpenToolCalls(chunk.tool_call_chunks, settled) > 0) {
    return false;
  }
  if ((chunk.invalid_tool_calls?.length ?? 0) > 0) {
    return false;
  }
  if (hasOpenGoogleServerToolCall(chunk.content)) {
    return false;
  }
  return hasNonEmptyTextContent(chunk.content);
}

/**
 * What a host's armed preempt request may do to the turn in flight.
 *
 * `seal` keeps the partial assistant turn and injects after it; `restart`
 * throws the partial away and re-issues the model call with the injected turn
 * appended to the prompt. They are not two flavors of the same move — one
 * preserves work, the other deliberately discards it — so the decision is
 * resolved once, here, rather than re-derived at each trigger.
 */
export type PreemptAction = 'none' | 'seal' | 'restart';

/**
 * True when the accumulated turn holds nothing a seal would have to preserve,
 * so the whole model call can be thrown away and re-issued instead.
 *
 * The membership test is a WHITELIST: every content block must be a known
 * reasoning block, and an unrecognized block refuses the restart. Blacklisting
 * would silently discard whatever a future provider adds, and the failure is
 * asymmetric — refusing costs the user a slower interrupt (exactly today's
 * behavior), while wrongly discarding destroys work the model already did and
 * the host may already have rendered.
 *
 * Tool machinery of any kind refuses, and deliberately without the settled-id
 * allowance {@link canSealPreempt} makes: a settled `web_search_tool_result`
 * means the provider already ran and billed a search, and re-issuing the call
 * would run it again. A seal there is free; a restart is not.
 *
 * An accumulation that carries visible text is not handled here at all — the
 * caller resolves `seal` first, because keeping the user's answer always beats
 * discarding it.
 *
 * `undefined` — the provider has sent nothing at all — is the case this whole
 * path exists for: it is the silent window between the request and the first
 * chunk, where there is by definition nothing to lose.
 */
export function canRestartPreempt(chunk: AIMessageChunk | undefined): boolean {
  if (chunk == null) {
    return true;
  }
  if ((chunk.tool_calls?.length ?? 0) > 0) {
    return false;
  }
  if ((chunk.tool_call_chunks?.length ?? 0) > 0) {
    return false;
  }
  if ((chunk.invalid_tool_calls?.length ?? 0) > 0) {
    return false;
  }
  const { content } = chunk;
  if (typeof content === 'string') {
    return content.trim() === '';
  }
  for (const block of content) {
    if (!isReasoningContentBlock(block)) {
      return false;
    }
  }
  return true;
}

/**
 * Normalizes a host-supplied restart grace, on the same rules as
 * {@link resolveMaxSeals}: `0` is honored as "never wait", anything not finite
 * falls back to the default. Negative values collapse to `0` rather than
 * inverting the comparison in {@link resolvePreemptAction}.
 */
export function resolveRestartGraceMs(graceMs: number | undefined): number {
  if (graceMs == null || !Number.isFinite(graceMs)) {
    return DEFAULT_PREEMPT_RESTART_GRACE_MS;
  }
  return graceMs > 0 ? graceMs : 0;
}

/**
 * The single decision point both preempt triggers share: the per-chunk poll in
 * the stream loop, and the host wake-up that fires while the provider is
 * silent.
 *
 * A SEAL is preferred wherever one is available, so a turn that has already
 * produced an answer never loses it to a restart. A RESTART converts only once
 * the request has outlived `graceMs`.
 *
 * The window is not politeness, it is correctness, and it applies to a silent
 * provider exactly as it does to a reasoning one:
 *  - reasoning usually precedes text by moments, and discarding a turn that
 *    was about to become sealable trades a kept answer for a re-issued
 *    request. Only a genuinely long thinking stretch — the one an interrupt
 *    can otherwise wait out entirely — should convert.
 *  - `chunk` is what the CONSUMER has accumulated, and the provider stream
 *    buffers a chunk ahead of it. An empty accumulation therefore does not
 *    prove the provider produced nothing; it can also mean the first chunk is
 *    in flight. Converting on emptiness alone would discard that chunk — text
 *    included — on a race no caller can see. A provider that is still silent a
 *    window later has no such chunk outstanding.
 *
 * Non-mutating and allocation-free, like the poll that guards it: the seal
 * budget is only spent once the caller acts on a non-`none` result.
 */
export function resolvePreemptAction({
  chunk,
  requestAgeMs,
  graceMs,
}: {
  chunk: AIMessageChunk | undefined;
  requestAgeMs: number;
  graceMs: number;
}): PreemptAction {
  if (canSealPreempt(chunk)) {
    return 'seal';
  }
  if (!canRestartPreempt(chunk)) {
    return 'none';
  }
  return requestAgeMs >= graceMs ? 'restart' : 'none';
}

/**
 * How long a restarted model run stays recorded when nothing consumes it.
 *
 * The consumer is LangChain's error callback, which is dispatched on a
 * non-awaited queue, so a marker may legitimately sit unread for a moment. A
 * COUNT cap cannot tell that apart from a leak: a burst of concurrent restarts
 * would evict markers whose callbacks were still queued, and each eviction
 * turns an expected restart back into an error in the host's traces — the very
 * thing the marker exists to prevent. Age can tell them apart, so the bound is
 * time.
 */
const PREEMPT_RESTARTED_RUN_TTL_MS = 60_000;

/**
 * Model runs whose provider stream was torn down for a restart, with the
 * accumulated turn so the close can carry its usage.
 *
 * Recorded rather than inferred from the thrown error: aborting makes the
 * provider adapter raise whatever IT raises for cancellation, and matching on
 * that shape would bind the tracing layer to per-provider error identity. The
 * run id is unambiguous and already captured for the seal path.
 *
 * The message rides along because the run closes through the error path, which
 * carries no output: without it a discarded attempt would report no usage at
 * all, and the reasoning tokens the provider already billed would vanish from
 * cost accounting. The later synthetic `CHAT_MODEL_END` cannot repair that —
 * by then the generation is closed.
 *
 * Insertion-ordered, so expiry sweeps from the front and stops at the first
 * live entry.
 */
const preemptRestartedRuns = new Map<string, PreemptRestartedRun>();

/** A torn-down model run awaiting its tracing close. */
export interface PreemptRestartedRun {
  /** The accumulated turn, carrying whatever usage was resolved for it. */
  message: AIMessageChunk;
  recordedAt: number;
}

/**
 * Armed while anything is recorded, so expiry does not depend on another
 * restart arriving to trigger it. Without this the final burst before a quiet
 * period would hold its messages for the life of the process — and a host that
 * enables preemption without tracing never consumes a single record, so that
 * burst is the common case, not the corner one.
 *
 * `unref`'d: reclaiming a few messages is never a reason to keep a process
 * alive.
 */
let sweepTimer: ReturnType<typeof setTimeout> | undefined;

function sweepExpiredRestartedRuns(now: number): void {
  for (const [runId, record] of preemptRestartedRuns) {
    if (now - record.recordedAt < PREEMPT_RESTARTED_RUN_TTL_MS) {
      break;
    }
    preemptRestartedRuns.delete(runId);
  }
  if (preemptRestartedRuns.size === 0) {
    if (sweepTimer != null) {
      clearTimeout(sweepTimer);
      sweepTimer = undefined;
    }
    return;
  }
  if (sweepTimer != null) {
    return;
  }
  sweepTimer = setTimeout(() => {
    sweepTimer = undefined;
    sweepExpiredRestartedRuns(Date.now());
  }, PREEMPT_RESTARTED_RUN_TTL_MS);
  sweepTimer.unref();
}

/** Records a model run whose stream was torn down for a restart. */
export function notePreemptRestartedRun(
  runId: string,
  message: AIMessageChunk
): void {
  const now = Date.now();
  preemptRestartedRuns.set(runId, { message, recordedAt: now });
  sweepExpiredRestartedRuns(now);
}

/**
 * The record for a run that ended because a preempt discarded it, or
 * `undefined`. Consuming: the run closes exactly once, and a stale id must not
 * reclassify a later genuine failure on a reused id.
 */
export function consumePreemptRestartedRun(
  runId: string
): PreemptRestartedRun | undefined {
  const record = preemptRestartedRuns.get(runId);
  if (record == null) {
    return undefined;
  }
  preemptRestartedRuns.delete(runId);
  if (preemptRestartedRuns.size === 0 && sweepTimer != null) {
    clearTimeout(sweepTimer);
    sweepTimer = undefined;
  }
  return record;
}
