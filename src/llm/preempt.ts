// src/llm/preempt.ts
import type { AIMessageChunk } from '@langchain/core/messages';
import { ContentTypes, DEFAULT_MAX_SEALS } from '@/common';

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
