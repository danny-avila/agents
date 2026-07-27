// src/llm/preempt.ts
import type { AIMessageChunk } from '@langchain/core/messages';
import { ContentTypes } from '@/common';

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
  if ((chunk.tool_calls?.length ?? 0) > 0) {
    return false;
  }
  if ((chunk.tool_call_chunks?.length ?? 0) > 0) {
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
