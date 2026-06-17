import {
  HumanMessage,
  AIMessage,
  ToolMessage,
  type BaseMessage,
  type MessageContentComplex,
} from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { ensureThinkingBlockInMessages } from './format';
import { toLangChainContent } from './langchain';
import { addTailCacheControl } from './cache';
import { Providers } from '@/common';

/**
 * Regression coverage for the single tail prompt-cache breakpoint surviving all
 * the way into the final Anthropic payload — i.e. the marker must land on a
 * block that actually ships, not one that downstream conversion / folding
 * removes. Two ways the breakpoint was silently lost:
 *
 *  - Foreign reasoning tail: addTailCacheControl anchored on a
 *    `reasoning_content`/`reasoning`/`think` block, which the Anthropic
 *    converter drops on assistant turns (cross-provider handoff).
 *  - Thinking-fold ordering: marking before ensureThinkingBlockInMessages let
 *    the fold rewrite the anchored AI→Tool tail into a `[Previous agent
 *    context]` HumanMessage that copies text but not cache_control.
 */

type PayloadMessage = { content: unknown };

function hasCacheControl(block: unknown): boolean {
  return (
    typeof block === 'object' && block !== null && 'cache_control' in block
  );
}

/** Does any block (top-level or nested in tool_result) carry cache_control? */
function breakpointSurvives(messages: PayloadMessage[]): boolean {
  for (const m of messages) {
    if (!Array.isArray(m.content)) {
      continue;
    }
    for (const block of m.content as unknown[]) {
      if (hasCacheControl(block)) {
        return true;
      }
      const inner = (block as { content?: unknown }).content;
      if (Array.isArray(inner) && inner.some(hasCacheControl)) {
        return true;
      }
    }
  }
  return false;
}

describe('tail breakpoint survives Anthropic conversion', () => {
  test('foreign reasoning tail keeps a usable breakpoint (anchored on text)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hello'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Here is my answer.' },
          { type: 'reasoning_content', reasoningText: { text: 'r' } },
        ] as MessageContentComplex[]),
      }),
    ];

    const payload = _convertMessagesToAnthropicPayload(
      addTailCacheControl(messages)
    );

    expect(breakpointSurvives(payload.messages as PayloadMessage[])).toBe(true);
  });

  test('string tool-result tail keeps a usable breakpoint', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('run it'),
      new AIMessage({
        content: 'calling',
        tool_calls: [{ id: 't1', name: 'search', args: {} }],
      }),
      new ToolMessage({ tool_call_id: 't1', content: 'result body' }),
    ];

    const payload = _convertMessagesToAnthropicPayload(
      addTailCacheControl(messages)
    );

    expect(breakpointSurvives(payload.messages as PayloadMessage[])).toBe(true);
  });

  test('marking AFTER the thinking fold preserves the breakpoint (Graph order)', () => {
    // A historical non-thinking AI→Tool chain at the tail (no trailing human).
    const messages: BaseMessage[] = [
      new HumanMessage('do the thing'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 't1', name: 'search', args: { q: 'x' } }],
      }),
      new ToolMessage({ tool_call_id: 't1', content: 'tool output text' }),
    ];

    // Graph applies the fold first, THEN the tail marker.
    const folded = ensureThinkingBlockInMessages(
      messages,
      Providers.ANTHROPIC,
      undefined,
      messages.length
    );
    const payload = _convertMessagesToAnthropicPayload(
      addTailCacheControl(folded)
    );

    expect(breakpointSurvives(payload.messages as PayloadMessage[])).toBe(true);
  });

  test('marking BEFORE the fold loses the breakpoint (guards the ordering)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('do the thing'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 't1', name: 'search', args: { q: 'x' } }],
      }),
      new ToolMessage({ tool_call_id: 't1', content: 'tool output text' }),
    ];

    // The buggy order: mark first, then fold drops the marker.
    const marked = addTailCacheControl(messages);
    const folded = ensureThinkingBlockInMessages(
      marked,
      Providers.ANTHROPIC,
      undefined,
      messages.length
    );
    const payload = _convertMessagesToAnthropicPayload(folded);

    expect(breakpointSurvives(payload.messages as PayloadMessage[])).toBe(
      false
    );
  });
});
