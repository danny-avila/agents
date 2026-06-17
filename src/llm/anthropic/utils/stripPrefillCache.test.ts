import type { AnthropicMessageCreateParams } from '../types';
import { stripUnsupportedAssistantPrefill } from './message_inputs';

/**
 * When a model disallows assistant prefill (Claude 4.6+), the trailing
 * assistant message is stripped right before the API call. If the single tail
 * prompt-cache breakpoint rode that assistant prefill, the survivors would lose
 * their only message-level `cache_control` — so the strip must re-anchor the
 * breakpoint onto the new tail.
 */

type Msgs = AnthropicMessageCreateParams['messages'];

function cacheControlBlocks(messages: Msgs): number {
  let n = 0;
  for (const m of messages) {
    if (!Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if ('cache_control' in b) n++;
    }
  }
  return n;
}

describe('stripUnsupportedAssistantPrefill — cache re-anchoring', () => {
  test('re-anchors the breakpoint onto the new tail when the prefill carried it', () => {
    const request = {
      model: 'claude-opus-4-6',
      max_tokens: 100,
      messages: [
        {
          role: 'user' as const,
          content: [{ type: 'text' as const, text: 'q' }],
        },
        {
          role: 'assistant' as const,
          content: [
            {
              type: 'text' as const,
              text: 'prefill',
              cache_control: { type: 'ephemeral' as const },
            },
          ],
        },
      ],
    };

    const out = stripUnsupportedAssistantPrefill(request);

    // Prefill removed, and exactly one breakpoint survives — on the new tail.
    expect(out.messages).toHaveLength(1);
    expect(out.messages[0].role).toBe('user');
    expect(cacheControlBlocks(out.messages)).toBe(1);
    const tail = out.messages[0].content as Array<{ cache_control?: unknown }>;
    expect(tail[tail.length - 1].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('does not add a breakpoint when caching was off (no marker present)', () => {
    const request = {
      model: 'claude-opus-4-6',
      max_tokens: 100,
      messages: [
        { role: 'user' as const, content: 'q' },
        { role: 'assistant' as const, content: 'prefill' },
      ],
    };

    const out = stripUnsupportedAssistantPrefill(request);

    expect(out.messages).toHaveLength(1);
    expect(cacheControlBlocks(out.messages)).toBe(0);
  });

  test('leaves a surviving breakpoint untouched (no double-anchor)', () => {
    const request = {
      model: 'claude-opus-4-6',
      max_tokens: 100,
      messages: [
        {
          role: 'user' as const,
          content: [
            {
              type: 'text' as const,
              text: 'q',
              cache_control: { type: 'ephemeral' as const },
            },
          ],
        },
        { role: 'assistant' as const, content: 'prefill' },
      ],
    };

    const out = stripUnsupportedAssistantPrefill(request);

    expect(out.messages).toHaveLength(1);
    expect(cacheControlBlocks(out.messages)).toBe(1);
  });

  test('older models keep the assistant prefill (no strip, no re-anchor)', () => {
    const request = {
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 100,
      messages: [
        { role: 'user' as const, content: 'q' },
        { role: 'assistant' as const, content: '{' },
      ],
    };

    expect(stripUnsupportedAssistantPrefill(request)).toBe(request);
  });
});
