import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicMessages } from '@/types/messages';
import { addTailCacheControl, addBedrockTailCacheControl } from './cache';
import { toLangChainContent } from './langchain';

type CacheControlBlock = MessageContentComplex & {
  cache_control?: { type: 'ephemeral'; ttl?: '1h' };
};

/** Count every block across all messages that carries a cache_control marker. */
function countCacheMarkers(
  messages: ReadonlyArray<{ content: unknown }>
): number {
  let count = 0;
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (block && typeof block === 'object' && 'cache_control' in block) {
        count++;
      }
    }
  }
  return count;
}

function blocksOf(message: { content: unknown }): CacheControlBlock[] {
  return message.content as CacheControlBlock[];
}

describe('addTailCacheControl (single tail breakpoint)', () => {
  test('places exactly one marker on the last message', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Hi there' }] },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Doing well' }] },
      { role: 'user', content: [{ type: 'text', text: 'Great!' }] },
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(
      (result[4].content[0] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
    expect(result[2].content[0]).not.toHaveProperty('cache_control');
  });

  test('anchors on a trailing tool_result block (tail is a tool turn)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Run the tool'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'Calling it' },
          { type: 'tool_use', id: 't1', name: 'search', input: {} },
        ] as MessageContentComplex[]),
        tool_calls: [{ id: 't1', name: 'search', args: {} }],
      }),
      new ToolMessage({
        tool_call_id: 't1',
        content: toLangChainContent([
          {
            type: 'tool_result',
            tool_use_id: 't1',
            content: 'result body',
          },
        ] as MessageContentComplex[]),
      }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[2])[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('strips ALL stale markers and re-anchors a single one at the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'old marker',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new HumanMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'another old marker',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new AIMessage({ content: 'reply' }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[2])[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocksOf(result[0])[0]).not.toHaveProperty('cache_control');
    expect(blocksOf(result[1])[0]).not.toHaveProperty('cache_control');
  });

  test('does not anchor on thinking blocks', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('Hi'),
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'thought through it' },
          { type: 'thinking', thinking: 'secret reasoning' },
        ] as MessageContentComplex[]),
      }),
    ];

    const result = addTailCacheControl(messages);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[1])[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(blocksOf(result[1])[1]).not.toHaveProperty('cache_control');
  });

  test.each(['reasoning_content', 'reasoning', 'think'])(
    'does not anchor on a trailing foreign reasoning block (%s)',
    (reasoningType) => {
      // Foreign reasoning (Bedrock/Google/LibreChat) is dropped by the
      // Anthropic converter on assistant turns; anchoring the only breakpoint
      // there would silently lose tail caching on a cross-provider handoff.
      const messages: BaseMessage[] = [
        new HumanMessage('Hi'),
        new AIMessage({
          content: toLangChainContent([
            { type: 'text', text: 'Here is my answer.' },
            { type: reasoningType, text: 'foreign reasoning' },
          ] as MessageContentComplex[]),
        }),
      ];

      const result = addTailCacheControl(messages);

      expect(countCacheMarkers(result)).toBe(1);
      // Marker must land on the surviving text block, not the reasoning block.
      expect(blocksOf(result[1])[0].cache_control).toEqual({
        type: 'ephemeral',
      });
      expect(blocksOf(result[1])[1]).not.toHaveProperty('cache_control');
    }
  );

  test('skips synthetic meta tail and anchors on the previous real message', () => {
    const realTail = new AIMessage({ content: 'real answer' });
    const metaTail = new HumanMessage({ content: 'reinjected skill body' });
    (
      metaTail as unknown as { additional_kwargs: Record<string, unknown> }
    ).additional_kwargs = { isMeta: true };

    const result = addTailCacheControl([
      new HumanMessage({ content: 'question' }),
      realTail,
      metaTail,
    ]);

    expect(countCacheMarkers(result)).toBe(1);
    expect(blocksOf(result[1])[0].cache_control).toEqual({ type: 'ephemeral' });
  });

  test('handles string content on the tail', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Final' },
    ];

    const result = addTailCacheControl(messages);

    expect(result[0].content).toBe('Hello');
    expect(result[1].content[0]).toEqual({
      type: 'text',
      text: 'Final',
      cache_control: { type: 'ephemeral' },
    });
  });

  test('does not mutate the original messages', () => {
    const original: AnthropicMessages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'World' }] },
    ];

    addTailCacheControl(original);

    expect(original[1].content[0]).not.toHaveProperty('cache_control');
  });

  test('returns input unchanged for empty array', () => {
    const messages: AnthropicMessages = [];
    expect(addTailCacheControl(messages)).toEqual([]);
  });
});

/** Count every Bedrock cachePoint block across all messages. */
function countCachePoints(
  messages: ReadonlyArray<{ content: unknown }>
): number {
  let count = 0;
  for (const message of messages) {
    if (!Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (block && typeof block === 'object' && 'cachePoint' in block) {
        count++;
      }
    }
  }
  return count;
}

describe('addBedrockTailCacheControl (single tail cachePoint)', () => {
  test('inserts exactly one cachePoint after the last text block of the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('First question'),
      new AIMessage('First answer'),
      new HumanMessage('Second question'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(countCachePoints(result)).toBe(1);
    const tail = blocksOf(result[2]);
    expect(tail[tail.length - 1]).toEqual({ cachePoint: { type: 'default' } });
  });

  test('strips stale cachePoints and re-anchors a single one at the tail', () => {
    const messages: BaseMessage[] = [
      new HumanMessage({
        content: toLangChainContent([
          { type: 'text', text: 'old' },
          { cachePoint: { type: 'default' } },
        ] as MessageContentComplex[]),
      }),
      new AIMessage('reply'),
      new HumanMessage('newest'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(countCachePoints(result)).toBe(1);
    const tail = blocksOf(result[2]);
    expect(tail[tail.length - 1]).toEqual({ cachePoint: { type: 'default' } });
    expect(blocksOf(result[0]).some((b) => 'cachePoint' in b)).toBe(false);
  });

  test('strips Anthropic cache_control from a system message but never anchors it', () => {
    const messages: BaseMessage[] = [
      new SystemMessage({
        content: toLangChainContent([
          {
            type: 'text',
            text: 'system rules',
            cache_control: { type: 'ephemeral' },
          },
        ] as MessageContentComplex[]),
      }),
      new HumanMessage('hi'),
    ];

    const result = addBedrockTailCacheControl(messages);

    expect(blocksOf(result[0])[0]).not.toHaveProperty('cache_control');
    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[1])[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  test('skips synthetic meta tail and anchors on the previous real message', () => {
    const metaTail = new HumanMessage({ content: 'reinjected skill body' });
    (
      metaTail as unknown as { additional_kwargs: Record<string, unknown> }
    ).additional_kwargs = { source: 'skill' };

    const result = addBedrockTailCacheControl([
      new HumanMessage('question'),
      new AIMessage('real answer'),
      metaTail,
    ]);

    expect(countCachePoints(result)).toBe(1);
    const realTail = blocksOf(result[1]);
    expect(realTail[realTail.length - 1]).toEqual({
      cachePoint: { type: 'default' },
    });
  });

  test('handles string content on the tail', () => {
    const result = addBedrockTailCacheControl([
      new HumanMessage('Hello'),
      new AIMessage('Final'),
    ]);

    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[1])).toEqual([
      { type: 'text', text: 'Final' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  test('anchors on a trailing string tool result (agent-loop tail)', () => {
    const result = addBedrockTailCacheControl([
      new HumanMessage('Run the tool'),
      new AIMessage({
        content: 'Calling it',
        tool_calls: [{ id: 't1', name: 'search', args: {} }],
      }),
      new ToolMessage({ tool_call_id: 't1', content: 'result body' }),
    ]);

    // The single cachePoint must land on the trailing tool result so the
    // tool output is part of the cached prefix; the converter later hoists it
    // out of toolResult.content (see toolResultCachePoint.test.ts).
    expect(countCachePoints(result)).toBe(1);
    expect(blocksOf(result[2])).toEqual([
      { type: 'text', text: 'result body' },
      { cachePoint: { type: 'default' } },
    ]);
  });
});
