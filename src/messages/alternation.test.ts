import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { Providers } from '@/common';
import {
  coalesceAdjacentUserTurns,
  strictAlternationProviders,
} from './alternation';

describe('strictAlternationProviders', () => {
  it('covers the providers that reject consecutive user turns', () => {
    expect(strictAlternationProviders.has(Providers.BEDROCK)).toBe(true);
    expect(strictAlternationProviders.has(Providers.MISTRAL)).toBe(true);
    expect(strictAlternationProviders.has(Providers.MISTRALAI)).toBe(true);
  });

  it('leaves providers that accept them alone', () => {
    expect(strictAlternationProviders.has(Providers.ANTHROPIC)).toBe(false);
    expect(strictAlternationProviders.has(Providers.OPENAI)).toBe(false);
    expect(strictAlternationProviders.has(Providers.GOOGLE)).toBe(false);
  });
});

describe('coalesceAdjacentUserTurns', () => {
  it('leaves an already-alternating run untouched', () => {
    const messages = [
      new HumanMessage({ content: 'question' }),
      new AIMessage({ content: 'answer' }),
      new HumanMessage({ content: 'follow-up' }),
    ];
    const result = coalesceAdjacentUserTurns(messages);
    expect(result.map((m) => m.getType())).toEqual(['human', 'ai', 'human']);
    expect(result[0]).toBe(messages[0]);
  });

  /** A boundary that drains two steers, which is the ordinary queue case. */
  it('merges a run of consecutive human turns', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({ content: 'question' }),
      new AIMessage({ content: 'partial' }),
      new HumanMessage({ content: 'steer 1' }),
      new HumanMessage({ content: 'steer 2' }),
    ]);
    expect(result.map((m) => m.getType())).toEqual(['human', 'ai', 'human']);
    expect(result[2].content).toBe('steer 1\n\nsteer 2');
  });

  it('merges context plus injected messages into one turn', () => {
    const result = coalesceAdjacentUserTurns([
      new AIMessage({ content: 'partial' }),
      new HumanMessage({
        content: 'hook context',
        additional_kwargs: { role: 'system', source: 'hook' },
      }),
      new HumanMessage({
        content: 'steer',
        additional_kwargs: { role: 'user', source: 'steer' },
      }),
    ]);
    expect(result).toHaveLength(2);
    expect(result[1].content).toBe('hook context\n\nsteer');
  });

  it('concatenates block content rather than stringifying it', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({ content: [{ type: 'text', text: 'look' }] }),
      new HumanMessage({
        content: [{ type: 'image_url', image_url: { url: 'data:,' } }],
      }),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].content).toEqual([
      { type: 'text', text: 'look' },
      { type: 'image_url', image_url: { url: 'data:,' } },
    ]);
  });

  it('normalizes a mixed string and block pair', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({ content: 'plain' }),
      new HumanMessage({ content: [{ type: 'text', text: 'blocks' }] }),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].content).toEqual([
      { type: 'text', text: 'plain' },
      { type: 'text', text: 'blocks' },
    ]);
  });

  /**
   * Both vendored converters merge adjacent tool-result runs themselves, and
   * folding one into a text turn would orphan the pairing.
   */
  it('never merges tool-result turns', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }],
      }),
      new HumanMessage({
        content: [{ type: 'tool_result', tool_use_id: 't2', content: 'ok' }],
      }),
    ]);
    expect(result).toHaveLength(2);
  });

  it('never merges a tool-result turn into a text turn', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }],
      }),
      new HumanMessage({ content: 'steer' }),
    ]);
    expect(result).toHaveLength(2);
  });

  it('does not mutate the input array or its messages', () => {
    const first = new HumanMessage({ content: 'a' });
    const second = new HumanMessage({ content: 'b' });
    const messages = [first, second];
    coalesceAdjacentUserTurns(messages);
    expect(messages).toHaveLength(2);
    expect(first.content).toBe('a');
    expect(second.content).toBe('b');
  });
});
