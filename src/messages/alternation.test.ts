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
  it('leaves an already-alternating run untouched, returning the SAME array', () => {
    const messages = [
      new HumanMessage({ content: 'question' }),
      new AIMessage({ content: 'answer' }),
      new HumanMessage({ content: 'follow-up' }),
    ];
    const result = coalesceAdjacentUserTurns(messages);
    /**
     * Identity, not just equality: the pass runs twice for a primary
     * Bedrock/Mistral call (createCallModel, then the attemptInvoke funnel),
     * so the normalized second pass must not reallocate a context-sized
     * array — and origin tracking early-exits on `before === after`.
     */
    expect(result).toBe(messages);
  });

  it('is idempotent by identity: re-coalescing merged output is a no-op', () => {
    const once = coalesceAdjacentUserTurns([
      new HumanMessage({ content: 'steer 1' }),
      new HumanMessage({ content: 'steer 2' }),
    ]);
    expect(once).toHaveLength(1);
    expect(coalesceAdjacentUserTurns(once)).toBe(once);
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

  /**
   * The prompt-cache tail anchor reasons positionally — it inserts the
   * breakpoint after the merged message's LAST text block — so the last
   * part's provenance flags must survive the merge. A skill body absorbed
   * into a real user turn stays anchorable; a real turn absorbed into a
   * trailing skill body must not let the anchor pin the volatile body.
   */
  it('keeps the last turn\'s additional_kwargs and the first turn\'s id', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'skill body',
        id: 'first-id',
        additional_kwargs: { isMeta: true, source: 'skill' },
      }),
      new HumanMessage({
        content: 'real user turn',
        id: 'second-id',
        additional_kwargs: { role: 'user' },
      }),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].additional_kwargs).toEqual({ role: 'user' });
    expect(result[0].id).toBe('first-id');
  });

  it('marks the merge meta when the trailing part is the volatile one', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'real steer',
        additional_kwargs: { source: 'steer' },
      }),
      new HumanMessage({
        content: 'skill body',
        additional_kwargs: { isMeta: true, source: 'skill', skillName: 'x' },
      }),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].additional_kwargs.source).toBe('skill');
    expect(result[0].additional_kwargs.isMeta).toBe(true);
  });

  /**
   * Only ALL-tool-result turns are excluded. A mixed turn is an ordinary
   * user turn that happens to carry a result block — merging preserves block
   * order, so the pairing survives, and excluding it would leave adjacent
   * user turns on the wire for exactly the shape Bedrock rejects.
   */
  it('merges a mixed text plus tool-result turn', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [
          { type: 'tool_result', tool_use_id: 't1', content: 'ok' },
          { type: 'text', text: 'and my comment' },
        ],
      }),
      new HumanMessage({ content: 'next turn' }),
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].content).toEqual([
      { type: 'tool_result', tool_use_id: 't1', content: 'ok' },
      { type: 'text', text: 'and my comment' },
      { type: 'text', text: 'next turn' },
    ]);
  });

  it('excludes the camelCase toolResult variant too', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [{ type: 'toolResult', toolResult: { content: 'ok' } }],
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
