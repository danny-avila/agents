import { AIMessage, HumanMessage } from '@langchain/core/messages';
import {
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
} from './provenance';
import {
  coalesceAdjacentUserTurns,
  strictAlternationProviders,
} from './alternation';
import { Providers } from '@/common';

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
    expect(result[0].additional_kwargs).toMatchObject({ role: 'user' });
    expect(result[0].id).toBe('first-id');
  });

  it('retains every formatted source id in stable content order', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'first',
        id: 'first-message',
        additional_kwargs: {
          sourceMessageId: 'first-message',
          sourceMessageIds: ['first-message'],
        },
      }),
      new HumanMessage({
        content: 'middle',
        additional_kwargs: {
          sourceMessageId: 'middle-message',
          sourceMessageIds: ['middle-message'],
        },
      }),
      new HumanMessage({
        content: 'last',
        additional_kwargs: {
          sourceMessageId: 'last-message',
          sourceMessageIds: ['last-message'],
        },
      }),
    ]);

    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('first-message');
    expect(result[0].additional_kwargs.sourceMessageId).toBe('last-message');
    expect(result[0].additional_kwargs.sourceMessageIds).toEqual([
      'first-message',
      'middle-message',
      'last-message',
    ]);
  });

  it('keeps user and synthetic contributors distinct across a merge', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'real user turn',
        additional_kwargs: {
          sourceMessageId: 'user-message',
          sourceMessageIds: ['user-message'],
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'user',
                sourceMessageId: 'user-message',
                sourceContentPartIndices: [1],
              },
            ],
          },
        },
      }),
      new HumanMessage({
        content: 'skill body',
        additional_kwargs: { isMeta: true, source: 'skill' },
      }),
    ]);

    expect(result).toHaveLength(1);
    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'user-message',
          sourceContentPartIndices: [1],
        },
        { attribution: 'synthetic' },
      ],
    });
    expect(result[0].additional_kwargs.sourceMessageIds).toEqual([
      'user-message',
    ]);
  });

  it('reconciles plural lineage when explicit legacy parts omit ids', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'legacy first',
        additional_kwargs: {
          sourceMessageIds: ['legacy-first'],
          provenance: {
            version: 1,
            parts: [{ attribution: 'user' }],
          },
        },
      }),
      new HumanMessage({
        content: 'second',
        additional_kwargs: { sourceMessageId: 'second' },
      }),
    ]);

    expect(result[0].additional_kwargs.sourceMessageIds).toEqual([
      'legacy-first',
      'second',
    ]);
    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'user', sourceMessageId: 'legacy-first' },
        { attribution: 'user', sourceMessageId: 'second' },
      ],
    });
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

  it('does not merge the LangChain server-tool-result block into user text', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [
          {
            type: 'server_tool_result',
            tool_call_id: 'server-call',
            status: 'success',
            output: 'tool bytes',
          },
        ],
      }),
      new HumanMessage({ content: 'next turn' }),
    ]);

    expect(result).toHaveLength(2);
  });

  it('does not merge a Gemini code-execution result into user text', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [
          {
            type: 'codeExecutionResult',
            codeExecutionResult: {
              outcome: 'OUTCOME_OK',
              output: 'tool bytes',
            },
          },
        ],
      }),
      new HumanMessage({ content: 'next turn' }),
    ]);

    expect(result).toHaveLength(2);
  });

  it('does not trust arbitrary source metadata or tool-result suffixes', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [{ type: 'attacker_tool_result', text: 'submitted text' }],
        additional_kwargs: {
          source: 'mobile',
          sourceMessageId: 'first',
        },
      }),
      new HumanMessage({
        content: 'next',
        additional_kwargs: { sourceMessageId: 'second' },
      }),
    ]);

    expect(result).toHaveLength(1);
    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'user', sourceMessageId: 'first' },
        { attribution: 'user', sourceMessageId: 'second' },
      ],
    });
  });

  it('keeps steer user attribution even when legacy metadata marks it meta', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: 'steer',
        additional_kwargs: {
          source: 'steer',
          isMeta: true,
          sourceMessageId: 'steer-row',
        },
      }),
      new HumanMessage({
        content: 'next',
        additional_kwargs: { sourceMessageId: 'next-row' },
      }),
    ]);

    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'user', sourceMessageId: 'steer-row' },
        { attribution: 'user', sourceMessageId: 'next-row' },
      ],
    });
  });

  it('does not fabricate indexed mappings across plural legacy sources', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: [{ type: 'text', text: 'merged legacy bytes' }],
        additional_kwargs: {
          sourceMessageIds: ['first-row', 'second-row'],
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'user',
                sourceContentPartIndices: [0],
              },
            ],
          },
        },
      }),
      new HumanMessage({
        content: 'next',
        additional_kwargs: { sourceMessageId: 'next-row' },
      }),
    ]);

    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'user', sourceContentPartIndices: [0] },
        { attribution: 'user', sourceMessageId: 'first-row' },
        { attribution: 'user', sourceMessageId: 'second-row' },
        { attribution: 'user', sourceMessageId: 'next-row' },
      ],
    });
  });

  it('does not attribute leading empty turns that add no provider bytes', () => {
    const result = coalesceAdjacentUserTurns([
      new HumanMessage({
        content: '',
        additional_kwargs: { sourceMessageId: 'empty-row' },
      }),
      new HumanMessage({
        content: 'visible',
        additional_kwargs: { sourceMessageId: 'visible-row' },
      }),
    ]);

    expect(result[0].content).toBe('visible');
    expect(getProviderSourceMessageIds(result[0])).toEqual(['visible-row']);
  });

  it('coalesces a large adjacent run in one pass without truncating lineage', () => {
    const count = 4_000;
    const messages = Array.from(
      { length: count },
      (_, index) =>
        new HumanMessage({
          content: [{ type: 'text', text: String(index) }],
          additional_kwargs: { sourceMessageId: `source-${index}` },
        })
    );

    const result = coalesceAdjacentUserTurns(messages);
    const sourceMessageIds = getProviderSourceMessageIds(result[0]);

    expect(result).toHaveLength(1);
    expect(result[0].content).toHaveLength(count);
    expect(sourceMessageIds).toHaveLength(count);
    expect(getProviderMessageProvenance(result[0])?.parts).toHaveLength(count);
    expect(sourceMessageIds[0]).toBe('source-0');
    expect(sourceMessageIds[count - 1]).toBe(`source-${count - 1}`);
  });

  it('coalesces a very large block array without call-argument spreading', () => {
    const blockCount = 150_000;
    const repeatedBlock = { type: 'text', text: 'x' } as const;
    const largeContent = new Array(blockCount).fill(repeatedBlock);
    const messages = [
      new HumanMessage({ content: [{ type: 'text', text: 'first' }] }),
      new HumanMessage({ content: largeContent }),
    ];

    const result = coalesceAdjacentUserTurns(messages);

    expect(result).toHaveLength(1);
    expect(result[0].content).toHaveLength(blockCount + 1);
    expect(messages[1].content).toBe(largeContent);
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
