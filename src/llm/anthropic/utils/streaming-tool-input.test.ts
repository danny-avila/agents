/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _makeMessageChunkFromAnthropicEvent } from './message_outputs';
import { _convertMessagesToAnthropicPayload } from './message_inputs';

/**
 * Regression for @langchain/core >= 1.1.46 streaming aggregation: a tool call's
 * input_json_delta is kept as a separate content block and v1-cast to a `text`
 * block carrying `input` but no `text`, leaving the sibling tool_use block with an
 * empty inline input. The assembled arguments live on `message.tool_calls`.
 * Re-serializing such a message previously threw "Unsupported message content format".
 */
describe('_convertMessagesToAnthropicPayload — aggregated streaming tool input', () => {
  const buildHistory = (): BaseMessage[] => [
    new HumanMessage('what is 12345 * 6789?'),
    new AIMessage({
      content: [
        { type: 'text', text: 'Let me calculate that.' },
        // tool_use block left with empty inline input by aggregation
        {
          type: 'tool_use',
          id: 'toolu_calc',
          name: 'calculator',
          input: '',
          index: 0,
        } as any,
        // orphaned input delta, v1-cast to `text` with `input` and no `text`
        { type: 'text', index: 0, input: '{"input": "12345 * 6789"}' } as any,
      ],
      tool_calls: [
        {
          id: 'toolu_calc',
          name: 'calculator',
          args: { input: '12345 * 6789' },
          type: 'tool_call',
        },
      ],
    }),
  ];

  it('does not throw on the orphaned text-with-input block', () => {
    expect(() =>
      _convertMessagesToAnthropicPayload(buildHistory())
    ).not.toThrow();
  });

  it('restores tool_use input from message.tool_calls and drops the orphan block', () => {
    const payload = _convertMessagesToAnthropicPayload(buildHistory());
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    expect(assistant).toBeDefined();
    const blocks = assistant!.content as any[];

    const toolUse = blocks.find((b) => b.type === 'tool_use');
    expect(toolUse).toMatchObject({
      type: 'tool_use',
      id: 'toolu_calc',
      name: 'calculator',
      input: { input: '12345 * 6789' },
    });

    // No leftover delta: no text block carrying `input`, no input_json_delta.
    expect(
      blocks.find(
        (b) =>
          (b.type === 'text' && 'input' in b) || b.type === 'input_json_delta'
      )
    ).toBeUndefined();

    // The real assistant text is preserved.
    expect(
      blocks.some(
        (b) => b.type === 'text' && b.text === 'Let me calculate that.'
      )
    ).toBe(true);
  });

  it('does not overwrite a tool_use block that already has inline input', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'tool_use',
            id: 'toolu_x',
            name: 'calculator',
            input: { input: '2 + 2' },
          } as any,
        ],
        tool_calls: [
          {
            id: 'toolu_x',
            name: 'calculator',
            args: { input: '999' },
            type: 'tool_call',
          },
        ],
      }),
    ];
    const payload = _convertMessagesToAnthropicPayload(history);
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    const toolUse = (assistant!.content as any[]).find(
      (b) => b.type === 'tool_use'
    );
    expect(toolUse.input).toEqual({ input: '2 + 2' });
  });

  // Adapted from @langchain/anthropic's
  // "partial tool input is correctly merged before calling Anthropic API".
  it('merges sibling input_json_delta blocks into tool_use input (persisted, no tool_calls)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('What\'s the weather in Seattle tomorrow?'),
      new AIMessage({
        content: [
          {
            type: 'text',
            index: 1,
            text: 'I need to call the get_weather tool',
          },
          {
            type: 'tool_use',
            index: 2,
            name: 'get_weather',
            id: 'tool_call_id',
            input: '',
          },
          { type: 'input_json_delta', index: 2, input: '{"city": "' },
          { type: 'input_json_delta', index: 2, input: 'Seattle", "da' },
          { type: 'input_json_delta', index: 2, input: 'te": "to' },
          { type: 'input_json_delta', index: 2, input: 'morrow"}' },
        ] as any,
      }),
    ];

    const payload = _convertMessagesToAnthropicPayload(messages);
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    const blocks = assistant!.content as any[];
    expect(blocks.filter((b) => b.type === 'input_json_delta')).toHaveLength(0);
    const toolUse = blocks.find((b) => b.type === 'tool_use');
    expect(toolUse).toMatchObject({
      type: 'tool_use',
      name: 'get_weather',
      id: 'tool_call_id',
      input: { city: 'Seattle', date: 'tomorrow' },
    });
  });
});

/**
 * Regression for the summarization-CI flake: context-pressure truncation
 * (`preFlightTruncateToolCallInputs` under a tight budget) used to null BOTH a
 * tool_use block's inline `input` and its `tool_calls` args in graph state.
 * Replaying that message shipped `"input": null` and Anthropic rejected the
 * request with 400 `tool_use.input: Input should be an object`. The payload
 * conversion must never emit a non-object input, whatever shape history is in.
 */
describe('_convertMessagesToAnthropicPayload — non-object tool_use input replay', () => {
  const buildHistory = (input: unknown, args: unknown): BaseMessage[] => [
    new HumanMessage('What is 9 * 9?'),
    new AIMessage({
      content: [
        { type: 'tool_use', id: 'toolu_x', name: 'calculator', input } as any,
      ],
      tool_calls: [
        {
          id: 'toolu_x',
          name: 'calculator',
          args: args as any,
          type: 'tool_call',
        },
      ],
    }),
  ];

  const getToolUse = (history: BaseMessage[]): any => {
    const payload = _convertMessagesToAnthropicPayload(history);
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    return (assistant!.content as any[]).find((b) => b.type === 'tool_use');
  };

  it('ships an empty object when input and args were both truncated to null', () => {
    const toolUse = getToolUse(buildHistory(null, null));
    expect(toolUse.input).toEqual({});
  });

  it('restores object args when only the inline input was nulled', () => {
    const toolUse = getToolUse(buildHistory(null, { input: '9 * 9' }));
    expect(toolUse.input).toEqual({ input: '9 * 9' });
  });

  it('restores intact args when the inline input degraded to an empty object', () => {
    // Asymmetric truncation: the raw string input serializes longer than the
    // args object, so a near-envelope cap can degrade the inline input to {}
    // while the tool_calls mirror survives. Replay must prefer the mirror.
    const toolUse = getToolUse(
      buildHistory({}, { input: '670592745 / 99991' })
    );
    expect(toolUse.input).toEqual({ input: '670592745 / 99991' });
  });

  it('ships an empty object when the inline input is {} and args were nulled too', () => {
    const toolUse = getToolUse(buildHistory({}, null));
    expect(toolUse.input).toEqual({});
  });

  it('coerces non-object inputs on the srvtoolu_ server-tool normalization branch', () => {
    for (const [raw, expected] of [
      ['123', {}],
      ['[1,2]', {}],
      ['{"query": "x"}', { query: 'x' }],
      [null, {}],
    ] as Array<[unknown, Record<string, unknown>]>) {
      const history: BaseMessage[] = [
        new HumanMessage('search'),
        new AIMessage({
          content: [
            {
              type: 'server_tool_use',
              id: 'srvtoolu_abc',
              name: 'web_search',
              input: raw,
            } as any,
          ],
        }),
      ];
      const payload = _convertMessagesToAnthropicPayload(history);
      const assistant = payload.messages.find(
        (m: any) => m.role === 'assistant'
      );
      const block = (assistant!.content as any[]).find(
        (b) => b.type === 'server_tool_use'
      );
      expect(block.input).toEqual(expected);
    }
  });

  it('coerces a string input that parses to a non-object', () => {
    for (const raw of ['123', '[1,2]', '"text"', 'null']) {
      const toolUse = getToolUse(buildHistory(raw, undefined));
      expect(toolUse.input).toEqual({});
    }
  });

  it('still parses a complete JSON-object string input', () => {
    const toolUse = getToolUse(buildHistory('{"input": "9 * 9"}', undefined));
    expect(toolUse.input).toEqual({ input: '9 * 9' });
  });

  it('coerces non-object args on the string-content tool_calls path', () => {
    const history: BaseMessage[] = [
      new HumanMessage('What is 9 * 9?'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'toolu_x',
            name: 'calculator',
            args: null as any,
            type: 'tool_call',
          },
        ],
      }),
    ];
    const payload = _convertMessagesToAnthropicPayload(history);
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    const toolUse = (assistant!.content as any[]).find(
      (b) => b.type === 'tool_use'
    );
    expect(toolUse.input).toEqual({});
  });
});

describe('_makeMessageChunkFromAnthropicEvent — streamed tool input merges into content', () => {
  const fields = { streamUsage: true, coerceContentToString: false };

  it('emits input deltas without a type so aggregation merges them into the tool_use block', () => {
    const events: any[] = [
      {
        type: 'content_block_start',
        index: 0,
        content_block: {
          type: 'tool_use',
          id: 'toolu_1',
          name: 'calculator',
          input: {},
        },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'input_json_delta', partial_json: '{"input"' },
      },
      {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'input_json_delta', partial_json: ': "2 + 2"}' },
      },
    ];
    const chunks = events
      .map((e) => _makeMessageChunkFromAnthropicEvent(e, fields)?.chunk)
      .filter((c): c is NonNullable<typeof c> => c != null);

    // input-delta chunks must not carry a `type` (so core merges them by index
    // into the sibling tool_use/server_tool_use block rather than orphaning them)
    const deltaBlocks = chunks
      .slice(1)
      .flatMap((c) => (Array.isArray(c.content) ? (c.content as any[]) : []))
      .filter((b) => 'input' in b);
    expect(deltaBlocks.length).toBeGreaterThan(0);
    deltaBlocks.forEach((b) => expect('type' in b).toBe(false));

    // aggregate the chunks the way core does during streaming
    const merged = chunks.reduce((acc, c) => acc.concat(c));
    const blocks = merged.content as any[];

    const toolUse = blocks.find((b) => b.type === 'tool_use');
    expect(toolUse).toMatchObject({
      type: 'tool_use',
      id: 'toolu_1',
      name: 'calculator',
    });
    const parsed =
      typeof toolUse.input === 'string'
        ? JSON.parse(toolUse.input)
        : toolUse.input;
    expect(parsed).toEqual({ input: '2 + 2' });

    // no orphaned delta block survives aggregation
    expect(
      blocks.filter((b) => b.type !== 'tool_use' && 'input' in b)
    ).toHaveLength(0);

    // tool_calls remain correctly aggregated
    expect(merged.tool_calls?.[0]).toMatchObject({
      id: 'toolu_1',
      name: 'calculator',
      args: { input: '2 + 2' },
    });
  });
});
