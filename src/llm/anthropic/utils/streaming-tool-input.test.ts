/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from './message_inputs';
import { _makeMessageChunkFromAnthropicEvent } from './message_outputs';

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
    expect(() => _convertMessagesToAnthropicPayload(buildHistory())).not.toThrow();
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
        (b) => (b.type === 'text' && 'input' in b) || b.type === 'input_json_delta'
      )
    ).toBeUndefined();

    // The real assistant text is preserved.
    expect(
      blocks.some((b) => b.type === 'text' && b.text === 'Let me calculate that.')
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
    const toolUse = (assistant!.content as any[]).find((b) => b.type === 'tool_use');
    expect(toolUse.input).toEqual({ input: '2 + 2' });
  });

  // Adapted from @langchain/anthropic's
  // "partial tool input is correctly merged before calling Anthropic API".
  it('merges sibling input_json_delta blocks into tool_use input (persisted, no tool_calls)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('What\'s the weather in Seattle tomorrow?'),
      new AIMessage({
        content: [
          { type: 'text', index: 1, text: 'I need to call the get_weather tool' },
          { type: 'tool_use', index: 2, name: 'get_weather', id: 'tool_call_id', input: '' },
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

describe('_makeMessageChunkFromAnthropicEvent — streamed tool input merges into content', () => {
  const fields = { streamUsage: true, coerceContentToString: false };

  it('emits input deltas without a type so aggregation merges them into the tool_use block', () => {
    const events: any[] = [
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'tool_use', id: 'toolu_1', name: 'calculator', input: {} },
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
    expect(toolUse).toMatchObject({ type: 'tool_use', id: 'toolu_1', name: 'calculator' });
    const parsed =
      typeof toolUse.input === 'string' ? JSON.parse(toolUse.input) : toolUse.input;
    expect(parsed).toEqual({ input: '2 + 2' });

    // no orphaned delta block survives aggregation
    expect(blocks.filter((b) => b.type !== 'tool_use' && 'input' in b)).toHaveLength(0);

    // tool_calls remain correctly aggregated
    expect(merged.tool_calls?.[0]).toMatchObject({
      id: 'toolu_1',
      name: 'calculator',
      args: { input: '2 + 2' },
    });
  });
});
