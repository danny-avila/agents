/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
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
});
