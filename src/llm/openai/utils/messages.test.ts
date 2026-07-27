import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
} from './index';
import { calculateMaxToolCallInputChars } from '@/messages/prune';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';

describe('_convertMessagesToOpenAIParams', () => {
  it('includes reasoning_content for assistant messages in tool-call context when requested', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
      new ToolMessage({
        content: '57531',
        tool_call_id: 'call_1',
      }),
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Calculator returned 57531.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(3);
    expect(params[0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'Need calculator.',
      })
    );
    expect(params[2]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        reasoning_content: 'Calculator returned 57531.',
      })
    );
  });

  it('does not include reasoning_content for no-tool assistant messages', () => {
    const messages = [
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Mental calculation.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(1);
    expect(params[0]).not.toHaveProperty('reasoning_content');
  });

  it('does not include reasoning_content unless explicitly requested', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro');

    expect(params).toHaveLength(1);
    expect(params[0]).not.toHaveProperty('reasoning_content');
  });

  it('keeps reasoning_content latched after tool-call context is established', () => {
    const messages = [
      new AIMessage({
        content: 'No tool was needed.',
        additional_kwargs: {
          reasoning_content: 'Initial no-tool reasoning.',
        },
      }),
      new HumanMessage('Use the calculator.'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'calculator',
            args: { input: '127 * 453' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          reasoning_content: 'Need calculator.',
        },
      }),
      new ToolMessage({
        content: '57531',
        tool_call_id: 'call_1',
      }),
      new AIMessage({
        content: '127 * 453 = 57531.',
        additional_kwargs: {
          reasoning_content: 'Calculator returned 57531.',
        },
      }),
      new HumanMessage('Was that correct?'),
      new AIMessage({
        content: 'Yes.',
        additional_kwargs: {
          reasoning_content: 'The prior calculator result is available.',
        },
      }),
    ];

    const params = _convertMessagesToOpenAIParams(messages, 'deepseek-v4-pro', {
      includeReasoningContent: true,
    });

    expect(params).toHaveLength(7);
    expect(params[0]).not.toHaveProperty('reasoning_content');
    expect(params[2]).toEqual(
      expect.objectContaining({
        reasoning_content: 'Need calculator.',
      })
    );
    expect(params[4]).toEqual(
      expect.objectContaining({
        reasoning_content: 'Calculator returned 57531.',
      })
    );
    expect(params[6]).toEqual(
      expect.objectContaining({
        reasoning_content: 'The prior calculator result is available.',
      })
    );
  });

  it('bounds structured function-tool output for Chat and Responses adapters', () => {
    let toJSONCalls = 0;
    const toolMessage = new ToolMessage({
      content: [
        { type: 'text', text: 'rendered chart' },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/png;base64,${'A'.repeat(
              HARD_MAX_TOOL_RESULT_CHARS + 1_000
            )}`,
          },
          toJSON() {
            toJSONCalls++;
            return { expanded: 'B'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2) };
          },
        },
      ],
      tool_call_id: 'call_structured',
    });

    const chat = _convertMessagesToOpenAIParams([toolMessage]);
    const responses = _convertMessagesToOpenAIResponsesParams([toolMessage]);
    const chatContent = chat[0].content;

    expect(typeof chatContent).toBe('string');
    expect((chatContent as string).length).toBeLessThanOrEqual(
      HARD_MAX_TOOL_RESULT_CHARS
    );
    expect(responses[0]).toMatchObject({
      type: 'function_call_output',
      call_id: 'call_structured',
      output: chatContent,
    });
    expect(toJSONCalls).toBe(0);
    expect(Array.isArray(toolMessage.content)).toBe(true);
  });

  it('bounds direct structured tool-call args without invoking accessors or toJSON', () => {
    const maxInputChars = calculateMaxToolCallInputChars();
    let getterCalls = 0;
    let toJSONCalls = 0;
    const args: Record<string, unknown> = {
      query: 'safe',
      payload: 'x'.repeat(maxInputChars + 1_000),
    };
    Object.defineProperty(args, 'expanded', {
      enumerable: true,
      get() {
        getterCalls++;
        return 'y'.repeat(maxInputChars * 2);
      },
    });
    args.self = args;
    args.toJSON = (): Record<string, string> => {
      toJSONCalls++;
      return { payload: 'z'.repeat(maxInputChars * 2) };
    };
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_adversarial',
          name: 'lookup',
          args,
        },
      ],
    });

    const chat = _convertMessagesToOpenAIParams([message]);
    const responses = _convertMessagesToOpenAIResponsesParams([message]);
    const chatArguments = (
      chat[0] as {
        tool_calls: Array<{ function: { arguments: string } }>;
      }
    ).tool_calls[0].function.arguments;
    const responseArguments = (
      responses.find((item) => item.type === 'function_call') as {
        arguments: string;
      }
    ).arguments;

    expect(chatArguments).toBe(responseArguments);
    expect(chatArguments.length).toBeLessThanOrEqual(maxInputChars);
    expect(() => JSON.parse(chatArguments)).not.toThrow();
    expect(getterCalls).toBe(0);
    expect(toJSONCalls).toBe(0);
    expect(args.self).toBe(args);
  });

  it('bounds and synchronizes raw Chat and Responses function-call arguments', () => {
    const maxInputChars = calculateMaxToolCallInputChars();
    const rawArguments = `{"payload":"${'x'.repeat(maxInputChars + 1_000)}"}`;
    const rawOnlyMessage = new AIMessage({
      content: '',
      additional_kwargs: {
        tool_calls: [
          {
            id: 'call_raw',
            type: 'function',
            function: { name: 'lookup', arguments: rawArguments },
          },
        ],
      },
      response_metadata: {
        output: [
          {
            type: 'function_call',
            id: 'fc_raw',
            call_id: 'call_raw',
            name: 'lookup',
            arguments: rawArguments,
          },
        ],
      },
    });

    const chat = _convertMessagesToOpenAIParams([rawOnlyMessage]);
    const responses = _convertMessagesToOpenAIResponsesParams([rawOnlyMessage]);
    const chatArguments = (
      chat[0] as {
        tool_calls: Array<{ function: { arguments: string } }>;
      }
    ).tool_calls[0].function.arguments;
    const responseArguments = (
      responses.find((item) => item.type === 'function_call') as {
        arguments: string;
      }
    ).arguments;

    expect(chatArguments.length).toBeLessThanOrEqual(maxInputChars);
    expect(responseArguments.length).toBeLessThanOrEqual(maxInputChars);
    expect(() => JSON.parse(chatArguments)).not.toThrow();
    expect(() => JSON.parse(responseArguments)).not.toThrow();
    expect(
      (
        rawOnlyMessage.additional_kwargs.tool_calls as Array<{
          function: { arguments: string };
        }>
      )[0].function.arguments
    ).toBe(rawArguments);
    expect(
      (
        rawOnlyMessage.response_metadata.output as Array<{
          arguments: string;
        }>
      )[0].arguments
    ).toBe(rawArguments);
  });

  it('makes Responses raw reuse match the projected LangChain tool-call args', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_synced',
          name: 'lookup',
          args: { query: 'canonical', payload: 'x'.repeat(500_000) },
        },
      ],
      response_metadata: {
        output: [
          {
            type: 'function_call',
            id: 'fc_synced',
            call_id: 'call_synced',
            name: 'lookup',
            arguments: '{"query":"stale"}',
          },
        ],
      },
    });

    const chat = _convertMessagesToOpenAIParams([message]);
    const responses = _convertMessagesToOpenAIResponsesParams([message]);
    const chatArguments = (
      chat[0] as {
        tool_calls: Array<{ function: { arguments: string } }>;
      }
    ).tool_calls[0].function.arguments;
    const responseArguments = (
      responses.find((item) => item.type === 'function_call') as {
        arguments: string;
      }
    ).arguments;

    expect(responseArguments).toBe(chatArguments);
    expect(() => JSON.parse(responseArguments)).not.toThrow();
    expect(
      (
        message.response_metadata.output as Array<{
          arguments: string;
        }>
      )[0].arguments
    ).toBe('{"query":"stale"}');
  });

  it('preserves native Responses computer-call output handling', () => {
    const toolMessage = new ToolMessage({
      content: [
        {
          type: 'computer_screenshot',
          image_url: 'data:image/png;base64,AA==',
        },
      ],
      tool_call_id: 'call_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });

    expect(_convertMessagesToOpenAIResponsesParams([toolMessage])[0]).toEqual({
      type: 'computer_call_output',
      call_id: 'call_computer',
      output: {
        type: 'computer_screenshot',
        image_url: 'data:image/png;base64,AA==',
      },
    });
  });
});
