import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
} from './index';

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
});

describe('_convertMessagesToOpenAIResponsesParams', () => {
  it('combines persisted reasoning items with reconstructed standard tool calls', () => {
    const firstReasoning = {
      type: 'reasoning' as const,
      id: 'rs_1',
      summary: [],
      encrypted_content: 'encrypted-1',
    };
    const finalReasoning = {
      type: 'reasoning' as const,
      id: 'rs_2',
      summary: [],
      encrypted_content: 'encrypted-2',
    };
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: {},
            type: 'tool_call',
          },
        ],
        response_metadata: {
          model_provider: 'openai',
          output: [firstReasoning],
        },
      }),
      new ToolMessage({
        content: '{"ok":true}',
        tool_call_id: 'call_1',
      }),
      new AIMessage({
        content: 'Done.',
        response_metadata: {
          model_provider: 'openai',
          output: [finalReasoning],
        },
      }),
    ];

    expect(
      _convertMessagesToOpenAIResponsesParams(messages, 'gpt-5.6', true)
    ).toEqual([
      firstReasoning,
      {
        type: 'function_call',
        name: 'lookup',
        arguments: '{}',
        call_id: 'call_1',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: '{"ok":true}',
      },
      finalReasoning,
      {
        type: 'message',
        role: 'assistant',
        content: 'Done.',
      },
    ]);
  });
});
