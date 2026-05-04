import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { _convertMessagesToOpenAIParams } from './index';

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

  describe('cross-provider thinking block handling', () => {
    it('flattens Anthropic thinking block to <thinking> text for OpenAI target', () => {
      const messages = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [
            {
              type: 'thinking',
              thinking: 'Reviewing whether 17 is prime.',
              signature: 'sig_abc',
            },
            { type: 'text', text: 'Yes, 17 is prime.' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[1]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: '<thinking>Reviewing whether 17 is prime.</thinking>',
            },
            { type: 'text', text: 'Yes, 17 is prime.' },
          ],
        })
      );
    });

    it('drops empty thinking block before forwarding to OpenAI', () => {
      const messages = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [
            { type: 'thinking', thinking: '', signature: 'sig_empty' },
            { type: 'text', text: 'response' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[1]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: [{ type: 'text', text: 'response' }],
        })
      );
    });

    it('drops redacted_thinking block for OpenAI target', () => {
      const messages = [
        new AIMessage({
          content: [
            {
              type: 'redacted_thinking',
              data: 'opaque-encrypted-blob',
            },
            { type: 'text', text: 'visible answer' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[0]).toEqual(
        expect.objectContaining({
          content: [{ type: 'text', text: 'visible answer' }],
        })
      );
    });

    it('preserves thinking block for Claude-via-OpenRouter target', () => {
      const messages = [
        new AIMessage({
          content: [
            {
              type: 'thinking',
              thinking: 'Considering options.',
              signature: 'sig_x',
            },
            { type: 'text', text: 'Done.' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(
        messages,
        'anthropic/claude-sonnet-4-5'
      );

      expect(params[0]).toEqual(
        expect.objectContaining({
          content: [
            {
              type: 'thinking',
              thinking: 'Considering options.',
              signature: 'sig_x',
            },
            { type: 'text', text: 'Done.' },
          ],
        })
      );
    });

    it('emits empty content string when thinking-only message is fully dropped alongside tool_calls', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: '', signature: 'sig' },
          ] as never,
          tool_calls: [
            {
              id: 'call_1',
              name: 'noop',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[0]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: '',
          tool_calls: expect.any(Array),
        })
      );
    });
  });
});
