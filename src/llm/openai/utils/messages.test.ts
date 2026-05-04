import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  flattenAnthropicThinkingForOpenAI,
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

    it('emits empty content string when thinking-only message is fully dropped (no tool_calls)', () => {
      const messages = [
        new AIMessage({
          content: [
            {
              type: 'redacted_thinking',
              data: 'opaque-encrypted-blob',
            },
            { type: 'thinking', thinking: '', signature: 'sig' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[0]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: '',
        })
      );
      expect(params[0]).not.toHaveProperty('tool_calls');
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

    it('defaults to flatten/drop when model is undefined', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: 'reasoning', signature: 'sig' },
            { type: 'redacted_thinking', data: 'opaque' },
            { type: 'text', text: 'answer' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages);

      expect(params[0]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: [
            { type: 'text', text: '<thinking>reasoning</thinking>' },
            { type: 'text', text: 'answer' },
          ],
        })
      );
    });

    it('preserves order across interleaved thinking, text, and redacted_thinking', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: 'first', signature: 'sig1' },
            { type: 'text', text: 'one' },
            { type: 'redacted_thinking', data: 'opaque' },
            { type: 'text', text: 'two' },
            { type: 'thinking', thinking: '', signature: 'sig2' },
            { type: 'text', text: 'three' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIParams(messages, 'gpt-5.4-mini');

      expect(params[0]).toEqual(
        expect.objectContaining({
          role: 'assistant',
          content: [
            { type: 'text', text: '<thinking>first</thinking>' },
            { type: 'text', text: 'one' },
            { type: 'text', text: 'two' },
            { type: 'text', text: 'three' },
          ],
        })
      );
    });
  });

  describe('flattenAnthropicThinkingForOpenAI', () => {
    it('flattens thinking blocks for non-Claude targets', () => {
      const out = flattenAnthropicThinkingForOpenAI(
        [
          new AIMessage({
            content: [
              { type: 'thinking', thinking: 'reason', signature: 'sig' },
              { type: 'text', text: 'answer' },
            ] as never,
          }),
        ],
        'gpt-5.4-mini'
      );

      expect(out[0].content).toEqual([
        { type: 'text', text: '<thinking>reason</thinking>' },
        { type: 'text', text: 'answer' },
      ]);
    });

    it('passes through unchanged for Claude targets', () => {
      const original = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: 'reason', signature: 'sig' },
            { type: 'text', text: 'answer' },
          ] as never,
        }),
      ];

      const out = flattenAnthropicThinkingForOpenAI(
        original,
        'anthropic/claude-sonnet-4-5'
      );

      expect(out).toBe(original);
    });

    it('returns the same reference when no message has thinking blocks', () => {
      const original = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [{ type: 'text', text: 'hello' }] as never,
        }),
      ];

      const out = flattenAnthropicThinkingForOpenAI(original, 'gpt-5.4-mini');

      expect(out).toBe(original);
    });

    it('preserves all AIMessage fields across the rewrite', () => {
      const original = new AIMessage({
        content: [
          { type: 'thinking', thinking: 'thinking', signature: 'sig' },
          { type: 'text', text: 'will call tool' },
        ] as never,
        tool_calls: [
          { id: 'call_1', name: 'f', args: { x: 1 }, type: 'tool_call' },
        ],
        invalid_tool_calls: [
          { id: 'call_bad', name: 'g', args: 'malformed', error: 'oops' },
        ],
        additional_kwargs: { reasoning_content: 'extra' },
        response_metadata: { model_name: 'src-model', finish_reason: 'stop' },
        name: 'agent-7',
        id: 'msg_abc',
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 20,
          total_tokens: 30,
        },
      });

      const out = flattenAnthropicThinkingForOpenAI([original], 'gpt-5.4-mini');

      expect(out[0]).not.toBe(original);
      expect(out[0]).toBeInstanceOf(AIMessage);
      const ai = out[0] as AIMessage;
      expect(ai.tool_calls).toEqual(original.tool_calls);
      expect(ai.invalid_tool_calls).toEqual(original.invalid_tool_calls);
      expect(ai.additional_kwargs).toEqual({ reasoning_content: 'extra' });
      expect(ai.response_metadata).toEqual({
        model_name: 'src-model',
        finish_reason: 'stop',
      });
      expect(ai.name).toBe('agent-7');
      expect(ai.id).toBe('msg_abc');
      expect(ai.usage_metadata).toEqual({
        input_tokens: 10,
        output_tokens: 20,
        total_tokens: 30,
      });
      expect(ai.content).toEqual([
        { type: 'text', text: '<thinking>thinking</thinking>' },
        { type: 'text', text: 'will call tool' },
      ]);
    });

    it('passes non-AI messages through untouched even if they carry thinking-shaped blocks', () => {
      const human = new HumanMessage({
        content: [
          { type: 'thinking', thinking: 'should be ignored', signature: 'sig' },
          { type: 'text', text: 'user prompt' },
        ] as never,
      });

      const out = flattenAnthropicThinkingForOpenAI([human], 'gpt-5.4-mini');

      expect(out).toHaveLength(1);
      expect(out[0]).toBe(human);
    });

    it('falls back to empty string content when filtering empties the array', () => {
      const out = flattenAnthropicThinkingForOpenAI(
        [
          new AIMessage({
            content: [
              { type: 'thinking', thinking: '', signature: 'sig' },
              { type: 'redacted_thinking', data: 'opaque' },
            ] as never,
          }),
        ],
        'gpt-5.4-mini'
      );

      expect(out[0]).toBeInstanceOf(AIMessage);
      expect(out[0].content).toBe('');
    });
  });

  describe('_convertMessagesToOpenAIResponsesParams cross-provider thinking', () => {
    it('flattens thinking blocks to <thinking> output_text for non-Claude targets', () => {
      const messages = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [
            { type: 'thinking', thinking: 'consider X', signature: 'sig' },
            { type: 'text', text: 'answer' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIResponsesParams(messages, 'gpt-5');
      const assistant = params.find(
        (p) => 'role' in p && p.role === 'assistant'
      ) as { content: unknown[] };

      expect(assistant.content).toEqual([
        {
          type: 'output_text',
          text: '<thinking>consider X</thinking>',
          annotations: [],
        },
        {
          type: 'output_text',
          text: 'answer',
          annotations: [],
        },
      ]);
    });

    it('drops empty thinking and redacted_thinking on the Responses path', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: '', signature: 'sig' },
            { type: 'redacted_thinking', data: 'opaque' },
            { type: 'text', text: 'visible' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIResponsesParams(messages, 'gpt-5');
      const assistant = params.find(
        (p) => 'role' in p && p.role === 'assistant'
      ) as { content: unknown[] };

      expect(assistant.content).toEqual([
        { type: 'output_text', text: 'visible', annotations: [] },
      ]);
    });

    it('preserves thinking and redacted_thinking blocks for Claude-via-OpenRouter Responses target', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: 'preserved', signature: 'sig' },
            { type: 'redacted_thinking', data: 'opaque' },
            { type: 'text', text: 'answer' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIResponsesParams(
        messages,
        'anthropic/claude-sonnet-4-5'
      );
      const assistant = params.find(
        (p) => 'role' in p && p.role === 'assistant'
      ) as { content: unknown[] };

      expect(assistant.content).toEqual([
        { type: 'thinking', thinking: 'preserved', signature: 'sig' },
        { type: 'redacted_thinking', data: 'opaque' },
        { type: 'output_text', text: 'answer', annotations: [] },
      ]);
    });

    it('falls back to empty string when Responses content is fully filtered', () => {
      const messages = [
        new AIMessage({
          content: [
            { type: 'thinking', thinking: '', signature: 'sig' },
            { type: 'redacted_thinking', data: 'opaque' },
          ] as never,
        }),
      ];

      const params = _convertMessagesToOpenAIResponsesParams(messages, 'gpt-5');
      const assistant = params.find(
        (p) => 'role' in p && p.role === 'assistant'
      ) as { content: unknown };

      expect(assistant.content).toBe('');
    });
  });
});
