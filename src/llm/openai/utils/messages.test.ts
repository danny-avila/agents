import { AIMessage, ToolMessage } from '@langchain/core/messages';
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
});
