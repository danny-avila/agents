import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

function toolCallPart(
  id: string,
  name = 'search',
  output = 'result'
): Record<string, unknown> {
  return {
    type: ContentTypes.TOOL_CALL,
    tool_call: { id, name, args: '{}', output },
  };
}

describe('formatAgentMessages steer replay', () => {
  it('replays a steer part as a user message after preceding tool messages', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Working on it.',
            tool_call_ids: ['call_1'],
          },
          toolCallPart('call_1'),
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Actually, focus on the second item.',
            steerId: 's1',
          } as unknown as Record<string, unknown>,
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Focusing on item two.',
          },
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    const steerIndex = messages.findIndex(
      (message) =>
        message instanceof HumanMessage &&
        message.additional_kwargs.source === 'steer'
    );
    expect(steerIndex).toBeGreaterThan(0);
    expect(messages[steerIndex].content).toBe(
      'Actually, focus on the second item.'
    );
    expect(messages[steerIndex - 1]).toBeInstanceOf(ToolMessage);
    const trailing = messages[steerIndex + 1];
    expect(trailing).toBeInstanceOf(AIMessage);
  });

  it('flushes accumulated assistant text before the steer user message', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Partial answer.' },
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Change of plans.',
          } as unknown as Record<string, unknown>,
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[0].content).toBe('Partial answer.');
    expect(messages[1]).toBeInstanceOf(HumanMessage);
    expect(messages[1].content).toBe('Change of plans.');
  });

  it('uses the host-stamped media array for multimodal steers', () => {
    const media = [
      { type: 'text', text: 'Look at this screenshot.' },
      {
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,abc123', detail: 'auto' },
      },
    ];
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Look at this screenshot.',
            media,
          } as unknown as Record<string, unknown>,
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    expect(messages).toHaveLength(1);
    expect(messages[0]).toBeInstanceOf(HumanMessage);
    expect(messages[0].content).toEqual(media);
    expect((messages[0] as HumanMessage).additional_kwargs.source).toBe(
      'steer'
    );
  });

  it('never leaks a steer part into assistant content sent to the provider', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Before.' },
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Steer text.',
          } as unknown as Record<string, unknown>,
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'After.' },
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    for (const message of messages) {
      if (message instanceof HumanMessage) {
        continue;
      }
      const content = message.content;
      if (Array.isArray(content)) {
        expect(
          content.some((part) => (part as { type?: string }).type === 'steer')
        ).toBe(false);
      }
    }
    // Trailing content after the steer keeps the end-of-loop array form.
    expect(messages.map((message) => message.content)).toEqual([
      'Before.',
      'Steer text.',
      [{ type: 'text', text: 'After.' }],
    ]);
  });
});
