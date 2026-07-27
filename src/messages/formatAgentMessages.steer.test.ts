import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { ContentTypes, Constants, Providers } from '@/common';
import { formatAgentMessages } from './format';

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

  it('mints a fresh assistant anchor for a tool call after a steer', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'First step.',
            tool_call_ids: ['call_1'],
          },
          toolCallPart('call_1'),
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Do the second thing instead.',
          } as unknown as Record<string, unknown>,
          toolCallPart('call_2', 'search', 'second result'),
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    // AI(call_1), Tool(call_1), Human(steer), AI(call_2), Tool(call_2) —
    // the post-steer tool call must NOT attach to the pre-steer anchor, or
    // its ToolMessage would trail the user turn while the call precedes it.
    expect(messages.map((message) => message.constructor.name)).toEqual([
      'AIMessage',
      'ToolMessage',
      'HumanMessage',
      'AIMessage',
      'ToolMessage',
    ]);
    const postSteerAnchor = messages[3] as AIMessage;
    expect(postSteerAnchor.tool_calls?.map((call) => call.id)).toEqual([
      'call_2',
    ]);
    const preSteerAnchor = messages[0] as AIMessage;
    expect(preSteerAnchor.tool_calls?.map((call) => call.id)).toEqual([
      'call_1',
    ]);
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

  it('preserves pending reasoning that directly precedes a steer', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'chain of thought',
          },
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Stop and reconsider.',
          } as unknown as Record<string, unknown>,
        ],
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
      }
    );

    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[0].additional_kwargs.reasoning_content).toBe(
      'chain of thought'
    );
    expect(messages[1]).toBeInstanceOf(HumanMessage);
    expect(messages[1].content).toBe('Stop and reconsider.');
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

  it('replays [THINK, STEER, TEXT] without duplicating the trailing text', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'chain of thought',
          },
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Redirect now.',
          } as unknown as Record<string, unknown>,
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Post-steer answer.',
          },
        ],
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
      }
    );

    expect(messages.map((message) => message.constructor.name)).toEqual([
      'AIMessage',
      'HumanMessage',
      'AIMessage',
    ]);
    expect(messages[0].additional_kwargs.reasoning_content).toBe(
      'chain of thought'
    );
    // Post-steer segment starts with fresh reasoning state: the trailing
    // text appears exactly once (end-of-loop array form) and carries no
    // pre-steer reasoning.
    expect(messages[2].content).toEqual([
      { type: 'text', text: 'Post-steer answer.' },
    ]);
    expect(messages[2].additional_kwargs.reasoning_content).toBeUndefined();
  });

  it('keeps a paired Anthropic server-tool use/result together after the steer', () => {
    const useId = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}search1`;
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: { id: useId, name: 'web_search', args: '{"query":"x"}' },
          },
          {
            type: ContentTypes.STEER,
            [ContentTypes.STEER]: 'Also check the docs.',
          } as unknown as Record<string, unknown>,
          {
            type: 'web_search_tool_result',
            tool_use_id: useId,
            content: [
              { type: 'web_search_result', url: 'https://example.com' },
            ],
          } as unknown as Record<string, unknown>,
        ],
      },
    ];

    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['web_search']),
      undefined,
      { provider: Providers.ANTHROPIC }
    );

    // The pair may not split across the user turn (invalid for Anthropic);
    // it emits together in the post-steer assistant segment instead.
    expect(messages[0]).toBeInstanceOf(HumanMessage);
    expect(messages[0].content).toBe('Also check the docs.');
    const assistant = messages[1];
    expect(assistant).toBeInstanceOf(AIMessage);
    const parts = assistant.content as Array<{
      type?: string;
      tool_use_id?: string;
      id?: string;
    }>;
    const useIndex = parts.findIndex((part) => part.type === 'server_tool_use');
    const resultIndex = parts.findIndex(
      (part) => part.type === 'web_search_tool_result'
    );
    expect(useIndex).toBeGreaterThanOrEqual(0);
    expect(resultIndex).toBe(useIndex + 1);
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

describe('formatAgentMessages trailing-steer anchor', () => {
  const steerPart = (text: string): Record<string, unknown> =>
    ({
      type: ContentTypes.STEER,
      [ContentTypes.STEER]: text,
    }) as unknown as Record<string, unknown>;

  /**
   * A steer that ends an assistant message leaves the replay on a
   * HumanMessage. Followed by the next turn's user message, that reaches the
   * provider as two adjacent user turns — a hard error on Bedrock and
   * Mistral. Reachable today through abort, not only through preemption.
   */
  it('anchors a trailing steer when another turn follows', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Working on it.' },
          steerPart('Actually, stop and do the other thing.'),
        ],
      },
      { role: 'user', content: 'Next turn' },
    ];

    const { messages } = formatAgentMessages(payload);

    const steerIndex = messages.findIndex(
      (message) =>
        message instanceof HumanMessage &&
        message.additional_kwargs.source === 'steer'
    );
    expect(steerIndex).toBeGreaterThan(0);

    const anchor = messages[steerIndex + 1];
    expect(anchor).toBeInstanceOf(AIMessage);
    expect(anchor.content).not.toBe('');

    const following = messages[steerIndex + 2];
    expect(following).toBeInstanceOf(HumanMessage);
    expect(following.content).toEqual([{ type: 'text', text: 'Next turn' }]);
  });

  /**
   * `messagesStateReducer` treats a repeated id as replace-in-place, so an
   * anchor stamped with the shared `sourceMessageId` would overwrite the very
   * steer it exists to protect. It must reach the reducer unstamped.
   */
  it('does not stamp the anchor with the shared source message id', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        messageId: 'assistant_1',
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Partial answer.' },
          steerPart('Change of plan.'),
        ],
      } as unknown as TPayload[number],
      { role: 'user', content: 'Next turn' },
    ];

    const { messages } = formatAgentMessages(payload);

    const steerIndex = messages.findIndex(
      (message) => message.additional_kwargs.source === 'steer'
    );
    expect(steerIndex).toBeGreaterThan(0);
    expect(messages[steerIndex].id).toBe('assistant_1');
    expect(messages[steerIndex - 1].id).toBe('assistant_1');

    const anchor = messages[steerIndex + 1];
    expect(anchor).toBeInstanceOf(AIMessage);
    expect(anchor.content).not.toBe('');
    expect(anchor.id).not.toBe('assistant_1');
  });

  /**
   * The anchor exists to keep the sequence provider-valid, so the assertion
   * that matters is what a provider actually receives. An empty-string
   * assistant turn with no tool calls survives the Anthropic converter
   * verbatim — the empty-text repair covers only array content and tool-call
   * turns — so an empty anchor would trade adjacent user turns for an
   * empty-content 400.
   */
  it('survives the Anthropic converter with non-empty assistant content', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Partial answer.' },
          steerPart('Change of plan.'),
        ],
      },
      { role: 'user', content: 'Next turn' },
    ];

    const { messages } = formatAgentMessages(payload);
    const { messages: anthropicMessages } =
      _convertMessagesToAnthropicPayload(messages);

    for (const message of anthropicMessages) {
      if (message.role !== 'assistant') {
        continue;
      }
      if (typeof message.content === 'string') {
        expect(message.content.trim()).not.toBe('');
        continue;
      }
      expect(message.content.length).toBeGreaterThan(0);
    }

    const roles = anthropicMessages.map((message) => message.role);
    for (let i = 1; i < roles.length; i++) {
      expect(roles[i] === 'user' && roles[i - 1] === 'user').toBe(false);
    }
  });

  it('leaves the user turn last when the steer ends the payload', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Working on it.' },
          steerPart('Actually, stop and do the other thing.'),
        ],
      },
    ];

    const { messages } = formatAgentMessages(payload);

    const last = messages[messages.length - 1];
    expect(last).toBeInstanceOf(HumanMessage);
    expect(last.additional_kwargs.source).toBe('steer');
  });

  it('does not anchor when assistant content follows the steer', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Working on it.' },
          steerPart('Focus on item two.'),
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Focusing.' },
        ],
      },
      { role: 'user', content: 'Next turn' },
    ];

    const { messages } = formatAgentMessages(payload);

    const emptyAssistants = messages.filter(
      (message) => message instanceof AIMessage && message.content === ''
    );
    expect(emptyAssistants).toHaveLength(0);
  });

  it('never emits two adjacent user turns around a trailing steer', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Original request' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Partial answer.' },
          steerPart('Change of plan.'),
        ],
      },
      { role: 'user', content: 'Next turn' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Second answer.' },
          steerPart('Change again.'),
        ],
      },
      { role: 'user', content: 'Third turn' },
    ];

    const { messages } = formatAgentMessages(payload);

    for (let i = 1; i < messages.length; i++) {
      const adjacentHumans =
        messages[i] instanceof HumanMessage &&
        messages[i - 1] instanceof HumanMessage;
      expect(adjacentHumans).toBe(false);
    }
  });
});
