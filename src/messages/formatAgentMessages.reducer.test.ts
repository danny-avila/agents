import { convertMessagesToResponsesInput } from '@langchain/openai';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { MessageContentComplex } from '@/types';
import { getProviderMessageProvenance } from './provenance';
import { messagesStateReducer } from './reducer';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

const formatToolTurn = () =>
  formatAgentMessages([
    {
      role: 'assistant',
      messageId: 'msg_assistant_1',
      content: [
        {
          type: ContentTypes.TEXT,
          [ContentTypes.TEXT]: 'Running tool',
          tool_call_ids: ['tool_1'],
        },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_1',
            name: 'search',
            args: '{"query":"hello"}',
            output: 'world',
          },
        },
      ],
    },
  ]).messages;

describe('formatAgentMessages reducer compatibility', () => {
  it('does not expose derived message ids as OpenAI Responses item ids', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'msg_provider_1',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Finished',
          },
        ],
      },
    ]).messages;
    const input = convertMessagesToResponsesInput({
      messages,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(
      input
        .filter((item) => item.type === 'message')
        .map((item) => ('id' in item ? item.id : undefined))
    ).toEqual(['msg_provider_1', undefined]);
  });

  it('does not let a later source id replace a derived message', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'a',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
        ],
      },
      {
        role: 'user',
        messageId: 'a:derived:1',
        content: 'Continue',
      },
    ]).messages;

    const merged = messagesStateReducer([], messages);

    expect(merged).toHaveLength(3);
    expect(merged[0]).toBeInstanceOf(AIMessage);
    expect(merged[1]).toBeInstanceOf(ToolMessage);
    expect(merged[2]).toBeInstanceOf(HumanMessage);
  });

  it('uses a valid id when the preferred messageId is blank', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: '   ',
        id: 'fallback-id',
        content: 'hello',
      },
    ]).messages;

    expect(message.additional_kwargs.sourceMessageId).toBe('fallback-id');
    expect(message.additional_kwargs.sourceMessageIds).toEqual(['fallback-id']);
  });

  it('lets the reducer identify derived messages while retaining source correlation', () => {
    const messages = formatToolTurn();

    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[1]).toBeInstanceOf(ToolMessage);
    expect(messages.map((message) => message.id)).toEqual([
      'msg_assistant_1',
      undefined,
    ]);
    expect(messages.map((message) => message.lc_kwargs.id)).toEqual([
      'msg_assistant_1',
      undefined,
    ]);
    expect(
      messages.map((message) => message.additional_kwargs.sourceMessageId)
    ).toEqual(['msg_assistant_1', 'msg_assistant_1']);
    expect(
      messages.map((message) => message.additional_kwargs.sourceMessageIds)
    ).toEqual([['msg_assistant_1'], ['msg_assistant_1']]);
    expect(messages[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'msg_assistant_1',
          sourceContentPartIndices: [0, 1],
        },
      ],
    });
    expect(messages[1].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'msg_assistant_1',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const merged = messagesStateReducer([], messages);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toBeInstanceOf(AIMessage);
    expect(merged[1]).toBeInstanceOf(ToolMessage);
    expect(merged[0].id).toBe('msg_assistant_1');
    expect(merged[1].id).toEqual(expect.any(String));
    expect(merged[1].id).not.toBe(merged[0].id);
  });

  it('preserves the tool-call pair through a StateGraph input write', async () => {
    const State = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
      }),
    });
    let observedMessages: BaseMessage[] = [];
    const graph = new StateGraph(State)
      .addNode('capture', (state) => {
        observedMessages = state.messages;
        return {};
      })
      .addEdge(START, 'capture')
      .addEdge('capture', END)
      .compile();

    await graph.invoke({ messages: formatToolTurn() });

    expect(observedMessages).toHaveLength(2);
    expect(observedMessages[0]).toBeInstanceOf(AIMessage);
    expect((observedMessages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect(observedMessages[1]).toBeInstanceOf(ToolMessage);
  });

  it('keeps per-derived lineage when one assistant row emits model and user turns', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'mixed-row',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prefix' },
          { type: ContentTypes.STEER, [ContentTypes.STEER]: 'user steer' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model suffix' },
        ],
      },
    ]).messages;

    expect(messages.map((message) => message.getType())).toEqual([
      'ai',
      'human',
      'ai',
    ]);
    expect(
      messages.map((message) => message.additional_kwargs.provenance)
    ).toEqual([
      {
        version: 1,
        parts: [
          {
            attribution: 'model',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [0],
          },
        ],
      },
      {
        version: 1,
        parts: [
          {
            attribution: 'user',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [1],
          },
        ],
      },
      {
        version: 1,
        parts: [
          {
            attribution: 'model',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [2],
          },
        ],
      },
    ]);
  });

  it.each([
    ['user', 'user'],
    ['assistant', 'model'],
  ] as const)(
    'splits tool-result and %s prose attribution within one provider message',
    (role, proseAttribution) => {
      const [message] = formatAgentMessages([
        {
          role,
          messageId: `${role}-tool-result-row`,
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'tool-result-id',
              content: 'tool bytes',
            },
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'ordinary prose' },
          ],
        },
      ]).messages;

      expect(message.additional_kwargs.provenance).toEqual({
        version: 1,
        parts: [
          {
            attribution: 'tool',
            sourceMessageId: `${role}-tool-result-row`,
            sourceContentPartIndices: [0],
          },
          {
            attribution: proseAttribution,
            sourceMessageId: `${role}-tool-result-row`,
            sourceContentPartIndices: [1],
          },
        ],
      });
    }
  );

  it.each([
    'server_tool_call_result',
    'server_tool_result',
    'codeExecutionResult',
    'code_execution_tool_result',
    'mcp_tool_result',
  ])('attributes the known %s envelope to tool output', (type) => {
    const [message] = formatAgentMessages([
      {
        role: 'assistant',
        messageId: `${type}-row`,
        content: [
          { type, content: 'tool bytes' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prose' },
        ],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: `${type}-row`,
          sourceContentPartIndices: [0],
        },
        {
          attribution: 'model',
          sourceMessageId: `${type}-row`,
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('keeps an unknown tool-result-like user block user-attributed', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: 'untrusted-type-row',
        content: [{ type: 'attacker_tool_result', text: 'submitted bytes' }],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'untrusted-type-row',
          sourceContentPartIndices: [0],
        },
      ],
    });
  });

  it('ignores null runtime content entries without shifting raw indices', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: 'nullable-row',
        content: [
          null,
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
        ] as unknown as MessageContentComplex[],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'nullable-row',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const [assistantMessage] = formatAgentMessages(
      [
        {
          role: 'assistant',
          messageId: 'nullable-assistant-row',
          content: [
            null,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
          ] as unknown as MessageContentComplex[],
        },
      ],
      undefined,
      new Set(['search'])
    ).messages;

    expect(assistantMessage.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'nullable-assistant-row',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const [assistantMessageWithSkills] = formatAgentMessages(
      [
        {
          role: 'assistant',
          messageId: 'nullable-assistant-skills-row',
          content: [
            null,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
          ] as unknown as MessageContentComplex[],
        },
      ],
      undefined,
      undefined,
      new Map([['known-skill', 'body']])
    ).messages;

    expect(assistantMessageWithSkills.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'nullable-assistant-skills-row',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('accumulates large homogeneous content provenance in one linear group', () => {
    const count = 4_000;
    const [message] = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'large-assistant-row',
        content: Array.from({ length: count }, (_, index) => ({
          type: ContentTypes.TEXT,
          [ContentTypes.TEXT]: String(index),
        })),
      },
    ]).messages;

    const parts = getProviderMessageProvenance(message)!.parts;
    expect(parts).toHaveLength(1);
    const sourceContentPartIndices = parts[0]?.sourceContentPartIndices;
    expect(sourceContentPartIndices).toHaveLength(count);
    expect(sourceContentPartIndices?.[0]).toBe(0);
    expect(sourceContentPartIndices?.[count - 1]).toBe(count - 1);
  });

  it(
    'formats 150k tool derivatives without call-argument spreading',
    () => {
      const count = 150_000;
      const result = formatAgentMessages(
        [
          {
            role: 'assistant',
            messageId: 'large-tool-row',
            content: Array.from({ length: count }, (_, index) => ({
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: `call-${index}`,
                name: 'allowed',
                args: {},
                output: 'x',
              },
            })),
          },
        ],
        undefined,
        new Set(['allowed'])
      );

      expect(result.messages).toHaveLength(count + 1);
      expect(result.messages[0]).toBeInstanceOf(AIMessage);
      expect(result.messages[count]).toBeInstanceOf(ToolMessage);
      expect(
        getProviderMessageProvenance(result.messages[count])?.parts[0]
          .sourceContentPartIndices
      ).toEqual([count - 1]);
    },
    30_000
  );
});
