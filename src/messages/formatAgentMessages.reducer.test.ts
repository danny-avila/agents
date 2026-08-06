import { convertMessagesToResponsesInput } from '@langchain/openai';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
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
});
