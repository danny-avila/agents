import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
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
  it('assigns stable unique ids while retaining the source message id', () => {
    const messages = formatToolTurn();

    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[1]).toBeInstanceOf(ToolMessage);
    expect(messages.map((message) => message.id)).toEqual([
      'msg_assistant_1',
      'msg_assistant_1:derived:1',
    ]);
    expect(messages.map((message) => message.lc_kwargs.id)).toEqual([
      'msg_assistant_1',
      'msg_assistant_1:derived:1',
    ]);
    expect(
      messages.map((message) => message.additional_kwargs.sourceMessageId)
    ).toEqual(['msg_assistant_1', 'msg_assistant_1']);

    const merged = messagesStateReducer([], messages);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toBeInstanceOf(AIMessage);
    expect(merged[1]).toBeInstanceOf(ToolMessage);
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
