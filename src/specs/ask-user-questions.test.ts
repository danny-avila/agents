import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, expect, it } from '@jest/globals';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { askUserQuestions, MAX_ASK_USER_QUESTIONS } from '@/hitl';
import { isAskUserQuestionsInterrupt } from '@/types/hitl';
import { ToolNode } from '@/tools/ToolNode';

type MessagesUpdate = { messages: BaseMessage[] };
type CompiledMessagesGraph = Runnable<unknown, MessagesUpdate> & {
  invoke(input: unknown, config?: RunnableConfig): Promise<unknown>;
};

const questions = [
  {
    id: 'metric',
    header: 'Metric',
    question: 'Which performance cost should be analyzed?',
    options: [
      { label: 'ClickHouse workload', value: 'workload' },
      { label: 'Website experience', value: 'website' },
    ],
  },
  {
    id: 'window',
    header: 'Window',
    question: 'Which time window should be used?',
    options: [
      { label: 'Last 24 hours', value: '24h' },
      { label: 'Last 7 days', value: '7d' },
    ],
  },
] satisfies t.AskUserQuestionBatchItem[];

const questionSchema = z.object({
  id: z.string(),
  header: z.string().optional(),
  question: z.string(),
  options: z
    .array(z.object({ label: z.string(), value: z.string() }))
    .optional(),
});

function buildGraph(): CompiledMessagesGraph {
  const askTool = tool(
    async (input: t.AskUserQuestionsRequest, config) => {
      const resolution = askUserQuestions(input, {
        toolCallId: config.toolCall?.id,
      });
      return JSON.stringify(resolution);
    },
    {
      name: 'ask_user_question',
      description: 'Ask several related questions in one interaction.',
      schema: z.object({ questions: z.array(questionSchema).min(1).max(4) }),
    }
  ) as unknown as StructuredToolInterface;
  const node = new ToolNode({
    tools: [askTool],
    directToolNames: new Set(['ask_user_question']),
    interruptingToolNames: new Set(['ask_user_question']),
  });

  return new StateGraph(MessagesAnnotation)
    .addNode(
      'agent',
      (): MessagesUpdate => ({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'batched-ask-call',
                name: 'ask_user_question',
                args: { questions },
              },
            ],
          }),
        ],
      })
    )
    .addNode('tools', node)
    .addEdge(START, 'agent')
    .addEdge('agent', 'tools')
    .addEdge('tools', END)
    .compile({
      checkpointer: new MemorySaver(),
    }) as unknown as CompiledMessagesGraph;
}

describe('askUserQuestions', () => {
  it('pauses once for a batch and resumes with keyed answers', async () => {
    const graph = buildGraph();
    const config = { configurable: { thread_id: 'batched-questions' } };

    const first = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(first)) {
      throw new Error('expected batched question interrupt');
    }
    expect(first.__interrupt__).toHaveLength(1);
    const payload = first.__interrupt__[0].value;
    expect(isAskUserQuestionsInterrupt(payload)).toBe(true);
    expect(payload).toMatchObject({
      type: 'ask_user_question',
      tool_call_id: 'batched-ask-call',
      question: { question: questions[0].question },
      questions,
    });

    const answers: t.AskUserQuestionsResolution = {
      answers: { metric: 'workload', window: '7d' },
    };
    const second = (await graph.invoke(
      new Command({ resume: answers }),
      config
    )) as MessagesUpdate;
    const result = second.messages.find(
      (message): message is ToolMessage =>
        message.getType() === 'tool' &&
        (message as ToolMessage).name === 'ask_user_question'
    );
    expect(result).toBeDefined();
    expect(JSON.parse(String(result!.content))).toEqual(answers);
  });

  it('rejects duplicate question ids before raising an interrupt', () => {
    const request: t.AskUserQuestionsRequest = {
      questions: [questions[0], { ...questions[1], id: questions[0].id }],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'requires unique question ids'
    );
  });

  it('distinguishes singular ask payloads from batched payloads', () => {
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
      })
    ).toBe(false);
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: [],
      })
    ).toBe(false);
  });

  it('rejects empty question ids before raising an interrupt', () => {
    const request: t.AskUserQuestionsRequest = {
      questions: [{ ...questions[0], id: '  ' }],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'requires every question to have an id'
    );
  });

  it('rejects batches larger than four questions', () => {
    expect(MAX_ASK_USER_QUESTIONS).toBe(4);
    const request: t.AskUserQuestionsRequest = {
      questions: [
        questions[0],
        questions[1],
        { ...questions[0], id: 'third' },
        { ...questions[0], id: 'fourth' },
        { ...questions[0], id: 'fifth' },
      ],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'accepts at most 4 questions'
    );
  });
});
