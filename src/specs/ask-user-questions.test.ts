import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, expect, it } from '@jest/globals';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
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
import type * as t from '@/types';
import {
  askUserQuestions,
  isAskUserQuestionsInterrupt,
  MAX_ASK_USER_QUESTIONS,
} from '@/hitl';
import { ToolNode } from '@/tools/ToolNode';

type MessagesUpdate = { messages: BaseMessage[] };
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
const askUserQuestionsSchema = z.object({
  questions: z.array(questionSchema).min(1).max(MAX_ASK_USER_QUESTIONS),
});
type AskUserQuestionsInput = z.infer<typeof askUserQuestionsSchema>;

function buildGraph() {
  const askTool = tool(
    async (input: AskUserQuestionsInput, config) => {
      const resolution = askUserQuestions(input, {
        toolCallId: config.toolCall?.id,
      });
      return JSON.stringify(resolution);
    },
    {
      name: 'ask_user_question',
      description: 'Ask several related questions in one interaction.',
      schema: askUserQuestionsSchema,
    }
  );
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
    });
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
    const sparseOptions: unknown[] = [];
    sparseOptions.length = 2;

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
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: [null],
      })
    ).toBe(false);
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: [{ id: '__proto__', question: 'Unsafe key?' }],
      })
    ).toBe(false);
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: Array.from(
          { length: MAX_ASK_USER_QUESTIONS + 1 },
          (_, index) => ({
            id: `question-${index}`,
            question: `Question ${index}?`,
          })
        ),
      })
    ).toBe(false);
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: [
          {
            id: 'choice',
            question: 'Choose?',
            options: [{ label: 'Missing value' }],
          },
        ],
      })
    ).toBe(false);
    expect(
      isAskUserQuestionsInterrupt({
        type: 'ask_user_question',
        question: { question: 'Proceed?' },
        questions: [
          {
            id: 'choice',
            question: 'Choose?',
            options: sparseOptions,
          },
        ],
      })
    ).toBe(false);
  });

  it('returns a tool error when a resumed batch omits an answer', async () => {
    const graph = buildGraph();
    const config = { configurable: { thread_id: 'incomplete-answers' } };

    await graph.invoke({ messages: [] }, config);
    const resumed = (await graph.invoke(
      new Command({ resume: { answers: { metric: 'workload' } } }),
      config
    )) as MessagesUpdate;
    const result = resumed.messages.find(
      (message): message is ToolMessage =>
        message.getType() === 'tool' &&
        (message as ToolMessage).name === 'ask_user_question'
    );

    expect(result?.status).toBe('error');
    expect(String(result?.content)).toContain(
      'requires a string answer for question id "window"'
    );
  });

  it('rejects empty question ids before raising an interrupt', () => {
    const request: t.AskUserQuestionsRequest = {
      questions: [{ ...questions[0], id: '  ' }],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'requires each question id to match'
    );
  });

  it('rejects unsafe answer-map keys before raising an interrupt', () => {
    const request: t.AskUserQuestionsRequest = {
      questions: [{ ...questions[0], id: '__proto__' }],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'requires each question id to match'
    );
  });

  it('rejects sparse option arrays before raising an interrupt', () => {
    const sparseOptions: t.AskUserQuestionOption[] = [];
    sparseOptions.length = 2;
    const request: t.AskUserQuestionsRequest = {
      questions: [{ ...questions[0], options: sparseOptions }],
    };

    expect(() => askUserQuestions(request)).toThrow(
      'requires each question and option to have valid string fields'
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
