/**
 * Live proof that Anthropic can produce one tool call containing several
 * questions, pause once, and continue from one keyed batch resolution.
 *
 * Run with:
 * RUN_ASK_USER_QUESTIONS_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- ask-user-questions.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig(
  process.env.DOTENV_CONFIG_PATH != null
    ? { path: process.env.DOTENV_CONFIG_PATH }
    : undefined
);

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';
import { describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Providers } from '@/common';
import { askUserQuestions } from '@/hitl';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_ASK_USER_QUESTIONS_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';
const describeIfLive = shouldRunLive ? describe : describe.skip;
const modelName =
  process.env.ANTHROPIC_BATCH_QUESTIONS_LIVE_MODEL ?? 'claude-sonnet-5';

const questionSchema = z.object({
  id: z.enum(['metric', 'window']),
  header: z.string().max(20),
  question: z.string(),
  options: z
    .array(z.object({ label: z.string().max(120), value: z.string() }))
    .min(2)
    .max(3),
  multiSelect: z.boolean(),
});
const askUserQuestionsSchema = z.object({
  questions: z.array(questionSchema).length(2),
});
type AskUserQuestionsInput = z.infer<typeof askUserQuestionsSchema>;

const askTool = tool(
  async (input: AskUserQuestionsInput, config) => {
    const resolution = askUserQuestions(input, {
      toolCallId: config.toolCall?.id,
    });
    return JSON.stringify(resolution);
  },
  {
    name: 'ask_user_question',
    description:
      'Ask the user one to four related questions in one interaction. Put every question in this single tool call.',
    schema: askUserQuestionsSchema,
  }
);

type LiveStreamConfig = Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
};

function streamConfig(threadId: string): LiveStreamConfig {
  return {
    configurable: { thread_id: threadId },
    streamMode: 'values',
    version: 'v2',
  };
}

function messageText(message: BaseMessage): string {
  if (typeof message.content === 'string') {
    return message.content;
  }
  if (!Array.isArray(message.content)) {
    return '';
  }
  return message.content
    .map((part) =>
      typeof part === 'object' &&
      'text' in part &&
      typeof part.text === 'string'
        ? part.text
        : ''
    )
    .join('');
}

describeIfLive('askUserQuestions live Anthropic integration', () => {
  jest.setTimeout(120_000);

  it('uses one batched call and continues after one composite answer', async () => {
    const nonce = `batch-questions-${Date.now()}`;
    const saver = new MemorySaver();
    const run = await Run.create<t.IState>({
      runId: `${nonce}-run`,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'clarifier',
            provider: Providers.ANTHROPIC,
            clientOptions: {
              modelName,
              apiKey: process.env.ANTHROPIC_API_KEY,
              maxTokens: 512,
              streaming: true,
            },
            instructions: `You are testing a batched clarification tool.
On the first turn, call ask_user_question exactly once. In that one call, ask exactly two questions:
- id "metric": whether to analyze "workload" or "website"
- id "window": whether to analyze "24h" or "7d"
Do not emit two tool calls and do not answer in prose before the tool result.
After the tool returns, reply exactly: LIVE_BATCH_OK metric=<metric>; window=<window>`,
            maxContextTokens: 8000,
            graphTools: [askTool],
          },
        ],
        compileOptions: { checkpointer: saver },
      },
      returnContent: true,
      skipCleanup: true,
      interruptingToolNames: ['ask_user_question'],
    });
    const config = streamConfig(`${nonce}-thread`);

    await run.processStream(
      {
        messages: [
          new HumanMessage(
            'Clarify both dimensions before doing any analysis.'
          ),
        ],
      },
      config
    );

    const pending = run.getInterrupt();
    expect(pending?.payload.type).toBe('ask_user_question');
    if (pending?.payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question interrupt');
    }
    expect(pending.payload.questions).toHaveLength(2);
    expect(pending.payload.questions?.map(({ id }) => id)).toEqual([
      'metric',
      'window',
    ]);

    await run.resume<t.AskUserQuestionsResolution>(
      { answers: { metric: 'workload', window: '7d' } },
      config
    );

    expect(run.getInterrupt()).toBeUndefined();
    const messages = run.getRunMessages() ?? [];
    const askCalls = messages.flatMap((message) => {
      if (message.getType() !== 'ai') {
        return [];
      }
      return ((message as AIMessage).tool_calls ?? []).filter(
        ({ name }) => name === 'ask_user_question'
      );
    });
    expect(askCalls).toHaveLength(1);
    expect(askCalls[0].args).toMatchObject({
      questions: expect.arrayContaining([
        expect.objectContaining({ id: 'metric' }),
        expect.objectContaining({ id: 'window' }),
      ]),
    });

    const finalText = messages
      .filter((message) => message.getType() === 'ai')
      .map(messageText)
      .join('\n');
    expect(finalText).toContain('LIVE_BATCH_OK metric=workload; window=7d');
  });
});
