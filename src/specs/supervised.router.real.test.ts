/* eslint-disable @typescript-eslint/no-explicit-any */

import { config as loadEnv } from 'dotenv';
loadEnv();
import { HumanMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import type * as t from '@/types';
import { StandardGraph } from '@/graphs/Graph';

describe('Supervised Router (real providers)', () => {
  jest.setTimeout(120000);

  test('routes Anthropic -> OpenAI across stages with real calls', async () => {
    const llmConfig = getLLMConfig(Providers.ANTHROPIC);

    const run = await Run.create<t.IState>({
      runId: 'supervised-real-router',
      graphConfig: {
        type: 'supervised',
        llmConfig,
        routerEnabled: true,
        routingPolicies: [
          { stage: 'reasoning', model: Providers.ANTHROPIC },
          { stage: 'summarize', model: Providers.OPENAI },
        ],
        models: {
          reasoning: {
            provider: Providers.ANTHROPIC,
            model: 'claude-3-5-sonnet-20240620',
            streaming: true,
            streamUsage: true,
          },
          summarize: {
            provider: Providers.OPENAI,
            model: 'gpt-4.1',
            streaming: true,
            streamUsage: true,
          },
        },
        instructions: 'You are a helpful assistant. Keep responses concise.',
        additional_instructions:
          'User is named Jo and is located in New York, NY.',
      },
      returnContent: true,
    });

    // Spy on the actual graph's getNewModel to capture providers used per stage
    const getNewModelSpy = jest.spyOn(
      StandardGraph.prototype as any,
      'getNewModel'
    );

    const threadId = 'router-real-thread-1';
    const configV2 = {
      configurable: { thread_id: threadId },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const history: HumanMessage[] = [];

    // Stage 0 (reasoning) - Anthropic
    history.push(new HumanMessage('Hello!'));
    await run.processStream({ messages: history }, configV2);

    // Stage 1 (summarize) - OpenAI
    history.push(
      new HumanMessage('Summarize the previous response in one sentence.')
    );
    await run.processStream({ messages: history }, configV2);

    const providersInvoked = getNewModelSpy.mock.calls.map(
      (call: any[]) => call[0]?.provider as string
    );
    expect(providersInvoked).toEqual(
      expect.arrayContaining([Providers.ANTHROPIC, Providers.OPENAI])
    );

    getNewModelSpy.mockRestore();
  });

  test('routes Anthropic -> OpenAI -> OpenRouter across three stages', async () => {
    const llmConfig = getLLMConfig(Providers.ANTHROPIC);

    const run = await Run.create<t.IState>({
      runId: 'supervised-real-router-3',
      graphConfig: {
        type: 'supervised',
        llmConfig,
        routerEnabled: true,
        routingPolicies: [
          { stage: 'reasoning', model: Providers.ANTHROPIC },
          { stage: 'summarize', model: Providers.OPENAI },
          { stage: 'finalize', model: Providers.OPENROUTER },
        ],
        models: {
          reasoning: {
            provider: Providers.ANTHROPIC,
            model: 'claude-3-5-sonnet-20240620',
            streaming: true,
            streamUsage: true,
          },
          summarize: {
            provider: Providers.OPENAI,
            model: 'gpt-4.1',
            streaming: true,
            streamUsage: true,
          },
          // OpenRouter requires api key and baseURL to be present on the stage config
          finalize: {
            provider: Providers.OPENROUTER,
            model: 'deepseek/deepseek-r1',
            streaming: true,
            streamUsage: true,
            // pass through env-driven client options for OpenRouter

            ...(process.env.OPENROUTER_API_KEY
              ? ({ openAIApiKey: process.env.OPENROUTER_API_KEY } as any)
              : {}),

            ...(process.env.OPENROUTER_BASE_URL
              ? ({
                configuration: {
                  baseURL: process.env.OPENROUTER_BASE_URL,
                  defaultHeaders: {
                    'HTTP-Referer': 'https://librechat.ai',
                    'X-Title': 'LibreChat',
                  },
                },
              } as any)
              : {}),
            include_reasoning: true as boolean,
          },
        },
        instructions: 'Be concise.',
      },
      returnContent: true,
    });

    const getNewModelSpy = jest.spyOn(
      StandardGraph.prototype as any,
      'getNewModel'
    );

    const configV2 = {
      configurable: { thread_id: 'router-real-thread-2' },
      streamMode: 'values',
      version: 'v2' as const,
    };
    const history: HumanMessage[] = [];

    history.push(new HumanMessage('Hello!'));
    await run.processStream({ messages: history }, configV2);
    history.push(new HumanMessage('Summarize that in one sentence.'));
    await run.processStream({ messages: history }, configV2);
    history.push(new HumanMessage('Polish into a concise final answer.'));
    await run.processStream({ messages: history }, configV2);

    const providersInvoked = getNewModelSpy.mock.calls.map(
      (call: any[]) => call[0]?.provider as string
    );
    expect(providersInvoked).toEqual(
      expect.arrayContaining([
        Providers.ANTHROPIC,
        Providers.OPENAI,
        Providers.OPENROUTER,
      ])
    );

    getNewModelSpy.mockRestore();
  });
});
