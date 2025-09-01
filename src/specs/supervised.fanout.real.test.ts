/* eslint-disable @typescript-eslint/no-explicit-any */

import { config as loadEnv } from 'dotenv';
loadEnv();
import { HumanMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers } from '@/common';
import type * as t from '@/types';
import { StandardGraph } from '@/graphs/Graph';

// Guard: require OpenRouter API (and OpenAI in practice) for this real-provider fan-out test
const hasOpenRouter = (process.env.OPENROUTER_API_KEY ?? '').trim() !== '';
const describeIfOpenRouter = hasOpenRouter ? describe : describe.skip;

describeIfOpenRouter(
  'Supervised Router Fan-out (real providers, guarded)',
  () => {
    jest.setTimeout(60000);

    test('fan-out aggregates outputs from OpenAI and OpenRouter when enabled', async () => {
      const run = await Run.create<t.IState>({
        runId: 'supervised-fanout-real',
        graphConfig: {
          type: 'supervised',
          routerEnabled: true,
          llmConfig: {
            // Base model; not used when fan-out aggregator is active for the policy
            provider: Providers.OPENAI,
            model: 'gpt-4.1',
            streaming: true,
            streamUsage: true,
          } as unknown as t.LLMConfig,
          featureFlags: { fan_out: true },
          routingPolicies: [
            {
              stage: 'compare',
              model: Providers.OPENAI,
              parallel: true,
              agents: ['openai_fast', 'openrouter_r1'],
            },
          ],
          models: {
            openai_fast: {
              provider: Providers.OPENAI,
              model: 'gpt-4.1',
              streaming: true,
              streamUsage: true,
            } as unknown as t.LLMConfig,
            openrouter_r1: {
              provider: Providers.OPENROUTER,
              model: 'deepseek/deepseek-r1',
              streaming: true,
              streamUsage: true,
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
            } as unknown as t.LLMConfig,
          },
          instructions:
            'Provide two short alternative summaries and keep it concise.',
        },
        returnContent: true,
      });

      // Spy to confirm both providers are instantiated
      const getNewModelSpy = jest.spyOn(
        StandardGraph.prototype as any,
        'getNewModel'
      );

      const threadId = 'router-fanout-thread-1';
      const configV2 = {
        configurable: { thread_id: threadId },
        streamMode: 'values' as const,
        version: 'v2' as const,
      };

      const history: HumanMessage[] = [];
      history.push(
        new HumanMessage(
          'Summarize the benefits of fan-out vs single-model briefly.'
        )
      );
      const parts = await run.processStream({ messages: history }, configV2);

      // Assert both providers were used to build stage agents
      const providersInvoked = getNewModelSpy.mock.calls.map(
        (call: any[]) => call[0]?.provider as string
      );
      expect(providersInvoked).toEqual(
        expect.arrayContaining([Providers.OPENAI, Providers.OPENROUTER])
      );

      // Optional sanity: aggregated content exists when returnContent is true
      expect(parts && parts.length >= 1).toBe(true);

      getNewModelSpy.mockRestore();
    });
  }
);
