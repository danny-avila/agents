/* eslint-disable no-console */
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Run } from '@/run';
import { Providers } from '@/common';
import { FanoutMetricsHandler } from '@/events';

const conversationHistory: BaseMessage[] = [];

async function runFanOutDemo(): Promise<void> {
  // Note: Durable checkpointer example omitted here to avoid module availability issues.

  const run = await Run.create<t.IState>({
    runId: 'supervised-fanout-demo',
    graphConfig: {
      type: 'supervised',
      routerEnabled: true,
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4.1',
        streaming: true,
        streamUsage: true,
      } as unknown as t.LLMConfig,
      featureFlags: { fan_out: true, fan_out_retries: 1, fan_out_backoff_ms: 400, fan_out_concurrency: 2 },
      routingPolicies: [
        { stage: 'compare', model: Providers.OPENAI, parallel: true, agents: ['openai_fast', 'openrouter_r1'] },
      ],
      models: {
        openai_fast: {
          provider: Providers.OPENAI,
          model: 'gpt-4.1',
          streaming: true,
          streamUsage: true,
        } as unknown as t.LLMConfig,
        // Synthesizer model to collapse fan-out outputs to one answer
        synth: {
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
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ...(process.env.OPENROUTER_API_KEY ? { openAIApiKey: process.env.OPENROUTER_API_KEY } as any : {}),
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ...(process.env.OPENROUTER_BASE_URL ? { configuration: { baseURL: process.env.OPENROUTER_BASE_URL, defaultHeaders: { 'HTTP-Referer': 'https://librechat.ai', 'X-Title': 'LibreChat' } } } as any : {}),
          include_reasoning: true as boolean,
        } as unknown as t.LLMConfig,
      },
      instructions: 'Provide two short alternative summaries and keep it concise.',
    },
    returnContent: true,
    customHandlers: { fanout_branch_end: new FanoutMetricsHandler() },
  });

  const configV2 = { configurable: { thread_id: 'fanout-demo-1' }, streamMode: 'values' as const, version: 'v2' as const };

  conversationHistory.push(new HumanMessage('Summarize the tradeoffs of using fan-out vs a single model.'));
  const parts = await run.processStream({ messages: conversationHistory }, configV2);

  console.log('--- Aggregated Output ---');
  if (parts) {
    for (const p of parts) {
      if (p?.type === 'text') {
        console.log(p.text);
      }
    }
  }
}

runFanOutDemo().catch((err) => {
  console.error(err);
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});


