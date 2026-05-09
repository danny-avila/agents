// src/agents/__tests__/AgentContext.openrouter.live.test.ts
/**
 * Live OpenRouter prompt-cache verification.
 *
 * Run with:
 * RUN_OPENROUTER_PROMPT_CACHE_LIVE_TESTS=1 OPENROUTER_API_KEY=... npm test -- AgentContext.openrouter.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig({ path: process.env.DOTENV_CONFIG_PATH ?? '.env' });

import { describe, expect, it } from '@jest/globals';
import type { ClientOptions } from '@langchain/openai';
import {
  runLiveTurn,
  assertSystemPayloadShape,
  buildDynamicInstructions,
  buildStableInstructions,
  waitForCachePropagation,
} from './promptCacheLiveHelpers';
import type { ChatOpenRouterInput } from '@/llm/openrouter';
import { Providers } from '@/common';

const apiKey = process.env.OPENROUTER_API_KEY ?? process.env.OPENROUTER_KEY;
const shouldRunLive =
  process.env.RUN_OPENROUTER_PROMPT_CACHE_LIVE_TESTS === '1' &&
  apiKey != null &&
  apiKey !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

const model =
  process.env.OPENROUTER_PROMPT_CACHE_MODEL ?? 'anthropic/claude-sonnet-4.6';
const providerLabel = 'OpenRouter';
type OpenRouterLiveClientOptions = ChatOpenRouterInput & {
  configuration?: ClientOptions;
};

function createClientOptions(): OpenRouterLiveClientOptions {
  if (apiKey == null || apiKey === '') {
    throw new Error('OPENROUTER_API_KEY is required');
  }

  const reasoning = model.startsWith('google/gemini-3')
    ? { max_tokens: 16 }
    : undefined;

  return {
    model,
    apiKey,
    temperature: 0,
    maxTokens: 256,
    streaming: true,
    streamUsage: true,
    promptCache: true,
    configuration: {
      baseURL:
        process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://librechat.ai',
        'X-Title': 'LibreChat OpenRouter Prompt Cache Live Test',
      },
    },
    ...(reasoning != null ? { reasoning } : {}),
  };
}

describeIfLive('AgentContext OpenRouter prompt cache live API', () => {
  it('keeps dynamic instructions outside the cached system prefix', async () => {
    const nonce = `agent-openrouter-cache-live-${Date.now()}`;
    const clientOptions = createClientOptions();
    const stableInstructions = buildStableInstructions({
      nonce,
      providerLabel,
    });
    const firstDynamicInstructions = buildDynamicInstructions({
      marker: 'alpha',
      tailDescription:
        'The Dynamic Marker line is runtime context and must remain outside the cached prefix.',
    });
    const secondDynamicInstructions = buildDynamicInstructions({
      marker: 'bravo',
      tailDescription:
        'The Dynamic Marker line is runtime context and must remain outside the cached prefix.',
    });

    await assertSystemPayloadShape({
      agentId: 'live-openrouter-cache-shape-check',
      provider: Providers.OPENROUTER,
      clientOptions,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
      expectedContent: [
        {
          type: 'text',
          text: stableInstructions,
          cache_control: { type: 'ephemeral' },
        },
      ],
    });

    const first = await runLiveTurn({
      provider: Providers.OPENROUTER,
      providerLabel,
      clientOptions,
      runId: `${nonce}-first`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
    });

    expect(first.text.toLowerCase()).toContain('alpha');

    await waitForCachePropagation();

    const second = await runLiveTurn({
      provider: Providers.OPENROUTER,
      providerLabel,
      clientOptions,
      runId: `${nonce}-second`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: secondDynamicInstructions,
    });

    expect(second.text.toLowerCase()).toContain('bravo');
    expect(second.usage.input_token_details?.cache_read).toBeGreaterThan(0);
  }, 120_000);
});
