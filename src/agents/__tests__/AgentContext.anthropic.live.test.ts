// src/agents/__tests__/AgentContext.anthropic.live.test.ts
/**
 * Live Anthropic prompt-cache verification.
 *
 * Run with:
 * RUN_ANTHROPIC_PROMPT_CACHE_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- AgentContext.anthropic.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { describe, expect, it } from '@jest/globals';
import type * as t from '@/types';
import {
  runLiveTurn,
  assertSystemPayloadShape,
  buildDynamicInstructions,
  buildStableInstructions,
  waitForCachePropagation,
} from './promptCacheLiveHelpers';
import { Providers } from '@/common';

const shouldRunLive =
  process.env.RUN_ANTHROPIC_PROMPT_CACHE_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

const modelName =
  process.env.ANTHROPIC_PROMPT_CACHE_MODEL ?? 'claude-sonnet-4-5';
const providerLabel = 'Anthropic';

function createClientOptions(): t.AnthropicClientOptions {
  return {
    modelName,
    temperature: 0,
    maxTokens: 8,
    streaming: true,
    streamUsage: true,
    promptCache: true,
    clientOptions: {
      defaultHeaders: {
        'anthropic-beta': 'prompt-caching-2024-07-31',
      },
    },
  };
}

describeIfLive('AgentContext Anthropic prompt cache live API', () => {
  it('caches only the stable system prefix while dynamic tail changes', async () => {
    const nonce = `agent-cache-live-${Date.now()}`;
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
      agentId: 'live-cache-shape-check',
      provider: Providers.ANTHROPIC,
      clientOptions,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
      expectedContent: [
        {
          type: 'text',
          text: stableInstructions,
          cache_control: { type: 'ephemeral' },
        },
        {
          type: 'text',
          text: firstDynamicInstructions,
        },
      ],
    });

    const first = await runLiveTurn({
      provider: Providers.ANTHROPIC,
      providerLabel,
      clientOptions,
      runId: `${nonce}-first`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
    });

    expect(first.text.toLowerCase()).toContain('alpha');
    expect(first.usage.input_token_details?.cache_creation).toBeGreaterThan(0);
    expect(first.usage.input_token_details?.cache_read ?? 0).toBe(0);

    await waitForCachePropagation();

    const second = await runLiveTurn({
      provider: Providers.ANTHROPIC,
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
