// src/agents/__tests__/AgentContext.bedrock.live.test.ts
/**
 * Live Bedrock prompt-cache verification.
 *
 * Run with:
 * RUN_BEDROCK_PROMPT_CACHE_LIVE_TESTS=1 BEDROCK_AWS_REGION=... BEDROCK_AWS_ACCESS_KEY_ID=... BEDROCK_AWS_SECRET_ACCESS_KEY=... npm test -- AgentContext.bedrock.live.test.ts --runInBand
 *
 * Standard AWS credential env vars or AWS_PROFILE can also be used.
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

const accessKeyId =
  process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? process.env.AWS_ACCESS_KEY_ID;
const secretAccessKey =
  process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ??
  process.env.AWS_SECRET_ACCESS_KEY;
const sessionToken =
  process.env.BEDROCK_AWS_SESSION_TOKEN ?? process.env.AWS_SESSION_TOKEN;
const hasCredentialPair =
  accessKeyId != null &&
  accessKeyId !== '' &&
  secretAccessKey != null &&
  secretAccessKey !== '';
const hasAmbientCredentials =
  process.env.AWS_PROFILE != null ||
  process.env.AWS_WEB_IDENTITY_TOKEN_FILE != null;

const shouldRunLive =
  process.env.RUN_BEDROCK_PROMPT_CACHE_LIVE_TESTS === '1' &&
  (hasCredentialPair || hasAmbientCredentials);

const describeIfLive = shouldRunLive ? describe : describe.skip;

const model =
  process.env.BEDROCK_PROMPT_CACHE_MODEL ??
  'us.anthropic.claude-sonnet-4-5-20250929-v1:0';
const region =
  process.env.BEDROCK_AWS_REGION ?? process.env.AWS_REGION ?? 'us-east-1';
const providerLabel = 'Bedrock';

function getCredentials():
  | t.BedrockAnthropicClientOptions['credentials']
  | undefined {
  if (!hasCredentialPair) {
    return undefined;
  }

  return {
    accessKeyId,
    secretAccessKey,
    ...(sessionToken != null && sessionToken !== '' ? { sessionToken } : {}),
  };
}

function createClientOptions(): t.BedrockAnthropicClientOptions {
  const credentials = getCredentials();
  return {
    model,
    region,
    maxTokens: 8,
    streaming: true,
    streamUsage: true,
    promptCache: true,
    ...(credentials != null ? { credentials } : {}),
  };
}

describeIfLive('AgentContext Bedrock prompt cache live API', () => {
  it('caches only the stable system prefix while dynamic tail changes', async () => {
    const nonce = `agent-bedrock-cache-live-${Date.now()}`;
    const clientOptions = createClientOptions();
    const stableInstructions = buildStableInstructions({
      nonce,
      providerLabel,
    });
    const firstDynamicInstructions = buildDynamicInstructions({
      marker: 'alpha',
      tailDescription:
        'The Dynamic Marker line is runtime context and must remain after the Bedrock cache point.',
    });
    const secondDynamicInstructions = buildDynamicInstructions({
      marker: 'bravo',
      tailDescription:
        'The Dynamic Marker line is runtime context and must remain after the Bedrock cache point.',
    });

    await assertSystemPayloadShape({
      agentId: 'live-bedrock-cache-shape-check',
      provider: Providers.BEDROCK,
      clientOptions,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
      expectedContent: [
        {
          type: 'text',
          text: stableInstructions,
        },
        {
          cachePoint: { type: 'default' },
        },
        {
          type: 'text',
          text: firstDynamicInstructions,
        },
      ],
    });

    const first = await runLiveTurn({
      provider: Providers.BEDROCK,
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
      provider: Providers.BEDROCK,
      providerLabel,
      clientOptions,
      runId: `${nonce}-second`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: secondDynamicInstructions,
    });

    expect(second.text.toLowerCase()).toContain('bravo');
    expect(second.usage.input_token_details?.cache_read).toBeGreaterThan(0);
  }, 180_000);
});
