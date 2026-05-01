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

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it } from '@jest/globals';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ModelEndHandler } from '@/events';
import { AgentContext } from '../AgentContext';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

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

function buildStableInstructions(nonce: string): string {
  const records = Array.from(
    { length: 360 },
    (_, index) =>
      `Stable Bedrock cache record ${index}: nonce ${nonce}; keep this reference in the cacheable prefix and do not use it as the dynamic marker.`
  );
  return [
    'You are a Bedrock prompt-cache verification assistant.',
    'When asked for the dynamic marker, answer with only the marker value from the Dynamic Marker line.',
    ...records,
  ].join('\n');
}

function buildDynamicInstructions(marker: string): string {
  return [
    `Dynamic Marker: ${marker}`,
    'The Dynamic Marker line is runtime context and must remain after the Bedrock cache point.',
  ].join('\n');
}

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
  return {
    model,
    region,
    maxTokens: 8,
    streaming: true,
    streamUsage: true,
    promptCache: true,
    ...(getCredentials() != null ? { credentials: getCredentials() } : {}),
  };
}

async function assertSystemPayloadShape({
  stableInstructions,
  dynamicInstructions,
}: {
  stableInstructions: string;
  dynamicInstructions: string;
}): Promise<void> {
  const ctx = AgentContext.fromConfig({
    agentId: 'live-bedrock-cache-shape-check',
    provider: Providers.BEDROCK,
    clientOptions: createClientOptions(),
    instructions: stableInstructions,
    additional_instructions: dynamicInstructions,
  });

  const messages = await ctx.systemRunnable!.invoke([
    new HumanMessage('What is the dynamic marker?'),
  ]);

  expect(messages[0].content).toEqual([
    {
      type: 'text',
      text: stableInstructions,
    },
    {
      cachePoint: { type: 'default' },
    },
    {
      type: 'text',
      text: dynamicInstructions,
    },
  ]);
}

function latestUsage(
  collectedUsage: UsageMetadata[],
  label: string
): UsageMetadata {
  if (collectedUsage.length === 0) {
    throw new Error(`Missing Bedrock usage metadata for ${label}`);
  }
  return collectedUsage[collectedUsage.length - 1];
}

function collectText(parts: t.MessageContentComplex[] | undefined): string {
  return (parts ?? []).reduce((text, part) => {
    if (part.type === 'text') {
      return text + part.text;
    }
    return text;
  }, '');
}

async function runLiveTurn({
  runId,
  threadId,
  stableInstructions,
  dynamicInstructions,
}: {
  runId: string;
  threadId: string;
  stableInstructions: string;
  dynamicInstructions: string;
}): Promise<{
  text: string;
  usage: UsageMetadata;
}> {
  const collectedUsage: UsageMetadata[] = [];
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.BEDROCK,
        ...createClientOptions(),
      },
      instructions: stableInstructions,
      additional_instructions: dynamicInstructions,
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    },
  });

  const config = {
    configurable: { thread_id: threadId },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const contentParts = await run.processStream(
    {
      messages: [
        new HumanMessage('What is the dynamic marker? Reply with only it.'),
      ],
    },
    config
  );

  return {
    text: collectText(contentParts),
    usage: latestUsage(collectedUsage, runId),
  };
}

describeIfLive('AgentContext Bedrock prompt cache live API', () => {
  it('caches only the stable system prefix while dynamic tail changes', async () => {
    const nonce = `agent-bedrock-cache-live-${Date.now()}`;
    const stableInstructions = buildStableInstructions(nonce);
    const firstDynamicInstructions = buildDynamicInstructions('alpha');
    const secondDynamicInstructions = buildDynamicInstructions('bravo');

    await assertSystemPayloadShape({
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
    });

    const first = await runLiveTurn({
      runId: `${nonce}-first`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: firstDynamicInstructions,
    });

    expect(first.text.toLowerCase()).toContain('alpha');
    expect(first.usage.input_token_details?.cache_creation).toBeGreaterThan(0);
    expect(first.usage.input_token_details?.cache_read ?? 0).toBe(0);

    const second = await runLiveTurn({
      runId: `${nonce}-second`,
      threadId: `${nonce}-thread`,
      stableInstructions,
      dynamicInstructions: secondDynamicInstructions,
    });

    expect(second.text.toLowerCase()).toContain('bravo');
    expect(second.usage.input_token_details?.cache_read).toBeGreaterThan(0);
  }, 180_000);
});
