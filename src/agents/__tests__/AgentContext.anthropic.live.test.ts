// src/agents/__tests__/AgentContext.anthropic.live.test.ts
/**
 * Live Anthropic prompt-cache verification.
 *
 * Run with:
 * RUN_ANTHROPIC_PROMPT_CACHE_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- AgentContext.anthropic.live.test.ts --runInBand
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

const shouldRunLive =
  process.env.RUN_ANTHROPIC_PROMPT_CACHE_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

const modelName =
  process.env.ANTHROPIC_PROMPT_CACHE_MODEL ?? 'claude-3-haiku-20240307';

function buildStableInstructions(nonce: string): string {
  const records = Array.from(
    { length: 360 },
    (_, index) =>
      `Stable cache record ${index}: nonce ${nonce}; keep this reference in the cacheable prefix and do not use it as the dynamic marker.`
  );
  return [
    'You are a prompt-cache verification assistant.',
    'When asked for the dynamic marker, answer with only the marker value from the Dynamic Marker line.',
    ...records,
  ].join('\n');
}

function buildDynamicInstructions(marker: string): string {
  return [
    `Dynamic Marker: ${marker}`,
    'The Dynamic Marker line is runtime context and must remain outside the cached prefix.',
  ].join('\n');
}

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

async function assertSystemPayloadShape({
  stableInstructions,
  dynamicInstructions,
}: {
  stableInstructions: string;
  dynamicInstructions: string;
}): Promise<void> {
  const ctx = AgentContext.fromConfig({
    agentId: 'live-cache-shape-check',
    provider: Providers.ANTHROPIC,
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
      cache_control: { type: 'ephemeral' },
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
    throw new Error(`Missing Anthropic usage metadata for ${label}`);
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
        provider: Providers.ANTHROPIC,
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

describeIfLive('AgentContext Anthropic prompt cache live API', () => {
  it('caches only the stable system prefix while dynamic tail changes', async () => {
    const nonce = `agent-cache-live-${Date.now()}`;
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
  }, 120_000);
});
