import { expect } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ModelEndHandler } from '@/events';
import { AgentContext } from '../AgentContext';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

type LivePromptCacheProvider = Providers.ANTHROPIC | Providers.BEDROCK;

type PromptCacheExpectedSystemBlock =
  | { type: 'text'; text: string; cache_control?: { type: 'ephemeral' } }
  | { cachePoint: { type: 'default' } };

type LivePromptCacheClientOptions =
  | t.ClientOptions
  | t.BedrockAnthropicClientOptions;

export function buildStableInstructions({
  nonce,
  providerLabel,
}: {
  nonce: string;
  providerLabel: string;
}): string {
  const records = Array.from(
    { length: 360 },
    (_, index) =>
      `Stable ${providerLabel} cache record ${index}: nonce ${nonce}; keep this reference in the cacheable prefix and do not use it as the dynamic marker.`
  );
  return [
    `You are a ${providerLabel} prompt-cache verification assistant.`,
    'When asked for the dynamic marker, answer with only the marker value from the Dynamic Marker line.',
    ...records,
  ].join('\n');
}

export function buildDynamicInstructions({
  marker,
  tailDescription,
}: {
  marker: string;
  tailDescription: string;
}): string {
  return [`Dynamic Marker: ${marker}`, tailDescription].join('\n');
}

export function waitForCachePropagation(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 2000));
}

export async function assertSystemPayloadShape({
  agentId,
  provider,
  clientOptions,
  stableInstructions,
  dynamicInstructions,
  expectedContent,
}: {
  agentId: string;
  provider: LivePromptCacheProvider;
  clientOptions: LivePromptCacheClientOptions;
  stableInstructions: string;
  dynamicInstructions: string;
  expectedContent: PromptCacheExpectedSystemBlock[];
}): Promise<void> {
  const ctx = AgentContext.fromConfig({
    agentId,
    provider,
    clientOptions,
    instructions: stableInstructions,
    additional_instructions: dynamicInstructions,
  });

  const messages = await ctx.systemRunnable!.invoke([
    new HumanMessage('What is the dynamic marker?'),
  ]);

  expect(messages[0].content).toEqual(expectedContent);
}

function latestUsage({
  collectedUsage,
  label,
  providerLabel,
}: {
  collectedUsage: UsageMetadata[];
  label: string;
  providerLabel: string;
}): UsageMetadata {
  if (collectedUsage.length === 0) {
    throw new Error(`Missing ${providerLabel} usage metadata for ${label}`);
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

export async function runLiveTurn({
  provider,
  providerLabel,
  clientOptions,
  runId,
  threadId,
  stableInstructions,
  dynamicInstructions,
}: {
  provider: LivePromptCacheProvider;
  providerLabel: string;
  clientOptions: LivePromptCacheClientOptions;
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
        provider,
        ...clientOptions,
      } as t.LLMConfig,
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
    usage: latestUsage({ collectedUsage, label: runId, providerLabel }),
  };
}
