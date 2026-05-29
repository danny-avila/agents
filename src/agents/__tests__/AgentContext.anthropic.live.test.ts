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
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type MessageContentComplex,
} from '@langchain/core/messages';
import type * as t from '@/types';
import {
  runLiveTurn,
  assertSystemPayloadShape,
  buildDynamicInstructions,
  buildStableInstructions,
  waitForCachePropagation,
} from './promptCacheLiveHelpers';
import { Providers } from '@/common';
import { addCacheControl } from '@/messages/cache';
import { toLangChainContent } from '@/messages/langchain';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';

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

type AnthropicCacheUsage = {
  inputTokens: number;
  outputTokens: number;
  cacheCreation: number;
  cacheRead: number;
  latencyMs: number;
};

type AnthropicUsageResponse = {
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_creation_input_tokens?: number | null;
    cache_read_input_tokens?: number | null;
  };
};

type AnthropicMessagesClient = {
  messages: {
    create: (
      request: Record<string, unknown>,
      options: { headers: Record<string, string> }
    ) => Promise<AnthropicUsageResponse>;
  };
};

const benchmarkTool = {
  name: 'lookup_cache_probe',
  description: 'Returns prompt cache benchmark data.',
  input_schema: {
    type: 'object',
    properties: {
      step: { type: 'integer' },
    },
    required: ['step'],
  },
};

function cloneLiveMessage(
  message: BaseMessage,
  content: MessageContentComplex[]
): BaseMessage {
  const baseParams = {
    content: toLangChainContent(content),
    additional_kwargs: { ...message.additional_kwargs },
    response_metadata: { ...message.response_metadata },
    id: message.id,
    name: message.name,
  };

  const messageType = message.getType();
  if (messageType === 'ai') {
    return new AIMessage({
      ...baseParams,
      tool_calls: (message as AIMessage).tool_calls,
    });
  }
  if (messageType === 'human') {
    return new HumanMessage(baseParams);
  }
  if (messageType === 'system') {
    return new SystemMessage(baseParams);
  }
  if (messageType === 'tool') {
    return new ToolMessage({
      ...baseParams,
      tool_call_id: (message as ToolMessage).tool_call_id,
    });
  }

  return message;
}

function addLatestUserOnlyAnthropicCacheControl(
  messages: BaseMessage[]
): BaseMessage[] {
  const updatedMessages = [...messages];
  let addedCacheControl = false;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const message = updatedMessages[i];
    const content = message.content;
    const hasArrayContent = Array.isArray(content);
    const canAddCache =
      !addedCacheControl &&
      message.getType() === 'human' &&
      (typeof content === 'string' || hasArrayContent);

    if (!canAddCache && !hasArrayContent) {
      continue;
    }

    let workingContent: MessageContentComplex[];
    let modified = false;

    if (hasArrayContent) {
      workingContent = [];
      let lastTextIndex = -1;
      for (const block of content as MessageContentComplex[]) {
        if ('cachePoint' in block && !('type' in block)) {
          modified = true;
          continue;
        }
        const cloned = { ...block };
        if ('cache_control' in cloned) {
          delete (cloned as Record<string, unknown>).cache_control;
          modified = true;
        }
        if ('type' in cloned && cloned.type === 'text') {
          const text = (cloned as { text?: string }).text;
          if (text != null && text.trim() !== '') {
            lastTextIndex = workingContent.length;
          }
        }
        workingContent.push(cloned as MessageContentComplex);
      }

      if (canAddCache && lastTextIndex >= 0) {
        (
          workingContent[lastTextIndex] as MessageContentComplex & {
            cache_control?: { type: 'ephemeral' };
          }
        ).cache_control = { type: 'ephemeral' };
        addedCacheControl = true;
        modified = true;
      }

      if (!modified) {
        continue;
      }
    } else if (typeof content === 'string' && content.trim() !== '' && canAddCache) {
      workingContent = [
        {
          type: 'text',
          text: content,
          cache_control: { type: 'ephemeral' },
        },
      ] as unknown as MessageContentComplex[];
      addedCacheControl = true;
    } else {
      continue;
    }

    updatedMessages[i] = cloneLiveMessage(message, workingContent);
  }

  return updatedMessages;
}

function repeated(label: string, count: number): string {
  return Array.from(
    { length: count },
    (_, index) =>
      `${label} reference ${index}: stable schema, metric definition, access policy, dashboard note, and query planning guidance.`
  ).join('\n');
}

function buildMultiTurnToolMessages({
  nonce,
  marker,
}: {
  nonce: string;
  marker: string;
}): BaseMessage[] {
  const stableFirstUser = [
    `Anthropic prompt cache placement benchmark ${nonce}.`,
    'This first user turn is intentionally stable across calls in the same benchmark case.',
    repeated(`${nonce} stable-user-context`, 190),
  ].join('\n');
  const latestUser = [
    `Current user request marker: ${marker}.`,
    'Use the final tool result to answer with the marker only.',
    repeated(`${nonce} latest-user-${marker}`, 18),
  ].join('\n');
  const volatileToolPayload = repeated(`${nonce} volatile-tool-${marker}`, 70);

  return [
    new HumanMessage(stableFirstUser),
    new AIMessage('I will keep this stable context in mind.'),
    new HumanMessage(latestUser),
    new AIMessage({
      content: `I will inspect cache probe step 1 for ${marker}.\n${volatileToolPayload}`,
      tool_calls: [
        {
          id: `call_${marker}_1`,
          name: 'lookup_cache_probe',
          args: { step: 1 },
        },
      ],
    }),
    new ToolMessage({
      content: `Tool result 1 for ${marker}.\n${volatileToolPayload}`,
      tool_call_id: `call_${marker}_1`,
    }),
    new AIMessage({
      content: `I will inspect cache probe step 2 for ${marker}.\n${volatileToolPayload}`,
      tool_calls: [
        {
          id: `call_${marker}_2`,
          name: 'lookup_cache_probe',
          args: { step: 2 },
        },
      ],
    }),
    new ToolMessage({
      content: [
        `Final tool result marker: ${marker}.`,
        'Reply with the marker and no extra explanation.',
        volatileToolPayload,
      ].join('\n'),
      tool_call_id: `call_${marker}_2`,
    }),
  ];
}

function extractCacheUsage(
  response: AnthropicUsageResponse,
  latencyMs: number
): AnthropicCacheUsage {
  if (response.usage == null) {
    throw new Error('Missing Anthropic usage metadata for cache benchmark');
  }

  return {
    inputTokens: response.usage.input_tokens ?? 0,
    outputTokens: response.usage.output_tokens ?? 0,
    cacheCreation: response.usage.cache_creation_input_tokens ?? 0,
    cacheRead: response.usage.cache_read_input_tokens ?? 0,
    latencyMs,
  };
}

async function runAnthropicCacheBenchmarkTurn({
  client,
  messages,
}: {
  client: AnthropicMessagesClient;
  messages: BaseMessage[];
}): Promise<AnthropicCacheUsage> {
  const payload = _convertMessagesToAnthropicPayload(messages);
  const startedAt = Date.now();
  const response = await client.messages.create(
    {
      ...payload,
      model: modelName,
      max_tokens: 16,
      temperature: 0,
      tools: [benchmarkTool],
    },
    {
      headers: {
        'anthropic-beta': 'prompt-caching-2024-07-31',
      },
    }
  );

  return extractCacheUsage(
    response as AnthropicUsageResponse,
    Date.now() - startedAt
  );
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

  it('compares current two-user cache placement against latest-user-only', async () => {
    const Anthropic = (await import('@anthropic-ai/sdk')).default;
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    }) as unknown as AnthropicMessagesClient;
    const nonce = `anthropic-cache-placement-${Date.now()}`;
    const currentNonce = `${nonce}-current`;
    const latestOnlyNonce = `${nonce}-latest-only`;

    const currentFirst = await runAnthropicCacheBenchmarkTurn({
      client,
      messages: addCacheControl(
        buildMultiTurnToolMessages({ nonce: currentNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const currentSecond = await runAnthropicCacheBenchmarkTurn({
      client,
      messages: addCacheControl(
        buildMultiTurnToolMessages({ nonce: currentNonce, marker: 'bravo' })
      ),
    });

    const latestOnlyFirst = await runAnthropicCacheBenchmarkTurn({
      client,
      messages: addLatestUserOnlyAnthropicCacheControl(
        buildMultiTurnToolMessages({ nonce: latestOnlyNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const latestOnlySecond = await runAnthropicCacheBenchmarkTurn({
      client,
      messages: addLatestUserOnlyAnthropicCacheControl(
        buildMultiTurnToolMessages({ nonce: latestOnlyNonce, marker: 'bravo' })
      ),
    });

    process.stdout.write(
      `Anthropic cache placement benchmark ${JSON.stringify({
        currentFirst,
        currentSecond,
        latestOnlyFirst,
        latestOnlySecond,
        cacheWriteDelta:
          currentSecond.cacheCreation - latestOnlySecond.cacheCreation,
      })}\n`
    );

    expect(currentSecond.cacheRead).toBeGreaterThan(0);
    expect(currentSecond.cacheRead).toBeGreaterThan(latestOnlySecond.cacheRead);
    expect(currentSecond.cacheCreation).toBeLessThan(
      latestOnlySecond.cacheCreation
    );
  }, 180_000);
});
