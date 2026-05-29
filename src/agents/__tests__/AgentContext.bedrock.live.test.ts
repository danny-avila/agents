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
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type MessageContentComplex,
} from '@langchain/core/messages';
import {
  BedrockRuntimeClient,
  ConverseCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type * as t from '@/types';
import {
  runLiveTurn,
  assertSystemPayloadShape,
  buildDynamicInstructions,
  buildStableInstructions,
  waitForCachePropagation,
} from './promptCacheLiveHelpers';
import { Providers } from '@/common';
import { addBedrockCacheControl } from '@/messages/cache';
import { toLangChainContent } from '@/messages/langchain';
import { convertToConverseMessages } from '@/llm/bedrock/utils';

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

type BedrockCacheUsage = {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cacheCreation: number;
  cacheRead: number;
  latencyMs: number;
};

type ConverseUsageResponse = {
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    totalTokens?: number;
    cacheReadInputTokens?: number;
    cacheWriteInputTokens?: number;
  };
};

const benchmarkToolConfig = {
  tools: [
    {
      toolSpec: {
        name: 'lookup_cache_probe',
        description: 'Returns prompt cache benchmark data.',
        inputSchema: {
          json: {
            type: 'object',
            properties: {
              step: { type: 'integer' },
            },
            required: ['step'],
          },
        },
      },
    },
  ],
};

function cachePointBlock(): MessageContentComplex {
  return { cachePoint: { type: 'default' } } as MessageContentComplex;
}

function stripCacheMarkers(
  content: MessageContentComplex[]
): MessageContentComplex[] {
  return content
    .filter((block) => !('cachePoint' in block && !('type' in block)))
    .map((block) => {
      const cloned = { ...block };
      delete (cloned as Record<string, unknown>).cache_control;
      return cloned as MessageContentComplex;
    });
}

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

function addLegacyMovingTailBedrockCacheControl(
  messages: BaseMessage[]
): BaseMessage[] {
  const updatedMessages = [...messages];
  let messagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const message = updatedMessages[i];
    const messageType = message.getType();
    if (messageType === 'system' || messageType === 'tool') {
      continue;
    }

    const content = message.content;
    if (typeof content === 'string') {
      if (content === '' || messagesModified >= 2) {
        continue;
      }
      updatedMessages[i] = cloneLiveMessage(message, [
        { type: 'text', text: content } as MessageContentComplex,
        cachePointBlock(),
      ]);
      messagesModified++;
      continue;
    }

    if (!Array.isArray(content)) {
      continue;
    }

    const workingContent = stripCacheMarkers(
      content as MessageContentComplex[]
    );
    const lastTextIndex = workingContent.findLastIndex((block) => {
      const type = (block as { type?: string }).type;
      const text = (block as { text?: string }).text;
      return (type === 'text' || type === 'input_text') && text?.trim() !== '';
    });

    if (messagesModified < 2 && lastTextIndex >= 0) {
      workingContent.splice(lastTextIndex + 1, 0, cachePointBlock());
      messagesModified++;
    }

    updatedMessages[i] = cloneLiveMessage(message, workingContent);
  }

  return updatedMessages;
}

function addLatestUserOnlyBedrockCacheControl(
  messages: BaseMessage[]
): BaseMessage[] {
  const updatedMessages = [...messages];
  let addedCachePoint = false;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const message = updatedMessages[i];
    const messageType = message.getType();
    if (messageType === 'system') {
      continue;
    }

    const content = message.content;
    const hasArrayContent = Array.isArray(content);
    const canAddCache =
      !addedCachePoint &&
      messageType === 'human' &&
      (typeof content === 'string' || hasArrayContent);

    if (!canAddCache && !hasArrayContent) {
      continue;
    }

    let workingContent: MessageContentComplex[];
    let modified = false;

    if (hasArrayContent) {
      workingContent = stripCacheMarkers(content as MessageContentComplex[]);
      modified = workingContent.length !== content.length;
      const lastTextIndex = workingContent.findLastIndex((block) => {
        const type = (block as { type?: string }).type;
        const text = (block as { text?: string }).text;
        return (
          (type === 'text' || type === 'input_text') && text?.trim() !== ''
        );
      });

      if (canAddCache && lastTextIndex >= 0) {
        workingContent.splice(lastTextIndex + 1, 0, cachePointBlock());
        addedCachePoint = true;
        modified = true;
      }

      if (!modified) {
        continue;
      }
    } else if (typeof content === 'string' && content.trim() !== '' && canAddCache) {
      workingContent = [
        { type: 'text', text: content } as MessageContentComplex,
        cachePointBlock(),
      ];
      addedCachePoint = true;
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

function buildToolLoopMessages({
  nonce,
  marker,
}: {
  nonce: string;
  marker: string;
}): BaseMessage[] {
  const stableUserContext = [
    `Bedrock prompt cache placement benchmark ${nonce}.`,
    'The first user turn is intentionally stable across calls in the same benchmark case.',
    repeated(`${nonce} user-context`, 190),
    'Use the final tool result to answer with the requested marker.',
  ].join('\n');
  const volatileToolPayload = repeated(`${nonce} volatile-${marker}`, 70);

  return [
    new HumanMessage(stableUserContext),
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

function buildMultiTurnToolMessages({
  nonce,
  marker,
}: {
  nonce: string;
  marker: string;
}): BaseMessage[] {
  const stableFirstUser = [
    `Bedrock multi-turn prompt cache benchmark ${nonce}.`,
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
  response: ConverseUsageResponse,
  latencyMs: number
): BedrockCacheUsage {
  if (response.usage == null) {
    throw new Error('Missing Bedrock usage metadata for cache benchmark');
  }

  const inputTokens = response.usage.inputTokens ?? 0;
  const outputTokens = response.usage.outputTokens ?? 0;
  return {
    inputTokens,
    outputTokens,
    totalTokens: response.usage.totalTokens ?? inputTokens + outputTokens,
    cacheCreation: response.usage.cacheWriteInputTokens ?? 0,
    cacheRead: response.usage.cacheReadInputTokens ?? 0,
    latencyMs,
  };
}

async function runConverseCacheBenchmarkTurn({
  client,
  messages,
}: {
  client: BedrockRuntimeClient;
  messages: BaseMessage[];
}): Promise<BedrockCacheUsage> {
  const { converseMessages, converseSystem } =
    convertToConverseMessages(messages);
  const startedAt = Date.now();
  const response = await client.send(
    new ConverseCommand({
      modelId: model,
      ...(converseSystem.length > 0 ? { system: converseSystem } : {}),
      messages: converseMessages,
      toolConfig: benchmarkToolConfig,
      inferenceConfig: { maxTokens: 16, temperature: 0 },
    })
  );

  return extractCacheUsage(
    response as ConverseUsageResponse,
    Date.now() - startedAt
  );
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

  it('reduces repeated cache writes versus the previous moving-tail placement', async () => {
    const credentials = getCredentials();
    const client = new BedrockRuntimeClient({
      region,
      ...(credentials != null ? { credentials } : {}),
    });
    const nonce = `bedrock-cache-placement-${Date.now()}`;
    const legacyNonce = `${nonce}-legacy`;
    const currentNonce = `${nonce}-current`;

    const legacyFirst = await runConverseCacheBenchmarkTurn({
      client,
      messages: addLegacyMovingTailBedrockCacheControl(
        buildToolLoopMessages({ nonce: legacyNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const legacySecond = await runConverseCacheBenchmarkTurn({
      client,
      messages: addLegacyMovingTailBedrockCacheControl(
        buildToolLoopMessages({ nonce: legacyNonce, marker: 'bravo' })
      ),
    });

    const currentFirst = await runConverseCacheBenchmarkTurn({
      client,
      messages: addBedrockCacheControl(
        buildToolLoopMessages({ nonce: currentNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const currentSecond = await runConverseCacheBenchmarkTurn({
      client,
      messages: addBedrockCacheControl(
        buildToolLoopMessages({ nonce: currentNonce, marker: 'bravo' })
      ),
    });

    const cacheWriteReduction =
      legacySecond.cacheCreation - currentSecond.cacheCreation;
    process.stdout.write(
      `Bedrock cache placement benchmark ${JSON.stringify({
        legacyFirst,
        legacySecond,
        currentFirst,
        currentSecond,
        cacheWriteReduction,
      })}\n`
    );

    expect(currentSecond.cacheRead).toBeGreaterThan(0);
    expect(cacheWriteReduction).toBeGreaterThan(0);
    expect(currentSecond.cacheCreation).toBeLessThan(
      Math.ceil(legacySecond.cacheCreation * 0.5)
    );
  }, 240_000);

  it('reuses prior user cache points when the latest user turn changes', async () => {
    const credentials = getCredentials();
    const client = new BedrockRuntimeClient({
      region,
      ...(credentials != null ? { credentials } : {}),
    });
    const nonce = `bedrock-multiturn-cache-placement-${Date.now()}`;
    const currentNonce = `${nonce}-current`;
    const latestOnlyNonce = `${nonce}-latest-only`;

    const currentFirst = await runConverseCacheBenchmarkTurn({
      client,
      messages: addBedrockCacheControl(
        buildMultiTurnToolMessages({ nonce: currentNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const currentSecond = await runConverseCacheBenchmarkTurn({
      client,
      messages: addBedrockCacheControl(
        buildMultiTurnToolMessages({ nonce: currentNonce, marker: 'bravo' })
      ),
    });

    const latestOnlyFirst = await runConverseCacheBenchmarkTurn({
      client,
      messages: addLatestUserOnlyBedrockCacheControl(
        buildMultiTurnToolMessages({ nonce: latestOnlyNonce, marker: 'alpha' })
      ),
    });

    await waitForCachePropagation();

    const latestOnlySecond = await runConverseCacheBenchmarkTurn({
      client,
      messages: addLatestUserOnlyBedrockCacheControl(
        buildMultiTurnToolMessages({ nonce: latestOnlyNonce, marker: 'bravo' })
      ),
    });

    process.stdout.write(
      `Bedrock multi-turn cache placement benchmark ${JSON.stringify({
        currentFirst,
        currentSecond,
        latestOnlyFirst,
        latestOnlySecond,
        cacheWriteDelta:
          currentSecond.cacheCreation - latestOnlySecond.cacheCreation,
      })}\n`
    );

    expect(currentSecond.cacheRead).toBeGreaterThan(
      latestOnlySecond.cacheRead
    );
    expect(currentSecond.cacheCreation).toBeLessThan(
      latestOnlySecond.cacheCreation
    );
  }, 240_000);
});
