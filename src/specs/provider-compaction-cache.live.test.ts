/* eslint-disable no-console */
import { performance } from 'node:perf_hooks';
import { config } from 'dotenv';
config();
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { applySummarizationHistoryCache } from '@/summarization/node';
import { prepareToolsForPromptCache } from '@/llm/promptCacheTools';
import { initializeModel } from '@/llm/init';
import { Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';

interface CacheProbe {
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
  inputTokens: number;
  latencyMs: number;
}

interface ProviderProbe {
  provider: t.ProviderName;
  clientOptions: t.ClientOptions;
}

const LARGE_PREFIX = 'stable compaction cache prefix token '.repeat(220);

function createTools(nonce: string): t.GraphTools {
  return Array.from({ length: 8 }, (_, index) => ({
    name: `cache_probe_${index}`,
    description: `${nonce} ${LARGE_PREFIX}`,
    schema: {
      type: 'object' as const,
      properties: {
        value: { type: 'string' },
      },
    },
  })) as t.GraphTools;
}

function readCacheProbe(message: AIMessage, latencyMs: number): CacheProbe {
  return {
    cacheCreationInputTokens:
      message.usage_metadata?.input_token_details?.cache_creation ?? 0,
    cacheReadInputTokens:
      message.usage_metadata?.input_token_details?.cache_read ?? 0,
    inputTokens: message.usage_metadata?.input_tokens ?? 0,
    latencyMs: Math.round(latencyMs),
  };
}

async function invokeProbe(params: {
  provider: t.ProviderName;
  clientOptions: t.ClientOptions;
  tools: t.GraphTools;
  history: HumanMessage[];
  instruction: string;
}): Promise<CacheProbe> {
  const model = initializeModel({
    provider: params.provider,
    clientOptions: params.clientOptions,
    tools: params.tools,
  }) as t.ChatModel;
  const startedAt = performance.now();
  const result = (await model.invoke([
    ...params.history,
    new HumanMessage(params.instruction),
  ])) as AIMessage;
  return readCacheProbe(result, performance.now() - startedAt);
}

async function runExplicitCacheProbe(probe: ProviderProbe): Promise<{
  prime: CacheProbe;
  baseline: CacheProbe;
  aligned: CacheProbe;
}> {
  const nonce = `${Date.now()}-${Math.random()}`;
  const tools = createTools(nonce);
  const cacheOptions = {
    ...probe.clientOptions,
    maxTokens: 8,
    promptCache: true,
  } as t.ClientOptions;
  const baselineOptions = {
    ...probe.clientOptions,
    maxTokens: 8,
    promptCache: false,
  } as t.ClientOptions;
  const cachedTools =
    prepareToolsForPromptCache({
      provider: probe.provider,
      clientOptions: cacheOptions,
      tools,
      isDeferred: () => false,
    }) ?? tools;
  const rawHistory = [new HumanMessage(`${nonce} ${LARGE_PREFIX}`)];
  const cachedHistory = applySummarizationHistoryCache({
    messages: rawHistory,
    provider: probe.provider,
    enabled: true,
    bedrockModelId: (cacheOptions as { model?: string }).model,
  }) as HumanMessage[];

  const prime = await invokeProbe({
    provider: probe.provider,
    clientOptions: cacheOptions,
    tools: cachedTools,
    history: cachedHistory,
    instruction: 'Prime the normal request prefix. Reply OK.',
  });
  await new Promise((resolve) => setTimeout(resolve, 2000));
  const baseline = await invokeProbe({
    provider: probe.provider,
    clientOptions: baselineOptions,
    tools,
    history: rawHistory,
    instruction: 'Unaligned compaction request. Reply OK.',
  });
  const aligned = await invokeProbe({
    provider: probe.provider,
    clientOptions: cacheOptions,
    tools: cachedTools,
    history: cachedHistory,
    instruction: 'Cache-aligned compaction request. Reply OK.',
  });

  return { prime, baseline, aligned };
}

async function runAutomaticOpenAIProbe(): Promise<{
  prime: CacheProbe;
  baseline: CacheProbe;
  aligned: CacheProbe;
}> {
  const nonce = `${Date.now()}-${Math.random()}`;
  const options = {
    ...getLLMConfig(Providers.OPENAI),
    model: 'gpt-4.1-mini',
    maxTokens: 8,
    streaming: false,
  } as t.ClientOptions;
  const tools = createTools(nonce);
  const history = [new HumanMessage(`${nonce} ${LARGE_PREFIX}`)];
  const prime = await invokeProbe({
    provider: Providers.OPENAI,
    clientOptions: options,
    tools,
    history,
    instruction: 'Prime the normal request prefix. Reply OK.',
  });
  const baseline = await invokeProbe({
    provider: Providers.OPENAI,
    clientOptions: options,
    tools: createTools(`${nonce}-unaligned`),
    history: [new HumanMessage(`${nonce}-unaligned ${LARGE_PREFIX}`)],
    instruction: 'Unaligned compaction request. Reply OK.',
  });
  const aligned = await invokeProbe({
    provider: Providers.OPENAI,
    clientOptions: options,
    tools,
    history,
    instruction: 'Cache-aligned compaction request. Reply OK.',
  });

  return { prime, baseline, aligned };
}

function expectCacheReuse(result: {
  baseline: CacheProbe;
  aligned: CacheProbe;
}): void {
  expect(result.aligned.cacheReadInputTokens).toBeGreaterThan(0);
  expect(result.aligned.cacheReadInputTokens).toBeGreaterThan(
    result.baseline.cacheReadInputTokens
  );
}

describe('Merged compaction provider cache validation', () => {
  jest.setTimeout(180_000);

  const hasBedrock =
    (process.env.BEDROCK_AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? '') !==
      '' &&
    (process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? '') !== '' &&
    (process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? '') !== '';
  const bedrockTest = hasBedrock ? test : test.skip;
  bedrockTest('Bedrock Claude reuses the aligned compaction prefix', async () => {
    const result = await runExplicitCacheProbe({
      provider: Providers.BEDROCK,
      clientOptions: {
        ...getLLMConfig(Providers.BEDROCK),
        region:
          process.env.BEDROCK_AWS_REGION ?? process.env.AWS_DEFAULT_REGION,
      } as t.ClientOptions,
    });
    console.log(`  Bedrock cache benchmark: ${JSON.stringify(result)}`);
    expectCacheReuse(result);
  });

  const hasOpenRouter = (process.env.OPENROUTER_API_KEY ?? '') !== '';
  const openRouterTest = hasOpenRouter ? test : test.skip;
  openRouterTest('OpenRouter Claude reuses the aligned compaction prefix', async () => {
    const result = await runExplicitCacheProbe({
      provider: Providers.OPENROUTER,
      clientOptions: {
        ...getLLMConfig(Providers.OPENROUTER),
        model:
          process.env.OPENROUTER_CACHE_MODEL ??
          'anthropic/claude-sonnet-4.6',
        streaming: false,
      } as t.ClientOptions,
    });
    console.log(`  OpenRouter cache benchmark: ${JSON.stringify(result)}`);
    expectCacheReuse(result);
  });

  const hasOpenAI = (process.env.OPENAI_API_KEY ?? '') !== '';
  const openAITest = hasOpenAI ? test : test.skip;
  openAITest('OpenAI automatically reuses the aligned compaction prefix', async () => {
    const result = await runAutomaticOpenAIProbe();
    console.log(`  OpenAI cache benchmark: ${JSON.stringify(result)}`);
    expectCacheReuse(result);
  });
});
