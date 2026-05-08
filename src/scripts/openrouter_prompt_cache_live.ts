import { config as loadEnv } from 'dotenv';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { AIMessage, BaseMessage } from '@langchain/core/messages';
import type { ClientOptions } from '@langchain/openai';
import type { ChatOpenRouterInput } from '@/llm/openrouter';
import { addCacheControl } from '@/messages/cache';
import { ChatOpenRouter } from '@/llm/openrouter';

loadEnv({ path: process.env.DOTENV_CONFIG_PATH ?? '.env' });

type ModelCase = {
  label: string;
  model: string;
};

type CacheUsage = {
  cacheCreation: number;
  cacheRead: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
};

const DEFAULT_MODEL_CASES: ModelCase[] = [
  { label: 'Anthropic Claude', model: 'anthropic/claude-haiku-4.5' },
  { label: 'Google Gemini', model: 'google/gemini-2.5-flash' },
  { label: 'Alibaba Qwen', model: 'qwen/qwen3-coder-flash' },
];

const apiKey = process.env.OPENROUTER_API_KEY;
const baseURL =
  process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1';
const attempts = Number(process.env.OPENROUTER_PROMPT_CACHE_ATTEMPTS ?? '3');
const modelCases = (
  process.env.OPENROUTER_PROMPT_CACHE_MODELS?.split(',').map((model) => ({
    label: 'Custom',
    model: model.trim(),
  })) ?? DEFAULT_MODEL_CASES
).filter(({ model }) => model.length > 0);

if (apiKey == null || apiKey.length === 0) {
  throw new Error('OPENROUTER_API_KEY is required');
}

function buildStableReference(): string {
  const paragraph =
    'LibreChat OpenRouter prompt caching live validation reference. This paragraph is deliberately stable across repeated requests so OpenRouter can route the conversation to the same provider endpoint and reuse cached prompt tokens. It describes cache breakpoints, provider sticky routing, cache write metrics, cache read metrics, model-specific minimum prompt sizes, and the expected behavior of explicit per-message cache_control markers for supported OpenRouter providers.';

  return Array.from({ length: 90 }, (_, index) => {
    const section = index + 1;
    return `Section ${section}. ${paragraph} Verification key ${section}: OPENROUTER_PROMPT_CACHE_LIVE_REFERENCE_${section}.`;
  }).join('\n');
}

function buildMessages(model: string): BaseMessage[] {
  const reference = buildStableReference();
  const messages: BaseMessage[] = [
    new SystemMessage(
      'You are validating prompt caching. Answer with one concise sentence.'
    ),
    new HumanMessage(
      [
        `For model ${model}, reply with exactly this format: cache live check ok.`,
        'Use the stable reference below only to make this request large enough to cache.',
        reference,
      ].join('\n\n')
    ),
  ];

  return addCacheControl<BaseMessage>(messages);
}

function getCacheUsage(message: AIMessage): CacheUsage {
  const usage = message.usage_metadata;
  const inputDetails = usage?.input_token_details;

  return {
    inputTokens: usage?.input_tokens ?? 0,
    outputTokens: usage?.output_tokens ?? 0,
    totalTokens: usage?.total_tokens ?? 0,
    cacheRead: inputDetails?.cache_read ?? 0,
    cacheCreation: inputDetails?.cache_creation ?? 0,
  };
}

function hasCacheHit(usages: CacheUsage[]): boolean {
  return usages.some(({ cacheRead }) => cacheRead > 0);
}

function hasCacheActivity(usages: CacheUsage[]): boolean {
  return usages.some(
    ({ cacheCreation, cacheRead }) => cacheCreation > 0 || cacheRead > 0
  );
}

function log(message = ''): void {
  process.stdout.write(`${message}\n`);
}

function logError(message: string): void {
  process.stderr.write(`${message}\n`);
}

async function runCase({ label, model }: ModelCase): Promise<CacheUsage[]> {
  const llmInput: ChatOpenRouterInput & { configuration: ClientOptions } = {
    model,
    apiKey,
    maxTokens: 12,
    temperature: 0,
    promptCache: true,
    streamUsage: true,
    configuration: {
      baseURL,
      defaultHeaders: {
        'HTTP-Referer': 'https://librechat.ai',
        'X-Title': 'LibreChat OpenRouter Prompt Cache Live Test',
      },
    },
  };
  const llm = new ChatOpenRouter(llmInput);
  const messages = buildMessages(model);
  const usages: CacheUsage[] = [];

  log(`\n${label}: ${model}`);

  for (let attempt = 1; attempt <= attempts; attempt++) {
    const started = Date.now();
    const response = (await llm.invoke(messages)) as AIMessage;
    const usage = getCacheUsage(response);
    usages.push(usage);

    log(
      [
        `attempt=${attempt}`,
        `ms=${Date.now() - started}`,
        `input=${usage.inputTokens}`,
        `output=${usage.outputTokens}`,
        `write=${usage.cacheCreation}`,
        `read=${usage.cacheRead}`,
        `total=${usage.totalTokens}`,
      ].join(' ')
    );

    if (hasCacheHit(usages)) {
      return usages;
    }
  }

  return usages;
}

async function main(): Promise<void> {
  const results: Array<ModelCase & { usages: CacheUsage[] }> = [];

  for (const modelCase of modelCases) {
    const usages = await runCase(modelCase);
    results.push({ ...modelCase, usages });
  }

  const failures = results.filter(({ usages }) => {
    return !hasCacheActivity(usages) || !hasCacheHit(usages);
  });

  log('\nSummary');
  for (const { label, model, usages } of results) {
    const writes = usages.map(({ cacheCreation }) => cacheCreation).join(',');
    const reads = usages.map(({ cacheRead }) => cacheRead).join(',');
    log(`${label} ${model}: writes=[${writes}] reads=[${reads}]`);
  }

  if (failures.length > 0) {
    const failedModels = failures.map(({ model }) => model).join(', ');
    throw new Error(`Prompt caching was not confirmed for: ${failedModels}`);
  }
}

main().catch((error: Error) => {
  logError(error.message);
  process.exit(1);
});
