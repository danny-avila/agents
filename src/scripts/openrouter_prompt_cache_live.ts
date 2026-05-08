import { config as loadEnv } from 'dotenv';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { AIMessage, BaseMessage } from '@langchain/core/messages';
import type { ClientOptions } from '@langchain/openai';
import type { GraphTools } from '@/types';
import type { ChatOpenRouterInput } from '@/llm/openrouter';
import { addCacheControl } from '@/messages/cache';
import { ChatOpenRouter } from '@/llm/openrouter';
import { partitionAndMarkOpenRouterToolCache } from '@/llm/openrouter/toolCache';

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

type OpenRouterTool = {
  type: 'function';
  function: {
    name: string;
  };
  cache_control?: { type: 'ephemeral' };
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

function buildStableToolDescription(): string {
  const paragraph =
    'Static OpenRouter tool contract for prompt cache validation. This tool description is stable across requests and intentionally verbose so provider-side prompt caching can write and then read a meaningful static tool-schema prefix while dynamic tools vary after the cache breakpoint.';

  return Array.from({ length: 90 }, (_, index) => {
    const section = index + 1;
    return `Tool section ${section}. ${paragraph} Stable tool key ${section}: OPENROUTER_STATIC_TOOL_CACHE_REFERENCE_${section}.`;
  }).join('\n');
}

function buildToolSet(attempt: number): GraphTools {
  return [
    {
      type: 'function',
      function: {
        name: 'stable_reference_lookup',
        description: buildStableToolDescription(),
        parameters: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'Stable lookup query.',
            },
          },
          required: ['query'],
          additionalProperties: false,
        },
      },
    },
    {
      type: 'function',
      function: {
        name: `dynamic_runtime_tool_${attempt}`,
        description: `Dynamic runtime tool ${attempt}; this varies between attempts and should sit after the cached static tool prefix.`,
        parameters: {
          type: 'object',
          properties: {
            value: {
              type: 'string',
            },
          },
          required: ['value'],
          additionalProperties: false,
        },
      },
    },
  ] as GraphTools;
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

async function runStaticToolCase(): Promise<CacheUsage[]> {
  const model = 'anthropic/claude-haiku-4.5';
  const usages: CacheUsage[] = [];

  log(`\nStatic tools through OpenRouter: ${model}`);

  for (let attempt = 1; attempt <= attempts; attempt++) {
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
    const tools = partitionAndMarkOpenRouterToolCache(
      buildToolSet(attempt),
      (name) => name.startsWith('dynamic_runtime_tool_')
    ) as OpenRouterTool[];
    const markedTool = tools.find((entry) => entry.cache_control != null);
    if (markedTool?.function.name !== 'stable_reference_lookup') {
      throw new Error('Static tool cache marker was not applied as expected');
    }

    const modelWithTools = llm.bindTools(tools);
    const started = Date.now();
    const response = (await modelWithTools.invoke([
      new SystemMessage('Reply with exactly: cache live check ok.'),
      new HumanMessage(
        `Attempt ${attempt}. Do not call tools; only answer with the requested text.`
      ),
    ])) as AIMessage;
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

  const staticToolUsages = await runStaticToolCase();
  results.push({
    label: 'Static tools',
    model: 'anthropic/claude-haiku-4.5',
    usages: staticToolUsages,
  });

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
