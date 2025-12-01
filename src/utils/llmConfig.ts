// src/utils/llmConfig.ts
import { Providers } from '@/common';
import type * as or from '@/llm/openrouter';
import type * as t from '@/types';

export const llmConfigs: Record<string, t.LLMConfig | undefined> = {
  [Providers.OPENAI]: {
    provider: Providers.OPENAI,
    model: 'gpt-4.1',
    // temperature: 0.7,
    streaming: true,
    streamUsage: true,
    // disableStreaming: true,
  },
  anthropicLITELLM: {
    provider: Providers.OPENAI,
    streaming: true,
    streamUsage: false,
    apiKey: 'sk-1234',
    model: 'claude-sonnet-4',
    maxTokens: 8192,
    modelKwargs: {
      metadata: {
        user_id: 'some_user_id',
      },
      thinking: {
        type: 'enabled',
        budget_tokens: 2000,
      },
    },
    configuration: {
      baseURL: 'http://host.docker.internal:4000/v1',
      defaultHeaders: {
        'anthropic-beta': 'prompt-caching-2024-07-31,context-1m-2025-08-07',
      },
    },
    // disableStreaming: true,
  },
  [Providers.XAI]: {
    provider: Providers.XAI,
    model: 'grok-2-latest',
    streaming: true,
    streamUsage: true,
  },
  alibaba: {
    provider: Providers.OPENAI,
    streaming: true,
    streamUsage: true,
    model: 'qwen-max',
    openAIApiKey: process.env.ALIBABA_API_KEY,
    configuration: {
      baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
    },
  },
  [Providers.OPENROUTER]: {
    provider: Providers.OPENROUTER,
    streaming: true,
    streamUsage: true,
    model: 'anthropic/claude-sonnet-4',
    apiKey: process.env.OPENROUTER_API_KEY,
    configuration: {
      baseURL: process.env.OPENROUTER_BASE_URL,
      defaultHeaders: {
        'HTTP-Referer': 'https://librechat.ai',
        'X-Title': 'LibreChat',
      },
    },
    include_reasoning: true,
    modelKwargs: {
      reasoning: {
        max_tokens: 8000,
      },
      max_tokens: 10000,
    },
  } as or.ChatOpenRouterCallOptions & t.LLMConfig,
  [Providers.AZURE]: {
    provider: Providers.AZURE,
    temperature: 0.7,
    streaming: true,
    streamUsage: true,
    azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
    azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE,
    azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT,
    azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
    model: process.env.AZURE_MODEL_NAME ?? 'gpt-4o',
  },
  ollama: {
    provider: Providers.OPENAI,
    model: 'gpt-oss:20b',
    streaming: true,
    streamUsage: true,
    configuration: {
      baseURL: 'http://localhost:11434/v1',
    },
  },
  lmstudio: {
    provider: Providers.OPENAI,
    model: 'gpt-oss-20b',
    streaming: true,
    streamUsage: true,
    configuration: {
      baseURL: 'http://192.168.254.183:1233/v1',
    },
  },
  zhipu: {
    provider: Providers.OPENAI,
    streaming: true,
    streamUsage: false,
    model: 'glm-4.5-air',
    apiKey: process.env.ZHIPU_API_KEY,
    configuration: {
      baseURL: 'https://open.bigmodel.cn/api/paas/v4',
    },
  },
  [Providers.DEEPSEEK]: {
    provider: Providers.DEEPSEEK,
    model: 'deepseek-reasoner',
    streaming: true,
    streamUsage: true,
  },
  [Providers.ANTHROPIC]: {
    provider: Providers.ANTHROPIC,
    model: 'claude-sonnet-4-5',
    streaming: true,
    streamUsage: true,
  },
  // [Providers.MISTRALAI]: {
  //   provider: Providers.MISTRALAI,
  //   model: 'mistral-large-latest',
  //   streaming: true,
  //   streamUsage: true,
  // },
  [Providers.MISTRAL]: {
    provider: Providers.OPENAI,
    streaming: true,
    streamUsage: false,
    // model: 'codestral-latest',
    model: 'mistral-large-latest',
    openAIApiKey: process.env.MISTRAL_API_KEY,
    configuration: {
      baseURL: 'https://api.mistral.ai/v1',
      defaultHeaders: {},
    },
  },
  [Providers.VERTEXAI]: {
    provider: Providers.VERTEXAI,
    model: 'gemini-2.5-flash',
    streaming: true,
    streamUsage: true,
    keyFile: process.env.VERTEXAI_KEY_FILE,
  } as t.VertexAIClientOptions & t.LLMConfig,
  [Providers.GOOGLE]: {
    provider: Providers.GOOGLE,
    model: 'gemini-2.5-flash',
    streaming: true,
    streamUsage: true,
  },
  [Providers.BEDROCK]: {
    provider: Providers.BEDROCK,
    // model: 'anthropic.claude-3-sonnet-20240229-v1:0',
    // model: 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    // model: 'us.amazon.nova-pro-v1:0',
    model: 'us.anthropic.claude-sonnet-4-20250514-v1:0',
    // additionalModelRequestFields: { thinking: { type: 'enabled', budget_tokens: 2000 } },
    region: process.env.BEDROCK_AWS_REGION,
    credentials: {
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
    },
    streaming: true,
    streamUsage: true,
  },
  perplexity: {
    provider: Providers.OPENAI,
    model: 'llama-3.1-sonar-small-128k-online',
    streaming: true,
    streamUsage: true,
    apiKey: process.env.PERPLEXITY_API_KEY,
    configuration: {
      baseURL: 'https://api.perplexity.ai/',
    },
  },
};

export function getLLMConfig(provider: string): t.LLMConfig {
  const config = llmConfigs[provider];
  if (config === undefined) {
    throw new Error(`Unsupported provider: ${provider}`);
  }
  return config;
}
