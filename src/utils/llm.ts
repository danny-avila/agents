// src/utils/llm.ts
import { Providers } from '@/common';
import type { ClientOptions, ProviderOptionsMap } from '@/types/llm';

export function isOpenAILike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return (
    [
      Providers.OPENAI,
      Providers.AZURE,
      Providers.OPENROUTER,
      Providers.XAI,
      Providers.DEEPSEEK,
      Providers.OLLAMA,
    ] as string[]
  ).includes(provider);
}

export function isGoogleLike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return ([Providers.GOOGLE, Providers.VERTEXAI] as string[]).includes(
    provider
  );
}

/**
 * Validates client options for a given provider, returning any validation errors.
 * @param options - The client options to validate.
 * @param provider - The LLM provider (e.g., OPENAI, ANTHROPIC).
 * @returns An array of error messages; empty if validation passes.
 * @throws Error if the provider is unsupported.
 */
export function validateClientOptions<T extends Providers>(
  options: ClientOptions,
  provider: T
): string[] {
  const errors: string[] = [];

  // Common validation for all providers
  if (options?.temperature != null && (options.temperature < 0 || options.temperature > 2)) {
    errors.push('Temperature must be between 0 and 2');
  }
  if (options?.maxTokens != null && options.maxTokens <= 0) {
    errors.push('maxTokens must be greater than 0');
  }

  // Provider-specific validation
  switch (provider) {
    case Providers.OPENAI:
      if (!options?.apiKey && !options?.openAIApiKey) {
        errors.push('OpenAI requires an API key');
      }
      break;
    case Providers.AZURE:
      if (!options?.azureOpenAIApiKey) {
        errors.push('Azure requires an API key (azureOpenAIApiKey)');
      }
      if (!options?.azureOpenAIApiInstanceName) {
        errors.push('Azure requires an API instance name (azureOpenAIApiInstanceName)');
      }
      if (!options?.azureOpenAIApiDeploymentName) {
        errors.push('Azure requires a deployment name (azureOpenAIApiDeploymentName)');
      }
      if (!options?.azureOpenAIApiVersion) {
        errors.push('Azure requires an API version (azureOpenAIApiVersion)');
      }
      break;
    case Providers.ANTHROPIC:
      if (!options?.apiKey) {
        errors.push('Anthropic requires an API key');
      }
      if (options?.thinkingBudget != null && options.thinkingBudget <= 0) {
        errors.push('thinkingBudget must be greater than 0 for Anthropic');
      }
      break;
    case Providers.GOOGLE:
      if (!options?.apiKey && !options?.customHeaders?.['x-goog-api-key']) {
        errors.push('Google requires an API key');
      }
      break;
    case Providers.VERTEXAI:
      if (!options?.keyFile && !options?.apiKey) {
        errors.push('VertexAI requires either a key file or API key');
      }
      break;
    case Providers.OLLAMA:
      if (!options?.baseUrl) {
        errors.push('Ollama requires a baseUrl');
      }
      break;
    case Providers.OPENROUTER:
      if (!options?.openAIApiKey) {
        errors.push('OpenRouter requires an API key (openAIApiKey)');
      }
      if (!options?.configuration?.baseURL) {
        errors.push('OpenRouter requires a baseURL in configuration');
      }
      break;
    case Providers.DEEPSEEK:
      if (!options?.apiKey) {
        errors.push('DeepSeek requires an API key');
      }
      break;
    case Providers.MISTRAL:
      if (!options?.openAIApiKey) {
        errors.push('Mistral requires an API key (openAIApiKey)');
      }
      if (!options?.configuration?.baseURL) {
        errors.push('Mistral requires a baseURL in configuration');
      }
      break;
    case Providers.BEDROCK:
      if (!options?.credentials?.accessKeyId || !options?.credentials?.secretAccessKey) {
        errors.push('Bedrock requires AWS credentials (accessKeyId and secretAccessKey)');
      }
      if (!options?.region) {
        errors.push('Bedrock requires a region');
      }
      break;
    case Providers.XAI:
      if (!options?.apiKey) {
        errors.push('XAI requires an API key');
      }
      break;
    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }

  return errors;
}
