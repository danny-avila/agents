// src/utils/llm.ts
import { Providers } from '@/common';

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
    ] as string[]
  ).includes(provider);
}

/** Checks if a model name refers to a Claude/Anthropic model (e.g. `anthropic/claude-sonnet-4`) */
export function isAnthropicModel(model?: string): boolean {
  return (
    model?.includes('claude') === true ||
    model?.includes('anthropic') === true
  );
}

export function isGoogleLike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return ([Providers.GOOGLE, Providers.VERTEXAI] as string[]).includes(
    provider
  );
}
