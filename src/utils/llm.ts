// src/utils/llm.ts
import type { ProviderName } from '@/types';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

export function isOpenAILike(provider?: ProviderName): boolean {
  if (provider == null) {
    return false;
  }
  return (
    getProviderFamily(provider) === 'openai' ||
    (
      [
        Providers.OPENAI,
        Providers.AZURE,
        Providers.OPENROUTER,
        Providers.XAI,
        Providers.DEEPSEEK,
      ] as string[]
    ).includes(provider)
  );
}

export function isGoogleLike(provider?: ProviderName): boolean {
  if (provider == null) {
    return false;
  }
  return (
    getProviderFamily(provider) === 'google' ||
    ([Providers.GOOGLE, Providers.VERTEXAI] as string[]).includes(provider)
  );
}

/** Returns true for native Anthropic or Bedrock running a Claude model. */
export function isAnthropicLike(
  provider?: ProviderName,
  clientOptions?: { model?: string }
): boolean {
  const family = provider == null ? undefined : getProviderFamily(provider);
  if (provider === Providers.ANTHROPIC || family === 'anthropic') return true;
  if (provider === Providers.BEDROCK || family === 'bedrock') {
    return (
      clientOptions?.model == null ||
      /claude/i.test(String(clientOptions.model))
    );
  }
  return false;
}
