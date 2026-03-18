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

export function isGoogleLike(provider?: string | Providers): boolean {
  if (provider == null) {
    return false;
  }
  return ([Providers.GOOGLE, Providers.VERTEXAI] as string[]).includes(
    provider
  );
}

/** Returns true for native Anthropic or Bedrock running a Claude model. */
export function isAnthropicLike(
  provider?: string | Providers,
  clientOptions?: { model?: string }
): boolean {
  if (provider === Providers.ANTHROPIC) return true;
  if (provider === Providers.BEDROCK) {
    return /claude/i.test(String(clientOptions?.model ?? ''));
  }
  return false;
}
