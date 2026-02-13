import type * as t from '@/types';
import { Providers } from '@/common';

/**
 * Returns true when the provider + clientOptions indicate extended thinking
 * is enabled.  Works across Anthropic (direct), Bedrock (additionalModelRequestFields),
 * and OpenAI-compat (modelKwargs.thinking).
 */
export function isThinkingEnabled(
  provider: Providers,
  clientOptions?: t.ClientOptions
): boolean {
  if (!clientOptions) return false;

  if (
    provider === Providers.ANTHROPIC &&
    (clientOptions as t.AnthropicClientOptions).thinking != null
  ) {
    return true;
  }

  if (
    provider === Providers.BEDROCK &&
    (clientOptions as t.BedrockAnthropicInput).additionalModelRequestFields?.[
      'thinking'
    ] != null
  ) {
    return true;
  }

  if (
    provider === Providers.OPENAI &&
    (
      (clientOptions as t.OpenAIClientOptions).modelKwargs
        ?.thinking as t.AnthropicClientOptions['thinking']
    )?.type === 'enabled'
  ) {
    return true;
  }

  return false;
}

/**
 * Returns the correct key for setting max output tokens on the model
 * constructor options.  Google/Vertex use `maxOutputTokens`, all others
 * use `maxTokens`.
 */
export function getMaxOutputTokensKey(
  provider: Providers | string
): 'maxOutputTokens' | 'maxTokens' {
  return provider === Providers.GOOGLE || provider === Providers.VERTEXAI
    ? 'maxOutputTokens'
    : 'maxTokens';
}
