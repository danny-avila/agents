import type * as t from '@/types';
import {
  resolvePromptCacheTtl,
  supportsBedrockToolCache,
} from '@/messages/cache';
import { partitionAndMarkAnthropicToolCache } from '@/messages/anthropicToolCache';
import { partitionAndMarkOpenRouterToolCache } from '@/llm/openrouter/toolCache';
import { partitionAndMarkBedrockToolCache } from '@/llm/bedrock/toolCache';
import { Providers } from '@/common';

export function prepareToolsForPromptCache(params: {
  provider: t.ProviderName;
  clientOptions?: t.ClientOptions;
  tools?: t.GraphTools;
  isDeferred: (toolName: string) => boolean;
}): t.GraphTools | undefined {
  const { provider, clientOptions, tools, isDeferred } = params;
  if (provider === Providers.ANTHROPIC) {
    const options = clientOptions as t.AnthropicClientOptions | undefined;
    if (options?.promptCache !== true) {
      return tools;
    }
    return (
      partitionAndMarkAnthropicToolCache(
        tools,
        isDeferred,
        resolvePromptCacheTtl(options.promptCacheTtl)
      ) ?? tools
    );
  }
  if (provider === Providers.OPENROUTER) {
    const options = clientOptions as
      | t.ProviderOptionsMap[Providers.OPENROUTER]
      | undefined;
    if (options?.promptCache !== true) {
      return tools;
    }
    return (
      partitionAndMarkOpenRouterToolCache(
        tools,
        isDeferred,
        resolvePromptCacheTtl(options.promptCacheTtl)
      ) ?? tools
    );
  }
  if (provider !== Providers.BEDROCK) {
    return tools;
  }
  const options = clientOptions as t.BedrockAnthropicClientOptions | undefined;
  if (options?.promptCache !== true) {
    return tools;
  }
  const model = (options as { model?: string }).model;
  if (model != null && !supportsBedrockToolCache(model)) {
    return tools;
  }
  return partitionAndMarkBedrockToolCache(tools, isDeferred) ?? tools;
}
