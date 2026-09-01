// src/utils/llm.ts
import type { ProviderName } from '@/types';
import { getProviderFamily } from '@/llm/providerRegistry';
import { Providers } from '@/common';

const OPENAI_FAMILY_LC_NAMES: ReadonlySet<string> = new Set([
  'LibreChatOpenAI',
  'LibreChatAzureOpenAI',
]);

/** Walks the constructor chain for LangChain's `lc_name` serialization ids.
 *  Constructor identity is unusable across lazily loaded module formats — the
 *  CJS and ESM builds carry distinct classes — so guards match on the ids the
 *  relevant classes declare instead of on `instanceof`. */
export function constructorChainHasLcName(
  model: unknown,
  names: ReadonlySet<string>
): boolean {
  if (typeof model !== 'object' || model == null) {
    return false;
  }
  let ctor: unknown = model.constructor;
  while (typeof ctor === 'function') {
    const lcName = (ctor as { lc_name?: () => string }).lc_name?.();
    if (lcName != null && names.has(lcName)) {
      return true;
    }
    ctor = Object.getPrototypeOf(ctor);
  }
  return false;
}

/** Matches exactly the instances that were `instanceof` this package's
 *  `ChatOpenAI`/`AzureChatOpenAI`: only those classes declare the
 *  `LibreChatOpenAI`/`LibreChatAzureOpenAI` ids, their subclasses keep them in
 *  the chain, and upstream `@langchain/openai` classes never carry them. */
export function isLibreChatOpenAIModel(
  model: unknown
): model is
  | import('@/llm/openai').ChatOpenAI
  | import('@/llm/openai').AzureChatOpenAI {
  return constructorChainHasLcName(model, OPENAI_FAMILY_LC_NAMES);
}

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
