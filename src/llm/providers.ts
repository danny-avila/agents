import type { ProviderModelConstructor, ProviderName } from '@/types';
import { requireLazyModule } from '@/lazyRequire';
import type { ProviderFamily } from '../provider-registration';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import {
  getRegisteredChatModelClass,
  registerBuiltInProviderLoader,
} from '@/llm/providerRegistry';
import { Providers } from '@/common';

/**
 * Built-in provider SDKs load on first model request, not at import time: eagerly
 * importing every provider cost hundreds of milliseconds of boot in hosts that
 * configure one or two of them. `createRequire` keeps resolution synchronous and
 * correct from both the CJS and ESM builds.
 */

type ProviderModelClass = new (config: never) => BaseChatModel;

type BuiltInProviderTraits = {
  family: ProviderFamily;
  manualToolStream?: boolean;
  strictAlternation?: boolean;
};

function initializeBuiltInProvider(
  provider: Providers,
  loadModel: () => ProviderModelClass,
  traits: BuiltInProviderTraits
): void {
  registerBuiltInProviderLoader({
    provider,
    loadModel,
    ...traits,
  });
}

const fromOpenAI =
  (name: string) => (): ProviderModelClass =>
    (requireLazyModule<Record<string, ProviderModelClass>>('@librechat/agents/llm/openai'))[
      name
    ];

initializeBuiltInProvider(Providers.XAI, fromOpenAI('ChatXAI'), { family: 'openai' });
initializeBuiltInProvider(Providers.OPENAI, fromOpenAI('ChatOpenAI'), { family: 'openai' });
initializeBuiltInProvider(Providers.AZURE, fromOpenAI('AzureChatOpenAI'), {
  family: 'openai',
});
initializeBuiltInProvider(
  Providers.VERTEXAI,
  () => (requireLazyModule<typeof import('@/llm/vertexai')>('@librechat/agents/llm/vertexai')).ChatVertexAI,
  { family: 'google' }
);
initializeBuiltInProvider(Providers.DEEPSEEK, fromOpenAI('ChatDeepSeek'), {
  family: 'openai',
});
const loadMistral = (): ProviderModelClass =>
  (requireLazyModule<typeof import('@/llm/mistral')>('@librechat/agents/llm/mistral')).CustomChatMistralAI;
initializeBuiltInProvider(Providers.MISTRALAI, loadMistral, {
  family: 'mistral',
  strictAlternation: true,
});
initializeBuiltInProvider(Providers.MISTRAL, loadMistral, {
  family: 'mistral',
  strictAlternation: true,
});
initializeBuiltInProvider(
  Providers.ANTHROPIC,
  () => (requireLazyModule<typeof import('@/llm/anthropic')>('@librechat/agents/llm/anthropic')).CustomAnthropic,
  { family: 'anthropic', manualToolStream: true }
);
initializeBuiltInProvider(
  Providers.OPENROUTER,
  () =>
    (requireLazyModule<typeof import('@/llm/openrouter')>('@librechat/agents/llm/openrouter')).ChatOpenRouter,
  { family: 'openai' }
);
initializeBuiltInProvider(
  Providers.BEDROCK,
  () =>
    (requireLazyModule<typeof import('@/llm/bedrock')>('@librechat/agents/llm/bedrock'))
      .CustomChatBedrockConverse,
  { family: 'bedrock', manualToolStream: true, strictAlternation: true }
);
initializeBuiltInProvider(
  Providers.GOOGLE,
  () =>
    (requireLazyModule<typeof import('@/llm/google')>('@librechat/agents/llm/google'))
      .CustomChatGoogleGenerativeAI,
  { family: 'google' }
);
initializeBuiltInProvider(Providers.MOONSHOT, fromOpenAI('ChatMoonshot'), {
  family: 'generic',
});

export const getChatModelClass = <P extends ProviderName>(
  provider: P
): ProviderModelConstructor<P> => getRegisteredChatModelClass(provider);

export {
  getProviderFamily,
  providerRequiresStrictAlternation,
  providerUsesManualToolStream,
  registerProvider,
} from '@/llm/providerRegistry';
export type {
  ProviderFamily,
  ProviderRegistrationOptions,
} from '../provider-registration';
