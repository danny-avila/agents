import type { ProviderModelConstructor, ProviderName } from '@/types';
import { requireInternalModule } from '@/lazyRequire';
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
    (requireInternalModule<Record<string, ProviderModelClass>>('llm/openai/index'))[
      name
    ];

initializeBuiltInProvider(Providers.XAI, fromOpenAI('ChatXAI'), { family: 'openai' });
initializeBuiltInProvider(Providers.OPENAI, fromOpenAI('ChatOpenAI'), { family: 'openai' });
initializeBuiltInProvider(Providers.AZURE, fromOpenAI('AzureChatOpenAI'), {
  family: 'openai',
});
initializeBuiltInProvider(
  Providers.VERTEXAI,
  () => (requireInternalModule<typeof import('@/llm/vertexai')>('llm/vertexai/index')).ChatVertexAI,
  { family: 'google' }
);
initializeBuiltInProvider(Providers.DEEPSEEK, fromOpenAI('ChatDeepSeek'), {
  family: 'openai',
});
const loadMistral = (): ProviderModelClass =>
  (requireInternalModule<typeof import('@/llm/mistral')>('llm/mistral/index')).CustomChatMistralAI;
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
  () => (requireInternalModule<typeof import('@/llm/anthropic')>('llm/anthropic/index')).CustomAnthropic,
  { family: 'anthropic', manualToolStream: true }
);
initializeBuiltInProvider(
  Providers.OPENROUTER,
  () =>
    (requireInternalModule<typeof import('@/llm/openrouter')>('llm/openrouter/index')).ChatOpenRouter,
  { family: 'openai' }
);
initializeBuiltInProvider(
  Providers.BEDROCK,
  () =>
    (requireInternalModule<typeof import('@/llm/bedrock')>('llm/bedrock/index'))
      .CustomChatBedrockConverse,
  { family: 'bedrock', manualToolStream: true, strictAlternation: true }
);
initializeBuiltInProvider(
  Providers.GOOGLE,
  () =>
    (requireInternalModule<typeof import('@/llm/google')>('llm/google/index'))
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
