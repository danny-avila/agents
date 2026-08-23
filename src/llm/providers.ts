import type {
  ChatModelConstructorMap,
  ProviderModelConstructor,
  ProviderName,
} from '@/types';
import type { ProviderFamily } from '../provider-registration';
import {
  AzureChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  ChatOpenAI,
  ChatXAI,
} from '@/llm/openai';
import {
  getRegisteredChatModelClass,
  registerBuiltInProvider,
} from '@/llm/providerRegistry';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { CustomChatMistralAI } from '@/llm/mistral';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatVertexAI } from '@/llm/vertexai';
import { Providers } from '@/common';

type BuiltInProviderTraits = {
  family: ProviderFamily;
  manualToolStream?: boolean;
  strictAlternation?: boolean;
};

function initializeBuiltInProvider<P extends Providers>(
  provider: P,
  model: ChatModelConstructorMap[P],
  traits: BuiltInProviderTraits
): void {
  registerBuiltInProvider({
    provider,
    model,
    ...traits,
  });
}

initializeBuiltInProvider(Providers.XAI, ChatXAI, { family: 'openai' });
initializeBuiltInProvider(Providers.OPENAI, ChatOpenAI, { family: 'openai' });
initializeBuiltInProvider(Providers.AZURE, AzureChatOpenAI, {
  family: 'openai',
});
initializeBuiltInProvider(Providers.VERTEXAI, ChatVertexAI, {
  family: 'google',
});
initializeBuiltInProvider(Providers.DEEPSEEK, ChatDeepSeek, {
  family: 'openai',
});
initializeBuiltInProvider(Providers.MISTRALAI, CustomChatMistralAI, {
  family: 'mistral',
  strictAlternation: true,
});
initializeBuiltInProvider(Providers.MISTRAL, CustomChatMistralAI, {
  family: 'mistral',
  strictAlternation: true,
});
initializeBuiltInProvider(Providers.ANTHROPIC, CustomAnthropic, {
  family: 'anthropic',
  manualToolStream: true,
});
initializeBuiltInProvider(Providers.OPENROUTER, ChatOpenRouter, {
  family: 'openai',
});
initializeBuiltInProvider(Providers.BEDROCK, CustomChatBedrockConverse, {
  family: 'bedrock',
  manualToolStream: true,
  strictAlternation: true,
});
initializeBuiltInProvider(Providers.GOOGLE, CustomChatGoogleGenerativeAI, {
  family: 'google',
});
initializeBuiltInProvider(Providers.MOONSHOT, ChatMoonshot, {
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
