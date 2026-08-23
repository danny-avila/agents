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
  registerProvider,
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

function registerBuiltInProvider<P extends Providers>(
  provider: P,
  model: ChatModelConstructorMap[P],
  traits: BuiltInProviderTraits
): void {
  registerProvider({
    provider,
    model,
    ...traits,
  });
}

registerBuiltInProvider(Providers.XAI, ChatXAI, { family: 'openai' });
registerBuiltInProvider(Providers.OPENAI, ChatOpenAI, { family: 'openai' });
registerBuiltInProvider(Providers.AZURE, AzureChatOpenAI, { family: 'openai' });
registerBuiltInProvider(Providers.VERTEXAI, ChatVertexAI, { family: 'google' });
registerBuiltInProvider(Providers.DEEPSEEK, ChatDeepSeek, { family: 'openai' });
registerBuiltInProvider(Providers.MISTRALAI, CustomChatMistralAI, {
  family: 'mistral',
  strictAlternation: true,
});
registerBuiltInProvider(Providers.MISTRAL, CustomChatMistralAI, {
  family: 'mistral',
  strictAlternation: true,
});
registerBuiltInProvider(Providers.ANTHROPIC, CustomAnthropic, {
  family: 'anthropic',
  manualToolStream: true,
});
registerBuiltInProvider(Providers.OPENROUTER, ChatOpenRouter, {
  family: 'openai',
});
registerBuiltInProvider(Providers.BEDROCK, CustomChatBedrockConverse, {
  family: 'bedrock',
  manualToolStream: true,
  strictAlternation: true,
});
registerBuiltInProvider(Providers.GOOGLE, CustomChatGoogleGenerativeAI, {
  family: 'google',
});
registerBuiltInProvider(Providers.MOONSHOT, ChatMoonshot, {
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
