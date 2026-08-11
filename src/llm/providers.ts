// src/llm/providers.ts
import { CustomChatMistralAI } from '@/llm/mistral';
import type {
  ChatModelConstructorMap,
  ProviderOptionsMap,
  ChatModelMap,
} from '@/types';
import {
  AzureChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  ChatOpenAI,
  ChatXAI,
} from '@/llm/openai';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatVertexAI } from '@/llm/vertexai';
import { Providers } from '@/common';
import type { Runnable } from '@langchain/core/runnables';

/** Constructor contract for providers registered by host applications. */
export type RegisteredChatModelConstructor = new (config: unknown) => Runnable;

const registeredChatModelProviders = new Map<string, RegisteredChatModelConstructor>();

export const llmProviders: Partial<ChatModelConstructorMap> = {
  [Providers.XAI]: ChatXAI,
  [Providers.OPENAI]: ChatOpenAI,
  [Providers.AZURE]: AzureChatOpenAI,
  [Providers.VERTEXAI]: ChatVertexAI,
  [Providers.DEEPSEEK]: ChatDeepSeek,
  [Providers.MISTRALAI]: CustomChatMistralAI,
  [Providers.MISTRAL]: CustomChatMistralAI,
  [Providers.ANTHROPIC]: CustomAnthropic,
  [Providers.OPENROUTER]: ChatOpenRouter,
  [Providers.BEDROCK]: CustomChatBedrockConverse,
  [Providers.GOOGLE]: CustomChatGoogleGenerativeAI,
  [Providers.MOONSHOT]: ChatMoonshot,
};

/** Register a host-provided provider without replacing built-in providers. */
export function registerChatModelProvider(
  provider: string,
  modelClass: RegisteredChatModelConstructor,
): void {
  if (typeof provider !== 'string' || provider.trim() === '') {
    throw new Error('Provider name must be a non-empty string');
  }
  if (typeof modelClass !== 'function') {
    throw new Error(`Provider constructor is invalid: ${provider}`);
  }
  if (
    Object.prototype.hasOwnProperty.call(llmProviders, provider) ||
    registeredChatModelProviders.has(provider)
  ) {
    throw new Error(`Provider already registered: ${provider}`);
  }
  registeredChatModelProviders.set(provider, modelClass);
}

/** Resolve either a built-in or explicitly registered host provider. */
export function getRegisteredChatModelClass(
  provider: string,
): RegisteredChatModelConstructor {
  const registered = registeredChatModelProviders.get(provider);
  if (registered) return registered;
  const builtin = llmProviders[provider as Providers];
  if (builtin) return builtin as unknown as RegisteredChatModelConstructor;
  throw new Error(`Unsupported LLM provider: ${provider}`);
}

export const manualToolStreamProviders = new Set<Providers | string>([
  Providers.ANTHROPIC,
  Providers.BEDROCK,
]);

export const getChatModelClass = <P extends Providers>(
  provider: P,
): new (config: ProviderOptionsMap[P]) => ChatModelMap[P] => {
  const ChatModelClass = llmProviders[provider];
  if (!ChatModelClass) {
    throw new Error(`Unsupported LLM provider: ${provider}`);
  }
  return ChatModelClass;
};
