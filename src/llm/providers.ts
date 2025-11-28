// src/llm/providers.ts
import { ChatMistralAI } from '@langchain/mistralai';
import type {
  ChatModelConstructorMap,
  ProviderOptionsMap,
  ChatModelMap,
} from '@/types';
import {
  AzureChatOpenAI,
  ChatDeepSeek,
  ChatOpenAI,
  ChatXAI,
} from '@/llm/openai';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatVertexAI } from '@/llm/vertexai';
import { Providers } from '@/common';

export const llmProviders: Partial<ChatModelConstructorMap> = {
  [Providers.XAI]: ChatXAI,
  [Providers.OPENAI]: ChatOpenAI,
  [Providers.AZURE]: AzureChatOpenAI,
  [Providers.VERTEXAI]: ChatVertexAI,
  [Providers.DEEPSEEK]: ChatDeepSeek,
  [Providers.MISTRALAI]: ChatMistralAI,
  [Providers.MISTRAL]: ChatMistralAI,
  [Providers.ANTHROPIC]: CustomAnthropic,
  [Providers.OPENROUTER]: ChatOpenRouter,
  [Providers.BEDROCK]: CustomChatBedrockConverse,
  // [Providers.ANTHROPIC]: ChatAnthropic,
  [Providers.GOOGLE]: CustomChatGoogleGenerativeAI,
};

export const manualToolStreamProviders = new Set<Providers | string>([
  Providers.ANTHROPIC,
  Providers.BEDROCK,
]);

export const getChatModelClass = <P extends Providers>(
  provider: P
): new (config: ProviderOptionsMap[P]) => ChatModelMap[P] => {
  const ChatModelClass = llmProviders[provider];
  if (!ChatModelClass) {
    throw new Error(`Unsupported LLM provider: ${provider}`);
  }

  return ChatModelClass;
};
