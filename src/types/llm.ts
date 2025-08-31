// src/types/llm.ts
import { ChatOllama } from '@langchain/ollama';
import { ChatAnthropic } from '@langchain/anthropic';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatMistralAI } from '@langchain/mistralai';
import { ChatBedrockConverse } from '@langchain/aws';
import { BedrockChat } from '@langchain/community/chat_models/bedrock/web';
import type {
  BindToolsInput,
  BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import type {
  OpenAIChatInput,
  ChatOpenAIFields,
  AzureOpenAIInput,
  ClientOptions as OAIClientOptions,
} from '@langchain/openai';
import type { BedrockChatFields } from '@langchain/community/chat_models/bedrock/web';
import type { GoogleGenerativeAIChatInput } from '@langchain/google-genai';
import type { GeminiGenerationConfig } from '@langchain/google-common';
import type { ChatVertexAIInput } from '@langchain/google-vertexai';
import type { ChatDeepSeekCallOptions } from '@langchain/deepseek';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import type { ChatMistralAIInput } from '@langchain/mistralai';
import type { RequestOptions } from '@google/generative-ai';
import type { StructuredTool } from '@langchain/core/tools';
import type { AnthropicInput } from '@langchain/anthropic';
import type { Runnable } from '@langchain/core/runnables';
import type { ChatOllamaInput } from '@langchain/ollama';
import type { OpenAI as OpenAIClient } from 'openai';
import type { ChatXAIInput } from '@langchain/xai';
import {
  AzureChatOpenAI,
  ChatDeepSeek,
  ChatOpenAI,
  ChatXAI,
} from '@/llm/openai';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ChatVertexAI } from '@/llm/vertexai';
import { Providers } from '@/common';

export type AzureClientOptions = Partial<OpenAIChatInput> &
  Partial<AzureOpenAIInput> & {
    openAIApiKey?: string;
    openAIApiVersion?: string;
    openAIBasePath?: string;
    deploymentName?: string;
  } & BaseChatModelParams & {
    configuration?: OAIClientOptions;
  };
export type ThinkingConfig = AnthropicInput['thinking'];
export type ChatOpenAIToolType =
  | BindToolsInput
  | OpenAIClient.ChatCompletionTool;
export type CommonToolType = StructuredTool | ChatOpenAIToolType;
export type AnthropicReasoning = {
  thinking?: ThinkingConfig | boolean;
  thinkingBudget?: number;
};
export type OpenAIClientOptions = ChatOpenAIFields;
export type OllamaClientOptions = ChatOllamaInput;
export type AnthropicClientOptions = AnthropicInput;
export type MistralAIClientOptions = ChatMistralAIInput;
export type VertexAIClientOptions = ChatVertexAIInput & {
  includeThoughts?: boolean;
};
export type BedrockClientOptions = BedrockChatFields;
export type BedrockAnthropicInput = ChatBedrockConverseInput & {
  additionalModelRequestFields?: ChatBedrockConverseInput['additionalModelRequestFields'] &
    AnthropicReasoning;
};
export type BedrockConverseClientOptions = ChatBedrockConverseInput;
export type GoogleClientOptions = GoogleGenerativeAIChatInput & {
  customHeaders?: RequestOptions['customHeaders'];
  thinkingConfig?: GeminiGenerationConfig['thinkingConfig'];
};
export type DeepSeekClientOptions = ChatDeepSeekCallOptions;
export type XAIClientOptions = ChatXAIInput;

export type ClientOptions =
  | OpenAIClientOptions
  | AzureClientOptions
  | OllamaClientOptions
  | AnthropicClientOptions
  | MistralAIClientOptions
  | VertexAIClientOptions
  | BedrockClientOptions
  | BedrockConverseClientOptions
  | GoogleClientOptions
  | DeepSeekClientOptions
  | XAIClientOptions;

export type SharedLLMConfig = {
  provider: Providers;
};

export type LLMConfig = SharedLLMConfig & ClientOptions & {
  /** Optional provider fallbacks in order of attempt */
  fallbacks?: Array<{ provider: Providers; clientOptions?: ClientOptions }>;
};

export type ProviderOptionsMap = {
  [Providers.AZURE]: AzureClientOptions;
  [Providers.OPENAI]: OpenAIClientOptions;
  [Providers.OLLAMA]: OllamaClientOptions;
  [Providers.GOOGLE]: GoogleClientOptions;
  [Providers.VERTEXAI]: VertexAIClientOptions;
  [Providers.DEEPSEEK]: DeepSeekClientOptions;
  [Providers.ANTHROPIC]: AnthropicClientOptions;
  [Providers.MISTRALAI]: MistralAIClientOptions;
  [Providers.MISTRAL]: MistralAIClientOptions;
  [Providers.OPENROUTER]: ChatOpenRouterCallOptions;
  [Providers.BEDROCK_LEGACY]: BedrockClientOptions;
  [Providers.BEDROCK]: BedrockConverseClientOptions;
  [Providers.XAI]: XAIClientOptions;
};

export type ChatModelMap = {
  [Providers.XAI]: ChatXAI;
  [Providers.OPENAI]: ChatOpenAI;
  [Providers.OLLAMA]: ChatOllama;
  [Providers.AZURE]: AzureChatOpenAI;
  [Providers.DEEPSEEK]: ChatDeepSeek;
  [Providers.VERTEXAI]: ChatVertexAI;
  [Providers.ANTHROPIC]: CustomAnthropic;
  [Providers.MISTRALAI]: ChatMistralAI;
  [Providers.MISTRAL]: ChatMistralAI;
  [Providers.OPENROUTER]: ChatOpenRouter;
  [Providers.BEDROCK_LEGACY]: BedrockChat;
  [Providers.BEDROCK]: ChatBedrockConverse;
  [Providers.GOOGLE]: CustomChatGoogleGenerativeAI;
};

export type ChatModelConstructorMap = {
  [P in Providers]: new (config: ProviderOptionsMap[P]) => ChatModelMap[P];
};

export type ChatModelInstance = ChatModelMap[Providers];

export type ModelWithTools = ChatModelInstance & {
  bindTools(tools: CommonToolType[]): Runnable;
};
