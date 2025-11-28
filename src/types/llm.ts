// src/types/llm.ts
import { ChatMistralAI } from '@langchain/mistralai';
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
import type { OpenAI as OpenAIClient } from 'openai';
import type { ChatXAIInput } from '@langchain/xai';
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
export type AnthropicClientOptions = AnthropicInput;
export type MistralAIClientOptions = ChatMistralAIInput;
export type VertexAIClientOptions = ChatVertexAIInput & {
  includeThoughts?: boolean;
};
export type BedrockAnthropicInput = ChatBedrockConverseInput & {
  additionalModelRequestFields?: ChatBedrockConverseInput['additionalModelRequestFields'] &
    AnthropicReasoning;
  promptCache?: boolean;
};
export type BedrockConverseClientOptions = ChatBedrockConverseInput;
export type BedrockAnthropicClientOptions = BedrockAnthropicInput;
export type GoogleClientOptions = GoogleGenerativeAIChatInput & {
  customHeaders?: RequestOptions['customHeaders'];
  thinkingConfig?: GeminiGenerationConfig['thinkingConfig'];
};
export type DeepSeekClientOptions = ChatDeepSeekCallOptions;
export type XAIClientOptions = ChatXAIInput;

export type ClientOptions =
  | OpenAIClientOptions
  | AzureClientOptions
  | AnthropicClientOptions
  | MistralAIClientOptions
  | VertexAIClientOptions
  | BedrockConverseClientOptions
  | GoogleClientOptions
  | DeepSeekClientOptions
  | XAIClientOptions;

export type SharedLLMConfig = {
  provider: Providers;
  _lc_stream_delay?: number;
};

export type LLMConfig = SharedLLMConfig &
  ClientOptions & {
    /** Optional provider fallbacks in order of attempt */
    fallbacks?: Array<{ provider: Providers; clientOptions?: ClientOptions }>;
  };

export type ProviderOptionsMap = {
  [Providers.AZURE]: AzureClientOptions;
  [Providers.OPENAI]: OpenAIClientOptions;
  [Providers.GOOGLE]: GoogleClientOptions;
  [Providers.VERTEXAI]: VertexAIClientOptions;
  [Providers.DEEPSEEK]: DeepSeekClientOptions;
  [Providers.ANTHROPIC]: AnthropicClientOptions;
  [Providers.MISTRALAI]: MistralAIClientOptions;
  [Providers.MISTRAL]: MistralAIClientOptions;
  [Providers.OPENROUTER]: ChatOpenRouterCallOptions;
  [Providers.BEDROCK]: BedrockConverseClientOptions;
  [Providers.XAI]: XAIClientOptions;
};

export type ChatModelMap = {
  [Providers.XAI]: ChatXAI;
  [Providers.OPENAI]: ChatOpenAI;
  [Providers.AZURE]: AzureChatOpenAI;
  [Providers.DEEPSEEK]: ChatDeepSeek;
  [Providers.VERTEXAI]: ChatVertexAI;
  [Providers.ANTHROPIC]: CustomAnthropic;
  [Providers.MISTRALAI]: ChatMistralAI;
  [Providers.MISTRAL]: ChatMistralAI;
  [Providers.OPENROUTER]: ChatOpenRouter;
  [Providers.BEDROCK]: CustomChatBedrockConverse;
  [Providers.GOOGLE]: CustomChatGoogleGenerativeAI;
};

export type ChatModelConstructorMap = {
  [P in Providers]: new (config: ProviderOptionsMap[P]) => ChatModelMap[P];
};

export type ChatModelInstance = ChatModelMap[Providers];

export type ModelWithTools = ChatModelInstance & {
  bindTools(tools: CommonToolType[]): Runnable;
};
