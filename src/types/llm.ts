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
import type { ChatVertexAIInput } from '@langchain/google-vertexai';
import type { ChatDeepSeekInput } from '@langchain/deepseek';
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

export type AzureClientOptions = Partial<OpenAIChatInput> &
  Partial<AzureOpenAIInput> & {
    openAIApiKey?: string;
    openAIApiVersion?: string;
    openAIBasePath?: string;
    deploymentName?: string;
  } & BaseChatModelParams & {
    configuration?: OAIClientOptions;
  };
/**
 * Controls whether Claude's reasoning content is returned in adaptive
 * thinking responses. Added for Claude Opus 4.7, which omits thinking by
 * default unless the caller opts in with `'summarized'`.
 * @see https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7#thinking-content-omitted-by-default
 */
export type ThinkingDisplay = 'summarized' | 'omitted';
export type ThinkingConfigAdaptive = {
  type: 'adaptive';
  display?: ThinkingDisplay;
};
export type ThinkingConfig =
  | NonNullable<AnthropicInput['thinking']>
  | ThinkingConfigAdaptive;
export type ChatOpenAIToolType =
  | BindToolsInput
  | OpenAIClient.ChatCompletionTool;
export type CommonToolType = StructuredTool | ChatOpenAIToolType;
export type AnthropicReasoning = {
  thinking?: ThinkingConfig | boolean;
  thinkingBudget?: number;
};
export type GoogleThinkingConfig = {
  thinkingBudget?: number;
  includeThoughts?: boolean;
  thinkingLevel?: 'THINKING_LEVEL_UNSPECIFIED' | 'LOW' | 'MEDIUM' | 'HIGH';
};
export type OpenAIClientOptions = ChatOpenAIFields;
export type AnthropicClientOptions = Omit<AnthropicInput, 'thinking'> & {
  thinking?: ThinkingConfig;
  promptCache?: boolean;
};
export type MistralAIClientOptions = ChatMistralAIInput;
export type VertexAIClientOptions = ChatVertexAIInput & {
  includeThoughts?: boolean;
  thinkingConfig?: GoogleThinkingConfig;
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
  thinkingConfig?: GoogleThinkingConfig;
};
export type DeepSeekClientOptions = Partial<ChatDeepSeekInput>;
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
  [Providers.MOONSHOT]: OpenAIClientOptions;
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
  [Providers.MOONSHOT]: ChatMoonshot;
};

export type ChatModelConstructorMap = {
  [P in Providers]: new (config: ProviderOptionsMap[P]) => ChatModelMap[P];
};

export type ChatModelInstance = ChatModelMap[Providers];

export type ModelWithTools = ChatModelInstance & {
  bindTools(tools: CommonToolType[]): Runnable;
};
