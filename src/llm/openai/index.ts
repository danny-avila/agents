import { AzureOpenAI as AzureOpenAIClient } from 'openai';
import { ChatXAI as OriginalChatXAI } from '@langchain/xai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  AIMessage,
  AIMessageChunk,
  isAIMessage,
} from '@langchain/core/messages';
import { ToolDefinition } from '@langchain/core/language_models/base';
import {
  convertToOpenAITool,
  isLangChainTool,
} from '@langchain/core/utils/function_calling';
import { ChatDeepSeek as OriginalChatDeepSeek } from '@langchain/deepseek';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  getEndpoint,
  OpenAIClient,
  getHeadersWithUserAgent,
  ChatOpenAI as OriginalChatOpenAI,
  ChatOpenAIResponses as OriginalChatOpenAIResponses,
  ChatOpenAICompletions as OriginalChatOpenAICompletions,
  AzureChatOpenAI as OriginalAzureChatOpenAI,
  AzureChatOpenAIResponses as OriginalAzureChatOpenAIResponses,
  AzureChatOpenAICompletions as OriginalAzureChatOpenAICompletions,
} from '@langchain/openai';
import type { HeaderValue, HeadersLike } from './types';
import type {
  BaseMessage,
  BaseMessageChunk,
  UsageMetadata,
} from '@langchain/core/messages';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { ChatGeneration, ChatResult } from '@langchain/core/outputs';
import type { ChatXAIInput } from '@langchain/xai';
import type * as t from '@langchain/openai';
import {
  isReasoningModel,
  flattenAnthropicThinkingForOpenAI,
  _convertMessagesToOpenAIParams,
} from './utils';
import { sleep } from '@/utils';

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
const iife = <T>(fn: () => T) => fn();

export function isHeaders(headers: unknown): headers is Headers {
  return (
    typeof Headers !== 'undefined' &&
    headers !== null &&
    typeof headers === 'object' &&
    Object.prototype.toString.call(headers) === '[object Headers]'
  );
}

export function normalizeHeaders(
  headers: HeadersLike
): Record<string, HeaderValue | readonly HeaderValue[]> {
  const output = iife(() => {
    // If headers is a Headers instance
    if (isHeaders(headers)) {
      return headers;
    }
    // If headers is an array of [key, value] pairs
    else if (Array.isArray(headers)) {
      return new Headers(headers);
    }
    // If headers is a NullableHeaders-like object (has 'values' property that is a Headers)
    else if (
      typeof headers === 'object' &&
      headers !== null &&
      'values' in headers &&
      isHeaders(headers.values)
    ) {
      return headers.values;
    }
    // If headers is a plain object
    else if (typeof headers === 'object' && headers !== null) {
      const entries: [string, string][] = Object.entries(headers)
        .filter(([, v]) => typeof v === 'string')
        .map(([k, v]) => [k, v as string]);
      return new Headers(entries);
    }
    return new Headers();
  });

  return Object.fromEntries(output.entries());
}

type OpenAICoreRequestOptions = OpenAIClient.RequestOptions;
type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;
type OpenAIClientConfig = NonNullable<
  ConstructorParameters<typeof OpenAIClient>[0]
>;
type LibreChatOpenAIFields = t.ChatOpenAIFields & {
  _lc_stream_delay?: number;
  includeReasoningContent?: boolean;
  includeReasoningDetails?: boolean;
  convertReasoningDetailsToContent?: boolean;
};
type LibreChatAzureOpenAIFields = t.AzureOpenAIInput & {
  _lc_stream_delay?: number;
};
type ReasoningCallOptions = {
  reasoning?: OpenAIClient.Reasoning;
  reasoningEffort?: OpenAIClient.Reasoning['effort'];
};
type OpenAIDeltaWithLibreChatFields = Record<string, unknown> & {
  reasoning?: unknown;
  reasoning_details?: unknown;
  provider_specific_fields?: unknown;
};
type OpenAIClientOwner = {
  client?: OpenAIClient;
  clientConfig: OpenAIClientConfig;
  timeout?: number;
};
type AbortableOpenAIClient = CustomOpenAIClient | CustomAzureOpenAIClient;
type OpenAIClientDelegate = {
  client?: AbortableOpenAIClient;
  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions;
};
type OpenAIChatCompletion = OpenAIClient.Chat.Completions.ChatCompletion;
type OpenAIChatCompletionChunk =
  OpenAIClient.Chat.Completions.ChatCompletionChunk;
type OpenAIChatCompletionStreamItem =
  | OpenAIChatCompletionChunk
  | {
      event: string;
      data?: unknown;
    };
type OpenAIChatCompletionRequest =
  | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
  | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming;
type OpenAIChatCompletionResult =
  | AsyncIterable<OpenAIChatCompletionChunk>
  | OpenAIChatCompletion;
type OpenAIChatCompletionRetry = (
  request: OpenAIChatCompletionRequest,
  requestOptions?: OpenAICoreRequestOptions
) => Promise<
  AsyncIterable<OpenAIChatCompletionStreamItem> | OpenAIChatCompletion
>;

function createUsageMetadata(
  usage?: OpenAIClient.Completions.CompletionUsage
): UsageMetadata {
  const usageMetadata: UsageMetadata = {
    input_tokens: usage?.prompt_tokens ?? 0,
    output_tokens: usage?.completion_tokens ?? 0,
    total_tokens: usage?.total_tokens ?? 0,
  };

  if (usage == null) {
    return usageMetadata;
  }

  const inputTokenDetails: UsageMetadata['input_token_details'] = {};
  const outputTokenDetails: UsageMetadata['output_token_details'] = {};
  let hasInputTokenDetails = false;
  let hasOutputTokenDetails = false;
  const audioInputTokens = usage.prompt_tokens_details?.audio_tokens;
  const cachedInputTokens = usage.prompt_tokens_details?.cached_tokens;
  const audioOutputTokens = usage.completion_tokens_details?.audio_tokens;
  const reasoningOutputTokens =
    usage.completion_tokens_details?.reasoning_tokens;

  if (audioInputTokens != null) {
    inputTokenDetails.audio = audioInputTokens;
    hasInputTokenDetails = true;
  }
  if (cachedInputTokens != null) {
    inputTokenDetails.cache_read = cachedInputTokens;
    hasInputTokenDetails = true;
  }
  if (audioOutputTokens != null) {
    outputTokenDetails.audio = audioOutputTokens;
    hasOutputTokenDetails = true;
  }
  if (reasoningOutputTokens != null) {
    outputTokenDetails.reasoning = reasoningOutputTokens;
    hasOutputTokenDetails = true;
  }

  if (hasInputTokenDetails) {
    usageMetadata.input_token_details = inputTokenDetails;
  }
  if (hasOutputTokenDetails) {
    usageMetadata.output_token_details = outputTokenDetails;
  }

  return usageMetadata;
}

function getExposedOpenAIClient(
  completions: OpenAIClientDelegate,
  responses: OpenAIClientDelegate,
  preferResponses: boolean
): AbortableOpenAIClient {
  const responsesClient = responses.client;
  if (responsesClient?.abortHandler != null) {
    return responsesClient;
  }
  const completionsClient = completions.client;
  if (completionsClient?.abortHandler != null) {
    return completionsClient;
  }

  const delegate = preferResponses ? responses : completions;
  delegate._getClientOptions(undefined);
  return delegate.client as AbortableOpenAIClient;
}

function getReasoningParams(
  baseReasoning: OpenAIClient.Reasoning | undefined,
  options?: ReasoningCallOptions
): OpenAIClient.Reasoning | undefined {
  let reasoning: OpenAIClient.Reasoning | undefined;
  if (baseReasoning !== undefined) {
    reasoning = {
      ...reasoning,
      ...baseReasoning,
    };
  }
  if (options?.reasoning !== undefined) {
    reasoning = {
      ...reasoning,
      ...options.reasoning,
    };
  }
  if (
    options?.reasoningEffort !== undefined &&
    reasoning?.effort === undefined
  ) {
    reasoning = {
      ...reasoning,
      effort: options.reasoningEffort,
    };
  }
  return reasoning;
}

function getGatedReasoningParams(
  model: string,
  baseReasoning: OpenAIClient.Reasoning | undefined,
  options?: ReasoningCallOptions
): OpenAIClient.Reasoning | undefined {
  if (!isReasoningModel(model)) {
    return;
  }
  return getReasoningParams(baseReasoning, options);
}

function isObject(value: unknown): value is object {
  return typeof value === 'object' && value !== null;
}

function isOpenAIChatCompletionChunk(
  value: unknown
): value is OpenAIChatCompletionChunk {
  if (!isObject(value)) {
    return false;
  }

  // Intentionally loose: downstream handlers already tolerate empty choices.
  const { choices } = value as { choices?: unknown };
  return Array.isArray(choices);
}

function getOpenAIChatCompletionChunk(
  value: OpenAIChatCompletionStreamItem
): OpenAIChatCompletionChunk | undefined {
  if (isOpenAIChatCompletionChunk(value)) {
    return value;
  }

  const { data } = value;
  if (isOpenAIChatCompletionChunk(data)) {
    return data;
  }

  return undefined;
}

async function* filterOpenAIChatCompletionStream(
  stream: AsyncIterable<OpenAIChatCompletionStreamItem>
): AsyncGenerator<OpenAIChatCompletionChunk> {
  for await (const item of stream) {
    const chunk = getOpenAIChatCompletionChunk(item);
    if (chunk == null) {
      continue;
    }
    yield chunk;
  }
}

async function completionWithFilteredOpenAIStream(
  request: OpenAIChatCompletionRequest,
  requestOptions: OpenAICoreRequestOptions | undefined,
  completionWithRetry: OpenAIChatCompletionRetry
): Promise<OpenAIChatCompletionResult> {
  if (request.stream !== true) {
    return (await completionWithRetry(
      request,
      requestOptions
    )) as OpenAIChatCompletion;
  }

  const stream = await completionWithRetry(request, requestOptions);
  return filterOpenAIChatCompletionStream(
    stream as AsyncIterable<OpenAIChatCompletionStreamItem>
  );
}

function attachLibreChatDeltaFields(
  chunk: BaseMessageChunk,
  delta: Record<string, unknown>
): BaseMessageChunk {
  if (!AIMessageChunk.isInstance(chunk)) {
    return chunk;
  }

  const libreChatDelta = delta as OpenAIDeltaWithLibreChatFields;
  if (
    libreChatDelta.reasoning != null &&
    chunk.additional_kwargs.reasoning_content == null
  ) {
    chunk.additional_kwargs.reasoning_content = libreChatDelta.reasoning;
  }
  if (libreChatDelta.reasoning_details != null) {
    chunk.additional_kwargs.reasoning_details =
      libreChatDelta.reasoning_details;
  }
  if (libreChatDelta.provider_specific_fields != null) {
    chunk.additional_kwargs.provider_specific_fields =
      libreChatDelta.provider_specific_fields;
  }
  return chunk;
}

function attachLibreChatMessageFields(
  message: BaseMessage,
  rawMessage: Record<string, unknown>
): BaseMessage {
  if (!isAIMessage(message)) {
    return message;
  }
  if (
    rawMessage.reasoning != null &&
    message.additional_kwargs.reasoning_content == null
  ) {
    message.additional_kwargs.reasoning_content = rawMessage.reasoning;
  }
  if (rawMessage.reasoning_details != null) {
    message.additional_kwargs.reasoning_details = rawMessage.reasoning_details;
  }
  if (rawMessage.provider_specific_fields != null) {
    message.additional_kwargs.provider_specific_fields =
      rawMessage.provider_specific_fields;
  }
  return message;
}

function getCustomOpenAIClientOptions(
  owner: OpenAIClientOwner,
  options?: OpenAICoreRequestOptions
): OpenAICoreRequestOptions {
  if (!(owner.client as OpenAIClient | undefined)) {
    const openAIEndpointConfig: t.OpenAIEndpointConfig = {
      baseURL: owner.clientConfig.baseURL,
    };

    const endpoint = getEndpoint(openAIEndpointConfig);
    const params = {
      ...owner.clientConfig,
      baseURL: endpoint,
      timeout: owner.timeout,
      maxRetries: 0,
    };
    if (params.baseURL == null) {
      delete params.baseURL;
    }

    params.defaultHeaders = getHeadersWithUserAgent(params.defaultHeaders);
    owner.client = new CustomOpenAIClient(params);
  }
  const requestOptions = {
    ...owner.clientConfig,
    ...options,
  } as OpenAICoreRequestOptions;
  return requestOptions;
}

async function* delayStreamChunks<T>(
  chunks: AsyncGenerator<T>,
  delay?: number
): AsyncGenerator<T> {
  for await (const chunk of chunks) {
    yield chunk;
    if (delay != null) {
      await sleep(delay);
    }
  }
}

function createAbortHandler(controller: AbortController): () => void {
  return function (): void {
    controller.abort();
  };
}
/**
 * Formats a tool in either OpenAI format, or LangChain structured tool format
 * into an OpenAI tool format. If the tool is already in OpenAI format, return without
 * any changes. If it is in LangChain structured tool format, convert it to OpenAI tool format
 * using OpenAI's `zodFunction` util, falling back to `convertToOpenAIFunction` if the parameters
 * returned from the `zodFunction` util are not defined.
 *
 * @param {BindToolsInput} tool The tool to convert to an OpenAI tool.
 * @param {Object} [fields] Additional fields to add to the OpenAI tool.
 * @returns {ToolDefinition} The inputted tool in OpenAI tool format.
 */
export function _convertToOpenAITool(
  tool: BindToolsInput,
  fields?: {
    /**
     * If `true`, model output is guaranteed to exactly match the JSON Schema
     * provided in the function definition.
     */
    strict?: boolean;
  }
): OpenAIClient.ChatCompletionTool {
  let toolDef: OpenAIClient.ChatCompletionTool | undefined;

  if (isLangChainTool(tool)) {
    toolDef = convertToOpenAITool(tool);
  } else {
    toolDef = tool as ToolDefinition;
  }

  if (fields?.strict !== undefined) {
    toolDef.function.strict = fields.strict;
  }

  return toolDef;
}
export class CustomOpenAIClient extends OpenAIClient {
  abortHandler?: () => void;
  async fetchWithTimeout(
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController
  ): Promise<Response> {
    const { signal, ...options } = init || {};
    const handler = createAbortHandler(controller);
    this.abortHandler = handler;
    if (signal) signal.addEventListener('abort', handler, { once: true });

    const timeout = setTimeout(() => handler, ms);

    const fetchOptions = {
      signal: controller.signal as AbortSignal,
      ...options,
    };
    if (fetchOptions.method != null) {
      // Custom methods like 'patch' need to be uppercased
      // See https://github.com/nodejs/undici/issues/2294
      fetchOptions.method = fetchOptions.method.toUpperCase();
    }

    return (
      // use undefined this binding; fetch errors if bound to something else in browser/cloudflare
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      this.fetch.call(undefined, url, fetchOptions).finally(() => {
        clearTimeout(timeout);
      })
    );
  }
}
export class CustomAzureOpenAIClient extends AzureOpenAIClient {
  abortHandler?: () => void;
  async fetchWithTimeout(
    url: RequestInfo,
    init: RequestInit | undefined,
    ms: number,
    controller: AbortController
  ): Promise<Response> {
    const { signal, ...options } = init || {};
    const handler = createAbortHandler(controller);
    this.abortHandler = handler;
    if (signal) signal.addEventListener('abort', handler, { once: true });

    const timeout = setTimeout(() => handler, ms);

    const fetchOptions = {
      signal: controller.signal as AbortSignal,
      ...options,
    };
    if (fetchOptions.method != null) {
      // Custom methods like 'patch' need to be uppercased
      // See https://github.com/nodejs/undici/issues/2294
      fetchOptions.method = fetchOptions.method.toUpperCase();
    }

    return (
      // use undefined this binding; fetch errors if bound to something else in browser/cloudflare
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      this.fetch.call(undefined, url, fetchOptions).finally(() => {
        clearTimeout(timeout);
      })
    );
  }
}

class LibreChatOpenAICompletions extends OriginalChatOpenAICompletions {
  private includeReasoningContent?: boolean;
  private includeReasoningDetails?: boolean;
  private convertReasoningDetailsToContent?: boolean;

  constructor(fields?: LibreChatOpenAIFields) {
    super(fields);
    this.includeReasoningContent = fields?.includeReasoningContent;
    this.includeReasoningDetails = fields?.includeReasoningDetails;
    this.convertReasoningDetailsToContent =
      fields?.convertReasoningDetailsToContent;
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.reasoning, options);
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    return getCustomOpenAIClientOptions(this, options);
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIChatCompletion>;
  async completionWithRetry(
    request:
      | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
      | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk> | OpenAIChatCompletion> {
    return completionWithFilteredOpenAIStream(
      request,
      requestOptions,
      super.completionWithRetry.bind(this) as OpenAIChatCompletionRetry
    );
  }

  protected _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: OpenAIClient.Chat.Completions.ChatCompletionChunk,
    defaultRole?: OpenAIClient.Chat.ChatCompletionRole
  ): BaseMessageChunk {
    return attachLibreChatDeltaFields(
      super._convertCompletionsDeltaToBaseMessageChunk(
        delta,
        rawResponse,
        defaultRole
      ),
      delta
    );
  }

  protected _convertCompletionsMessageToBaseMessage(
    message: OpenAIClient.ChatCompletionMessage,
    rawResponse: OpenAIClient.ChatCompletion
  ): BaseMessage {
    return attachLibreChatMessageFields(
      super._convertCompletionsMessageToBaseMessage(message, rawResponse),
      message as unknown as Record<string, unknown>
    );
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    if (
      this.includeReasoningContent !== true &&
      this.includeReasoningDetails !== true
    ) {
      return super._generate(messages, options, runManager);
    }

    options.signal?.throwIfAborted();
    const usageMetadata: Partial<UsageMetadata> = {};
    const params = this.invocationParams(options);
    const messagesMapped = _convertMessagesToOpenAIParams(
      messages,
      this.model,
      {
        includeReasoningContent: this.includeReasoningContent,
        includeReasoningDetails: this.includeReasoningDetails,
        convertReasoningDetailsToContent: this.convertReasoningDetailsToContent,
      }
    );

    if (params.stream === true) {
      const stream = this._streamResponseChunks(messages, options, runManager);
      const finalChunks = new Map<number, ChatGenerationChunk>();
      for await (const chunk of stream) {
        chunk.message.response_metadata = {
          ...chunk.generationInfo,
          ...chunk.message.response_metadata,
        };
        const index =
          typeof chunk.generationInfo?.completion === 'number'
            ? chunk.generationInfo.completion
            : 0;
        const existingChunk = finalChunks.get(index);
        if (existingChunk == null) {
          finalChunks.set(index, chunk);
        } else {
          finalChunks.set(index, existingChunk.concat(chunk));
        }
      }
      const generations = Array.from(finalChunks.entries())
        .sort(([aKey], [bKey]) => aKey - bKey)
        .map(([, value]) => value);
      const { functions, function_call } = this.invocationParams(options);
      const promptTokenUsage = await this._getEstimatedTokenCountFromPrompt(
        messages,
        functions,
        function_call
      );
      const completionTokenUsage =
        await this._getNumTokensFromGenerations(generations);
      usageMetadata.input_tokens = promptTokenUsage;
      usageMetadata.output_tokens = completionTokenUsage;
      usageMetadata.total_tokens = promptTokenUsage + completionTokenUsage;
      return {
        generations,
        llmOutput: {
          estimatedTokenUsage: {
            promptTokens: usageMetadata.input_tokens,
            completionTokens: usageMetadata.output_tokens,
            totalTokens: usageMetadata.total_tokens,
          },
        },
      };
    }

    const data = await this.completionWithRetry(
      {
        ...params,
        stream: false,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );
    const {
      completion_tokens: completionTokens,
      prompt_tokens: promptTokens,
      total_tokens: totalTokens,
      prompt_tokens_details: promptTokensDetails,
      completion_tokens_details: completionTokensDetails,
    } = data.usage ?? {};

    if (completionTokens != null) {
      usageMetadata.output_tokens =
        (usageMetadata.output_tokens ?? 0) + completionTokens;
    }
    if (promptTokens != null) {
      usageMetadata.input_tokens =
        (usageMetadata.input_tokens ?? 0) + promptTokens;
    }
    if (totalTokens != null) {
      usageMetadata.total_tokens =
        (usageMetadata.total_tokens ?? 0) + totalTokens;
    }
    if (
      promptTokensDetails?.audio_tokens != null ||
      promptTokensDetails?.cached_tokens != null
    ) {
      usageMetadata.input_token_details = {
        ...(promptTokensDetails.audio_tokens != null && {
          audio: promptTokensDetails.audio_tokens,
        }),
        ...(promptTokensDetails.cached_tokens != null && {
          cache_read: promptTokensDetails.cached_tokens,
        }),
      };
    }
    if (
      completionTokensDetails?.audio_tokens != null ||
      completionTokensDetails?.reasoning_tokens != null
    ) {
      usageMetadata.output_token_details = {
        ...(completionTokensDetails.audio_tokens != null && {
          audio: completionTokensDetails.audio_tokens,
        }),
        ...(completionTokensDetails.reasoning_tokens != null && {
          reasoning: completionTokensDetails.reasoning_tokens,
        }),
      };
    }

    const generations: ChatGeneration[] = [];
    for (const part of data.choices) {
      const generation: ChatGeneration = {
        text: part.message.content ?? '',
        message: this._convertCompletionsMessageToBaseMessage(
          part.message,
          data
        ),
      };
      generation.generationInfo = {
        finish_reason: part.finish_reason,
        ...(part.logprobs ? { logprobs: part.logprobs } : {}),
      };
      if (isAIMessage(generation.message)) {
        generation.message.usage_metadata = usageMetadata as UsageMetadata;
      }
      generation.message = new AIMessage(
        Object.fromEntries(
          Object.entries(generation.message).filter(
            ([key]) => !key.startsWith('lc_')
          )
        )
      );
      generations.push(generation);
    }
    return {
      generations,
      llmOutput: {
        tokenUsage: {
          promptTokens: usageMetadata.input_tokens,
          completionTokens: usageMetadata.output_tokens,
          totalTokens: usageMetadata.total_tokens,
        },
      },
    };
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (
      this.includeReasoningContent !== true &&
      this.includeReasoningDetails !== true
    ) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }

    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model, {
        includeReasoningContent: this.includeReasoningContent,
        includeReasoningDetails: this.includeReasoningDetails,
        convertReasoningDetailsToContent: this.convertReasoningDetailsToContent,
      });

    const params = {
      ...this.invocationParams(options, {
        streaming: true,
      }),
      messages: messagesMapped,
      stream: true as const,
    };
    let defaultRole: OpenAIClient.Chat.ChatCompletionRole | undefined;

    const streamIterable = await this.completionWithRetry(params, options);
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;
    for await (const data of streamIterable) {
      if (options.signal?.aborted === true) {
        return;
      }
      type StreamChoice = Omit<
        OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice,
        'delta'
      > & {
        delta?: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['delta'];
      };
      const choices = data.choices as StreamChoice[] | undefined;
      const choice = choices?.[0];
      if (data.usage != null) {
        usage = data.usage;
      }
      if (choice == null) {
        continue;
      }

      const { delta } = choice;
      if (delta == null) {
        continue;
      }
      const chunk = this._convertCompletionsDeltaToBaseMessageChunk(
        delta as unknown as Record<string, unknown>,
        data,
        defaultRole
      );
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: options.promptIndex ?? 0,
        completion: choice.index,
      };
      if (typeof chunk.content !== 'string') {
        // eslint-disable-next-line no-console
        console.log(
          '[WARNING]: Received non-string content from OpenAI. This is currently not supported.'
        );
        continue;
      }
      const generationInfo: Record<string, unknown> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
        generationInfo.service_tier = data.service_tier;
      }
      if (this.logprobs === true) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text,
        newTokenIndices,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (usage) {
      const inputTokenDetails = {
        ...(usage.prompt_tokens_details?.audio_tokens != null && {
          audio: usage.prompt_tokens_details.audio_tokens,
        }),
        ...(usage.prompt_tokens_details?.cached_tokens != null && {
          cache_read: usage.prompt_tokens_details.cached_tokens,
        }),
      };
      const outputTokenDetails = {
        ...(usage.completion_tokens_details?.audio_tokens != null && {
          audio: usage.completion_tokens_details.audio_tokens,
        }),
        ...(usage.completion_tokens_details?.reasoning_tokens != null && {
          reasoning: usage.completion_tokens_details.reasoning_tokens,
        }),
      };
      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          response_metadata: { usage: { ...usage } },
          usage_metadata: {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            ...(Object.keys(inputTokenDetails).length > 0 && {
              input_token_details: inputTokenDetails,
            }),
            ...(Object.keys(outputTokenDetails).length > 0 && {
              output_token_details: outputTokenDetails,
            }),
          },
        }),
        text: '',
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text,
        {
          prompt: 0,
          completion: 0,
        },
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }
}

class LibreChatOpenAIResponses extends OriginalChatOpenAIResponses {
  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.reasoning, options);
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    return getCustomOpenAIClientOptions(this, options);
  }
}

class LibreChatAzureOpenAICompletions extends OriginalAzureChatOpenAICompletions {
  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(this.model, this.reasoning, options);
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<OpenAIChatCompletion>;
  async completionWithRetry(
    request:
      | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
      | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAICoreRequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk> | OpenAIChatCompletion> {
    return completionWithFilteredOpenAIStream(
      request,
      requestOptions,
      super.completionWithRetry.bind(this) as OpenAIChatCompletionRetry
    );
  }
}

class LibreChatAzureOpenAIResponses extends OriginalAzureChatOpenAIResponses {
  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(this.model, this.reasoning, options);
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }
}

function withLibreChatOpenAIFields(
  fields?: LibreChatOpenAIFields
): LibreChatOpenAIFields {
  const nextFields = fields ?? {};
  return {
    ...nextFields,
    completions:
      nextFields.completions ?? new LibreChatOpenAICompletions(nextFields),
    responses: nextFields.responses ?? new LibreChatOpenAIResponses(nextFields),
  };
}

export class ChatOpenAI extends OriginalChatOpenAI<t.ChatOpenAICallOptions> {
  _lc_stream_delay?: number;

  constructor(
    fields?: LibreChatOpenAIFields & t.OpenAIChatInput['modelKwargs']
  ) {
    super(withLibreChatOpenAIFields(fields));
    this._lc_stream_delay = fields?._lc_stream_delay;
  }

  public get exposedClient(): CustomOpenAIClient {
    return getExposedOpenAIClient(
      this.completions as OpenAIClientDelegate,
      this.responses as OpenAIClientDelegate,
      this._useResponsesApi(undefined)
    ) as CustomOpenAIClient;
  }
  static lc_name(): string {
    return 'LibreChatOpenAI';
  }
  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  /**
   * Returns backwards compatible reasoning parameters from constructor params and call options
   * @internal
   */
  getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getReasoningParams(this.reasoning, options);
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    return super._generate(
      flattenAnthropicThinkingForOpenAI(messages, this.model),
      options,
      runManager
    );
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(
        flattenAnthropicThinkingForOpenAI(messages, this.model),
        options,
        runManager
      ),
      this._lc_stream_delay
    );
  }
}

export class AzureChatOpenAI extends OriginalAzureChatOpenAI {
  _lc_stream_delay?: number;

  constructor(fields?: LibreChatAzureOpenAIFields) {
    super(fields);
    this.completions = new LibreChatAzureOpenAICompletions(fields);
    this.responses = new LibreChatAzureOpenAIResponses(fields);
    this._lc_stream_delay = fields?._lc_stream_delay;
  }

  public get exposedClient(): CustomOpenAIClient {
    return getExposedOpenAIClient(
      this.completions as OpenAIClientDelegate,
      this.responses as OpenAIClientDelegate,
      this._useResponsesApi(undefined)
    ) as CustomOpenAIClient;
  }
  static lc_name(): 'LibreChatAzureOpenAI' {
    return 'LibreChatAzureOpenAI';
  }
  /**
   * Returns backwards compatible reasoning parameters from constructor params and call options
   * @internal
   */
  getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return getGatedReasoningParams(this.model, this.reasoning, options);
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  _getClientOptions(
    options: OpenAICoreRequestOptions | undefined
  ): OpenAICoreRequestOptions {
    if (!(this.client as unknown as AzureOpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        azureOpenAIApiDeploymentName: this.azureOpenAIApiDeploymentName,
        azureOpenAIApiInstanceName: this.azureOpenAIApiInstanceName,
        azureOpenAIApiKey: this.azureOpenAIApiKey,
        azureOpenAIBasePath: this.azureOpenAIBasePath,
        azureADTokenProvider: this.azureADTokenProvider,
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);

      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };

      if (!this.azureADTokenProvider) {
        params.apiKey = openAIEndpointConfig.azureOpenAIApiKey;
      }

      if (params.baseURL == null) {
        delete params.baseURL;
      }

      const defaultHeaders = normalizeHeaders(params.defaultHeaders);
      params.defaultHeaders = {
        ...params.defaultHeaders,
        'User-Agent':
          defaultHeaders['User-Agent'] != null
            ? `${defaultHeaders['User-Agent']}: librechat-azure-openai-v2`
            : 'librechat-azure-openai-v2',
      };

      this.client = new CustomAzureOpenAIClient({
        apiVersion: this.azureOpenAIApiVersion,
        azureADTokenProvider: this.azureADTokenProvider,
        ...(params as t.AzureOpenAIInput),
      }) as unknown as CustomOpenAIClient;
    }

    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    if (this.azureOpenAIApiKey != null) {
      requestOptions.headers = {
        'api-key': this.azureOpenAIApiKey,
        ...requestOptions.headers,
      };
      requestOptions.query = {
        'api-version': this.azureOpenAIApiVersion,
        ...requestOptions.query,
      };
    }
    return requestOptions;
  }
  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    return super._generate(
      flattenAnthropicThinkingForOpenAI(messages, this.model),
      options,
      runManager
    );
  }
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(
        flattenAnthropicThinkingForOpenAI(messages, this.model),
        options,
        runManager
      ),
      this._lc_stream_delay
    );
  }
}
export class ChatDeepSeek extends OriginalChatDeepSeek {
  _lc_stream_delay?: number;

  constructor(
    fields?: ConstructorParameters<typeof OriginalChatDeepSeek>[0] & {
      _lc_stream_delay?: number;
    }
  ) {
    super(fields);
    this._lc_stream_delay = fields?._lc_stream_delay;
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }
  static lc_name(): 'LibreChatDeepSeek' {
    return 'LibreChatDeepSeek';
  }

  protected _convertDeepSeekMessages(
    messages: BaseMessage[]
  ): OpenAICompletionParam[] {
    return _convertMessagesToOpenAIParams(messages, this.model, {
      includeReasoningContent: true,
    });
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    options.signal?.throwIfAborted();
    const params = this.invocationParams(options);

    messages = flattenAnthropicThinkingForOpenAI(messages, this.model);

    if (params.stream === true) {
      return super._generate(messages, options, runManager);
    }

    const messagesMapped = this._convertDeepSeekMessages(messages);
    const response = await this.completionWithRetry(
      {
        ...params,
        stream: false,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );

    const usageMetadata = createUsageMetadata(response.usage);

    const generations: ChatGeneration[] = response.choices.map((part) => {
      const text = part.message.content ?? '';
      const generation: ChatGeneration = {
        text,
        message: this._convertCompletionsMessageToBaseMessage(
          part.message,
          response
        ),
      };
      generation.generationInfo = {
        finish_reason: part.finish_reason,
        ...(part.logprobs != null ? { logprobs: part.logprobs } : {}),
      };
      if (isAIMessage(generation.message)) {
        generation.message.usage_metadata = usageMetadata;
      }
      generation.message = new AIMessage(
        Object.fromEntries(
          Object.entries(generation.message).filter(
            ([key]) => !key.startsWith('lc_')
          )
        )
      );
      return generation;
    });

    return {
      generations,
      llmOutput: {
        tokenUsage: {
          promptTokens: usageMetadata.input_tokens,
          completionTokens: usageMetadata.output_tokens,
          totalTokens: usageMetadata.total_tokens,
        },
      },
    };
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      this._streamResponseChunksWithReasoning(
        flattenAnthropicThinkingForOpenAI(messages, this.model),
        options,
        runManager
      ),
      this._lc_stream_delay
    );
  }

  /** Parses raw `<think>` fallback tags across chunks and emits sanitized DeepSeek stream chunks. */
  protected async *_streamResponseChunksWithReasoning(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const stream = this._streamResponseChunksFromReasoningMessages(
      messages,
      options
    );
    const thinkStartTag = '<think>';
    const thinkEndTag = '</think>';
    let tokensBuffer = '';
    let isThinking = false;

    for await (const chunk of stream) {
      if (options.signal?.aborted === true) {
        throw new Error('AbortError');
      }

      const reasoningContent =
        chunk.message.additional_kwargs.reasoning_content;
      if (reasoningContent != null && reasoningContent !== '') {
        yield* this._yieldDeepSeekStreamChunk(chunk, runManager);
        continue;
      }

      const text = chunk.text;
      if (text === '') {
        yield* this._yieldDeepSeekStreamChunk(chunk, runManager);
        continue;
      }

      tokensBuffer += text;

      while (tokensBuffer !== '') {
        if (isThinking) {
          const thinkEndIndex = tokensBuffer.indexOf(thinkEndTag);
          if (thinkEndIndex !== -1) {
            const thoughtContent = tokensBuffer.substring(0, thinkEndIndex);
            if (thoughtContent !== '') {
              yield* this._yieldDeepSeekReasoningText(
                chunk,
                thoughtContent,
                runManager
              );
            }

            tokensBuffer = tokensBuffer.substring(
              thinkEndIndex + thinkEndTag.length
            );
            isThinking = false;
            continue;
          }

          const splitIndex = this._getDeepSeekPartialTagSplitIndex(
            tokensBuffer,
            thinkEndTag
          );
          if (splitIndex !== -1) {
            const safeToYield = tokensBuffer.substring(0, splitIndex);
            if (safeToYield !== '') {
              yield* this._yieldDeepSeekReasoningText(
                chunk,
                safeToYield,
                runManager
              );
            }
            tokensBuffer = tokensBuffer.substring(splitIndex);
            break;
          }

          yield* this._yieldDeepSeekReasoningText(
            chunk,
            tokensBuffer,
            runManager
          );
          tokensBuffer = '';
          break;
        }

        const thinkStartIndex = tokensBuffer.indexOf(thinkStartTag);
        if (thinkStartIndex !== -1) {
          const beforeThink = tokensBuffer.substring(0, thinkStartIndex);
          if (beforeThink !== '') {
            yield* this._yieldDeepSeekStreamChunk(
              this._createDeepSeekStreamChunk(chunk, beforeThink),
              runManager
            );
          }

          tokensBuffer = tokensBuffer.substring(
            thinkStartIndex + thinkStartTag.length
          );
          isThinking = true;
          continue;
        }

        const splitIndex = this._getDeepSeekPartialTagSplitIndex(
          tokensBuffer,
          thinkStartTag
        );
        if (splitIndex !== -1) {
          const safeToYield = tokensBuffer.substring(0, splitIndex);
          if (safeToYield !== '') {
            yield* this._yieldDeepSeekStreamChunk(
              this._createDeepSeekStreamChunk(chunk, safeToYield),
              runManager
            );
          }
          tokensBuffer = tokensBuffer.substring(splitIndex);
          break;
        }

        yield* this._yieldDeepSeekStreamChunk(
          this._createDeepSeekStreamChunk(chunk, tokensBuffer),
          runManager
        );
        tokensBuffer = '';
        break;
      }
    }

    if (tokensBuffer === '') {
      return;
    }

    if (isThinking) {
      yield* this._yieldDeepSeekStreamChunk(
        new ChatGenerationChunk({
          message: new AIMessageChunk({
            content: '',
            additional_kwargs: {
              reasoning_content: tokensBuffer,
            },
          }),
          text: '',
        }),
        runManager
      );
      return;
    }

    yield* this._yieldDeepSeekStreamChunk(
      new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: tokensBuffer,
        }),
        text: tokensBuffer,
      }),
      runManager
    );
  }

  protected async *_streamResponseChunksFromReasoningMessages(
    messages: BaseMessage[],
    options: this['ParsedCallOptions']
  ): AsyncGenerator<ChatGenerationChunk> {
    const params = {
      ...this.invocationParams(options, { streaming: true }),
      stream: true as const,
    };
    const messagesMapped = this._convertDeepSeekMessages(messages);
    const streamIterable = await this.completionWithRetry(
      {
        ...params,
        messages: messagesMapped,
      },
      {
        signal: options.signal,
        ...options.options,
      }
    );

    let defaultRole:
      | OpenAIClient.Chat.Completions.ChatCompletionRole
      | undefined;
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;

    for await (const data of streamIterable) {
      if (options.signal?.aborted === true) {
        throw new Error('AbortError');
      }

      if (data.usage != null) {
        usage = data.usage;
      }

      if (data.choices.length === 0) {
        continue;
      }

      const choice = data.choices[0];
      const { delta } = choice;
      const messageChunk = this._convertCompletionsDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      defaultRole = delta.role ?? defaultRole;

      if (typeof messageChunk.content !== 'string') {
        continue;
      }

      const messageText = messageChunk.content;
      const newTokenIndices = {
        prompt: options.promptIndex ?? 0,
        completion: choice.index,
      };
      const generationInfo = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        Object.assign(generationInfo, {
          finish_reason: choice.finish_reason,
          system_fingerprint: data.system_fingerprint,
          model_name: data.model,
          service_tier: data.service_tier,
        });
      }
      if (this.logprobs === true) {
        Object.assign(generationInfo, { logprobs: choice.logprobs });
      }

      const generationChunk = new ChatGenerationChunk({
        message: messageChunk,
        text: messageText,
        generationInfo,
      });

      yield generationChunk;
    }

    if (usage != null) {
      const usageMetadata = createUsageMetadata(usage);

      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          response_metadata: {
            usage: { ...usage },
          },
          usage_metadata: usageMetadata,
        }),
        text: '',
        generationInfo: {
          prompt: 0,
          completion: 0,
        },
      });

      yield generationChunk;
    }

    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }

  protected _createDeepSeekStreamChunk(
    chunk: ChatGenerationChunk,
    content: string,
    additionalKwargs?: AIMessageChunk['additional_kwargs'],
    text = content
  ): ChatGenerationChunk {
    if (!(chunk.message instanceof AIMessageChunk)) {
      return new ChatGenerationChunk({
        message: new AIMessageChunk({
          content,
          additional_kwargs:
            additionalKwargs ?? chunk.message.additional_kwargs,
          response_metadata: chunk.message.response_metadata,
          id: chunk.message.id,
        }),
        text,
        generationInfo: chunk.generationInfo,
      });
    }

    const message = chunk.message;
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        content,
        additional_kwargs: additionalKwargs ?? message.additional_kwargs,
        response_metadata: message.response_metadata,
        tool_calls: message.tool_calls,
        tool_call_chunks: message.tool_call_chunks,
        id: message.id,
      }),
      text,
      generationInfo: chunk.generationInfo,
    });
  }

  protected _createDeepSeekReasoningStreamChunk(
    chunk: ChatGenerationChunk,
    reasoningContent: string
  ): ChatGenerationChunk {
    return this._createDeepSeekStreamChunk(
      chunk,
      '',
      {
        ...chunk.message.additional_kwargs,
        reasoning_content: reasoningContent,
      },
      ''
    );
  }

  protected async *_yieldDeepSeekReasoningText(
    chunk: ChatGenerationChunk,
    reasoningContent: string,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* this._yieldDeepSeekStreamChunk(
      this._createDeepSeekReasoningStreamChunk(chunk, reasoningContent),
      runManager
    );
  }

  protected async *_yieldDeepSeekStreamChunk(
    chunk: ChatGenerationChunk,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield chunk;
    await runManager?.handleLLMNewToken(
      chunk.text,
      this._getDeepSeekTokenIndices(chunk),
      undefined,
      undefined,
      undefined,
      { chunk }
    );
  }

  protected _getDeepSeekTokenIndices(
    chunk: ChatGenerationChunk
  ): { prompt: number; completion: number } | undefined {
    const prompt = chunk.generationInfo?.prompt;
    const completion = chunk.generationInfo?.completion;

    if (typeof prompt === 'number' && typeof completion === 'number') {
      return { prompt, completion };
    }

    return undefined;
  }

  protected _getDeepSeekPartialTagSplitIndex(
    text: string,
    tag: string
  ): number {
    for (let i = tag.length - 1; i >= 1; i--) {
      if (text.endsWith(tag.substring(0, i))) {
        return text.length - i;
      }
    }

    return -1;
  }
}

/** xAI-specific usage metadata type */
export interface XAIUsageMetadata
  extends OpenAIClient.Completions.CompletionUsage {
  prompt_tokens_details?: {
    audio_tokens?: number;
    cached_tokens?: number;
    text_tokens?: number;
    image_tokens?: number;
  };
  completion_tokens_details?: {
    audio_tokens?: number;
    reasoning_tokens?: number;
    accepted_prediction_tokens?: number;
    rejected_prediction_tokens?: number;
  };
  num_sources_used?: number;
}

export class ChatMoonshot extends ChatOpenAI {
  constructor(
    fields?: LibreChatOpenAIFields & t.OpenAIChatInput['modelKwargs']
  ) {
    super({
      ...fields,
      includeReasoningContent: true,
    });
  }

  static lc_name(): 'LibreChatMoonshot' {
    return 'LibreChatMoonshot';
  }
}

export class ChatXAI extends OriginalChatXAI {
  _lc_stream_delay?: number;

  constructor(
    fields?: Partial<ChatXAIInput> & {
      configuration?: { baseURL?: string };
      clientConfig?: { baseURL?: string };
      _lc_stream_delay?: number;
    }
  ) {
    super(fields);
    this._lc_stream_delay = fields?._lc_stream_delay;
    const customBaseURL =
      fields?.configuration?.baseURL ?? fields?.clientConfig?.baseURL;
    if (customBaseURL != null && customBaseURL) {
      this.clientConfig = {
        ...this.clientConfig,
        baseURL: customBaseURL,
      };
      // Reset the client to force recreation with new config
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.client = undefined as any;
    }
  }

  static lc_name(): 'LibreChatXAI' {
    return 'LibreChatXAI';
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }

  _getClientOptions(
    options?: OpenAICoreRequestOptions
  ): OpenAICoreRequestOptions {
    if (!(this.client as OpenAIClient | undefined)) {
      const openAIEndpointConfig: t.OpenAIEndpointConfig = {
        baseURL: this.clientConfig.baseURL,
      };

      const endpoint = getEndpoint(openAIEndpointConfig);
      const params = {
        ...this.clientConfig,
        baseURL: endpoint,
        timeout: this.timeout,
        maxRetries: 0,
      };
      if (params.baseURL == null) {
        delete params.baseURL;
      }

      this.client = new CustomOpenAIClient(params);
    }
    const requestOptions = {
      ...this.clientConfig,
      ...options,
    } as OpenAICoreRequestOptions;
    return requestOptions;
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    return super._generate(
      flattenAnthropicThinkingForOpenAI(messages, this.model),
      options,
      runManager
    );
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(
        flattenAnthropicThinkingForOpenAI(messages, this.model),
        options,
        runManager
      ),
      this._lc_stream_delay
    );
  }
}
