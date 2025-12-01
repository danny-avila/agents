import { AzureOpenAI as AzureOpenAIClient } from 'openai';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatXAI as OriginalChatXAI } from '@langchain/xai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ToolDefinition } from '@langchain/core/language_models/base';
import { isLangChainTool } from '@langchain/core/utils/function_calling';
import { ChatDeepSeek as OriginalChatDeepSeek } from '@langchain/deepseek';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import {
  getEndpoint,
  OpenAIClient,
  formatToOpenAITool,
  ChatOpenAI as OriginalChatOpenAI,
  AzureChatOpenAI as OriginalAzureChatOpenAI,
} from '@langchain/openai';
import type {
  OpenAIChatCallOptions,
  OpenAIRoleEnum,
  HeaderValue,
  HeadersLike,
} from './types';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { ChatXAIInput } from '@langchain/xai';
import type * as t from '@langchain/openai';
import {
  isReasoningModel,
  _convertMessagesToOpenAIParams,
  _convertMessagesToOpenAIResponsesParams,
  _convertOpenAIResponsesDeltaToBaseMessageChunk,
  type ResponseReturnStreamEvents,
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

type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;

type OpenAICoreRequestOptions = OpenAIClient.RequestOptions;

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
    toolDef = formatToOpenAITool(tool);
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

/** @ts-expect-error We are intentionally overriding `getReasoningParams` */
export class ChatOpenAI extends OriginalChatOpenAI<t.ChatOpenAICallOptions> {
  _lc_stream_delay?: number;

  constructor(
    fields?: t.ChatOpenAICallOptions & {
      _lc_stream_delay?: number;
    } & t.OpenAIChatInput['modelKwargs']
  ) {
    super(fields);
    this._lc_stream_delay = fields?._lc_stream_delay;
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }
  static lc_name(): string {
    return 'LibreChatOpenAI';
  }
  protected _getClientOptions(
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
    // apply options in reverse order of importance -- newer options supersede older options
    let reasoning: OpenAIClient.Reasoning | undefined;
    if (this.reasoning !== undefined) {
      reasoning = {
        ...reasoning,
        ...this.reasoning,
      };
    }
    if (options?.reasoning !== undefined) {
      reasoning = {
        ...reasoning,
        ...options.reasoning,
      };
    }

    return reasoning;
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (!this._useResponseApi(options)) {
      return yield* this._streamResponseChunks2(messages, options, runManager);
    }
    const streamIterable = await this.responseApiWithRetry(
      {
        ...this.invocationParams<'responses'>(options, { streaming: true }),
        input: _convertMessagesToOpenAIResponsesParams(
          messages,
          this.model,
          this.zdrEnabled
        ),
        stream: true,
      },
      options
    );

    for await (const data of streamIterable) {
      const chunk = _convertOpenAIResponsesDeltaToBaseMessageChunk(
        data as ResponseReturnStreamEvents
      );
      if (chunk == null) continue;
      yield chunk;
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
      await runManager?.handleLLMNewToken(
        chunk.text || '',
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
    }

    return;
  }

  async *_streamResponseChunks2(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model);

    const params = {
      ...this.invocationParams(options, {
        streaming: true,
      }),
      messages: messagesMapped,
      stream: true as const,
    };
    let defaultRole: OpenAIRoleEnum | undefined;

    const streamIterable = await this.completionWithRetry(params, options);
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;
    for await (const data of streamIterable) {
      const choice = data.choices[0] as
        | Partial<OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice>
        | undefined;
      if (data.usage) {
        usage = data.usage;
      }
      if (!choice) {
        continue;
      }

      const { delta } = choice;
      if (!delta) {
        continue;
      }
      const chunk = this._convertOpenAIDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      if ('reasoning_content' in delta) {
        chunk.additional_kwargs.reasoning_content = delta.reasoning_content;
      } else if ('reasoning' in delta) {
        chunk.additional_kwargs.reasoning_content = delta.reasoning;
      }
      if ('provider_specific_fields' in delta) {
        chunk.additional_kwargs.provider_specific_fields =
          delta.provider_specific_fields;
      }
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: options.promptIndex ?? 0,
        completion: choice.index ?? 0,
      };
      if (typeof chunk.content !== 'string') {
        // eslint-disable-next-line no-console
        console.log(
          '[WARNING]: Received non-string content from OpenAI. This is currently not supported.'
        );
        continue;
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generationInfo: Record<string, any> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        // Only include system fingerprint in the last chunk for now
        // to avoid concatenation issues
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
        generationInfo.service_tier = data.service_tier;
      }
      if (this.logprobs == true) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
      await runManager?.handleLLMNewToken(
        generationChunk.text || '',
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
          response_metadata: {
            usage: { ...usage },
          },
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
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }
}

/** @ts-expect-error We are intentionally overriding `getReasoningParams` */
export class AzureChatOpenAI extends OriginalAzureChatOpenAI {
  _lc_stream_delay?: number;

  constructor(fields?: t.AzureOpenAIInput & { _lc_stream_delay?: number }) {
    super(fields);
    this._lc_stream_delay = fields?._lc_stream_delay;
  }

  public get exposedClient(): CustomOpenAIClient {
    return this.client;
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
    if (!isReasoningModel(this.model)) {
      return;
    }

    // apply options in reverse order of importance -- newer options supersede older options
    let reasoning: OpenAIClient.Reasoning | undefined;
    if (this.reasoning !== undefined) {
      reasoning = {
        ...reasoning,
        ...this.reasoning,
      };
    }
    if (options?.reasoning !== undefined) {
      reasoning = {
        ...reasoning,
        ...options.reasoning,
      };
    }

    return reasoning;
  }

  protected _getReasoningParams(
    options?: this['ParsedCallOptions']
  ): OpenAIClient.Reasoning | undefined {
    return this.getReasoningParams(options);
  }

  protected _getClientOptions(
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
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (!this._useResponseApi(options)) {
      return yield* super._streamResponseChunks(messages, options, runManager);
    }
    const streamIterable = await this.responseApiWithRetry(
      {
        ...this.invocationParams<'responses'>(options, { streaming: true }),
        input: _convertMessagesToOpenAIResponsesParams(
          messages,
          this.model,
          this.zdrEnabled
        ),
        stream: true,
      },
      options
    );

    for await (const data of streamIterable) {
      const chunk = _convertOpenAIResponsesDeltaToBaseMessageChunk(
        data as ResponseReturnStreamEvents
      );
      if (chunk == null) continue;
      yield chunk;
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
      await runManager?.handleLLMNewToken(
        chunk.text || '',
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
    }

    return;
  }
}
export class ChatDeepSeek extends OriginalChatDeepSeek {
  public get exposedClient(): CustomOpenAIClient {
    return this.client;
  }
  static lc_name(): 'LibreChatDeepSeek' {
    return 'LibreChatDeepSeek';
  }
  protected _getClientOptions(
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
    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model, {
        includeReasoningContent: true,
      });

    const params = {
      ...this.invocationParams(options, {
        streaming: true,
      }),
      messages: messagesMapped,
      stream: true as const,
    };
    let defaultRole: OpenAIRoleEnum | undefined;

    const streamIterable = await this.completionWithRetry(params, options);
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;
    for await (const data of streamIterable) {
      const choice = data.choices[0] as
        | Partial<OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice>
        | undefined;
      if (data.usage) {
        usage = data.usage;
      }
      if (!choice) {
        continue;
      }

      const { delta } = choice;
      if (!delta) {
        continue;
      }
      const chunk = this._convertOpenAIDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      if ('reasoning_content' in delta) {
        chunk.additional_kwargs.reasoning_content = delta.reasoning_content;
      }
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: (options as OpenAIChatCallOptions).promptIndex ?? 0,
        completion: choice.index ?? 0,
      };
      if (typeof chunk.content !== 'string') {
        // eslint-disable-next-line no-console
        console.log(
          '[WARNING]: Received non-string content from OpenAI. This is currently not supported.'
        );
        continue;
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generationInfo: Record<string, any> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
        generationInfo.service_tier = data.service_tier;
      }
      if (this.logprobs == true) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      await runManager?.handleLLMNewToken(
        generationChunk.text || '',
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
          response_metadata: {
            usage: { ...usage },
          },
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
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
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

  protected _getClientOptions(
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
    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model);

    const params = {
      ...this.invocationParams(options, {
        streaming: true,
      }),
      messages: messagesMapped,
      stream: true as const,
    };
    let defaultRole: OpenAIRoleEnum | undefined;

    const streamIterable = await this.completionWithRetry(params, options);
    let usage: OpenAIClient.Completions.CompletionUsage | undefined;
    for await (const data of streamIterable) {
      const choice = data.choices[0] as
        | Partial<OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice>
        | undefined;
      if (data.usage) {
        usage = data.usage;
      }
      if (!choice) {
        continue;
      }

      const { delta } = choice;
      if (!delta) {
        continue;
      }
      const chunk = this._convertOpenAIDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );
      if (chunk.usage_metadata != null) {
        chunk.usage_metadata = {
          input_tokens:
            (chunk.usage_metadata as Partial<UsageMetadata>).input_tokens ?? 0,
          output_tokens:
            (chunk.usage_metadata as Partial<UsageMetadata>).output_tokens ?? 0,
          total_tokens:
            (chunk.usage_metadata as Partial<UsageMetadata>).total_tokens ?? 0,
        };
      }
      if ('reasoning_content' in delta) {
        chunk.additional_kwargs.reasoning_content = delta.reasoning_content;
      }
      defaultRole = delta.role ?? defaultRole;
      const newTokenIndices = {
        prompt: (options as OpenAIChatCallOptions).promptIndex ?? 0,
        completion: choice.index ?? 0,
      };
      if (typeof chunk.content !== 'string') {
        // eslint-disable-next-line no-console
        console.log(
          '[WARNING]: Received non-string content from OpenAI. This is currently not supported.'
        );
        continue;
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generationInfo: Record<string, any> = { ...newTokenIndices };
      if (choice.finish_reason != null) {
        generationInfo.finish_reason = choice.finish_reason;
        // Only include system fingerprint in the last chunk for now
        // to avoid concatenation issues
        generationInfo.system_fingerprint = data.system_fingerprint;
        generationInfo.model_name = data.model;
        generationInfo.service_tier = data.service_tier;
      }
      if (this.logprobs == true) {
        generationInfo.logprobs = choice.logprobs;
      }
      const generationChunk = new ChatGenerationChunk({
        message: chunk,
        text: chunk.content,
        generationInfo,
      });
      yield generationChunk;
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
      await runManager?.handleLLMNewToken(
        generationChunk.text || '',
        newTokenIndices,
        undefined,
        undefined,
        undefined,
        { chunk: generationChunk }
      );
    }
    if (usage) {
      // Type assertion for xAI-specific usage structure
      const xaiUsage = usage as XAIUsageMetadata;
      const inputTokenDetails = {
        // Standard OpenAI fields
        ...(usage.prompt_tokens_details?.audio_tokens != null && {
          audio: usage.prompt_tokens_details.audio_tokens,
        }),
        ...(usage.prompt_tokens_details?.cached_tokens != null && {
          cache_read: usage.prompt_tokens_details.cached_tokens,
        }),
        // Add xAI-specific prompt token details if they exist
        ...(xaiUsage.prompt_tokens_details?.text_tokens != null && {
          text: xaiUsage.prompt_tokens_details.text_tokens,
        }),
        ...(xaiUsage.prompt_tokens_details?.image_tokens != null && {
          image: xaiUsage.prompt_tokens_details.image_tokens,
        }),
      };
      const outputTokenDetails = {
        // Standard OpenAI fields
        ...(usage.completion_tokens_details?.audio_tokens != null && {
          audio: usage.completion_tokens_details.audio_tokens,
        }),
        ...(usage.completion_tokens_details?.reasoning_tokens != null && {
          reasoning: usage.completion_tokens_details.reasoning_tokens,
        }),
        // Add xAI-specific completion token details if they exist
        ...(xaiUsage.completion_tokens_details?.accepted_prediction_tokens !=
          null && {
          accepted_prediction:
            xaiUsage.completion_tokens_details.accepted_prediction_tokens,
        }),
        ...(xaiUsage.completion_tokens_details?.rejected_prediction_tokens !=
          null && {
          rejected_prediction:
            xaiUsage.completion_tokens_details.rejected_prediction_tokens,
        }),
      };
      const generationChunk = new ChatGenerationChunk({
        message: new AIMessageChunk({
          content: '',
          response_metadata: {
            usage: { ...usage },
            // Include xAI-specific metadata if it exists
            ...(xaiUsage.num_sources_used != null && {
              num_sources_used: xaiUsage.num_sources_used,
            }),
          },
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
      if (this._lc_stream_delay != null) {
        await sleep(this._lc_stream_delay);
      }
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }
}
