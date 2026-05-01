import { AzureOpenAI as AzureOpenAIClient } from 'openai';
import { ChatXAI as OriginalChatXAI } from '@langchain/xai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
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
  ChatOpenAI as OriginalChatOpenAI,
  AzureChatOpenAI as OriginalAzureChatOpenAI,
} from '@langchain/openai';
import type { HeaderValue, HeadersLike } from './types';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { ChatXAIInput } from '@langchain/xai';
import type * as t from '@langchain/openai';
import { isReasoningModel } from './utils';
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

  if (fields?.strict !== undefined && toolDef.type === 'function') {
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
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, runManager),
      this._lc_stream_delay
    );
  }
}

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
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, runManager),
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
      super._streamResponseChunks(messages, options, runManager),
      this._lc_stream_delay
    );
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

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    yield* delayStreamChunks(
      super._streamResponseChunks(messages, options, runManager),
      this._lc_stream_delay
    );
  }
}
