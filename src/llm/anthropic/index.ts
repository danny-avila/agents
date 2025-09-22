import { AIMessageChunk } from '@langchain/core/messages';
import { ChatAnthropicMessages } from '@langchain/anthropic';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseChatModelParams } from '@langchain/core/language_models/chat_models';
import type {
  BaseMessage,
  UsageMetadata,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { AnthropicInput } from '@langchain/anthropic';
import type { Anthropic } from '@anthropic-ai/sdk';
import type {
  AnthropicMessageCreateParams,
  AnthropicStreamingMessageCreateParams,
  AnthropicStreamUsage,
  AnthropicMessageStartEvent,
  AnthropicMessageDeltaEvent,
} from '@/llm/anthropic/types';
import { _makeMessageChunkFromAnthropicEvent } from './utils/message_outputs';
import { _convertMessagesToAnthropicPayload } from './utils/message_inputs';
import { handleToolChoice } from './utils/tools';
import { TextStream } from '@/llm/text';

function _toolsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(params.tools && params.tools.length > 0);
}
function _documentsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  for (const message of params.messages ?? []) {
    if (typeof message.content === 'string') {
      continue;
    }
    for (const block of message.content ?? []) {
      if (
        typeof block === 'object' &&
        block != null &&
        block.type === 'document' &&
        typeof block.citations === 'object' &&
        block.citations.enabled
      ) {
        return true;
      }
    }
  }
  return false;
}

function _thinkingInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(params.thinking && params.thinking.type === 'enabled');
}

function extractToken(
  chunk: AIMessageChunk
): [string, 'string' | 'input' | 'content'] | [undefined] {
  if (typeof chunk.content === 'string') {
    return [chunk.content, 'string'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'input' in chunk.content[0]
  ) {
    return typeof chunk.content[0].input === 'string'
      ? [chunk.content[0].input, 'input']
      : [JSON.stringify(chunk.content[0].input), 'input'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'text' in chunk.content[0]
  ) {
    return [chunk.content[0].text, 'content'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'thinking' in chunk.content[0]
  ) {
    return [chunk.content[0].thinking, 'content'];
  }
  return [undefined];
}

function cloneChunk(
  text: string,
  tokenType: string,
  chunk: AIMessageChunk
): AIMessageChunk {
  if (tokenType === 'string') {
    return new AIMessageChunk(Object.assign({}, chunk, { content: text }));
  } else if (tokenType === 'input') {
    return chunk;
  }
  const content = chunk.content[0] as MessageContentComplex;
  if (tokenType === 'content' && content.type === 'text') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (tokenType === 'content' && content.type === 'text_delta') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (tokenType === 'content' && content.type?.startsWith('thinking')) {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { thinking: text })],
      })
    );
  }

  return chunk;
}

export type CustomAnthropicInput = AnthropicInput & {
  _lc_stream_delay?: number;
} & BaseChatModelParams;

/**
 * A type representing additional parameters that can be passed to the
 * Anthropic API.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Kwargs = Record<string, any>;

export class CustomAnthropic extends ChatAnthropicMessages {
  _lc_stream_delay: number;
  private message_start: AnthropicMessageStartEvent | undefined;
  private message_delta: AnthropicMessageDeltaEvent | undefined;
  private tools_in_params?: boolean;
  private emitted_usage?: boolean;
  constructor(fields?: CustomAnthropicInput) {
    super(fields);
    this.resetTokenEvents();
    this._lc_stream_delay = fields?._lc_stream_delay ?? 25;
  }

  static lc_name(): 'LibreChatAnthropic' {
    return 'LibreChatAnthropic';
  }

  /**
   * Get the parameters used to invoke the model
   */
  override invocationParams(
    options?: this['ParsedCallOptions']
  ): Omit<
    AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams,
    'messages'
  > &
    Kwargs {
    const tool_choice:
      | Anthropic.Messages.ToolChoiceAuto
      | Anthropic.Messages.ToolChoiceAny
      | Anthropic.Messages.ToolChoiceTool
      | undefined = handleToolChoice(options?.tool_choice);

    if (this.thinking.type === 'enabled') {
      if (this.topK !== -1 && (this.topK as number | undefined) != null) {
        throw new Error('topK is not supported when thinking is enabled');
      }
      if (this.topP !== -1 && (this.topP as number | undefined) != null) {
        throw new Error('topP is not supported when thinking is enabled');
      }
      if (
        this.temperature !== 1 &&
        (this.temperature as number | undefined) != null
      ) {
        throw new Error(
          'temperature is not supported when thinking is enabled'
        );
      }

      return {
        model: this.model,
        stop_sequences: options?.stop ?? this.stopSequences,
        stream: this.streaming,
        max_tokens: this.maxTokens,
        tools: this.formatStructuredToolToAnthropic(options?.tools),
        tool_choice,
        thinking: this.thinking,
        ...this.invocationKwargs,
      };
    }
    return {
      model: this.model,
      temperature: this.temperature,
      top_k: this.topK,
      top_p: this.topP,
      stop_sequences: options?.stop ?? this.stopSequences,
      stream: this.streaming,
      max_tokens: this.maxTokens,
      tools: this.formatStructuredToolToAnthropic(options?.tools),
      tool_choice,
      thinking: this.thinking,
      ...this.invocationKwargs,
    };
  }

  /**
   * Get stream usage as returned by this client's API response.
   * @returns The stream usage object.
   */
  getStreamUsage(): UsageMetadata | undefined {
    if (this.emitted_usage === true) {
      return;
    }
    const inputUsage = this.message_start?.message.usage as
      | undefined
      | AnthropicStreamUsage;
    const outputUsage = this.message_delta?.usage as
      | undefined
      | Partial<AnthropicStreamUsage>;
    if (!outputUsage) {
      return;
    }
    const totalUsage: UsageMetadata = {
      input_tokens: inputUsage?.input_tokens ?? 0,
      output_tokens: outputUsage.output_tokens ?? 0,
      total_tokens:
        (inputUsage?.input_tokens ?? 0) + (outputUsage.output_tokens ?? 0),
    };

    if (
      inputUsage?.cache_creation_input_tokens != null ||
      inputUsage?.cache_read_input_tokens != null
    ) {
      totalUsage.input_token_details = {
        cache_creation: inputUsage.cache_creation_input_tokens ?? 0,
        cache_read: inputUsage.cache_read_input_tokens ?? 0,
      };
    }

    this.emitted_usage = true;
    return totalUsage;
  }

  resetTokenEvents(): void {
    this.message_start = undefined;
    this.message_delta = undefined;
    this.emitted_usage = undefined;
    this.tools_in_params = undefined;
  }

  private createGenerationChunk({
    token,
    chunk,
    usageMetadata,
    shouldStreamUsage,
  }: {
    token?: string;
    chunk: AIMessageChunk;
    shouldStreamUsage: boolean;
    usageMetadata?: UsageMetadata;
  }): ChatGenerationChunk {
    const usage_metadata = shouldStreamUsage
      ? (usageMetadata ?? chunk.usage_metadata)
      : undefined;
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        // Just yield chunk as it is and tool_use will be concat by BaseChatModel._generateUncached().
        content: chunk.content,
        additional_kwargs: chunk.additional_kwargs,
        tool_call_chunks: chunk.tool_call_chunks,
        response_metadata: chunk.response_metadata,
        usage_metadata,
        id: chunk.id,
      }),
      text: token ?? '',
    });
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.resetTokenEvents();
    const params = this.invocationParams(options);
    const formattedMessages = _convertMessagesToAnthropicPayload(messages);
    const payload = {
      ...params,
      ...formattedMessages,
      stream: true,
    } as const;
    const coerceContentToString =
      !_toolsInParams(payload) &&
      !_documentsInParams(payload) &&
      !_thinkingInParams(payload);

    const stream = await this.createStreamWithRetry(payload, {
      headers: options.headers,
    });

    const shouldStreamUsage = this.streamUsage ?? options.streamUsage;

    for await (const data of stream) {
      if (options.signal?.aborted === true) {
        stream.controller.abort();
        throw new Error('AbortError: User aborted the request.');
      }

      if (data.type === 'message_start') {
        this.message_start = data as AnthropicMessageStartEvent;
      } else if (data.type === 'message_delta') {
        this.message_delta = data as AnthropicMessageDeltaEvent;
      }

      let usageMetadata: UsageMetadata | undefined;
      if (this.tools_in_params !== true && this.emitted_usage !== true) {
        usageMetadata = this.getStreamUsage();
      }

      const result = _makeMessageChunkFromAnthropicEvent(data, {
        streamUsage: shouldStreamUsage,
        coerceContentToString,
      });
      if (!result) continue;

      const { chunk } = result;
      const [token = '', tokenType] = extractToken(chunk);

      if (
        !tokenType ||
        tokenType === 'input' ||
        (token === '' && (usageMetadata != null || chunk.id != null))
      ) {
        const generationChunk = this.createGenerationChunk({
          token,
          chunk,
          usageMetadata,
          shouldStreamUsage,
        });
        yield generationChunk;
        await runManager?.handleLLMNewToken(
          token,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: generationChunk }
        );
        continue;
      }

      const textStream = new TextStream(token, {
        delay: this._lc_stream_delay,
        firstWordChunk: true,
        minChunkSize: 4,
        maxChunkSize: 8,
      });

      const generator = textStream.generateText(options.signal);
      try {
        let emittedUsage = false;
        for await (const currentToken of generator) {
          if ((options.signal as AbortSignal | undefined)?.aborted === true) {
            break;
          }
          const newChunk = cloneChunk(currentToken, tokenType, chunk);

          const generationChunk = this.createGenerationChunk({
            token: currentToken,
            chunk: newChunk,
            usageMetadata: emittedUsage ? undefined : usageMetadata,
            shouldStreamUsage,
          });

          if (usageMetadata && !emittedUsage) {
            emittedUsage = true;
          }
          yield generationChunk;

          await runManager?.handleLLMNewToken(
            currentToken,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: generationChunk }
          );
        }
      } finally {
        await generator.return();
      }
    }

    this.resetTokenEvents();
  }
}
