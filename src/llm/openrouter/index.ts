import { ChatOpenAI } from '@/llm/openai';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { AIMessageChunk as AIMessageChunkClass } from '@langchain/core/messages';
import type {
  FunctionMessageChunk,
  SystemMessageChunk,
  HumanMessageChunk,
  ToolMessageChunk,
  ChatMessageChunk,
  AIMessageChunk,
  BaseMessage,
} from '@langchain/core/messages';
import type {
  ChatOpenAICallOptions,
  OpenAIChatInput,
  OpenAIClient,
} from '@langchain/openai';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';

type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;

type OpenAIRoleEnum =
  | 'system'
  | 'developer'
  | 'assistant'
  | 'user'
  | 'function'
  | 'tool';

export interface ChatOpenRouterCallOptions extends ChatOpenAICallOptions {
  include_reasoning?: boolean;
  modelKwargs?: OpenAIChatInput['modelKwargs'];
}
export class ChatOpenRouter extends ChatOpenAI {
  constructor(_fields: Partial<ChatOpenRouterCallOptions>) {
    const { include_reasoning, modelKwargs = {}, ...fields } = _fields;
    super({
      ...fields,
      modelKwargs: {
        ...modelKwargs,
        include_reasoning,
      },
    });
  }
  static lc_name(): 'LibreChatOpenRouter' {
    return 'LibreChatOpenRouter';
  }
  protected override _convertOpenAIDeltaToBaseMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    rawResponse: OpenAIClient.ChatCompletionChunk,
    defaultRole?:
      | 'function'
      | 'user'
      | 'system'
      | 'developer'
      | 'assistant'
      | 'tool'
  ):
    | AIMessageChunk
    | HumanMessageChunk
    | SystemMessageChunk
    | FunctionMessageChunk
    | ToolMessageChunk
    | ChatMessageChunk {
    const messageChunk = super._convertOpenAIDeltaToBaseMessageChunk(
      delta,
      rawResponse,
      defaultRole
    );
    if (delta.reasoning != null) {
      messageChunk.additional_kwargs.reasoning = delta.reasoning;
    }
    if (delta.reasoning_details != null) {
      messageChunk.additional_kwargs.reasoning_details =
        delta.reasoning_details;
    }
    return messageChunk;
  }

  async *_streamResponseChunks2(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const messagesMapped: OpenAICompletionParam[] =
      _convertMessagesToOpenAIParams(messages, this.model, {
        includeReasoningDetails: true,
        convertReasoningDetailsToContent: true,
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

    // Store reasoning_details keyed by unique identifier to prevent incorrect merging
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const reasoningTextByIndex: Map<number, Record<string, any>> = new Map();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const reasoningEncryptedById: Map<string, Record<string, any>> = new Map();

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

      // Accumulate reasoning_details from each delta
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const deltaAny = delta as Record<string, any>;
      if (
        deltaAny.reasoning_details != null &&
        Array.isArray(deltaAny.reasoning_details)
      ) {
        for (const detail of deltaAny.reasoning_details) {
          // For encrypted reasoning (thought signatures), store by ID - MUST be separate
          if (detail.type === 'reasoning.encrypted' && detail.id) {
            reasoningEncryptedById.set(detail.id, {
              type: detail.type,
              id: detail.id,
              data: detail.data,
              format: detail.format,
              index: detail.index,
            });
          } else if (detail.type === 'reasoning.text') {
            // For text reasoning, accumulate text by index
            const idx = detail.index ?? 0;
            const existing = reasoningTextByIndex.get(idx);
            if (existing) {
              // Only append text, keep other fields from first entry
              existing.text = (existing.text || '') + (detail.text || '');
            } else {
              reasoningTextByIndex.set(idx, {
                type: detail.type,
                text: detail.text || '',
                format: detail.format,
                index: idx,
              });
            }
          }
        }
      }

      const chunk = this._convertOpenAIDeltaToBaseMessageChunk(
        delta,
        data,
        defaultRole
      );

      // IMPORTANT: Only set reasoning_details on the FINAL chunk to prevent
      // LangChain's chunk concatenation from corrupting the array
      // Check if this is the final chunk (has finish_reason)
      if (choice.finish_reason != null) {
        // Build properly structured reasoning_details array
        // Text entries first (but we only need the encrypted ones for thought signatures)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const finalReasoningDetails: Record<string, any>[] = [
          ...reasoningTextByIndex.values(),
          ...reasoningEncryptedById.values(),
        ];

        if (finalReasoningDetails.length > 0) {
          chunk.additional_kwargs.reasoning_details = finalReasoningDetails;
        }
      } else {
        // Clear reasoning_details from intermediate chunks to prevent concatenation issues
        delete chunk.additional_kwargs.reasoning_details;
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
        await new Promise((resolve) =>
          setTimeout(resolve, this._lc_stream_delay)
        );
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
        message: new AIMessageChunkClass({
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
        await new Promise((resolve) =>
          setTimeout(resolve, this._lc_stream_delay)
        );
      }
    }
    if (options.signal?.aborted === true) {
      throw new Error('AbortError');
    }
  }
}
