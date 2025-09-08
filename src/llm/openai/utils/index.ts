/* eslint-disable @typescript-eslint/ban-ts-comment */
/* eslint-disable @typescript-eslint/explicit-function-return-type */
import { type OpenAI as OpenAIClient } from 'openai';
import type {
  ChatCompletionContentPartText,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartInputAudio,
  ChatCompletionContentPart,
} from 'openai/resources/chat/completions';
import {
  AIMessage,
  AIMessageChunk,
  type BaseMessage,
  ChatMessage,
  ToolMessage,
  isAIMessage,
  type UsageMetadata,
  type BaseMessageFields,
  type MessageContent,
  type InvalidToolCall,
  type MessageContentImageUrl,
  StandardContentBlockConverter,
  parseBase64DataUrl,
  parseMimeType,
  convertToProviderContentBlock,
  isDataContentBlock,
} from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  convertLangChainToolCallToOpenAI,
  makeInvalidToolCall,
  parseToolCall,
} from '@langchain/core/output_parsers/openai_tools';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type {
  OpenAICallOptions,
  OpenAIChatInput,
  ChatOpenAIReasoningSummary,
} from '@langchain/openai';

export type { OpenAICallOptions, OpenAIChatInput };

// Utility types to get hidden OpenAI response types
type ExtractAsyncIterableType<T> = T extends AsyncIterable<infer U> ? U : never;
type ExcludeController<T> = T extends { controller: unknown } ? never : T;
type ExcludeNonController<T> = T extends { controller: unknown } ? T : never;

type ResponsesCreate = OpenAIClient.Responses['create'];
type ResponsesParse = OpenAIClient.Responses['parse'];

type ResponsesInputItem = OpenAIClient.Responses.ResponseInputItem;

type ResponsesCreateInvoke = ExcludeController<
  Awaited<ReturnType<ResponsesCreate>>
>;

type ResponsesParseInvoke = ExcludeController<
  Awaited<ReturnType<ResponsesParse>>
>;

type ResponsesCreateStream = ExcludeNonController<
  Awaited<ReturnType<ResponsesCreate>>
>;

export type ResponseReturnStreamEvents =
  ExtractAsyncIterableType<ResponsesCreateStream>;

// TODO import from SDK when available
type OpenAIRoleEnum =
  | 'system'
  | 'developer'
  | 'assistant'
  | 'user'
  | 'function'
  | 'tool';

type OpenAICompletionParam =
  OpenAIClient.Chat.Completions.ChatCompletionMessageParam;

function extractGenericMessageCustomRole(message: ChatMessage) {
  if (
    message.role !== 'system' &&
    message.role !== 'developer' &&
    message.role !== 'assistant' &&
    message.role !== 'user' &&
    message.role !== 'function' &&
    message.role !== 'tool'
  ) {
    console.warn(`Unknown message role: ${message.role}`);
  }

  return message.role as OpenAIRoleEnum;
}

export function messageToOpenAIRole(message: BaseMessage): OpenAIRoleEnum {
  const type = message._getType();
  switch (type) {
  case 'system':
    return 'system';
  case 'ai':
    return 'assistant';
  case 'human':
    return 'user';
  case 'function':
    return 'function';
  case 'tool':
    return 'tool';
  case 'generic': {
    if (!ChatMessage.isInstance(message))
      throw new Error('Invalid generic chat message');
    return extractGenericMessageCustomRole(message);
  }
  default:
    throw new Error(`Unknown message type: ${type}`);
  }
}

const completionsApiContentBlockConverter: StandardContentBlockConverter<{
  text: ChatCompletionContentPartText;
  image: ChatCompletionContentPartImage;
  audio: ChatCompletionContentPartInputAudio;
  file: ChatCompletionContentPart.File;
}> = {
  providerName: 'ChatOpenAI',

  fromStandardTextBlock(block): ChatCompletionContentPartText {
    return { type: 'text', text: block.text };
  },

  fromStandardImageBlock(block): ChatCompletionContentPartImage {
    if (block.source_type === 'url') {
      return {
        type: 'image_url',
        image_url: {
          url: block.url,
          ...(block.metadata?.detail
            ? { detail: block.metadata.detail as 'auto' | 'low' | 'high' }
            : {}),
        },
      };
    }

    if (block.source_type === 'base64') {
      const url = `data:${block.mime_type ?? ''};base64,${block.data}`;
      return {
        type: 'image_url',
        image_url: {
          url,
          ...(block.metadata?.detail
            ? { detail: block.metadata.detail as 'auto' | 'low' | 'high' }
            : {}),
        },
      };
    }

    throw new Error(
      `Image content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },

  fromStandardAudioBlock(block): ChatCompletionContentPartInputAudio {
    if (block.source_type === 'url') {
      const data = parseBase64DataUrl({ dataUrl: block.url });
      if (!data) {
        throw new Error(
          `URL audio blocks with source_type ${block.source_type} must be formatted as a data URL for ChatOpenAI`
        );
      }

      const rawMimeType = data.mime_type || block.mime_type || '';
      let mimeType: { type: string; subtype: string };

      try {
        mimeType = parseMimeType(rawMimeType);
      } catch {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      if (
        mimeType.type !== 'audio' ||
        (mimeType.subtype !== 'wav' && mimeType.subtype !== 'mp3')
      ) {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      return {
        type: 'input_audio',
        input_audio: {
          format: mimeType.subtype,
          data: data.data,
        },
      };
    }

    if (block.source_type === 'base64') {
      let mimeType: { type: string; subtype: string };

      try {
        mimeType = parseMimeType(block.mime_type ?? '');
      } catch {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      if (
        mimeType.type !== 'audio' ||
        (mimeType.subtype !== 'wav' && mimeType.subtype !== 'mp3')
      ) {
        throw new Error(
          `Audio blocks with source_type ${block.source_type} must have mime type of audio/wav or audio/mp3`
        );
      }

      return {
        type: 'input_audio',
        input_audio: {
          format: mimeType.subtype,
          data: block.data,
        },
      };
    }

    throw new Error(
      `Audio content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },

  fromStandardFileBlock(block): ChatCompletionContentPart.File {
    if (block.source_type === 'url') {
      const data = parseBase64DataUrl({ dataUrl: block.url });
      if (!data) {
        throw new Error(
          `URL file blocks with source_type ${block.source_type} must be formatted as a data URL for ChatOpenAI`
        );
      }

      return {
        type: 'file',
        file: {
          file_data: block.url, // formatted as base64 data URL
          ...(block.metadata?.filename || block.metadata?.name
            ? {
              filename: (block.metadata.filename ||
                  block.metadata.name) as string,
            }
            : {}),
        },
      };
    }

    if (block.source_type === 'base64') {
      return {
        type: 'file',
        file: {
          file_data: `data:${block.mime_type ?? ''};base64,${block.data}`,
          ...(block.metadata?.filename ||
          block.metadata?.name ||
          block.metadata?.title
            ? {
              filename: (block.metadata.filename ||
                  block.metadata.name ||
                  block.metadata.title) as string,
            }
            : {}),
        },
      };
    }

    if (block.source_type === 'id') {
      return {
        type: 'file',
        file: {
          file_id: block.id,
        },
      };
    }

    throw new Error(
      `File content blocks with source_type ${block.source_type} are not supported for ChatOpenAI`
    );
  },
};

// Used in LangSmith, export is important here
export function _convertMessagesToOpenAIParams(
  messages: BaseMessage[],
  model?: string
): OpenAICompletionParam[] {
  // TODO: Function messages do not support array content, fix cast
  return messages.flatMap((message) => {
    let role = messageToOpenAIRole(message);
    if (role === 'system' && isReasoningModel(model)) {
      role = 'developer';
    }

    let hasAnthropicThinkingBlock: boolean = false;

    const content =
      typeof message.content === 'string'
        ? message.content
        : message.content.map((m) => {
          if ('type' in m && m.type === 'thinking') {
            hasAnthropicThinkingBlock = true;
            return m;
          }
          if (isDataContentBlock(m)) {
            return convertToProviderContentBlock(
              m,
              completionsApiContentBlockConverter
            );
          }
          return m;
        });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const completionParam: Record<string, any> = {
      role,
      content,
    };
    if (message.name != null) {
      completionParam.name = message.name;
    }
    if (message.additional_kwargs.function_call != null) {
      completionParam.function_call = message.additional_kwargs.function_call;
      completionParam.content = '';
    }
    if (isAIMessage(message) && !!message.tool_calls?.length) {
      completionParam.tool_calls = message.tool_calls.map(
        convertLangChainToolCallToOpenAI
      );
      completionParam.content = hasAnthropicThinkingBlock ? content : '';
    } else {
      if (message.additional_kwargs.tool_calls != null) {
        completionParam.tool_calls = message.additional_kwargs.tool_calls;
      }
      if ((message as ToolMessage).tool_call_id != null) {
        completionParam.tool_call_id = (message as ToolMessage).tool_call_id;
      }
    }

    if (
      message.additional_kwargs.audio &&
      typeof message.additional_kwargs.audio === 'object' &&
      'id' in message.additional_kwargs.audio
    ) {
      const audioMessage = {
        role: 'assistant',
        audio: {
          id: message.additional_kwargs.audio.id,
        },
      };
      return [completionParam, audioMessage] as OpenAICompletionParam[];
    }

    return completionParam as OpenAICompletionParam;
  });
}

const _FUNCTION_CALL_IDS_MAP_KEY = '__openai_function_call_ids__';

function _convertReasoningSummaryToOpenAIResponsesParams(
  reasoning: ChatOpenAIReasoningSummary
): OpenAIClient.Responses.ResponseReasoningItem {
  // combine summary parts that have the the same index and then remove the indexes
  const summary = (
    reasoning.summary.length > 1
      ? reasoning.summary.reduce(
        (acc, curr) => {
          const last = acc.at(-1);

          if (last!.index === curr.index) {
              last!.text += curr.text;
          } else {
            acc.push(curr);
          }
          return acc;
        },
        [{ ...reasoning.summary[0] }]
      )
      : reasoning.summary
  ).map((s) =>
    Object.fromEntries(Object.entries(s).filter(([k]) => k !== 'index'))
  ) as OpenAIClient.Responses.ResponseReasoningItem.Summary[];

  return {
    ...reasoning,
    summary,
  } as OpenAIClient.Responses.ResponseReasoningItem;
}

export function _convertMessagesToOpenAIResponsesParams(
  messages: BaseMessage[],
  model?: string,
  zdrEnabled?: boolean
): ResponsesInputItem[] {
  return messages.flatMap(
    (lcMsg): ResponsesInputItem | ResponsesInputItem[] => {
      const additional_kwargs =
        lcMsg.additional_kwargs as BaseMessageFields['additional_kwargs'] & {
          [_FUNCTION_CALL_IDS_MAP_KEY]?: Record<string, string>;
          reasoning?: OpenAIClient.Responses.ResponseReasoningItem;
          type?: string;
          refusal?: string;
        };

      let role = messageToOpenAIRole(lcMsg);
      if (role === 'system' && isReasoningModel(model)) role = 'developer';

      if (role === 'function') {
        throw new Error('Function messages are not supported in Responses API');
      }

      if (role === 'tool') {
        const toolMessage = lcMsg as ToolMessage;

        // Handle computer call output
        if (additional_kwargs.type === 'computer_call_output') {
          const output = (() => {
            if (typeof toolMessage.content === 'string') {
              return {
                type: 'computer_screenshot' as const,
                image_url: toolMessage.content,
              };
            }

            if (Array.isArray(toolMessage.content)) {
              const oaiScreenshot = toolMessage.content.find(
                (i) => i.type === 'computer_screenshot'
              ) as { type: 'computer_screenshot'; image_url: string };

              if (oaiScreenshot) return oaiScreenshot;

              const lcImage = toolMessage.content.find(
                (i) => i.type === 'image_url'
              ) as MessageContentImageUrl;

              if (lcImage) {
                return {
                  type: 'computer_screenshot' as const,
                  image_url:
                    typeof lcImage.image_url === 'string'
                      ? lcImage.image_url
                      : lcImage.image_url.url,
                };
              }
            }

            throw new Error('Invalid computer call output');
          })();

          return {
            type: 'computer_call_output',
            output,
            call_id: toolMessage.tool_call_id,
          };
        }

        return {
          type: 'function_call_output',
          call_id: toolMessage.tool_call_id,
          id: toolMessage.id?.startsWith('fc_') ? toolMessage.id : undefined,
          output:
            typeof toolMessage.content !== 'string'
              ? JSON.stringify(toolMessage.content)
              : toolMessage.content,
        };
      }

      if (role === 'assistant') {
        // if we have the original response items, just reuse them
        if (
          !zdrEnabled &&
          lcMsg.response_metadata.output != null &&
          Array.isArray(lcMsg.response_metadata.output) &&
          lcMsg.response_metadata.output.length > 0 &&
          lcMsg.response_metadata.output.every((item) => 'type' in item)
        ) {
          return lcMsg.response_metadata.output;
        }

        // otherwise, try to reconstruct the response from what we have

        const input: ResponsesInputItem[] = [];

        // reasoning items
        if (additional_kwargs.reasoning && !zdrEnabled) {
          const reasoningItem = _convertReasoningSummaryToOpenAIResponsesParams(
            additional_kwargs.reasoning
          );
          input.push(reasoningItem);
        }

        // ai content
        let { content } = lcMsg;
        if (additional_kwargs.refusal) {
          if (typeof content === 'string') {
            content = [{ type: 'output_text', text: content, annotations: [] }];
          }
          content = [
            ...content,
            { type: 'refusal', refusal: additional_kwargs.refusal },
          ];
        }

        input.push({
          type: 'message',
          role: 'assistant',
          ...(lcMsg.id && !zdrEnabled ? { id: lcMsg.id } : {}),
          content:
            typeof content === 'string'
              ? content
              : content.flatMap((item) => {
                if (item.type === 'text') {
                  return {
                    type: 'output_text',
                    text: item.text,
                    // @ts-expect-error TODO: add types for `annotations`
                    annotations: item.annotations ?? [],
                  };
                }

                if (item.type === 'output_text' || item.type === 'refusal') {
                  return item;
                }

                return [];
              }),
        });

        const functionCallIds = additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY];

        if (isAIMessage(lcMsg) && !!lcMsg.tool_calls?.length) {
          input.push(
            ...lcMsg.tool_calls.map(
              (toolCall): ResponsesInputItem => ({
                type: 'function_call',
                name: toolCall.name,
                arguments: JSON.stringify(toolCall.args),
                call_id: toolCall.id!,
                ...(zdrEnabled ? { id: functionCallIds?.[toolCall.id!] } : {}),
              })
            )
          );
        } else if (additional_kwargs.tool_calls) {
          input.push(
            ...additional_kwargs.tool_calls.map(
              (toolCall): ResponsesInputItem => ({
                type: 'function_call',
                name: toolCall.function.name,
                call_id: toolCall.id,
                arguments: toolCall.function.arguments,
                ...(zdrEnabled ? { id: functionCallIds?.[toolCall.id] } : {}),
              })
            )
          );
        }

        const toolOutputs =
          ((
            lcMsg.response_metadata.output as
              | Array<ResponsesInputItem>
              | undefined
          )?.length ?? 0) > 0
            ? lcMsg.response_metadata.output
            : additional_kwargs.tool_outputs;

        const fallthroughCallTypes: ResponsesInputItem['type'][] = [
          'computer_call',
          /** @ts-ignore */
          'mcp_call',
          /** @ts-ignore */
          'code_interpreter_call',
          /** @ts-ignore */
          'image_generation_call',
        ];

        if (toolOutputs != null) {
          const castToolOutputs = toolOutputs as Array<ResponsesInputItem>;
          const fallthroughCalls = castToolOutputs.filter((item) =>
            fallthroughCallTypes.includes(item.type)
          );
          if (fallthroughCalls.length > 0) input.push(...fallthroughCalls);
        }

        return input;
      }

      if (role === 'user' || role === 'system' || role === 'developer') {
        if (typeof lcMsg.content === 'string') {
          return { type: 'message', role, content: lcMsg.content };
        }

        const messages: ResponsesInputItem[] = [];
        const content = lcMsg.content.flatMap((item) => {
          if (item.type === 'mcp_approval_response') {
            messages.push({
              // @ts-ignore
              type: 'mcp_approval_response',
              approval_request_id: item.approval_request_id,
              approve: item.approve,
            });
          }
          if (isDataContentBlock(item)) {
            return convertToProviderContentBlock(
              item,
              completionsApiContentBlockConverter
            );
          }
          if (item.type === 'text') {
            return {
              type: 'input_text',
              text: item.text,
            };
          }
          if (item.type === 'image_url') {
            return {
              type: 'input_image',
              image_url:
                typeof item.image_url === 'string'
                  ? item.image_url
                  : item.image_url.url,
              detail:
                typeof item.image_url === 'string'
                  ? 'auto'
                  : item.image_url.detail,
            };
          }
          if (
            item.type === 'input_text' ||
            item.type === 'input_image' ||
            item.type === 'input_file'
          ) {
            return item;
          }
          return [];
        });

        if (content.length > 0) {
          messages.push({ type: 'message', role, content });
        }
        return messages;
      }

      console.warn(
        `Unsupported role found when converting to OpenAI Responses API: ${role}`
      );
      return [];
    }
  );
}

export function isReasoningModel(model?: string) {
  return model != null && model !== '' && /\b(o\d|gpt-[5-9])\b/i.test(model);
}

function _convertOpenAIResponsesMessageToBaseMessage(
  response: ResponsesCreateInvoke | ResponsesParseInvoke
): BaseMessage {
  if (response.error) {
    // TODO: add support for `addLangChainErrorFields`
    const error = new Error(response.error.message);
    error.name = response.error.code;
    throw error;
  }

  let messageId: string | undefined;
  const content: MessageContent = [];
  const tool_calls: ToolCall[] = [];
  const invalid_tool_calls: InvalidToolCall[] = [];
  const response_metadata: Record<string, unknown> = {
    model: response.model,
    created_at: response.created_at,
    id: response.id,
    incomplete_details: response.incomplete_details,
    metadata: response.metadata,
    object: response.object,
    status: response.status,
    user: response.user,
    service_tier: response.service_tier,

    // for compatibility with chat completion calls.
    model_name: response.model,
  };

  const additional_kwargs: {
    [key: string]: unknown;
    refusal?: string;
    reasoning?: OpenAIClient.Responses.ResponseReasoningItem;
    tool_outputs?: unknown[];
    parsed?: unknown;
    [_FUNCTION_CALL_IDS_MAP_KEY]?: Record<string, string>;
  } = {};

  for (const item of response.output) {
    if (item.type === 'message') {
      messageId = item.id;
      content.push(
        ...item.content.flatMap((part) => {
          if (part.type === 'output_text') {
            if ('parsed' in part && part.parsed != null) {
              additional_kwargs.parsed = part.parsed;
            }
            return {
              type: 'text',
              text: part.text,
              annotations: part.annotations,
            };
          }

          if (part.type === 'refusal') {
            additional_kwargs.refusal = part.refusal;
            return [];
          }

          return part;
        })
      );
    } else if (item.type === 'function_call') {
      const fnAdapter = {
        function: { name: item.name, arguments: item.arguments },
        id: item.call_id,
      };

      try {
        tool_calls.push(parseToolCall(fnAdapter, { returnId: true }));
      } catch (e: unknown) {
        let errMessage: string | undefined;
        if (
          typeof e === 'object' &&
          e != null &&
          'message' in e &&
          typeof e.message === 'string'
        ) {
          errMessage = e.message;
        }
        invalid_tool_calls.push(makeInvalidToolCall(fnAdapter, errMessage));
      }

      additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] ??= {};
      if (item.id) {
        additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY][item.call_id] = item.id;
      }
    } else if (item.type === 'reasoning') {
      additional_kwargs.reasoning = item;
    } else {
      additional_kwargs.tool_outputs ??= [];
      additional_kwargs.tool_outputs.push(item);
    }
  }

  return new AIMessage({
    id: messageId,
    content,
    tool_calls,
    invalid_tool_calls,
    usage_metadata: response.usage,
    additional_kwargs,
    response_metadata,
  });
}

export function _convertOpenAIResponsesDeltaToBaseMessageChunk(
  chunk: ResponseReturnStreamEvents
) {
  const content: Record<string, unknown>[] = [];
  let generationInfo: Record<string, unknown> = {};
  let usage_metadata: UsageMetadata | undefined;
  const tool_call_chunks: ToolCallChunk[] = [];
  const response_metadata: Record<string, unknown> = {};
  const additional_kwargs: {
    [key: string]: unknown;
    reasoning?: Partial<ChatOpenAIReasoningSummary>;
    tool_outputs?: unknown[];
  } = {};
  let id: string | undefined;
  if (chunk.type === 'response.output_text.delta') {
    content.push({
      type: 'text',
      text: chunk.delta,
      index: chunk.content_index,
    });
    /** @ts-ignore */
  } else if (chunk.type === 'response.output_text_annotation.added') {
    content.push({
      type: 'text',
      text: '',
      /** @ts-ignore */
      annotations: [chunk.annotation],
      /** @ts-ignore */
      index: chunk.content_index,
    });
  } else if (
    chunk.type === 'response.output_item.added' &&
    chunk.item.type === 'message'
  ) {
    id = chunk.item.id;
  } else if (
    chunk.type === 'response.output_item.added' &&
    chunk.item.type === 'function_call'
  ) {
    tool_call_chunks.push({
      type: 'tool_call_chunk',
      name: chunk.item.name,
      args: chunk.item.arguments,
      id: chunk.item.call_id,
      index: chunk.output_index,
    });

    additional_kwargs[_FUNCTION_CALL_IDS_MAP_KEY] = {
      [chunk.item.call_id]: chunk.item.id,
    };
  } else if (
    chunk.type === 'response.output_item.done' &&
    [
      'web_search_call',
      'file_search_call',
      'computer_call',
      'code_interpreter_call',
      'mcp_call',
      'mcp_list_tools',
      'mcp_approval_request',
      'image_generation_call',
    ].includes(chunk.item.type)
  ) {
    additional_kwargs.tool_outputs = [chunk.item];
  } else if (chunk.type === 'response.created') {
    response_metadata.id = chunk.response.id;
    response_metadata.model_name = chunk.response.model;
    response_metadata.model = chunk.response.model;
  } else if (chunk.type === 'response.completed') {
    const msg = _convertOpenAIResponsesMessageToBaseMessage(chunk.response);

    usage_metadata = chunk.response.usage;
    if (chunk.response.text?.format?.type === 'json_schema') {
      additional_kwargs.parsed ??= JSON.parse(msg.text);
    }
    for (const [key, value] of Object.entries(chunk.response)) {
      if (key !== 'id') response_metadata[key] = value;
    }
  } else if (chunk.type === 'response.function_call_arguments.delta') {
    tool_call_chunks.push({
      type: 'tool_call_chunk',
      args: chunk.delta,
      index: chunk.output_index,
    });
  } else if (
    chunk.type === 'response.web_search_call.completed' ||
    chunk.type === 'response.file_search_call.completed'
  ) {
    generationInfo = {
      tool_outputs: {
        id: chunk.item_id,
        type: chunk.type.replace('response.', '').replace('.completed', ''),
        status: 'completed',
      },
    };
  } else if (chunk.type === 'response.refusal.done') {
    additional_kwargs.refusal = chunk.refusal;
  } else if (
    chunk.type === 'response.output_item.added' &&
    'item' in chunk &&
    chunk.item.type === 'reasoning'
  ) {
    const summary: ChatOpenAIReasoningSummary['summary'] | undefined = chunk
      .item.summary
      ? chunk.item.summary.map((s, index) => ({
        ...s,
        index,
      }))
      : undefined;

    additional_kwargs.reasoning = {
      // We only capture ID in the first chunk or else the concatenated result of all chunks will
      // have an ID field that is repeated once per chunk. There is special handling for the `type`
      // field that prevents this, however.
      id: chunk.item.id,
      type: chunk.item.type,
      ...(summary ? { summary } : {}),
    };
  } else if (chunk.type === 'response.reasoning_summary_part.added') {
    additional_kwargs.reasoning = {
      type: 'reasoning',
      summary: [{ ...chunk.part, index: chunk.summary_index }],
    };
  } else if (chunk.type === 'response.reasoning_summary_text.delta') {
    additional_kwargs.reasoning = {
      type: 'reasoning',
      summary: [
        { text: chunk.delta, type: 'summary_text', index: chunk.summary_index },
      ],
    };
    /** @ts-ignore */
  } else if (chunk.type === 'response.image_generation_call.partial_image') {
    // noop/fixme: retaining partial images in a message chunk means that _all_
    // partial images get kept in history, so we don't do anything here.
    return null;
  } else {
    return null;
  }

  return new ChatGenerationChunk({
    // Legacy reasons, `onLLMNewToken` should pulls this out
    text: content.map((part) => part.text).join(''),
    message: new AIMessageChunk({
      id,
      content,
      tool_call_chunks,
      usage_metadata,
      additional_kwargs,
      response_metadata,
    }),
    generationInfo,
  });
}
