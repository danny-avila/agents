import { GraphEvents } from '@/common';
import type * as t from '@/types';

export interface OpenAICompatibleWriter {
  write(data: string): void | Promise<void>;
}

export interface OpenAIResponseContext {
  requestId: string;
  model: string;
  created: number;
}

export interface OpenAICompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
}

export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAIChatCompletionChunkChoice {
  index: number;
  delta: {
    role?: 'assistant';
    content?: string | null;
    reasoning?: string | null;
    tool_calls?: Array<{
      index: number;
      id?: string;
      type?: 'function';
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null;
}

export interface OpenAIChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: OpenAIChatCompletionChunkChoice[];
  usage?: OpenAICompletionUsage;
}

export interface OpenAIStreamTracker {
  hasText: boolean;
  hasReasoning: boolean;
  toolCalls: Map<number, OpenAIToolCall>;
  usage: {
    promptTokens: number;
    completionTokens: number;
    reasoningTokens: number;
  };
}

export interface OpenAIHandlerConfig {
  writer: OpenAICompatibleWriter;
  context: OpenAIResponseContext;
  tracker: OpenAIStreamTracker;
}

export function createOpenAIStreamTracker(): OpenAIStreamTracker {
  return {
    hasText: false,
    hasReasoning: false,
    toolCalls: new Map(),
    usage: {
      promptTokens: 0,
      completionTokens: 0,
      reasoningTokens: 0,
    },
  };
}

export function createChatCompletionChunk(
  context: OpenAIResponseContext,
  delta: OpenAIChatCompletionChunkChoice['delta'],
  finishReason: OpenAIChatCompletionChunkChoice['finish_reason'] = null,
  usage?: OpenAICompletionUsage
): OpenAIChatCompletionChunk {
  return {
    id: context.requestId,
    object: 'chat.completion.chunk',
    created: context.created,
    model: context.model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...(usage ? { usage } : {}),
  };
}

export async function writeOpenAISSE(
  writer: OpenAICompatibleWriter,
  data: OpenAIChatCompletionChunk | '[DONE]'
): Promise<void> {
  await writer.write(
    `data: ${data === '[DONE]' ? data : JSON.stringify(data)}\n\n`
  );
}

function getTextParts(
  data: t.MessageDeltaEvent | t.ReasoningDeltaEvent
): string[] {
  const parts = data.delta.content ?? [];
  const text: string[] = [];
  for (const part of parts) {
    if ('text' in part && typeof part.text === 'string') {
      text.push(part.text);
      continue;
    }
    if ('think' in part && typeof part.think === 'string') {
      text.push(part.think);
    }
  }
  return text;
}

export function createOpenAIHandlers(
  config: OpenAIHandlerConfig
): Record<string, t.EventHandler> {
  return {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        for (const text of getTextParts(data as t.MessageDeltaEvent)) {
          config.tracker.hasText = true;
          await writeOpenAISSE(
            config.writer,
            createChatCompletionChunk(config.context, { content: text })
          );
        }
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        for (const text of getTextParts(data as t.ReasoningDeltaEvent)) {
          config.tracker.hasReasoning = true;
          await writeOpenAISSE(
            config.writer,
            createChatCompletionChunk(config.context, { reasoning: text })
          );
        }
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        const delta = (data as t.RunStepDeltaEvent).delta;
        if (delta.type !== 'tool_calls') {
          return;
        }
        for (const toolCall of delta.tool_calls ?? []) {
          await writeOpenAISSE(
            config.writer,
            createChatCompletionChunk(config.context, {
              tool_calls: [
                {
                  index: toolCall.index ?? 0,
                  ...(toolCall.id != null && toolCall.id !== ''
                    ? { id: toolCall.id }
                    : {}),
                  type: 'function',
                  function: {
                    ...(toolCall.name != null && toolCall.name !== ''
                      ? { name: toolCall.name }
                      : {}),
                    ...(toolCall.args != null && toolCall.args !== ''
                      ? { arguments: toolCall.args }
                      : {}),
                  },
                },
              ],
            })
          );
        }
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event, data): void => {
        const usage = (data as t.ModelEndData)?.output?.usage_metadata;
        if (!usage) {
          return;
        }
        config.tracker.usage.promptTokens += usage.input_tokens;
        config.tracker.usage.completionTokens += usage.output_tokens;
      },
    },
  };
}

export async function sendOpenAIFinalChunk(
  config: OpenAIHandlerConfig,
  finishReason: OpenAIChatCompletionChunkChoice['finish_reason'] = 'stop'
): Promise<void> {
  const usage: OpenAICompletionUsage = {
    prompt_tokens: config.tracker.usage.promptTokens,
    completion_tokens: config.tracker.usage.completionTokens,
    total_tokens:
      config.tracker.usage.promptTokens + config.tracker.usage.completionTokens,
  };
  if (config.tracker.usage.reasoningTokens > 0) {
    usage.completion_tokens_details = {
      reasoning_tokens: config.tracker.usage.reasoningTokens,
    };
  }
  await writeOpenAISSE(
    config.writer,
    createChatCompletionChunk(config.context, {}, finishReason, usage)
  );
  await writeOpenAISSE(config.writer, '[DONE]');
}
