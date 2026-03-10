import { concat } from '@langchain/core/utils/stream';
import { AIMessageChunk } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { manualToolStreamProviders } from '@/llm/providers';
import { modifyDeltaProperties } from '@/messages';
import { ChatModelStreamHandler } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { initializeModel } from '@/llm/init';

/**
 * Context passed to `attemptInvoke` for stream event handling.
 * Matches the subset of Graph that `ChatModelStreamHandler.handle` needs.
 */
export type InvokeContext = Parameters<ChatModelStreamHandler['handle']>[3];

/**
 * Invokes a chat model with the given messages, handling both streaming and
 * non-streaming paths.  Stream chunks are processed through a local
 * `ChatModelStreamHandler` so that run steps (MESSAGE_CREATION, TOOL_CALLS)
 * are fully populated before the graph transitions to the next node.
 */
export async function attemptInvoke(
  {
    model,
    messages,
    provider,
    context,
  }: {
    model: t.ChatModel;
    messages: BaseMessage[];
    provider: Providers;
    context?: InvokeContext;
  },
  config?: RunnableConfig
): Promise<Partial<t.BaseGraphState>> {
  if (model.stream) {
    const metadata = config?.metadata as Record<string, unknown> | undefined;
    const streamHandler = new ChatModelStreamHandler();
    const stream = await model.stream(messages, config);
    let finalChunk: AIMessageChunk | undefined;
    for await (const chunk of stream) {
      await streamHandler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        { chunk },
        metadata,
        context
      );
      finalChunk = finalChunk ? concat(finalChunk, chunk) : chunk;
    }

    if (manualToolStreamProviders.has(provider)) {
      finalChunk = modifyDeltaProperties(provider, finalChunk);
    }

    if ((finalChunk?.tool_calls?.length ?? 0) > 0) {
      finalChunk!.tool_calls = finalChunk!.tool_calls?.filter(
        (tool_call: ToolCall) => !!tool_call.name
      );
    }

    return { messages: [finalChunk as AIMessageChunk] };
  }

  const finalMessage = await model.invoke(messages, config);
  if ((finalMessage.tool_calls?.length ?? 0) > 0) {
    finalMessage.tool_calls = finalMessage.tool_calls?.filter(
      (tool_call: ToolCall) => !!tool_call.name
    );
  }
  return { messages: [finalMessage] };
}

/**
 * Attempts each fallback provider in order until one succeeds.
 * Throws the last error if all fallbacks fail.
 */
export async function tryFallbackProviders({
  fallbacks,
  tools,
  messages,
  config,
  primaryError,
  context,
}: {
  fallbacks: Array<{ provider: Providers; clientOptions?: t.ClientOptions }>;
  tools?: t.GraphTools;
  messages: BaseMessage[];
  config?: RunnableConfig;
  primaryError: unknown;
  context?: InvokeContext;
}): Promise<Partial<t.BaseGraphState> | undefined> {
  let lastError: unknown = primaryError;
  for (const fb of fallbacks) {
    try {
      const fbModel = initializeModel({
        provider: fb.provider,
        clientOptions: fb.clientOptions,
        tools,
      });
      const result = await attemptInvoke(
        {
          model: fbModel as t.ChatModel,
          messages,
          provider: fb.provider,
          context,
        },
        config
      );
      return result;
    } catch (e) {
      lastError = e;
      continue;
    }
  }
  if (lastError !== undefined) {
    throw lastError;
  }
  return undefined;
}
