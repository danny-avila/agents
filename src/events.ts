/* eslint-disable no-console */
// src/events.ts
import type {
  ToolMessage,
  UsageMetadata,
  BaseMessageFields,
} from '@langchain/core/messages';
import type { MultiAgentGraph, StandardGraph } from '@/graphs';
import type { Logger } from 'winston';
import type * as t from '@/types';
import { handleToolCalls } from '@/tools/handlers';
import { flushThinkingBuffer } from '@/stream';
import { Constants, Providers } from '@/common';

export class HandlerRegistry {
  private handlers: Map<string, t.EventHandler> = new Map();

  register(eventType: string, handler: t.EventHandler): void {
    this.handlers.set(eventType, handler);
  }

  getHandler(eventType: string): t.EventHandler | undefined {
    return this.handlers.get(eventType);
  }
}

export class ModelEndHandler implements t.EventHandler {
  collectedUsage?: UsageMetadata[];
  constructor(collectedUsage?: UsageMetadata[]) {
    if (collectedUsage && !Array.isArray(collectedUsage)) {
      throw new Error('collectedUsage must be an array');
    }
    this.collectedUsage = collectedUsage;
  }

  async handle(
    event: string,
    data: t.ModelEndData,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): Promise<void> {
    if (!graph || !metadata) {
      console.warn(`Graph or metadata not found in ${event} event`);
      return;
    }

    const agentContext = graph.getAgentContext(metadata);

    // Flush any remaining buffered content from thinking tag state machine
    await flushThinkingBuffer(agentContext, graph as StandardGraph);

    const usage = data?.output?.usage_metadata;
    if (usage != null && this.collectedUsage != null) {
      this.collectedUsage.push(usage);
    }

    if (metadata.ls_provider === 'FakeListChatModel') {
      return handleToolCalls(data?.output?.tool_calls, metadata, graph);
    }

    console.log(`====== ${event.toUpperCase()} ======`);
    console.dir(
      {
        usage,
      },
      { depth: null }
    );

    if (
      agentContext.provider !== Providers.GOOGLE &&
      agentContext.provider !== Providers.BEDROCK
    ) {
      return;
    }

    await handleToolCalls(data?.output?.tool_calls, metadata, graph);
  }
}

export class ToolEndHandler implements t.EventHandler {
  private callback?: t.ToolEndCallback;
  private logger?: Logger;
  private omitOutput?: (name?: string) => boolean;
  constructor(
    callback?: t.ToolEndCallback,
    logger?: Logger,
    omitOutput?: (name?: string) => boolean
  ) {
    this.callback = callback;
    this.logger = logger;
    this.omitOutput = omitOutput;
  }
  async handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): Promise<void> {
    try {
      if (!graph || !metadata) {
        if (this.logger) {
          this.logger.warn(`Graph or metadata not found in ${event} event`);
        } else {
          console.warn(`Graph or metadata not found in ${event} event`);
        }
        return;
      }

      const toolEndData = data as t.ToolEndData | undefined;
      if (!toolEndData?.output) {
        if (this.logger) {
          this.logger.warn('No output found in tool_end event');
        } else {
          console.warn('No output found in tool_end event');
        }
        return;
      }

      if (metadata[Constants.PROGRAMMATIC_TOOL_CALLING] === true) {
        return;
      }

      if (this.callback) {
        await this.callback(toolEndData, metadata);
      }
      await graph.handleToolCallCompleted(
        { input: toolEndData.input, output: toolEndData.output },
        metadata,
        this.omitOutput?.((toolEndData.output as ToolMessage | undefined)?.name)
      );
    } catch (error) {
      if (this.logger) {
        this.logger.error('Error handling tool_end event:', error);
      } else {
        console.error('Error handling tool_end event:', error);
      }
    }
  }
}

export class TestLLMStreamHandler implements t.EventHandler {
  handle(event: string, data: t.StreamEventData | undefined): void {
    const chunk = data?.chunk;
    const isMessageChunk = !!(chunk && 'message' in chunk);
    const msg = isMessageChunk ? chunk.message : undefined;
    if (msg && msg.tool_call_chunks && msg.tool_call_chunks.length > 0) {
      console.log(msg.tool_call_chunks);
    } else if (msg && msg.content) {
      if (typeof msg.content === 'string') {
        process.stdout.write(msg.content);
      }
    }
  }
}

export class TestChatStreamHandler implements t.EventHandler {
  handle(event: string, data: t.StreamEventData | undefined): void {
    const chunk = data?.chunk;
    const isContentChunk = !!(chunk && 'content' in chunk);
    const content = isContentChunk && chunk.content;

    if (!content || !isContentChunk) {
      return;
    }

    if (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) {
      console.dir(chunk.tool_call_chunks, { depth: null });
    }

    if (typeof content === 'string') {
      process.stdout.write(content);
    } else {
      console.dir(content, { depth: null });
    }
  }
}

export class LLMStreamHandler implements t.EventHandler {
  handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>
  ): void {
    const chunk = data?.chunk;
    const isMessageChunk = !!(chunk && 'message' in chunk);
    const msg = isMessageChunk && chunk.message;
    if (metadata) {
      console.log(metadata);
    }
    if (msg && msg.tool_call_chunks && msg.tool_call_chunks.length > 0) {
      console.log(msg.tool_call_chunks);
    } else if (msg && msg.content) {
      if (typeof msg.content === 'string') {
        // const text_delta = msg.content;
        // dispatchCustomEvent(GraphEvents.CHAT_MODEL_STREAM, { chunk }, config);
        process.stdout.write(msg.content);
      }
    }
  }
}

export const createMetadataAggregator = (
  _collected?: Record<
    string,
    NonNullable<BaseMessageFields['response_metadata']>
  >[]
): t.MetadataAggregatorResult => {
  const collected = _collected || [];

  const handleLLMEnd: t.HandleLLMEnd = (output) => {
    const { generations } = output;
    const lastMessageOutput = (
      generations[generations.length - 1] as
        | (t.StreamGeneration | undefined)[]
        | undefined
    )?.[0];
    if (!lastMessageOutput) {
      return;
    }
    const { message } = lastMessageOutput;
    if (message?.response_metadata) {
      collected.push(message.response_metadata);
    }
  };

  return { handleLLMEnd, collected };
};
