/* eslint-disable no-console */
// src/events.ts
import type {
  UsageMetadata,
  BaseMessageFields,
} from '@langchain/core/messages';
import type { Graph } from '@/graphs';
import type * as t from '@/types';
import { handleToolCalls } from '@/tools/handlers';
import { Providers, StepTypes, ToolCallTypes } from '@/common';
import { getMessageId } from '@/messages';

export class HandlerRegistry {
  private handlers: Map<string, t.EventHandler> = new Map();

  register(eventType: string, handler: t.EventHandler): void {
    this.handlers.set(eventType, handler);
  }

  getHandler(eventType: string): t.EventHandler | undefined {
    return this.handlers.get(eventType);
  }
}

export class FanoutMetricsHandler implements t.EventHandler {
  handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>,
  ): void {
    if (event !== 'fanout_branch_end') return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const detail = data as any;
    const provider = detail?.provider ?? '';
    const duration = detail?.duration ?? 0;
    const status = detail?.status ?? 'unknown';
    // Minimal structured log; can be wired to metrics sink
    console.log('[fanout]', { provider, duration, status, metadata });
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

  handle(
    event: string,
    data: t.ModelEndData,
    metadata?: Record<string, unknown>,
    graph?: Graph
  ): void {
    if (!graph || !metadata) {
      console.warn(`Graph or metadata not found in ${event} event`);
      return;
    }

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

    if (metadata.provider !== Providers.GOOGLE) {
      return;
    }

    handleToolCalls(data?.output?.tool_calls, metadata, graph);
  }
}

export class ToolEndHandler implements t.EventHandler {
  private callback?: t.ToolEndCallback;
  private omitOutput?: (name?: string) => boolean;
  constructor(
    callback?: t.ToolEndCallback,
    omitOutput?: (name?: string) => boolean
  ) {
    this.callback = callback;
    this.omitOutput = omitOutput;
  }
  handle(
    event: string,
    data: t.StreamEventData | undefined,
    metadata?: Record<string, unknown>,
    graph?: Graph
  ): void {
    if (!graph || !metadata) {
      console.warn(`Graph or metadata not found in ${event} event`);
      return;
    }

    const toolEndData = data as t.ToolEndData | undefined;
    if (!toolEndData?.output) {
      console.warn('No output found in tool_end event');
      return;
    }

    this.callback?.(toolEndData, metadata);

    // Backfill mapping if tool_call_id wasn't registered or missing (e.g., some providers omit it)
    const rawId = toolEndData.output?.tool_call_id ?? '';
    let toolCallId = rawId;
    try {
      const stepKey = graph.getStepKey(metadata);
      let lastStepId = '';
      try {
        lastStepId = graph.getStepIdByKey(stepKey, graph.contentData.length - 1);
      } catch {
        // Ensure a message creation step exists
        const messageId = getMessageId(stepKey, graph, true) ?? '';
        lastStepId = graph.dispatchRunStep(stepKey, {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: { message_id: messageId },
        });
      }

      // Synthesize an ID if provider didn't supply one
      if (!toolCallId || toolCallId === '') {
        toolCallId = `${toolEndData.output?.name ?? 'tool'}_${Date.now()}`;
      }

      if (!graph.toolCallStepIds?.has(toolCallId)) {
        // Attach tool_call_ids to the last message so UI/state understands association
        graph.dispatchMessageDelta(lastStepId, {
          content: [
            { type: 'text', text: '', tool_call_ids: [toolCallId] },
          ],
        });

        // Create a TOOL_CALLS run step so Graph maps toolCallId -> stepId
        graph.dispatchRunStep(stepKey, {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: toolCallId,
              name: toolEndData.output?.name ?? '',
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              args: {},
              type: ToolCallTypes.TOOL_CALL,
            } as unknown as t.AgentToolCall,
          ],
        });
      }
    } catch (e) {
      // If backfill fails, continue; the completion call below will surface issues
    }

    // Ensure handleToolCallCompleted receives a populated tool_call_id
    const outputForDispatch = {
      ...toolEndData.output,
      tool_call_id: toolCallId,
    } as NonNullable<typeof toolEndData.output>;

    // Mark tool call as invoked to prevent router loops
    if (graph.invokedToolIds == null) {
      graph.invokedToolIds = new Set<string>();
    }
    if (toolCallId) {
      graph.invokedToolIds.add(toolCallId);
    }

    graph.handleToolCallCompleted(
      { input: toolEndData.input, output: outputForDispatch },
      metadata,
      this.omitOutput?.(toolEndData.output.name)
    );
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
