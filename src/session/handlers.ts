import { GraphEvents } from '@/common';
import { ModelEndHandler, ToolEndHandler } from '@/events';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import type * as t from '@/types';
import type {
  AgentSessionHandlersResult,
  AgentSessionStreamEvent,
  AgentSessionUsage,
} from './types';
import { createTimestamp } from './ids';
import { toJsonValue } from './messageSerialization';

function createEventFactory(params: {
  runId: string;
  threadId: string;
}): (
  type: AgentSessionStreamEvent['type'],
  data?: unknown
) => AgentSessionStreamEvent {
  let sequence = 0;
  return (type, data) => ({
    type,
    sequence: sequence++,
    runId: params.runId,
    threadId: params.threadId,
    timestamp: createTimestamp(),
    ...(typeof data !== 'undefined' ? { data: toJsonValue(data) } : {}),
  });
}

function updateUsage(usage: AgentSessionUsage, data: t.ModelEndData): void {
  const metadata = data?.output?.usage_metadata;
  if (!metadata) {
    return;
  }
  const inputTokens = metadata.input_tokens;
  const outputTokens = metadata.output_tokens;
  usage.inputTokens += inputTokens;
  usage.outputTokens += outputTokens;
  usage.totalTokens += inputTokens + outputTokens;
}

async function callUserHandler(params: {
  userHandlers?: Record<string, t.EventHandler>;
  event: string;
  data: Parameters<t.EventHandler['handle']>[1];
  metadata?: Record<string, unknown>;
  graph?: Parameters<t.EventHandler['handle']>[3];
}): Promise<void> {
  const handler = params.userHandlers?.[params.event];
  if (!handler) {
    return;
  }
  await handler.handle(
    params.event,
    params.data,
    params.metadata,
    params.graph
  );
}

export function createRunHandlers(params: {
  runId: string;
  threadId: string;
  userHandlers?: Record<string, t.EventHandler>;
  onEvent?: (event: AgentSessionStreamEvent) => void;
}): AgentSessionHandlersResult {
  const { contentParts, aggregateContent } = createContentAggregator();
  const steps: t.RunStep[] = [];
  const usage: AgentSessionUsage = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  };
  const events: AgentSessionStreamEvent[] = [];
  const createEvent = createEventFactory({
    runId: params.runId,
    threadId: params.threadId,
  });
  const emitEvent = (event: AgentSessionStreamEvent): void => {
    events.push(event);
    params.onEvent?.(event);
  };
  const chatModelStreamHandler = new ChatModelStreamHandler();
  const toolEndHandler = new ToolEndHandler();
  const modelEndHandler = new ModelEndHandler();

  emitEvent(createEvent('run.started'));

  const handlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_STREAM]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        await chatModelStreamHandler.handle(
          event,
          data as t.StreamEventData,
          metadata,
          graph
        );
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        await modelEndHandler.handle(
          event,
          data as t.ModelEndData,
          metadata,
          graph
        );
        updateUsage(usage, data as t.ModelEndData);
        emitEvent(createEvent('usage.updated', usage));
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.TOOL_END]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        await toolEndHandler.handle(
          event,
          data as t.StreamEventData,
          metadata,
          graph
        );
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        const runStep = data as t.RunStep;
        steps.push(runStep);
        aggregateContent({ event: GraphEvents.ON_RUN_STEP, data: runStep });
        if (runStep.stepDetails.type === 'tool_calls') {
          emitEvent(createEvent('tool.started', runStep));
        }
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        const delta = data as t.RunStepDeltaEvent;
        aggregateContent({ event: GraphEvents.ON_RUN_STEP_DELTA, data: delta });
        emitEvent(createEvent('tool.delta', delta));
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        const completed = data as unknown as {
          result:
            | t.ToolEndEvent
            | (t.SummaryCompleted & { id: string; index: number });
        };
        aggregateContent({
          event: GraphEvents.ON_RUN_STEP_COMPLETED,
          data: completed as { result: t.ToolEndEvent },
        });
        emitEvent(createEvent('tool.completed', completed));
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        const delta = data as t.MessageDeltaEvent;
        aggregateContent({ event: GraphEvents.ON_MESSAGE_DELTA, data: delta });
        emitEvent(createEvent('message.delta', delta));
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        const delta = data as t.ReasoningDeltaEvent;
        aggregateContent({
          event: GraphEvents.ON_REASONING_DELTA,
          data: delta,
        });
        emitEvent(createEvent('reasoning.delta', delta));
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_SUMMARIZE_DELTA]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        aggregateContent({
          event: GraphEvents.ON_SUMMARIZE_DELTA,
          data: data as t.SummarizeDeltaData,
        });
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
    [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        aggregateContent({
          event: GraphEvents.ON_SUMMARIZE_COMPLETE,
          data: data as t.SummarizeCompleteEvent,
        });
        await callUserHandler({
          userHandlers: params.userHandlers,
          event,
          data,
          metadata,
          graph,
        });
      },
    },
  };

  return { contentParts, steps, usage, events, handlers };
}
