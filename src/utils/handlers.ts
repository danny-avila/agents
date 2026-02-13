/**
 * Multi-Agent Handler Utilities
 *
 * Provides a simple helper to create handlers with content aggregation for multi-agent scripts.
 *
 * Usage:
 * ```typescript
 * const { contentParts, aggregateContent, handlers } = createHandlers();
 *
 * // With callbacks
 * const { contentParts, aggregateContent, handlers } = createHandlers({
 *   onRunStep: (event, data) => console.log('Step:', data),
 *   onRunStepCompleted: (event, data) => console.log('Completed:', data)
 * });
 * ```
 */

import { GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

interface HandlerCallbacks {
  onRunStep?: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData) => void;
  onRunStepCompleted?: (
    event: GraphEvents.ON_RUN_STEP_COMPLETED,
    data: t.StreamEventData
  ) => void;
  onRunStepDelta?: (
    event: GraphEvents.ON_RUN_STEP_DELTA,
    data: t.StreamEventData
  ) => void;
  onMessageDelta?: (
    event: GraphEvents.ON_MESSAGE_DELTA,
    data: t.StreamEventData
  ) => void;
}

/**
 * Creates handlers with content aggregation for multi-agent scripts
 */
export function createHandlers(callbacks?: HandlerCallbacks): {
  contentParts: Array<t.MessageContentComplex | undefined>;
  aggregateContent: ReturnType<
    typeof createContentAggregator
  >['aggregateContent'];
  handlers: Record<string, t.EventHandler>;
} {
  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Create the handlers object
  const handlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),

    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStep });
        callbacks?.onRunStep?.(event, data);
      },
    },

    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
        callbacks?.onRunStepCompleted?.(event, data);
      },
    },

    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
        callbacks?.onRunStepDelta?.(event, data);
      },
    },

    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
        callbacks?.onMessageDelta?.(event, data);
      },
    },

    [GraphEvents.ON_SUMMARIZE_DELTA]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.SummarizeDeltaData,
        });
      },
    },

    [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.SummarizeCompleteEvent,
        });
      },
    },
  };

  return {
    contentParts,
    aggregateContent,
    handlers,
  };
}
