import { HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createContentAggregator } from '@/stream';
import { Run } from '@/run';

describe('Custom event handler awaitHandlers behavior', () => {
  jest.setTimeout(15000);

  const llmConfig: t.LLMConfig = {
    provider: Providers.OPENAI,
    streaming: true,
    streamUsage: false,
  };

  const config = {
    configurable: {
      thread_id: 'test-thread',
    },
    streamMode: 'values' as const,
    version: 'v2' as const,
  };

  it('does not redispatch SDK custom events yielded by streamEvents', async () => {
    const handledEvents: GraphEvents[] = [];
    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (event: GraphEvents): void => {
          handledEvents.push(event);
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: 'test-custom-events-skip-stream-loop',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      skipCleanup: true,
      customHandlers,
    });

    async function* streamEvents(): AsyncGenerator<t.StreamEvent> {
      yield {
        event: GraphEvents.ON_MESSAGE_DELTA,
        name: 'custom-event',
        run_id: 'custom-event-run',
        metadata: {},
        data: {},
      };
    }

    run.graphRunnable = { streamEvents } as unknown as t.CompiledStateWorkflow;

    await run.processStream({ messages: [new HumanMessage('hello')] }, config);

    expect(handledEvents).toEqual([]);
  });

  it('dispatches message deltas through the graph handler registry', async () => {
    const handledEvents: Array<{
      data: t.MessageDeltaEvent;
      metadata?: Record<string, unknown>;
    }> = [];
    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: (_event, data, metadata): void => {
          handledEvents.push({
            data: data as t.MessageDeltaEvent,
            metadata,
          });
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: 'test-message-delta-direct-dispatch',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      skipCleanup: true,
      customHandlers,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const metadata = { thread_id: 'thread_direct', langgraph_step: 1 };
    run.Graph.config = { configurable: { thread_id: 'thread_direct' } };
    await run.Graph.dispatchMessageDelta(
      'step_direct',
      {
        content: [{ type: ContentTypes.TEXT, text: 'hello' }],
      },
      metadata
    );

    expect(handledEvents).toEqual([
      {
        data: {
          id: 'step_direct',
          delta: {
            content: [{ type: ContentTypes.TEXT, text: 'hello' }],
          },
        },
        metadata,
      },
    ]);
  });

  it('should fully aggregate all content before processStream returns', async () => {
    const longResponse =
      'The quick brown fox jumps over the lazy dog and then runs across the field to find shelter from the rain';

    let aggregateCallCount = 0;
    const { contentParts, aggregateContent } = createContentAggregator();

    const wrappedAggregate: t.ContentAggregator = (params) => {
      aggregateCallCount++;
      aggregateContent(params);
    };

    let messageDeltaCount = 0;

    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (
          event: GraphEvents.ON_RUN_STEP_COMPLETED,
          data: t.StreamEventData
        ) => {
          wrappedAggregate({
            event,
            data: data as unknown as { result: t.ToolEndEvent },
          });
        },
      },
      [GraphEvents.ON_RUN_STEP]: {
        handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData) => {
          wrappedAggregate({ event, data: data as t.RunStep });
        },
      },
      [GraphEvents.ON_RUN_STEP_DELTA]: {
        handle: (
          event: GraphEvents.ON_RUN_STEP_DELTA,
          data: t.StreamEventData
        ) => {
          wrappedAggregate({ event, data: data as t.RunStepDeltaEvent });
        },
      },
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: async (
          event: GraphEvents.ON_MESSAGE_DELTA,
          data: t.StreamEventData
        ) => {
          messageDeltaCount++;
          wrappedAggregate({ event, data: data as t.MessageDeltaEvent });
        },
      },
      [GraphEvents.ON_REASONING_DELTA]: {
        handle: (
          event: GraphEvents.ON_REASONING_DELTA,
          data: t.StreamEventData
        ) => {
          wrappedAggregate({ event, data: data as t.ReasoningDeltaEvent });
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: 'test-await-handlers',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    run.Graph!.overrideTestModel([longResponse]);

    const inputs = { messages: [new HumanMessage('hello')] };
    const finalContentParts = await run.processStream(inputs, config);

    expect(finalContentParts).toBeDefined();
    expect(finalContentParts!.length).toBeGreaterThan(0);

    expect(messageDeltaCount).toBeGreaterThan(0);
    expect(aggregateCallCount).toBeGreaterThan(0);

    const typedParts = contentParts as t.MessageContentComplex[];
    const textParts = typedParts.filter(
      (p: t.MessageContentComplex | undefined) =>
        p !== undefined && p.type === ContentTypes.TEXT
    );
    expect(textParts.length).toBeGreaterThan(0);

    const aggregatedText = textParts
      .map(
        (p) =>
          (p as { type: string; [ContentTypes.TEXT]: string })[
            ContentTypes.TEXT
          ]
      )
      .join('');
    expect(aggregatedText).toBe(longResponse);
  });

  it('should aggregate content from async handlers before processStream returns', async () => {
    const response =
      'This is a test of async handler aggregation with multiple tokens';

    const { contentParts, aggregateContent } = createContentAggregator();

    let asyncHandlerCompletions = 0;

    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_RUN_STEP_COMPLETED]: {
        handle: (
          event: GraphEvents.ON_RUN_STEP_COMPLETED,
          data: t.StreamEventData
        ) => {
          aggregateContent({
            event,
            data: data as unknown as { result: t.ToolEndEvent },
          });
        },
      },
      [GraphEvents.ON_RUN_STEP]: {
        handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData) => {
          aggregateContent({ event, data: data as t.RunStep });
        },
      },
      [GraphEvents.ON_RUN_STEP_DELTA]: {
        handle: (
          event: GraphEvents.ON_RUN_STEP_DELTA,
          data: t.StreamEventData
        ) => {
          aggregateContent({ event, data: data as t.RunStepDeltaEvent });
        },
      },
      [GraphEvents.ON_MESSAGE_DELTA]: {
        handle: async (
          event: GraphEvents.ON_MESSAGE_DELTA,
          data: t.StreamEventData
        ) => {
          await new Promise<void>((resolve) => setTimeout(resolve, 5));
          aggregateContent({ event, data: data as t.MessageDeltaEvent });
          asyncHandlerCompletions++;
        },
      },
      [GraphEvents.ON_REASONING_DELTA]: {
        handle: (
          event: GraphEvents.ON_REASONING_DELTA,
          data: t.StreamEventData
        ) => {
          aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: 'test-async-handlers',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    run.Graph!.overrideTestModel([response]);

    const inputs = { messages: [new HumanMessage('hello')] };
    await run.processStream(inputs, config);

    expect(asyncHandlerCompletions).toBeGreaterThan(0);

    const typedParts = contentParts as t.MessageContentComplex[];
    const textParts = typedParts.filter(
      (p: t.MessageContentComplex | undefined) =>
        p !== undefined && p.type === ContentTypes.TEXT
    );
    expect(textParts.length).toBeGreaterThan(0);

    const aggregatedText = textParts
      .map(
        (p) =>
          (p as { type: string; [ContentTypes.TEXT]: string })[
            ContentTypes.TEXT
          ]
      )
      .join('');
    expect(aggregatedText).toBe(response);
  });
});
