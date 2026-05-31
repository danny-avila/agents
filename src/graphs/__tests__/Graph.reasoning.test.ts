import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { Run } from '@/run';

class InvokeOnlyReasoningModel implements t.ChatModel {
  constructor(
    private readonly response: {
      content: string;
      reasoningContent: string;
    }
  ) {}

  async invoke(
    _messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AIMessageChunk> {
    return new AIMessageChunk({
      content: this.response.content,
      additional_kwargs: {
        reasoning_content: this.response.reasoningContent,
      },
    });
  }
}

function createReasoningHandlers(
  aggregateContent: t.ContentAggregator,
  reasoningDeltas: t.ReasoningDeltaEvent[]
): Record<string | GraphEvents, t.EventHandler> {
  return {
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData): void => {
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => {
        const reasoningDelta = data as t.ReasoningDeltaEvent;
        reasoningDeltas.push(reasoningDelta);
        aggregateContent({ event, data: reasoningDelta });
      },
    },
  };
}

describe('StandardGraph final response reasoning fallback', () => {
  const config = {
    configurable: {
      thread_id: 'reasoning-fallback-thread',
    },
    streamMode: 'values' as const,
    version: 'v2' as const,
  };
  const llmConfig: t.LLMConfig = {
    provider: Providers.OPENAI,
    disableStreaming: true,
    streamUsage: false,
  };

  it('emits reasoning_content from invoke-only final responses', async () => {
    const reasoningText = 'Need to inspect the Home Assistant tool state.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-empty-content',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createReasoningHandlers(
        aggregateContent,
        reasoningDeltas
      ),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new InvokeOnlyReasoningModel({
      content: '',
      reasoningContent: reasoningText,
    });

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('turn on the bedroom light')] },
      config
    );

    expect(finalContentParts).toBeDefined();
    expect(reasoningDeltas).toHaveLength(1);
    expect(reasoningDeltas[0].delta.content?.[0]).toEqual({
      type: ContentTypes.THINK,
      think: reasoningText,
    });
    expect(contentParts).toContainEqual({
      type: ContentTypes.THINK,
      think: reasoningText,
    });
  });

  it('keeps final reasoning before final text when both are present', async () => {
    const text = 'Done.';
    const reasoningText = 'Decide whether a tool is needed first.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-with-text',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createReasoningHandlers(
        aggregateContent,
        reasoningDeltas
      ),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new InvokeOnlyReasoningModel({
      content: text,
      reasoningContent: reasoningText,
    });

    await run.processStream(
      { messages: [new HumanMessage('say done')] },
      config
    );

    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
  });
});
