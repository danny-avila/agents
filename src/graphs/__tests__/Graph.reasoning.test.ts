import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { Run } from '@/run';

type ReasoningKey = 'reasoning_content' | 'reasoning';

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

class StreamingReasoningModel implements t.ChatModel {
  constructor(private readonly chunks: AIMessageChunk[]) {}

  async invoke(
    _messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AIMessageChunk> {
    return this.chunks[this.chunks.length - 1] ?? new AIMessageChunk('');
  }

  async stream(
    _messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AsyncIterable<AIMessageChunk>> {
    const chunks = this.chunks;
    return (async function* streamChunks(): AsyncGenerator<AIMessageChunk> {
      for (const chunk of chunks) {
        yield chunk;
      }
    })();
  }
}

function createReasoningChunk(
  reasoningKey: ReasoningKey,
  reasoningText: string
): AIMessageChunk {
  return new AIMessageChunk({
    content: '',
    additional_kwargs: {
      [reasoningKey]: reasoningText,
    },
  });
}

function createOpenAIReasoningSummaryChunk(reasoningText: string): AIMessageChunk {
  return new AIMessageChunk({
    content: '',
    additional_kwargs: {
      reasoning: {
        summary: [{ text: reasoningText }],
      },
    },
  });
}

function createReasoningHandlers(
  aggregateContent: t.ContentAggregator,
  reasoningDeltas: t.ReasoningDeltaEvent[],
  messageDeltas?: t.MessageDeltaEvent[]
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
        const messageDelta = data as t.MessageDeltaEvent;
        messageDeltas?.push(messageDelta);
        aggregateContent({ event, data: messageDelta });
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

  it.each([
    {
      providerName: 'DeepSeek',
      provider: Providers.DEEPSEEK,
      reasoningKey: 'reasoning_content' as const,
    },
    {
      providerName: 'OpenRouter',
      provider: Providers.OPENROUTER,
      reasoningKey: 'reasoning' as const,
    },
  ])(
    'does not replay streamed $providerName reasoning from the final fallback',
    async ({ provider, providerName, reasoningKey }) => {
      const text = 'Done.';
      const reasoningText = 'Check the provider reasoning stream first.';
      const firstReasoningChunk = reasoningText.slice(0, 19);
      const secondReasoningChunk = reasoningText.slice(19);
      const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
      const messageDeltas: t.MessageDeltaEvent[] = [];
      const { contentParts, aggregateContent } = createContentAggregator();
      const run = await Run.create<t.IState>({
        runId: `reasoning-fallback-${providerName.toLowerCase()}-stream`,
        graphConfig: {
          type: 'standard',
          llmConfig: {
            provider,
            streamUsage: false,
          },
          reasoningKey,
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: createReasoningHandlers(
          aggregateContent,
          reasoningDeltas,
          messageDeltas
        ),
      });

      if (!run.Graph) {
        throw new Error('Expected graph to be initialized');
      }

      run.Graph.overrideModel = new StreamingReasoningModel([
        createReasoningChunk(reasoningKey, firstReasoningChunk),
        createReasoningChunk(reasoningKey, secondReasoningChunk),
        new AIMessageChunk({ content: text }),
      ]);

      await run.processStream(
        { messages: [new HumanMessage('stream provider reasoning')] },
        {
          ...config,
          configurable: {
            thread_id: `reasoning-fallback-${providerName.toLowerCase()}-stream`,
          },
        }
      );

      expect(reasoningDeltas).toHaveLength(2);
      expect(messageDeltas).toHaveLength(1);
      expect(contentParts).toEqual([
        { type: ContentTypes.THINK, think: reasoningText },
        { type: ContentTypes.TEXT, text },
      ]);
    }
  );

  it.each([
    {
      providerName: 'DeepSeek',
      provider: Providers.DEEPSEEK,
      reasoningKey: 'reasoning_content' as const,
    },
    {
      providerName: 'OpenRouter',
      provider: Providers.OPENROUTER,
      reasoningKey: 'reasoning' as const,
    },
  ])(
    'does not replay streamed reasoning-only $providerName output',
    async ({ provider, providerName, reasoningKey }) => {
      const reasoningText = 'The answer is still being considered.';
      const firstReasoningChunk = reasoningText.slice(0, 14);
      const secondReasoningChunk = reasoningText.slice(14);
      const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
      const messageDeltas: t.MessageDeltaEvent[] = [];
      const { contentParts, aggregateContent } = createContentAggregator();
      const run = await Run.create<t.IState>({
        runId: `reasoning-only-${providerName.toLowerCase()}-stream`,
        graphConfig: {
          type: 'standard',
          llmConfig: {
            provider,
            streamUsage: false,
          },
          reasoningKey,
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: createReasoningHandlers(
          aggregateContent,
          reasoningDeltas,
          messageDeltas
        ),
      });

      if (!run.Graph) {
        throw new Error('Expected graph to be initialized');
      }

      run.Graph.overrideModel = new StreamingReasoningModel([
        createReasoningChunk(reasoningKey, firstReasoningChunk),
        createReasoningChunk(reasoningKey, secondReasoningChunk),
      ]);

      await run.processStream(
        { messages: [new HumanMessage('stream provider reasoning only')] },
        {
          ...config,
          configurable: {
            thread_id: `reasoning-only-${providerName.toLowerCase()}-stream`,
          },
        }
      );

      expect(reasoningDeltas).toHaveLength(2);
      expect(messageDeltas).toHaveLength(0);
      expect(contentParts).toEqual([
        { type: ContentTypes.THINK, think: reasoningText },
      ]);
    }
  );

  it('does not replay streamed OpenAI reasoning summaries from the final fallback', async () => {
    const text = 'Done.';
    const reasoningText = 'Use the summary reasoning channel.';
    const firstReasoningChunk = reasoningText.slice(0, 15);
    const secondReasoningChunk = reasoningText.slice(15);
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openai-summary-stream',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createReasoningHandlers(
        aggregateContent,
        reasoningDeltas,
        messageDeltas
      ),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new StreamingReasoningModel([
      createOpenAIReasoningSummaryChunk(firstReasoningChunk),
      createOpenAIReasoningSummaryChunk(secondReasoningChunk),
      new AIMessageChunk({ content: text }),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('stream OpenAI summary reasoning')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openai-summary-stream',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(2);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
  });
});
