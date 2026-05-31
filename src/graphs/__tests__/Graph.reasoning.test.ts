import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { ModelEndHandler, ToolEndHandler } from '@/events';
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

class InvokeOnlyMessageModel implements t.ChatModel {
  constructor(private readonly message: AIMessageChunk) {}

  async invoke(
    _messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AIMessageChunk> {
    return this.message;
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

class CallbackStreamingReasoningModel extends FakeListChatModel {
  constructor(private readonly chunks: AIMessageChunk[]) {
    super({ responses: [''] });
  }

  _llmType(): string {
    return 'callback-streaming-reasoning';
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    for (const chunk of this.chunks) {
      const text = typeof chunk.content === 'string' ? chunk.content : '';
      yield new ChatGenerationChunk({
        text,
        generationInfo: {},
        message: chunk,
      });
      void runManager?.handleLLMNewToken(text);
    }
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

function createLibreChatLikeHandlers({
  aggregateContent,
  collectedUsage,
  emittedEvents,
}: {
  aggregateContent: t.ContentAggregator;
  collectedUsage: UsageMetadata[];
  emittedEvents: Array<{ event: string; data: unknown }>;
}): Record<string | GraphEvents, t.EventHandler> {
  const modelEndHandler = new ModelEndHandler(collectedUsage);
  const toolEndHandler = new ToolEndHandler();
  const aggregateAndEmit = (
    event: GraphEvents,
    data: t.StreamEventData
  ): void => {
    aggregateContent({
      event,
      data: data as
        | t.RunStep
        | t.MessageDeltaEvent
        | t.ReasoningDeltaEvent
        | t.RunStepDeltaEvent
        | { result: t.ToolEndEvent },
    });
    emittedEvents.push({
      event,
      data,
    });
  };

  return {
    [GraphEvents.CHAT_MODEL_END]: {
      handle: async (event, data, metadata, graph): Promise<void> => {
        await modelEndHandler.handle(
          event,
          data as t.ModelEndData,
          metadata,
          graph
        );
        emittedEvents.push({
          event,
          data,
        });
      },
    },
    [GraphEvents.TOOL_END]: toolEndHandler,
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData): void =>
        aggregateAndEmit(event, data),
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => aggregateAndEmit(event, data),
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => aggregateAndEmit(event, data),
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => aggregateAndEmit(event, data),
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => aggregateAndEmit(event, data),
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

    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
    ]);
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

  it('returns reasoning content without a custom aggregator', async () => {
    const reasoningText = 'Reasoning should persist for returnContent.';
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-return-content',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new InvokeOnlyReasoningModel({
      content: '',
      reasoningContent: reasoningText,
    });

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('return reasoning content')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-return-content',
        },
      }
    );

    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
    ]);
  });

  it('emits every OpenAI reasoning summary segment in invoke-only fallback', async () => {
    const reasoningText = 'First summary. Second summary.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openai-multi-summary',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          disableStreaming: true,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
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

    run.Graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({
        content: '',
        additional_kwargs: {
          reasoning: {
            summary: [{ text: 'First summary. ' }, { text: 'Second summary.' }],
          },
        },
      })
    );

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('return multi summary reasoning')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openai-multi-summary',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(1);
    expect(reasoningDeltas[0].delta.content?.[0]).toEqual({
      type: ContentTypes.THINK,
      think: reasoningText,
    });
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
    ]);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
    ]);
  });

  it('emits OpenRouter reasoning_details in invoke-only fallback', async () => {
    const reasoningText = 'OpenRouter detail reasoning.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-details',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
          disableStreaming: true,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
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

    run.Graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({
        content: '',
        additional_kwargs: {
          reasoning_details: [
            { type: 'reasoning.text', text: 'OpenRouter detail ' },
            { type: 'reasoning.encrypted', id: 'encrypted' },
            { type: 'reasoning.text', text: 'reasoning.' },
          ],
        },
      })
    );

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('return OpenRouter reasoning details')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-details',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(1);
    expect(reasoningDeltas[0].delta.content?.[0]).toEqual({
      type: ContentTypes.THINK,
      think: reasoningText,
    });
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
    ]);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
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

  it('preserves LibreChat-like callbacks including model_end usage collection', async () => {
    const text = 'Visible answer.';
    const reasoningText = 'Visible reasoning.';
    const usage: UsageMetadata = {
      input_tokens: 7,
      output_tokens: 5,
      total_tokens: 12,
      output_token_details: {
        reasoning: 3,
      },
    };
    const collectedUsage: UsageMetadata[] = [];
    const emittedEvents: Array<{ event: string; data: unknown }> = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-librechat-callbacks',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.DEEPSEEK,
          streamUsage: false,
        },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createLibreChatLikeHandlers({
        aggregateContent,
        collectedUsage,
        emittedEvents,
      }),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new CallbackStreamingReasoningModel([
      createReasoningChunk('reasoning_content', reasoningText.slice(0, 8)),
      createReasoningChunk('reasoning_content', reasoningText.slice(8)),
      new AIMessageChunk({
        content: text,
        usage_metadata: usage,
      }),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('stream with LibreChat handlers')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-librechat-callbacks',
        },
      }
    );

    const countEvents = (event: GraphEvents): number =>
      emittedEvents.filter((entry) => entry.event === event).length;

    expect(countEvents(GraphEvents.ON_REASONING_DELTA)).toBe(2);
    expect(countEvents(GraphEvents.ON_MESSAGE_DELTA)).toBe(1);
    expect(countEvents(GraphEvents.CHAT_MODEL_END)).toBe(1);
    expect(collectedUsage).toHaveLength(1);
    expect(collectedUsage[0]).toMatchObject(usage);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
  });
});
