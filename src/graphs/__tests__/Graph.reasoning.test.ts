import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type {
  BaseMessage,
  MessageContent,
  UsageMetadata,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { OpenAIClient } from '@langchain/openai';
import type * as t from '@/types';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
} from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ModelEndHandler, ToolEndHandler } from '@/events';
import { toLangChainContent } from '@/messages/langchain';
import { filterToolsForCurrentTurn } from '../Graph';
import { ChatOpenRouter } from '@/llm/openrouter';
import { ToolNode } from '@/tools/ToolNode';
import { Run } from '@/run';

type ReasoningKey = 'reasoning_content' | 'reasoning';
type StreamingCompletionBackedModel = {
  completions: {
    completionWithRetry: () => Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    >;
  };
};

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

class CapturingMessageModel implements t.ChatModel {
  readonly invocations: BaseMessage[][] = [];

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.invocations.push(messages);
    return new AIMessageChunk('ok');
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

function createOpenRouterStreamChunk(
  content: string,
  finishReason: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['finish_reason'] = null
): OpenAIClient.Chat.Completions.ChatCompletionChunk {
  return {
    id: 'chatcmpl-openrouter-test',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'google/gemini-test',
    choices: [
      {
        index: 0,
        delta: { content },
        finish_reason: finishReason,
      },
    ],
  };
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

function createOpenAIReasoningSummaryChunk(
  reasoningText: string
): AIMessageChunk {
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
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
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

  it('emits final text from invoke-only Gemini server-side context responses', async () => {
    const text = 'Search complete.';
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-gemini-server-side-context',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.GOOGLE,
          disableStreaming: true,
          streamUsage: false,
        },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createReasoningHandlers(aggregateContent, []),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const messageContent: t.MessageContentComplex[] = [
      {
        type: 'toolCall',
        toolCall: {
          id: 'server-search-1',
          name: 'google_search',
          args: {},
        },
      },
      { type: ContentTypes.TEXT, text },
      {
        type: 'toolResponse',
        toolResponse: {
          id: 'server-search-1',
          name: 'google_search',
          response: { results: [] },
        },
      },
    ];
    run.Graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({
        content: toLangChainContent(messageContent),
      })
    );

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('search and answer')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-gemini-server-side-context',
        },
      }
    );

    expect(finalContentParts).toEqual(messageContent);
    expect(contentParts).toEqual(messageContent);
  });

  it('does not preserve Gemini server-side context blocks for non-Google invoke fallbacks', async () => {
    const text = 'Search complete.';
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-non-google-server-side-shape',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: createReasoningHandlers(aggregateContent, []),
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const messageContent: t.MessageContentComplex[] = [
      {
        type: 'toolCall',
        toolCall: {
          id: 'server-search-1',
          name: 'google_search',
          args: {},
        },
      },
      { type: ContentTypes.TEXT, text },
      {
        type: 'toolResponse',
        toolResponse: {
          id: 'server-search-1',
          name: 'google_search',
          response: { results: [] },
        },
      },
    ];
    run.Graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({
        content: toLangChainContent(messageContent),
      })
    );

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('search and answer')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-non-google-server-side-shape',
        },
      }
    );

    expect(finalContentParts).toEqual(messageContent);
    expect(contentParts).toEqual([]);
  });

  it('uses the final fallback for one-shot disableStreaming mixed chunks', async () => {
    const text = 'Cloudflare content check.';
    const reasoningText = 'Plan the exact final wording.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-disable-streaming-mixed-final-chunk',
      graphConfig: {
        type: 'standard',
        llmConfig,
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
      new AIMessageChunk({
        content: text,
        additional_kwargs: {
          reasoning_content: reasoningText,
        },
      }),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('return mixed final chunk')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-disable-streaming-mixed-final-chunk',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(1);
    expect(messageDeltas).toHaveLength(1);
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

  it('streams OpenRouter reasoning_content before visible text', async () => {
    const text = '391.';
    const reasoningText = 'Use the difference of squares.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-reasoning-content-stream',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      createReasoningChunk('reasoning_content', reasoningText),
      new AIMessageChunk({ content: text }),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('stream OpenRouter reasoning_content')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-reasoning-content-stream',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(1);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
  });

  it('does not replay streamed OpenRouter text when the final chunk has reasoning_details', async () => {
    const text = '391.';
    const reasoningText = '17 times 23 equals 391.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-final-details-stream',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      new AIMessageChunk({
        content: text,
        additional_kwargs: {
          reasoning_details: [{ type: 'reasoning.text', text: reasoningText }],
        },
      }),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('think briefly, what is 17 x 23?')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-final-details-stream',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(1);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
  });

  it('does not keep an OpenRouter streamed text replay before final reasoning_details fallback', async () => {
    const text = '391.';
    const reasoningText = '17 times 23 equals 391.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-streamed-text-final-details',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      new AIMessageChunk({ content: text }),
      new AIMessageChunk({
        content: text,
        additional_kwargs: {
          reasoning_details: [{ type: 'reasoning.text', text: reasoningText }],
        },
      }),
    ]);

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('think briefly, what is 17 x 23?')] },
      {
        ...config,
        configurable: {
          thread_id:
            'reasoning-fallback-openrouter-streamed-text-final-details',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([{ type: ContentTypes.TEXT, text }]);
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
    expect(run.getRunMessages()?.[0]?.content).toBe(text);
  });

  it('does not replay streamed text when late OpenRouter reasoning_content arrives', async () => {
    const text = '391.';
    const reasoningText = 'Confirm the arithmetic after visible text.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-text-before-reasoning-content',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      new AIMessageChunk({ content: text }),
      createReasoningChunk('reasoning_content', reasoningText),
    ]);

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('think briefly, what is 17 x 23?')] },
      {
        ...config,
        configurable: {
          thread_id:
            'reasoning-fallback-openrouter-text-before-reasoning-content',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([{ type: ContentTypes.TEXT, text }]);
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
    expect(run.getRunMessages()?.[0]?.content).toBe(text);
  });

  it('does not replay streamed text when the final fallback is on the reasoning key', async () => {
    const text = '391.';
    const reasoningText = 'Confirm the arithmetic after visible text.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-text-before-reasoning-key',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      new AIMessageChunk({ content: text }),
      createReasoningChunk('reasoning', reasoningText),
    ]);

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('think briefly, what is 17 x 23?')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-text-before-reasoning-key',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([{ type: ContentTypes.TEXT, text }]);
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text },
    ]);
    expect(run.getRunMessages()?.[0]?.content).toBe(text);
  });

  it('keeps new OpenRouter final text suffixes that carry reasoning_details', async () => {
    const firstText = 'Hello ';
    const replayWithSuffix = 'Hello world';
    const reasoningText = 'The greeting needs one more word.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-final-suffix-details',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
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
      new AIMessageChunk({ content: firstText }),
      new AIMessageChunk({
        content: replayWithSuffix,
        additional_kwargs: {
          reasoning_details: [{ type: 'reasoning.text', text: reasoningText }],
        },
      }),
    ]);

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('finish the greeting')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-final-suffix-details',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(2);
    expect(messageDeltas[1].delta.content?.[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'world',
    });
    expect(contentParts).toEqual([
      { type: ContentTypes.TEXT, text: replayWithSuffix },
    ]);
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text: replayWithSuffix },
    ]);
    expect(run.getRunMessages()?.[0]?.content).toBe(replayWithSuffix);
  });

  it('keeps post-tool final text when an earlier invocation streamed text', async () => {
    const streamedBeforeTool = 'I will check that. ';
    const finalAfterTool = 'The answer is 391.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-post-tool-final-text',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          disableStreaming: true,
          streamUsage: false,
        },
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

    const graph = run.Graph;
    const metadata = {
      thread_id: 'reasoning-fallback-post-tool-final-text',
      langgraph_node: 'agent=default',
      langgraph_step: 1,
      checkpoint_ns: '',
    };
    const runConfig = {
      ...config,
      configurable: {
        thread_id: 'reasoning-fallback-post-tool-final-text',
      },
      metadata,
    };
    graph.config = runConfig;
    const streamedStepKey = graph.getStepKey(metadata);
    await graph.dispatchRunStep(
      streamedStepKey,
      {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'msg-before-tool' },
      },
      metadata
    );
    await graph.dispatchMessageDelta(
      graph.getStepIdByKey(streamedStepKey),
      {
        content: [
          {
            type: ContentTypes.TEXT,
            text: streamedBeforeTool,
          },
        ],
      },
      metadata
    );

    graph.invokedToolIds = new Set(['call_calculator']);
    graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({ content: finalAfterTool })
    );

    await graph.createCallModel('default')(
      { messages: [new HumanMessage('continue after tool')] },
      runConfig
    );

    expect(messageDeltas.map((delta) => delta.delta.content?.[0])).toEqual([
      {
        type: ContentTypes.TEXT,
        text: streamedBeforeTool,
      },
      {
        type: ContentTypes.TEXT,
        text: finalAfterTool,
      },
    ]);
  });

  it('does not replay streamed text block variants from the final fallback', async () => {
    const text = 'Variant text.';
    const textDeltaContent: t.MessageDelta['content'] = [
      { type: 'text_delta', text },
    ];
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-text-delta-block',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          disableStreaming: true,
          streamUsage: false,
        },
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

    const graph = run.Graph;
    const metadata = {
      thread_id: 'reasoning-fallback-text-delta-block',
      langgraph_node: 'agent=default',
      langgraph_step: 1,
      checkpoint_ns: '',
    };
    const runConfig = {
      ...config,
      configurable: {
        thread_id: 'reasoning-fallback-text-delta-block',
      },
      metadata,
    };
    graph.config = runConfig;
    const stepKey = graph.getStepKey(metadata);
    await graph.dispatchRunStep(
      stepKey,
      {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: 'msg-text-delta-block' },
      },
      metadata
    );
    await graph.dispatchMessageDelta(
      graph.getStepIdByKey(stepKey),
      { content: textDeltaContent },
      metadata
    );

    graph.overrideModel = new InvokeOnlyMessageModel(
      new AIMessageChunk({ content: textDeltaContent as never })
    );

    await graph.createCallModel('default')(
      { messages: [new HumanMessage('finish text delta block')] },
      runConfig
    );

    expect(messageDeltas.map((delta) => delta.delta.content?.[0])).toEqual([
      { type: 'text_delta', text },
    ]);
  });

  it('sanitizes OpenRouter final reasoning chunks for registered stream handlers', async () => {
    const firstText = 'Hello ';
    const replayWithSuffix = 'Hello world';
    const reasoningText = 'The greeting needs one more word.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { aggregateContent } = createContentAggregator();
    const streamHandler = new ChatModelStreamHandler();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-registered-handler-suffix',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: streamHandler,
        ...createReasoningHandlers(
          aggregateContent,
          reasoningDeltas,
          messageDeltas
        ),
      },
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    run.Graph.overrideModel = new StreamingReasoningModel([
      new AIMessageChunk({ content: firstText }),
      new AIMessageChunk({
        content: replayWithSuffix,
        additional_kwargs: {
          reasoning_details: [{ type: 'reasoning.text', text: reasoningText }],
        },
      }),
    ]);

    const finalContentParts = await run.processStream(
      { messages: [new HumanMessage('stream OpenRouter suffix once')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-registered-handler-suffix',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas.map((delta) => delta.delta.content?.[0])).toEqual([
      { type: ContentTypes.TEXT, text: 'world' },
    ]);
    expect(finalContentParts).toEqual([
      { type: ContentTypes.THINK, think: reasoningText },
      { type: ContentTypes.TEXT, text: replayWithSuffix },
    ]);
    expect(run.getRunMessages()?.[0]?.content).toBe(replayWithSuffix);
  });

  it('composes observer chat stream handlers with default graph dispatch', async () => {
    const firstText = 'Vis';
    const secondText = 'ible answer.';
    const observedTextChunks: string[] = [];
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-observer-stream',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            const content = (data.chunk as AIMessageChunk | undefined)?.content;
            if (typeof content === 'string' && content !== '') {
              observedTextChunks.push(content);
            }
          },
        },
        ...createReasoningHandlers(
          aggregateContent,
          reasoningDeltas,
          messageDeltas
        ),
      },
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const model = new ChatOpenRouter({
      model: 'google/gemini-test',
      apiKey: 'test-key',
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createOpenRouterStreamChunk(firstText);
      yield createOpenRouterStreamChunk(secondText, 'stop');
    }

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => streamChunks();
    run.Graph.overrideModel = model as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('stream OpenRouter text to observer')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-observer-stream',
        },
      }
    );

    expect(observedTextChunks).toEqual([firstText, secondText]);
    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(2);
    expect(contentParts).toEqual([
      { type: ContentTypes.TEXT, text: `${firstText}${secondText}` },
    ]);
  });

  it('does not handle OpenRouter callback-emitted chunks twice inside graph streaming', async () => {
    const text = 'Visible answer.';
    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();
    const streamHandler = new ChatModelStreamHandler();
    const run = await Run.create<t.IState>({
      runId: 'reasoning-fallback-openrouter-callback-yield-stream',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENROUTER,
          streamUsage: false,
        },
        reasoningKey: 'reasoning',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.CHAT_MODEL_STREAM]: streamHandler,
        ...createReasoningHandlers(
          aggregateContent,
          reasoningDeltas,
          messageDeltas
        ),
      },
    });

    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const model = new ChatOpenRouter({
      model: 'google/gemini-test',
      apiKey: 'test-key',
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createOpenRouterStreamChunk(text, 'stop');
    }

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => streamChunks();
    run.Graph.overrideModel = model as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('stream OpenRouter text once')] },
      {
        ...config,
        configurable: {
          thread_id: 'reasoning-fallback-openrouter-callback-yield-stream',
        },
      }
    );

    expect(reasoningDeltas).toHaveLength(0);
    expect(messageDeltas).toHaveLength(1);
    expect(contentParts).toEqual([{ type: ContentTypes.TEXT, text }]);
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

  it('keeps an image-only latest user turn as the exact message passed to the provider', async () => {
    const fiveImages: MessageContent = Array.from(
      { length: 5 },
      (_, index) => ({
        type: 'image_url',
        image_url: { url: `https://example.com/image-${index + 1}.png` },
      })
    );
    const older = new HumanMessage('older turn');
    const latest = new HumanMessage({ content: fiveImages });
    const run = await Run.create<t.IState>({
      runId: 'latest-image-turn-retained',
      graphConfig: { type: 'standard', llmConfig },
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const graph = run.Graph;
    const agentContext = graph.agentContexts.get('default');
    if (!agentContext) {
      throw new Error('Expected default agent context');
    }
    agentContext.pruneMessages = () => ({
      context: [latest],
      indexTokenCountMap: { 1: 5 },
      messagesToRefine: [older],
      prePruneContextTokens: 20,
      remainingContextTokens: 5,
      contextPressure: 0.8,
      calibrationRatio: 1,
      contextBudget: 25,
      effectiveInstructionTokens: 0,
    });
    const model = new CapturingMessageModel();
    graph.overrideModel = model;

    await graph.createCallModel('default')(
      { messages: [older, latest] },
      {
        ...config,
        metadata: {
          thread_id: 'latest-image-turn-retained',
          langgraph_node: 'agent=default',
          langgraph_step: 0,
          checkpoint_ns: '',
        },
      }
    );

    expect(model.invocations).toHaveLength(1);
    expect(model.invocations[0]).toHaveLength(1);
    expect(model.invocations[0][0]).toBe(latest);
    expect(model.invocations[0][0].content).toBe(fiveImages);
    expect(model.invocations[0][0].content).toHaveLength(5);
  });

  it('fails before provider invocation when pruning loses the latest user turn', async () => {
    const older = new HumanMessage('older turn');
    const latest = new HumanMessage({
      content: [
        {
          type: 'image_url',
          image_url: { url: 'https://example.com/latest.png' },
        },
      ],
    });
    const run = await Run.create<t.IState>({
      runId: 'latest-image-turn-lost',
      graphConfig: { type: 'standard', llmConfig },
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const graph = run.Graph;
    const agentContext = graph.agentContexts.get('default');
    if (!agentContext) {
      throw new Error('Expected default agent context');
    }
    agentContext.pruneMessages = () => ({
      context: [older],
      indexTokenCountMap: { 0: 5 },
      messagesToRefine: [latest],
      prePruneContextTokens: 20,
      remainingContextTokens: 5,
      contextPressure: 0.8,
      calibrationRatio: 1,
      contextBudget: 25,
      effectiveInstructionTokens: 0,
    });
    const model = new CapturingMessageModel();
    graph.overrideModel = model;

    await expect(
      graph.createCallModel('default')(
        { messages: [older, latest] },
        {
          ...config,
          metadata: {
            thread_id: 'latest-image-turn-lost',
            langgraph_node: 'agent=default',
            langgraph_step: 0,
            checkpoint_ns: '',
          },
        }
      )
    ).rejects.toThrow('current_turn_exceeds_context');
    expect(model.invocations).toHaveLength(0);
  });

  it('removes only explicitly disabled tools from the next current-turn binding', () => {
    const tools = [
      { name: 'file_search' },
      { name: 'calculator' },
    ] as t.GraphTools;
    const result = filterToolsForCurrentTurn(
      [
        new HumanMessage('search these files'),
        new ToolMessage({
          content: 'search budget exhausted',
          tool_call_id: 'file-search-1',
          name: 'file_search',
          artifact: {
            toolControl: { disableTools: ['file_search'] },
          },
        }),
      ],
      tools
    );

    expect(
      (result as Array<{ name?: string }>).map((tool) => tool.name)
    ).toEqual(['calculator']);
  });

  it('retains tool controls from a content_and_artifact ToolNode result', async () => {
    const fileSearch = tool(
      async () => [
        'search budget exhausted',
        { toolControl: { disableTools: ['file_search'] } },
      ],
      {
        name: 'file_search',
        description: 'Search files',
        schema: z.object({}),
        responseFormat: Constants.CONTENT_AND_ARTIFACT,
      }
    );
    const toolNode = new ToolNode({ tools: [fileSearch] });
    const result = (await toolNode.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [{ id: 'file-search-1', name: 'file_search', args: {} }],
        }),
      ],
    })) as { messages: ToolMessage[] };

    expect(result.messages[0].artifact).toEqual({
      toolControl: { disableTools: ['file_search'] },
    });
    expect(
      filterToolsForCurrentTurn(
        [new HumanMessage('search'), ...result.messages],
        [{ name: 'file_search' }] as t.GraphTools
      )
    ).toHaveLength(0);
  });

  it('ignores malformed tool-control artifacts', () => {
    const tools = [{ name: 'file_search' }] as t.GraphTools;
    const result = filterToolsForCurrentTurn(
      [
        new HumanMessage('search these files'),
        new ToolMessage({
          content: 'malformed control',
          tool_call_id: 'file-search-1',
          name: 'file_search',
          artifact: { toolControl: { disableTools: 'file_search' } },
        }),
      ],
      tools
    );

    expect(result).toBe(tools);
  });

  it('does not carry a tool-control artifact into a newer user turn', () => {
    const tools = [{ name: 'file_search' }] as t.GraphTools;
    const result = filterToolsForCurrentTurn(
      [
        new HumanMessage('first request'),
        new ToolMessage({
          content: 'search budget exhausted',
          tool_call_id: 'file-search-1',
          name: 'file_search',
          artifact: {
            toolControl: { disableTools: ['file_search'] },
          },
        }),
        new HumanMessage('new request'),
      ],
      tools
    );

    expect(result).toBe(tools);
  });
});
