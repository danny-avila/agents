// src/run.ts
import './instrumentation';
import { CallbackHandler } from '@langfuse/langchain';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';
import { AzureChatOpenAI, ChatOpenAI } from '@langchain/openai';
import type {
  MessageContentComplex,
  BaseMessage,
} from '@langchain/core/messages';
import type { StringPromptValue } from '@langchain/core/prompt_values';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import {
  createCompletionTitleRunnable,
  createTitleRunnable,
} from '@/utils/title';
import { GraphEvents, Callback, TitleMethod } from '@/common';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { createTokenCounter } from '@/utils/tokens';
import { StandardGraph } from '@/graphs/Graph';
import { HandlerRegistry } from '@/events';
import { isOpenAILike } from '@/utils/llm';
import { isPresent } from '@/utils/misc';

export const defaultOmitOptions = new Set([
  'stream',
  'thinking',
  'streaming',
  'maxTokens',
  'clientOptions',
  'thinkingConfig',
  'thinkingBudget',
  'includeThoughts',
  'maxOutputTokens',
  'additionalModelRequestFields',
]);

export class Run<_T extends t.BaseGraphState> {
  id: string;
  private tokenCounter?: t.TokenCounter;
  private handlerRegistry?: HandlerRegistry;
  private indexTokenCountMap?: Record<string, number>;
  graphRunnable?: t.CompiledStateWorkflow;
  Graph: StandardGraph | MultiAgentGraph | undefined;
  returnContent: boolean = false;

  private constructor(config: Partial<t.RunConfig>) {
    const runId = config.runId ?? '';
    if (!runId) {
      throw new Error('Run ID not provided');
    }

    this.id = runId;
    this.tokenCounter = config.tokenCounter;
    this.indexTokenCountMap = config.indexTokenCountMap;

    const handlerRegistry = new HandlerRegistry();

    if (config.customHandlers) {
      for (const [eventType, handler] of Object.entries(
        config.customHandlers
      )) {
        handlerRegistry.register(eventType, handler);
      }
    }

    this.handlerRegistry = handlerRegistry;

    if (!config.graphConfig) {
      throw new Error('Graph config not provided');
    }

    /** Handle different graph types */
    if (config.graphConfig.type === 'multi-agent') {
      this.graphRunnable = this.createMultiAgentGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.handlerRegistry = handlerRegistry;
      }
    } else {
      /** Default to legacy graph for 'standard' or undefined type */
      this.graphRunnable = this.createLegacyGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.compileOptions =
          config.graphConfig.compileOptions ?? this.Graph.compileOptions;
        this.Graph.handlerRegistry = handlerRegistry;
      }
    }

    this.returnContent = config.returnContent ?? false;
  }

  private createLegacyGraph(
    config: t.LegacyGraphConfig | t.StandardGraphConfig
  ): t.CompiledStateWorkflow {
    let agentConfig: t.AgentInputs;
    let signal: AbortSignal | undefined;

    /** Check if this is a multi-agent style config (has agents array) */
    if ('agents' in config && Array.isArray(config.agents)) {
      if (config.agents.length === 0) {
        throw new Error('At least one agent must be provided');
      }
      agentConfig = config.agents[0];
      signal = config.signal;
    } else {
      /** Legacy path: build agent config from llmConfig */
      const {
        type: _type,
        llmConfig,
        signal: legacySignal,
        tools = [],
        ...agentInputs
      } = config as t.LegacyGraphConfig;
      const { provider, ...clientOptions } = llmConfig;

      agentConfig = {
        ...agentInputs,
        tools,
        provider,
        clientOptions,
        agentId: 'default',
      };
      signal = legacySignal;
    }

    const standardGraph = new StandardGraph({
      signal,
      runId: this.id,
      agents: [agentConfig],
      tokenCounter: this.tokenCounter,
      indexTokenCountMap: this.indexTokenCountMap,
    });
    /** Propagate compile options from graph config */
    standardGraph.compileOptions = config.compileOptions;
    this.Graph = standardGraph;
    return standardGraph.createWorkflow();
  }

  private createMultiAgentGraph(
    config: t.MultiAgentGraphConfig
  ): t.CompiledStateWorkflow {
    const { agents, edges, compileOptions } = config;

    const multiAgentGraph = new MultiAgentGraph({
      runId: this.id,
      agents,
      edges,
      tokenCounter: this.tokenCounter,
      indexTokenCountMap: this.indexTokenCountMap,
    });

    if (compileOptions != null) {
      multiAgentGraph.compileOptions = compileOptions;
    }

    this.Graph = multiAgentGraph;
    return multiAgentGraph.createWorkflow();
  }

  static async create<T extends t.BaseGraphState>(
    config: t.RunConfig
  ): Promise<Run<T>> {
    /** Create tokenCounter if indexTokenCountMap is provided but tokenCounter is not */
    if (config.indexTokenCountMap && !config.tokenCounter) {
      config.tokenCounter = await createTokenCounter();
    }
    return new Run<T>(config);
  }

  getRunMessages(): BaseMessage[] | undefined {
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    return this.Graph.getRunMessages();
  }

  /**
   * Creates a custom event callback handler that intercepts custom events
   * and processes them through our handler registry instead of EventStreamCallbackHandler
   */
  private createCustomEventCallback() {
    return async (
      eventName: string,
      data: unknown,
      runId: string,
      tags?: string[],
      metadata?: Record<string, unknown>
    ): Promise<void> => {
      if (
        (data as t.StreamEventData)['emitted'] === true &&
        eventName === GraphEvents.CHAT_MODEL_STREAM
      ) {
        return;
      }
      const handler = this.handlerRegistry?.getHandler(eventName);
      if (handler && this.Graph) {
        return await handler.handle(
          eventName,
          data as
            | t.StreamEventData
            | t.ModelEndData
            | t.RunStep
            | t.RunStepDeltaEvent
            | t.MessageDeltaEvent
            | t.ReasoningDeltaEvent
            | { result: t.ToolEndEvent },
          metadata,
          this.Graph
        );
      }
    };
  }

  async processStream(
    inputs: t.IState,
    config: Partial<RunnableConfig> & { version: 'v1' | 'v2'; run_id?: string },
    streamOptions?: t.EventStreamOptions
  ): Promise<MessageContentComplex[] | undefined> {
    if (this.graphRunnable == null) {
      throw new Error(
        'Run not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }

    this.Graph.resetValues(streamOptions?.keepContent);

    /** Custom event callback to intercept and handle custom events */
    const customEventCallback = this.createCustomEventCallback();

    const baseCallbacks = (config.callbacks as t.ProvidedCallbacks) ?? [];
    const streamCallbacks = streamOptions?.callbacks
      ? this.getCallbacks(streamOptions.callbacks)
      : [];

    config.callbacks = baseCallbacks.concat(streamCallbacks).concat({
      [Callback.CUSTOM_EVENT]: customEventCallback,
    });

    if (
      isPresent(process.env.LANGFUSE_SECRET_KEY) &&
      isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
      isPresent(process.env.LANGFUSE_BASE_URL)
    ) {
      const userId = config.configurable?.user_id;
      const sessionId = config.configurable?.thread_id;
      const traceMetadata = {
        messageId: this.id,
        parentMessageId: config.configurable?.requestBody?.parentMessageId,
      };
      const handler = new CallbackHandler({
        userId,
        sessionId,
        traceMetadata,
      });
      config.callbacks = (
        (config.callbacks as t.ProvidedCallbacks) ?? []
      ).concat([handler]);
    }

    if (!this.id) {
      throw new Error('Run ID not provided');
    }

    config.run_id = this.id;
    config.configurable = Object.assign(config.configurable ?? {}, {
      run_id: this.id,
    });

    const stream = this.graphRunnable.streamEvents(inputs, config, {
      raiseError: true,
      /**
       * Prevent EventStreamCallbackHandler from processing custom events.
       * Custom events are already handled via our createCustomEventCallback()
       * which routes them through the handlerRegistry.
       * Without this flag, EventStreamCallbackHandler throws errors when
       * custom events are dispatched for run IDs not in its internal map
       * (due to timing issues in parallel execution or after run cleanup).
       */
      ignoreCustomEvent: true,
    });

    for await (const event of stream) {
      const { data, metadata, ...info } = event;

      const eventName: t.EventName = info.event;

      /** Skip custom events as they're handled by our callback */
      if (eventName === GraphEvents.ON_CUSTOM_EVENT) {
        continue;
      }

      const handler = this.handlerRegistry?.getHandler(eventName);
      if (handler) {
        await handler.handle(eventName, data, metadata, this.Graph);
      }
    }

    if (this.returnContent) {
      return this.Graph.getContentParts();
    }
  }

  private createSystemCallback<K extends keyof t.ClientCallbacks>(
    clientCallbacks: t.ClientCallbacks,
    key: K
  ): t.SystemCallbacks[K] {
    return ((...args: unknown[]) => {
      const clientCallback = clientCallbacks[key];
      if (clientCallback && this.Graph) {
        (clientCallback as (...args: unknown[]) => void)(this.Graph, ...args);
      }
    }) as t.SystemCallbacks[K];
  }

  getCallbacks(clientCallbacks: t.ClientCallbacks): t.SystemCallbacks {
    return {
      [Callback.TOOL_ERROR]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_ERROR
      ),
      [Callback.TOOL_START]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_START
      ),
      [Callback.TOOL_END]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_END
      ),
    };
  }

  async generateTitle({
    provider,
    inputText,
    contentParts,
    titlePrompt,
    clientOptions,
    chainOptions,
    skipLanguage,
    titleMethod = TitleMethod.COMPLETION,
    titlePromptTemplate,
  }: t.RunTitleOptions): Promise<{ language?: string; title?: string }> {
    if (
      chainOptions != null &&
      isPresent(process.env.LANGFUSE_SECRET_KEY) &&
      isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
      isPresent(process.env.LANGFUSE_BASE_URL)
    ) {
      const userId = chainOptions.configurable?.user_id;
      const sessionId = chainOptions.configurable?.thread_id;
      const traceMetadata = {
        messageId: 'title-' + this.id,
      };
      const handler = new CallbackHandler({
        userId,
        sessionId,
        traceMetadata,
      });
      chainOptions.callbacks = (
        (chainOptions.callbacks as t.ProvidedCallbacks) ?? []
      ).concat([handler]);
    }

    const convoTemplate = PromptTemplate.fromTemplate(
      titlePromptTemplate ?? 'User: {input}\nAI: {output}'
    );

    const response = contentParts
      .map((part) => {
        if (part?.type === 'text') return part.text;
        return '';
      })
      .join('\n');

    const model = this.Graph?.getNewModel({
      provider,
      clientOptions,
    });
    if (!model) {
      return { language: '', title: '' };
    }
    if (
      isOpenAILike(provider) &&
      (model instanceof ChatOpenAI || model instanceof AzureChatOpenAI)
    ) {
      model.temperature = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.temperature as number;
      model.topP = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.topP as number;
      model.frequencyPenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.frequencyPenalty as number;
      model.presencePenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.presencePenalty as number;
      model.n = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.n as number;
    }

    const convoToTitleInput = new RunnableLambda({
      func: (
        promptValue: StringPromptValue
      ): { convo: string; inputText: string; skipLanguage?: boolean } => ({
        convo: promptValue.value,
        inputText,
        skipLanguage,
      }),
    }).withConfig({ runName: 'ConvoTransform' });

    const titleChain =
      titleMethod === TitleMethod.COMPLETION
        ? await createCompletionTitleRunnable(model, titlePrompt)
        : await createTitleRunnable(model, titlePrompt);

    /** Pipes `convoTemplate` -> `transformer` -> `titleChain` */
    const fullChain = convoTemplate
      .withConfig({ runName: 'ConvoTemplate' })
      .pipe(convoToTitleInput)
      .pipe(titleChain)
      .withConfig({ runName: 'TitleChain' });

    const invokeConfig = Object.assign({}, chainOptions, {
      run_id: this.id,
      runId: this.id,
    });

    try {
      return await fullChain.invoke(
        { input: inputText, output: response },
        invokeConfig
      );
    } catch (_e) {
      // Fallback: strip callbacks to avoid EventStream tracer errors in certain environments
      // But preserve langfuse handler if it exists
      const langfuseHandler = (
        invokeConfig.callbacks as t.ProvidedCallbacks
      )?.find((cb) => cb instanceof CallbackHandler);
      const { callbacks: _cb, ...rest } = invokeConfig;
      const safeConfig = Object.assign({}, rest, {
        callbacks: langfuseHandler ? [langfuseHandler] : [],
      });
      return await fullChain.invoke(
        { input: inputText, output: response },
        safeConfig as Partial<RunnableConfig>
      );
    }
  }
}
