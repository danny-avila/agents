// src/run.ts
import { PromptTemplate } from '@langchain/core/prompts';
import { AzureChatOpenAI, ChatOpenAI } from '@langchain/openai';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { ClientCallbacks, SystemCallbacks } from '@/graphs/Graph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Callback, TitleMethod } from '@/common';
import { createTokenCounter } from '@/utils/tokens';
import {
  createCompletionTitleRunnable,
  createTitleRunnable,
} from '@/utils/title';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { StandardGraph } from '@/graphs/Graph';
import { HandlerRegistry } from '@/events';
import { isOpenAILike } from '@/utils/llm';

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
  private handlerRegistry: HandlerRegistry;
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
      // Default to legacy graph for 'standard' or undefined type
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
    config: t.LegacyGraphConfig
  ): t.CompiledStateWorkflow {
    const {
      type: _type,
      llmConfig,
      signal,
      tools = [],
      ...agentInputs
    } = config;
    const { provider, ...clientOptions } = llmConfig;

    /** TEMP: Create agent configuration for the single agent */
    const agentConfig: t.AgentInputs = {
      ...agentInputs,
      tools,
      provider,
      clientOptions,
      agentId: 'default',
    };

    const standardGraph = new StandardGraph({
      signal,
      runId: this.id,
      agents: [agentConfig],
      tokenCounter: this.tokenCounter,
      indexTokenCountMap: this.indexTokenCountMap,
    });
    // propagate compile options from graph config
    standardGraph.compileOptions = (
      config as t.LegacyGraphConfig
    ).compileOptions;
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
    // Create tokenCounter if indexTokenCountMap is provided but tokenCounter is not
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

    if (streamOptions?.callbacks) {
      /* TODO: conflicts with callback manager */
      const callbacks = (config.callbacks as t.ProvidedCallbacks) ?? [];
      config.callbacks = callbacks.concat(
        this.getCallbacks(streamOptions.callbacks)
      );
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
    });

    for await (const event of stream) {
      const { data, name, metadata, ...info } = event;

      let eventName: t.EventName = info.event;
      // First normalize custom events to their named variant
      if (eventName && eventName === GraphEvents.ON_CUSTOM_EVENT) {
        eventName = name;
      }
      /**
       * Suppress CHAT_MODEL_STREAM if custom event, meaning the provider
       * is in `manualToolStreamProviders` and `tools` are present, which
       * creates a double-call of the event.
       */
      if (
        (data as t.StreamEventData)['emitted'] === true &&
        eventName === GraphEvents.CHAT_MODEL_STREAM
      ) {
        continue;
      }

      const handler = this.handlerRegistry.getHandler(eventName);
      if (handler) {
        handler.handle(eventName, data, metadata, this.Graph);
      }
    }

    if (this.returnContent) {
      return this.Graph.getContentParts();
    }
  }

  private createSystemCallback<K extends keyof ClientCallbacks>(
    clientCallbacks: ClientCallbacks,
    key: K
  ): SystemCallbacks[K] {
    return ((...args: unknown[]) => {
      const clientCallback = clientCallbacks[key];
      if (clientCallback && this.Graph) {
        (clientCallback as (...args: unknown[]) => void)(this.Graph, ...args);
      }
    }) as SystemCallbacks[K];
  }

  getCallbacks(clientCallbacks: ClientCallbacks): SystemCallbacks {
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
    const convoTemplate = PromptTemplate.fromTemplate(
      titlePromptTemplate ?? 'User: {input}\nAI: {output}'
    );
    const response = contentParts
      .map((part) => {
        if (part?.type === 'text') return part.text;
        return '';
      })
      .join('\n');
    const convo = (
      await convoTemplate.invoke({ input: inputText, output: response })
    ).value;
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
    const chain =
      titleMethod === TitleMethod.COMPLETION
        ? await createCompletionTitleRunnable(model, titlePrompt)
        : await createTitleRunnable(model, titlePrompt);
    const invokeConfig = Object.assign({}, chainOptions, {
      run_id: this.id,
      runId: this.id,
    });
    try {
      return await chain.invoke(
        { convo, inputText, skipLanguage },
        invokeConfig
      );
    } catch (_e) {
      // Fallback: strip callbacks to avoid EventStream tracer errors in certain environments
      const { callbacks: _cb, ...rest } = invokeConfig;
      const safeConfig = Object.assign({}, rest, { callbacks: [] });
      return await chain.invoke(
        { convo, inputText, skipLanguage },
        safeConfig as Partial<RunnableConfig>
      );
    }
  }
}
