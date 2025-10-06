// src/run.ts
import { zodToJsonSchema } from 'zod-to-json-schema';
import { PromptTemplate } from '@langchain/core/prompts';
import { SystemMessage } from '@langchain/core/messages';
import { AzureChatOpenAI, ChatOpenAI } from '@langchain/openai';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { ClientCallbacks, SystemCallbacks } from '@/graphs/Graph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Providers, Callback, TitleMethod } from '@/common';
import { manualToolStreamProviders } from '@/llm/providers';
import { shiftIndexTokenCountMap } from '@/messages/format';
import {
  createTitleRunnable,
  createCompletionTitleRunnable,
} from '@/utils/title';
import { createTokenCounter } from '@/utils/tokens';
import { StandardGraph } from '@/graphs/Graph';
import { HandlerRegistry } from '@/events';
import { isOpenAILike, validateClientOptions } from '@/utils/llm'; // 🔥 added validateClientOptions

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

export class Run<T extends t.BaseGraphState> {
  graphRunnable?: t.CompiledWorkflow<T, Partial<T>, string>;
  // private collab!: CollabGraph;
  // private taskManager!: TaskManager;
  private handlerRegistry: HandlerRegistry;
  id: string;
  Graph: StandardGraph | undefined;
  provider: Providers | undefined;
  returnContent: boolean = false;

  private constructor(config: Partial<t.RunConfig>) {
    const runId = config.runId ?? '';
    if (!runId) {
      throw new Error('Run ID not provided');
    }

    this.id = runId;

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

    if (config.graphConfig.type === 'standard' || !config.graphConfig.type) {
      this.provider = config.graphConfig.llmConfig.provider;
      this.graphRunnable = this.createStandardGraph(
        config.graphConfig
      ) as unknown as t.CompiledWorkflow<T, Partial<T>, string>;
      if (this.Graph) {
        this.Graph.handlerRegistry = handlerRegistry;
      }
    }

    this.returnContent = config.returnContent ?? false;
  }

  private createStandardGraph(
    config: t.StandardGraphConfig
  ): t.CompiledWorkflow<t.IState, Partial<t.IState>, string> {
    const { llmConfig, tools = [], ...graphInput } = config;
    const { provider, ...clientOptions } = llmConfig;

    // 🔥 Validate client options before creating the graph
    const validationErrors = validateClientOptions(clientOptions, provider);
    if (validationErrors.length > 0) {
      throw new Error(
        `Invalid client options for provider ${provider}: ${validationErrors.join(
          ', '
        )}`
      );
    }

    const standardGraph = new StandardGraph({
      tools,
      provider,
      clientOptions,
      ...graphInput,
      runId: this.id,
    });
    this.Graph = standardGraph;
    return standardGraph.createWorkflow();
  }

  static async create<T extends t.BaseGraphState>(
    config: t.RunConfig
  ): Promise<Run<T>> {
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
    if (!this.graphRunnable) {
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
    const provider = this.Graph.provider;
    const hasTools = this.Graph.tools ? this.Graph.tools.length > 0 : false;
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

    const tokenCounter =
      streamOptions?.tokenCounter ??
      (streamOptions?.indexTokenCountMap
        ? await createTokenCounter()
        : undefined);
    const tools = this.Graph.tools as
      | Array<t.GenericTool | undefined>
      | undefined;
    const toolTokens = tokenCounter
      ? (tools?.reduce((acc, tool) => {
        if (!(tool as Partial<t.GenericTool>).schema) {
          return acc;
        }

        const jsonSchema = zodToJsonSchema(
          (tool?.schema as t.ZodObjectAny).describe(tool?.description ?? ''),
          tool?.name ?? ''
        );
        return (
          acc + tokenCounter(new SystemMessage(JSON.stringify(jsonSchema)))
        );
      }, 0) ?? 0)
      : 0;
    let instructionTokens = toolTokens;
    if (this.Graph.systemMessage && tokenCounter) {
      instructionTokens += tokenCounter(this.Graph.systemMessage);
    }
    const tokenMap = streamOptions?.indexTokenCountMap ?? {};
    if (this.Graph.systemMessage && instructionTokens > 0) {
      this.Graph.indexTokenCountMap = shiftIndexTokenCountMap(
        tokenMap,
        instructionTokens
      );
    } else if (instructionTokens > 0) {
      tokenMap[0] = tokenMap[0] + instructionTokens;
      this.Graph.indexTokenCountMap = tokenMap;
    } else {
      this.Graph.indexTokenCountMap = tokenMap;
    }

    this.Graph.maxContextTokens = streamOptions?.maxContextTokens;
    this.Graph.tokenCounter = tokenCounter;

    config.run_id = this.id;
    config.configurable = Object.assign(config.configurable ?? {}, {
      run_id: this.id,
      provider: this.provider,
    });

    const stream = this.graphRunnable.streamEvents(inputs, config, {
      raiseError: true,
    });

    for await (const event of stream) {
      const { data, name, metadata, ...info } = event;

      let eventName: t.EventName = info.event;
      if (
        hasTools &&
        manualToolStreamProviders.has(provider) &&
        eventName === GraphEvents.CHAT_MODEL_STREAM
      ) {
        /* Skipping CHAT_MODEL_STREAM event due to double-call edge case */
        continue;
      }

      if (eventName && eventName === GraphEvents.ON_CUSTOM_EVENT) {
        eventName = name;
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
    omitOptions = defaultOmitOptions,
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
      omitOptions,
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
    return await chain.invoke({ convo, inputText, skipLanguage }, chainOptions);
  }
}
