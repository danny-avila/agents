/* eslint-disable no-console */
// src/graphs/Graph.ts
import { nanoid } from 'nanoid';
import { concat } from '@langchain/core/utils/stream';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatVertexAI } from '@langchain/google-vertexai';
import { START, END, StateGraph } from '@langchain/langgraph';
import { Runnable, RunnableConfig } from '@langchain/core/runnables';
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import {
  AIMessageChunk,
  ToolMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type {
  BaseMessage,
  BaseMessageFields,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import {
  Providers,
  GraphEvents,
  GraphNodeKeys,
  StepTypes,
  Callback,
  ContentTypes,
} from '@/common';
import type { ToolCall } from '@langchain/core/messages/tool';
import { getChatModelClass, manualToolStreamProviders } from '@/llm/providers';
import { ToolNode as CustomToolNode, toolsCondition } from '@/tools/ToolNode';
import {
  createPruneMessages,
  modifyDeltaProperties,
  formatToolArrayContent,
  convertMessagesToContent,
  formatAnthropicArtifactContent,
} from '@/messages';
import {
  resetIfNotEmpty,
  isOpenAILike,
  isGoogleLike,
  joinKeys,
  sleep,
} from '@/utils';
import { ChatOpenAI, AzureChatOpenAI } from '@/llm/openai';
import { createFakeStreamingLLM } from '@/llm/fake';
import { HandlerRegistry } from '@/events';

const { AGENT, TOOLS } = GraphNodeKeys;
export type GraphNode = GraphNodeKeys | typeof START;
export type ClientCallback<T extends unknown[]> = (
  graph: StandardGraph,
  ...args: T
) => void;
export type ClientCallbacks = {
  [Callback.TOOL_ERROR]?: ClientCallback<[Error, string]>;
  [Callback.TOOL_START]?: ClientCallback<unknown[]>;
  [Callback.TOOL_END]?: ClientCallback<unknown[]>;
};
export type SystemCallbacks = {
  [K in keyof ClientCallbacks]: ClientCallbacks[K] extends ClientCallback<
    infer Args
  >
    ? (...args: Args) => void
    : never;
};

export abstract class Graph<
  T extends t.BaseGraphState = t.BaseGraphState,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  TNodeName extends string = string,
> {
  abstract resetValues(): void;
  abstract createGraphState(): t.GraphStateChannels<T>;
  abstract initializeTools(): CustomToolNode<T> | ToolNode<T>;
  abstract initializeModel(): Runnable;
  abstract getRunMessages(): BaseMessage[] | undefined;
  abstract getContentParts(): t.MessageContentComplex[] | undefined;
  abstract generateStepId(stepKey: string): [string, number];
  abstract getKeyList(
    metadata: Record<string, unknown> | undefined
  ): (string | number | undefined)[];
  abstract getStepKey(metadata: Record<string, unknown> | undefined): string;
  abstract checkKeyList(keyList: (string | number | undefined)[]): boolean;
  abstract getStepIdByKey(stepKey: string, index?: number): string;
  abstract getRunStep(stepId: string): t.RunStep | undefined;
  abstract dispatchRunStep(stepKey: string, stepDetails: t.StepDetails): string;
  abstract dispatchRunStepDelta(id: string, delta: t.ToolCallDelta): void;
  abstract dispatchMessageDelta(id: string, delta: t.MessageDelta): void;
  abstract dispatchReasoningDelta(
    stepId: string,
    delta: t.ReasoningDelta
  ): void;
  abstract handleToolCallCompleted(
    data: t.ToolEndData,
    metadata?: Record<string, unknown>,
    omitOutput?: boolean
  ): void;

  abstract createCallModel(): (
    state: T,
    config?: RunnableConfig
  ) => Promise<Partial<T>>;
  abstract createWorkflow(): t.CompiledWorkflow<T>;
  lastToken?: string;
  tokenTypeSwitch?: 'reasoning' | 'content';
  reasoningKey: 'reasoning_content' | 'reasoning' = 'reasoning_content';
  currentTokenType: ContentTypes.TEXT | ContentTypes.THINK | 'think_and_text' =
    ContentTypes.TEXT;
  messageStepHasToolCalls: Map<string, boolean> = new Map();
  messageIdsByStepKey: Map<string, string> = new Map();
  prelimMessageIdsByStepKey: Map<string, string> = new Map();
  config: RunnableConfig | undefined;
  contentData: t.RunStep[] = [];
  stepKeyIds: Map<string, string[]> = new Map<string, string[]>();
  contentIndexMap: Map<string, number> = new Map();
  toolCallStepIds: Map<string, string> = new Map();
  currentUsage: Partial<UsageMetadata> | undefined;
  indexTokenCountMap: Record<string, number | undefined> = {};
  maxContextTokens: number | undefined;
  pruneMessages?: ReturnType<typeof createPruneMessages>;
  /** The amount of time that should pass before another consecutive API call */
  streamBuffer: number | undefined;
  tokenCounter?: t.TokenCounter;
  signal?: AbortSignal;
  /** Set of invoked tool call IDs from non-message run steps completed mid-run, if any */
  invokedToolIds?: Set<string>;
  handlerRegistry: HandlerRegistry | undefined;
}

export class StandardGraph extends Graph<t.BaseGraphState, GraphNode> {
  private graphState: t.GraphStateChannels<t.BaseGraphState>;
  clientOptions: t.ClientOptions;
  boundModel?: Runnable;
  /** The last recorded timestamp that a stream API call was invoked */
  lastStreamCall: number | undefined;
  systemMessage: SystemMessage | undefined;
  messages: BaseMessage[] = [];
  runId: string | undefined;
  tools?: t.GraphTools;
  toolMap?: t.ToolMap;
  startIndex: number = 0;
  provider: Providers;
  toolEnd: boolean;
  signal?: AbortSignal;

  constructor({
    runId,
    tools,
    signal,
    toolMap,
    provider,
    streamBuffer,
    instructions,
    reasoningKey,
    clientOptions,
    toolEnd = false,
    additional_instructions = '',
  }: t.StandardGraphInput) {
    super();
    this.runId = runId;
    this.tools = tools;
    this.signal = signal;
    this.toolEnd = toolEnd;
    this.toolMap = toolMap;
    this.provider = provider;
    this.streamBuffer = streamBuffer;
    this.clientOptions = clientOptions;
    this.graphState = this.createGraphState();
    this.boundModel = this.initializeModel();
    if (reasoningKey) {
      this.reasoningKey = reasoningKey;
    }

    let finalInstructions: string | BaseMessageFields | undefined =
      instructions;
    if (additional_instructions) {
      finalInstructions =
        finalInstructions != null && finalInstructions
          ? `${finalInstructions}\n\n${additional_instructions}`
          : additional_instructions;
    }

    if (
      finalInstructions != null &&
      finalInstructions &&
      provider === Providers.ANTHROPIC &&
      ((
        (clientOptions as t.AnthropicClientOptions).clientOptions
          ?.defaultHeaders as Record<string, string> | undefined
      )?.['anthropic-beta']?.includes('prompt-caching') ??
        false)
    ) {
      finalInstructions = {
        content: [
          {
            type: 'text',
            text: instructions,
            cache_control: { type: 'ephemeral' },
          },
        ],
      };
    }

    if (finalInstructions != null && finalInstructions !== '') {
      this.systemMessage = new SystemMessage(finalInstructions);
    }
  }

  /* Init */

  resetValues(keepContent?: boolean): void {
    this.messages = [];
    this.config = resetIfNotEmpty(this.config, undefined);
    if (keepContent !== true) {
      this.contentData = resetIfNotEmpty(this.contentData, []);
      this.contentIndexMap = resetIfNotEmpty(this.contentIndexMap, new Map());
    }
    this.stepKeyIds = resetIfNotEmpty(this.stepKeyIds, new Map());
    this.toolCallStepIds = resetIfNotEmpty(this.toolCallStepIds, new Map());
    this.messageIdsByStepKey = resetIfNotEmpty(
      this.messageIdsByStepKey,
      new Map()
    );
    this.messageStepHasToolCalls = resetIfNotEmpty(
      this.prelimMessageIdsByStepKey,
      new Map()
    );
    this.prelimMessageIdsByStepKey = resetIfNotEmpty(
      this.prelimMessageIdsByStepKey,
      new Map()
    );
    this.currentTokenType = resetIfNotEmpty(
      this.currentTokenType,
      ContentTypes.TEXT
    );
    this.lastToken = resetIfNotEmpty(this.lastToken, undefined);
    this.tokenTypeSwitch = resetIfNotEmpty(this.tokenTypeSwitch, undefined);
    this.indexTokenCountMap = resetIfNotEmpty(this.indexTokenCountMap, {});
    this.currentUsage = resetIfNotEmpty(this.currentUsage, undefined);
    this.tokenCounter = resetIfNotEmpty(this.tokenCounter, undefined);
    this.maxContextTokens = resetIfNotEmpty(this.maxContextTokens, undefined);
    this.invokedToolIds = resetIfNotEmpty(this.invokedToolIds, undefined);
  }

  /* Run Step Processing */

  getRunStep(stepId: string): t.RunStep | undefined {
    const index = this.contentIndexMap.get(stepId);
    if (index !== undefined) {
      return this.contentData[index];
    }
    return undefined;
  }

  getStepKey(metadata: Record<string, unknown> | undefined): string {
    if (!metadata) return '';

    const keyList = this.getKeyList(metadata);
    if (this.checkKeyList(keyList)) {
      throw new Error('Missing metadata');
    }

    return joinKeys(keyList);
  }

  getStepIdByKey(stepKey: string, index?: number): string {
    const stepIds = this.stepKeyIds.get(stepKey);
    if (!stepIds) {
      throw new Error(`No step IDs found for stepKey ${stepKey}`);
    }

    if (index === undefined) {
      return stepIds[stepIds.length - 1];
    }

    return stepIds[index];
  }

  generateStepId(stepKey: string): [string, number] {
    const stepIds = this.stepKeyIds.get(stepKey);
    let newStepId: string | undefined;
    let stepIndex = 0;
    if (stepIds) {
      stepIndex = stepIds.length;
      newStepId = `step_${nanoid()}`;
      stepIds.push(newStepId);
      this.stepKeyIds.set(stepKey, stepIds);
    } else {
      newStepId = `step_${nanoid()}`;
      this.stepKeyIds.set(stepKey, [newStepId]);
    }

    return [newStepId, stepIndex];
  }

  getKeyList(
    metadata: Record<string, unknown> | undefined
  ): (string | number | undefined)[] {
    if (!metadata) return [];

    const keyList = [
      metadata.run_id as string,
      metadata.thread_id as string,
      metadata.langgraph_node as string,
      metadata.langgraph_step as number,
      metadata.checkpoint_ns as string,
    ];
    if (
      this.currentTokenType === ContentTypes.THINK ||
      this.currentTokenType === 'think_and_text'
    ) {
      keyList.push('reasoning');
    }

    if (this.invokedToolIds != null && this.invokedToolIds.size > 0) {
      keyList.push(this.invokedToolIds.size + '');
    }

    return keyList;
  }

  checkKeyList(keyList: (string | number | undefined)[]): boolean {
    return keyList.some((key) => key === undefined);
  }

  /* Misc.*/

  getRunMessages(): BaseMessage[] | undefined {
    return this.messages.slice(this.startIndex);
  }

  getContentParts(): t.MessageContentComplex[] | undefined {
    return convertMessagesToContent(this.messages.slice(this.startIndex));
  }

  /* Graph */

  createGraphState(): t.GraphStateChannels<t.BaseGraphState> {
    return {
      messages: {
        value: (x: BaseMessage[], y: BaseMessage[]): BaseMessage[] => {
          if (!x.length) {
            if (this.systemMessage) {
              x.push(this.systemMessage);
            }

            this.startIndex = x.length + y.length;
          }
          const current = x.concat(y);
          this.messages = current;
          return current;
        },
        default: () => [],
      },
    };
  }

  initializeTools():
    | CustomToolNode<t.BaseGraphState>
    | ToolNode<t.BaseGraphState> {
    // return new ToolNode<t.BaseGraphState>(this.tools);
    return new CustomToolNode<t.BaseGraphState>({
      tools: (this.tools as t.GenericTool[] | undefined) || [],
      toolMap: this.toolMap,
      toolCallStepIds: this.toolCallStepIds,
      errorHandler: (data, metadata) =>
        StandardGraph.handleToolCallErrorStatic(this, data, metadata),
    });
  }

  initializeModel(): Runnable {
    const ChatModelClass = getChatModelClass(this.provider);
    const model = new ChatModelClass(this.clientOptions);

    if (
      isOpenAILike(this.provider) &&
      (model instanceof ChatOpenAI || model instanceof AzureChatOpenAI)
    ) {
      model.temperature = (this.clientOptions as t.OpenAIClientOptions)
        .temperature as number;
      model.topP = (this.clientOptions as t.OpenAIClientOptions).topP as number;
      model.frequencyPenalty = (this.clientOptions as t.OpenAIClientOptions)
        .frequencyPenalty as number;
      model.presencePenalty = (this.clientOptions as t.OpenAIClientOptions)
        .presencePenalty as number;
      model.n = (this.clientOptions as t.OpenAIClientOptions).n as number;
    } else if (
      this.provider === Providers.VERTEXAI &&
      model instanceof ChatVertexAI
    ) {
      model.temperature = (this.clientOptions as t.VertexAIClientOptions)
        .temperature as number;
      model.topP = (this.clientOptions as t.VertexAIClientOptions)
        .topP as number;
      model.topK = (this.clientOptions as t.VertexAIClientOptions)
        .topK as number;
      model.topLogprobs = (this.clientOptions as t.VertexAIClientOptions)
        .topLogprobs as number;
      model.frequencyPenalty = (this.clientOptions as t.VertexAIClientOptions)
        .frequencyPenalty as number;
      model.presencePenalty = (this.clientOptions as t.VertexAIClientOptions)
        .presencePenalty as number;
      model.maxOutputTokens = (this.clientOptions as t.VertexAIClientOptions)
        .maxOutputTokens as number;
    }

    if (!this.tools || this.tools.length === 0) {
      return model as unknown as Runnable;
    }

    return (model as t.ModelWithTools).bindTools(this.tools);
  }
  overrideTestModel(
    responses: string[],
    sleep?: number,
    toolCalls?: ToolCall[]
  ): void {
    this.boundModel = createFakeStreamingLLM({
      responses,
      sleep,
      toolCalls,
    });
  }

  getNewModel({
    provider,
    clientOptions,
    omitOptions,
  }: {
    provider: Providers;
    clientOptions?: t.ClientOptions;
    omitOptions?: Set<string>;
  }): t.ChatModelInstance {
    const ChatModelClass = getChatModelClass(provider);
    const options =
      omitOptions && clientOptions == null
        ? Object.assign(
          Object.fromEntries(
            Object.entries(this.clientOptions).filter(
              ([key]) => !omitOptions.has(key)
            )
          ),
          clientOptions
        )
        : (clientOptions ?? this.clientOptions);
    return new ChatModelClass(options);
  }

  storeUsageMetadata(finalMessage?: BaseMessage): void {
    if (
      finalMessage &&
      'usage_metadata' in finalMessage &&
      finalMessage.usage_metadata != null
    ) {
      this.currentUsage = finalMessage.usage_metadata as Partial<UsageMetadata>;
    }
  }

  cleanupSignalListener(): void {
    if (!this.signal) {
      return;
    }
    if (!this.boundModel) {
      return;
    }
    const client = (this.boundModel as ChatOpenAI | undefined)?.exposedClient;
    if (!client?.abortHandler) {
      return;
    }
    this.signal.removeEventListener('abort', client.abortHandler);
    client.abortHandler = undefined;
  }

  createCallModel() {
    return async (
      state: t.BaseGraphState,
      config?: RunnableConfig
    ): Promise<Partial<t.BaseGraphState>> => {
      const { provider = '' } =
        (config?.configurable as t.GraphConfig | undefined) ?? {};
      if (this.boundModel == null) {
        throw new Error('No Graph model found');
      }
      if (!config || !provider) {
        throw new Error(`No ${config ? 'provider' : 'config'} provided`);
      }
      if (!config.signal) {
        config.signal = this.signal;
      }
      this.config = config;
      const { messages } = state;

      let messagesToUse = messages;
      if (
        !this.pruneMessages &&
        this.tokenCounter &&
        this.maxContextTokens != null &&
        this.indexTokenCountMap[0] != null
      ) {
        const isAnthropicWithThinking =
          (this.provider === Providers.ANTHROPIC &&
            (this.clientOptions as t.AnthropicClientOptions).thinking !=
              null) ||
          (this.provider === Providers.BEDROCK &&
            (this.clientOptions as t.BedrockAnthropicInput)
              .additionalModelRequestFields?.['thinking'] != null);

        this.pruneMessages = createPruneMessages({
          provider: this.provider,
          indexTokenCountMap: this.indexTokenCountMap,
          maxTokens: this.maxContextTokens,
          tokenCounter: this.tokenCounter,
          startIndex: this.startIndex,
          thinkingEnabled: isAnthropicWithThinking,
        });
      }
      if (this.pruneMessages) {
        const { context, indexTokenCountMap } = this.pruneMessages({
          messages,
          usageMetadata: this.currentUsage,
          // startOnMessageType: 'human',
        });
        this.indexTokenCountMap = indexTokenCountMap;
        messagesToUse = context;
      }

      const finalMessages = messagesToUse;
      const lastMessageX =
        finalMessages.length >= 2
          ? finalMessages[finalMessages.length - 2]
          : null;
      const lastMessageY =
        finalMessages.length >= 1
          ? finalMessages[finalMessages.length - 1]
          : null;

      if (
        provider === Providers.BEDROCK &&
        lastMessageX instanceof AIMessageChunk &&
        lastMessageY instanceof ToolMessage &&
        typeof lastMessageX.content === 'string'
      ) {
        finalMessages[finalMessages.length - 2].content = '';
      }

      const isLatestToolMessage = lastMessageY instanceof ToolMessage;

      if (isLatestToolMessage && provider === Providers.ANTHROPIC) {
        formatAnthropicArtifactContent(finalMessages);
      } else if (
        isLatestToolMessage &&
        (isOpenAILike(provider) || isGoogleLike(provider))
      ) {
        formatToolArrayContent(finalMessages);
      }

      if (this.lastStreamCall != null && this.streamBuffer != null) {
        const timeSinceLastCall = Date.now() - this.lastStreamCall;
        if (timeSinceLastCall < this.streamBuffer) {
          const timeToWait =
            Math.ceil((this.streamBuffer - timeSinceLastCall) / 1000) * 1000;
          await sleep(timeToWait);
        }
      }

      this.lastStreamCall = Date.now();

      let result: Partial<t.BaseGraphState>;
      if (
        (this.tools?.length ?? 0) > 0 &&
        manualToolStreamProviders.has(provider)
      ) {
        const stream = await this.boundModel.stream(finalMessages, config);
        let finalChunk: AIMessageChunk | undefined;
        for await (const chunk of stream) {
          dispatchCustomEvent(GraphEvents.CHAT_MODEL_STREAM, { chunk }, config);
          if (!finalChunk) {
            finalChunk = chunk;
          } else {
            finalChunk = concat(finalChunk, chunk);
          }
        }

        finalChunk = modifyDeltaProperties(this.provider, finalChunk);
        result = { messages: [finalChunk as AIMessageChunk] };
      } else {
        const finalMessage = (await this.boundModel.invoke(
          finalMessages,
          config
        )) as AIMessageChunk;
        if ((finalMessage.tool_calls?.length ?? 0) > 0) {
          finalMessage.tool_calls = finalMessage.tool_calls?.filter(
            (tool_call) => {
              if (!tool_call.name) {
                return false;
              }
              return true;
            }
          );
        }
        result = { messages: [finalMessage] };
      }

      this.storeUsageMetadata(result.messages?.[0]);
      this.cleanupSignalListener();
      return result;
    };
  }

  createWorkflow(): t.CompiledWorkflow<t.BaseGraphState> {
    const routeMessage = (
      state: t.BaseGraphState,
      config?: RunnableConfig
    ): string => {
      this.config = config;
      return toolsCondition(state, this.invokedToolIds);
    };

    const workflow = new StateGraph<t.BaseGraphState>({
      channels: this.graphState,
    })
      .addNode(AGENT, this.createCallModel())
      .addNode(TOOLS, this.initializeTools())
      .addEdge(START, AGENT)
      .addConditionalEdges(AGENT, routeMessage)
      .addEdge(TOOLS, this.toolEnd ? END : AGENT);

    return workflow.compile();
  }

  /* Dispatchers */

  /**
   * Dispatches a run step to the client, returns the step ID
   */
  dispatchRunStep(stepKey: string, stepDetails: t.StepDetails): string {
    if (!this.config) {
      throw new Error('No config provided');
    }

    const [stepId, stepIndex] = this.generateStepId(stepKey);
    if (stepDetails.type === StepTypes.TOOL_CALLS && stepDetails.tool_calls) {
      for (const tool_call of stepDetails.tool_calls) {
        const toolCallId = tool_call.id ?? '';
        if (!toolCallId || this.toolCallStepIds.has(toolCallId)) {
          continue;
        }
        this.toolCallStepIds.set(toolCallId, stepId);
      }
    }

    const runStep: t.RunStep = {
      stepIndex,
      id: stepId,
      type: stepDetails.type,
      index: this.contentData.length,
      stepDetails,
      usage: null,
    };

    const runId = this.runId ?? '';
    if (runId) {
      runStep.runId = runId;
    }

    this.contentData.push(runStep);
    this.contentIndexMap.set(stepId, runStep.index);
    dispatchCustomEvent(GraphEvents.ON_RUN_STEP, runStep, this.config);
    return stepId;
  }

  handleToolCallCompleted(
    data: t.ToolEndData,
    metadata?: Record<string, unknown>,
    omitOutput?: boolean
  ): void {
    if (!this.config) {
      throw new Error('No config provided');
    }

    if (!data.output) {
      return;
    }

    const { input, output } = data;
    const { tool_call_id } = output;
    const stepId = this.toolCallStepIds.get(tool_call_id) ?? '';
    if (!stepId) {
      throw new Error(`No stepId found for tool_call_id ${tool_call_id}`);
    }

    const runStep = this.getRunStep(stepId);
    if (!runStep) {
      throw new Error(`No run step found for stepId ${stepId}`);
    }

    const dispatchedOutput =
      typeof output.content === 'string'
        ? output.content
        : JSON.stringify(output.content);

    const args = typeof input === 'string' ? input : input.input;
    const tool_call = {
      args: typeof args === 'string' ? args : JSON.stringify(args),
      name: output.name ?? '',
      id: output.tool_call_id,
      output: omitOutput === true ? '' : dispatchedOutput,
      progress: 1,
    };

    this.handlerRegistry?.getHandler(GraphEvents.ON_RUN_STEP_COMPLETED)?.handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: stepId,
          index: runStep.index,
          type: 'tool_call',
          tool_call,
        } as t.ToolCompleteEvent,
      },
      metadata,
      this
    );
  }
  /**
   * Static version of handleToolCallError to avoid creating strong references
   * that prevent garbage collection
   */
  static handleToolCallErrorStatic(
    graph: StandardGraph,
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): void {
    if (!graph.config) {
      throw new Error('No config provided');
    }

    if (!data.id) {
      console.warn('No Tool ID provided for Tool Error');
      return;
    }

    const stepId = graph.toolCallStepIds.get(data.id) ?? '';
    if (!stepId) {
      throw new Error(`No stepId found for tool_call_id ${data.id}`);
    }

    const { name, input: args, error } = data;

    const runStep = graph.getRunStep(stepId);
    if (!runStep) {
      throw new Error(`No run step found for stepId ${stepId}`);
    }

    const tool_call: t.ProcessedToolCall = {
      id: data.id,
      name: name || '',
      args: typeof args === 'string' ? args : JSON.stringify(args),
      output: `Error processing tool${error?.message != null ? `: ${error.message}` : ''}`,
      progress: 1,
    };

    graph.handlerRegistry
      ?.getHandler(GraphEvents.ON_RUN_STEP_COMPLETED)
      ?.handle(
        GraphEvents.ON_RUN_STEP_COMPLETED,
        {
          result: {
            id: stepId,
            index: runStep.index,
            type: 'tool_call',
            tool_call,
          } as t.ToolCompleteEvent,
        },
        metadata,
        graph
      );
  }

  /**
   * Instance method that delegates to the static method
   * Kept for backward compatibility
   */
  handleToolCallError(
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): void {
    StandardGraph.handleToolCallErrorStatic(this, data, metadata);
  }

  dispatchRunStepDelta(id: string, delta: t.ToolCallDelta): void {
    if (!this.config) {
      throw new Error('No config provided');
    } else if (!id) {
      throw new Error('No step ID found');
    }
    const runStepDelta: t.RunStepDeltaEvent = {
      id,
      delta,
    };
    dispatchCustomEvent(
      GraphEvents.ON_RUN_STEP_DELTA,
      runStepDelta,
      this.config
    );
  }

  dispatchMessageDelta(id: string, delta: t.MessageDelta): void {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const messageDelta: t.MessageDeltaEvent = {
      id,
      delta,
    };
    dispatchCustomEvent(
      GraphEvents.ON_MESSAGE_DELTA,
      messageDelta,
      this.config
    );
  }

  dispatchReasoningDelta = (stepId: string, delta: t.ReasoningDelta): void => {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const reasoningDelta: t.ReasoningDeltaEvent = {
      id: stepId,
      delta,
    };
    dispatchCustomEvent(
      GraphEvents.ON_REASONING_DELTA,
      reasoningDelta,
      this.config
    );
  };
}
