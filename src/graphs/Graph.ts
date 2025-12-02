/* eslint-disable no-console */
// src/graphs/Graph.ts
import { nanoid } from 'nanoid';
import { concat } from '@langchain/core/utils/stream';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatVertexAI } from '@langchain/google-vertexai';
import {
  START,
  END,
  Command,
  StateGraph,
  Annotation,
  messagesStateReducer,
} from '@langchain/langgraph';
import {
  Runnable,
  RunnableConfig,
  RunnableLambda,
} from '@langchain/core/runnables';
import {
  ToolMessage,
  SystemMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import type {
  BaseMessageFields,
  UsageMetadata,
  BaseMessage,
} from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  GraphNodeKeys,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
} from '@/common';
import {
  formatAnthropicArtifactContent,
  ensureThinkingBlockInMessages,
  convertMessagesToContent,
  addBedrockCacheControl,
  modifyDeltaProperties,
  formatArtifactPayload,
  formatContentStrings,
  createPruneMessages,
  addCacheControl,
} from '@/messages';
import {
  resetIfNotEmpty,
  isOpenAILike,
  isGoogleLike,
  joinKeys,
  sleep,
} from '@/utils';
import { getChatModelClass, manualToolStreamProviders } from '@/llm/providers';
import { ToolNode as CustomToolNode, toolsCondition } from '@/tools/ToolNode';
import { ChatOpenAI, AzureChatOpenAI } from '@/llm/openai';
import { safeDispatchCustomEvent } from '@/utils/events';
import { AgentContext } from '@/agents/AgentContext';
import { createFakeStreamingLLM } from '@/llm/fake';
import { HandlerRegistry } from '@/events';

const { AGENT, TOOLS } = GraphNodeKeys;

export abstract class Graph<
  T extends t.BaseGraphState = t.BaseGraphState,
  _TNodeName extends string = string,
> {
  abstract resetValues(): void;
  abstract initializeTools({
    currentTools,
    currentToolMap,
  }: {
    currentTools?: t.GraphTools;
    currentToolMap?: t.ToolMap;
  }): CustomToolNode<T> | ToolNode<T>;
  abstract initializeModel({
    currentModel,
    tools,
    clientOptions,
  }: {
    currentModel?: t.ChatModel;
    tools?: t.GraphTools;
    clientOptions?: t.ClientOptions;
  }): Runnable;
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
  abstract dispatchRunStep(
    stepKey: string,
    stepDetails: t.StepDetails,
    metadata?: Record<string, unknown>
  ): Promise<string>;
  abstract dispatchRunStepDelta(
    id: string,
    delta: t.ToolCallDelta
  ): Promise<void>;
  abstract dispatchMessageDelta(
    id: string,
    delta: t.MessageDelta
  ): Promise<void>;
  abstract dispatchReasoningDelta(
    stepId: string,
    delta: t.ReasoningDelta
  ): Promise<void>;
  abstract handleToolCallCompleted(
    data: t.ToolEndData,
    metadata?: Record<string, unknown>,
    omitOutput?: boolean
  ): Promise<void>;

  abstract createCallModel(
    agentId?: string,
    currentModel?: t.ChatModel
  ): (state: T, config?: RunnableConfig) => Promise<Partial<T>>;
  messageStepHasToolCalls: Map<string, boolean> = new Map();
  messageIdsByStepKey: Map<string, string> = new Map();
  prelimMessageIdsByStepKey: Map<string, string> = new Map();
  config: RunnableConfig | undefined;
  contentData: t.RunStep[] = [];
  stepKeyIds: Map<string, string[]> = new Map<string, string[]>();
  contentIndexMap: Map<string, number> = new Map();
  toolCallStepIds: Map<string, string> = new Map();
  signal?: AbortSignal;
  /** Set of invoked tool call IDs from non-message run steps completed mid-run, if any */
  invokedToolIds?: Set<string>;
  handlerRegistry: HandlerRegistry | undefined;
}

export class StandardGraph extends Graph<t.BaseGraphState, t.GraphNode> {
  overrideModel?: t.ChatModel;
  /** Optional compile options passed into workflow.compile() */
  compileOptions?: t.CompileOptions | undefined;
  messages: BaseMessage[] = [];
  runId: string | undefined;
  startIndex: number = 0;
  signal?: AbortSignal;
  /** Map of agent contexts by agent ID */
  agentContexts: Map<string, AgentContext> = new Map();
  /** Default agent ID to use */
  defaultAgentId: string;

  constructor({
    // parent-level graph inputs
    runId,
    signal,
    agents,
    tokenCounter,
    indexTokenCountMap,
  }: t.StandardGraphInput) {
    super();
    this.runId = runId;
    this.signal = signal;

    if (agents.length === 0) {
      throw new Error('At least one agent configuration is required');
    }

    for (const agentConfig of agents) {
      const agentContext = AgentContext.fromConfig(
        agentConfig,
        tokenCounter,
        indexTokenCountMap
      );

      this.agentContexts.set(agentConfig.agentId, agentContext);
    }

    this.defaultAgentId = agents[0].agentId;
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
      this.messageStepHasToolCalls,
      new Map()
    );
    this.prelimMessageIdsByStepKey = resetIfNotEmpty(
      this.prelimMessageIdsByStepKey,
      new Map()
    );
    this.invokedToolIds = resetIfNotEmpty(this.invokedToolIds, undefined);
    for (const context of this.agentContexts.values()) {
      context.reset();
    }
  }

  /* Run Step Processing */

  getRunStep(stepId: string): t.RunStep | undefined {
    const index = this.contentIndexMap.get(stepId);
    if (index !== undefined) {
      return this.contentData[index];
    }
    return undefined;
  }

  getAgentContext(metadata: Record<string, unknown> | undefined): AgentContext {
    if (!metadata) {
      throw new Error('No metadata provided to retrieve agent context');
    }

    const currentNode = metadata.langgraph_node as string;
    if (!currentNode) {
      throw new Error(
        'No langgraph_node in metadata to retrieve agent context'
      );
    }

    let agentId: string | undefined;
    if (currentNode.startsWith(AGENT)) {
      agentId = currentNode.substring(AGENT.length);
    } else if (currentNode.startsWith(TOOLS)) {
      agentId = currentNode.substring(TOOLS.length);
    }

    const agentContext = this.agentContexts.get(agentId ?? '');
    if (!agentContext) {
      throw new Error(`No agent context found for agent ID ${agentId}`);
    }

    return agentContext;
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

    const agentContext = this.getAgentContext(metadata);
    if (
      agentContext.currentTokenType === ContentTypes.THINK ||
      agentContext.currentTokenType === 'think_and_text'
    ) {
      keyList.push('reasoning');
    } else if (agentContext.tokenTypeSwitch === 'content') {
      keyList.push('post-reasoning');
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

  /**
   * Get all run steps, optionally filtered by agent ID
   */
  getRunSteps(agentId?: string): t.RunStep[] {
    if (agentId == null || agentId === '') {
      return [...this.contentData];
    }
    return this.contentData.filter((step) => step.agentId === agentId);
  }

  /**
   * Get run steps grouped by agent ID
   */
  getRunStepsByAgent(): Map<string, t.RunStep[]> {
    const stepsByAgent = new Map<string, t.RunStep[]>();

    for (const step of this.contentData) {
      if (step.agentId == null || step.agentId === '') continue;

      const steps = stepsByAgent.get(step.agentId) ?? [];
      steps.push(step);
      stepsByAgent.set(step.agentId, steps);
    }

    return stepsByAgent;
  }

  /**
   * Get agent IDs that participated in this run
   */
  getActiveAgentIds(): string[] {
    const agentIds = new Set<string>();
    for (const step of this.contentData) {
      if (step.agentId != null && step.agentId !== '') {
        agentIds.add(step.agentId);
      }
    }
    return Array.from(agentIds);
  }

  /**
   * Maps contentPart indices to agent IDs for post-run analysis
   * Returns a map where key is the contentPart index and value is the agentId
   */
  getContentPartAgentMap(): Map<number, string> {
    const contentPartAgentMap = new Map<number, string>();

    for (const step of this.contentData) {
      if (
        step.agentId != null &&
        step.agentId !== '' &&
        Number.isFinite(step.index)
      ) {
        contentPartAgentMap.set(step.index, step.agentId);
      }
    }

    return contentPartAgentMap;
  }

  /* Graph */

  createSystemRunnable({
    provider,
    clientOptions,
    instructions,
    additional_instructions,
  }: {
    provider?: Providers;
    clientOptions?: t.ClientOptions;
    instructions?: string;
    additional_instructions?: string;
  }): t.SystemRunnable | undefined {
    let finalInstructions: string | BaseMessageFields | undefined =
      instructions;
    if (additional_instructions != null && additional_instructions !== '') {
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
      const systemMessage = new SystemMessage(finalInstructions);
      return RunnableLambda.from((messages: BaseMessage[]) => {
        return [systemMessage, ...messages];
      }).withConfig({ runName: 'prompt' });
    }
  }

  initializeTools({
    currentTools,
    currentToolMap,
  }: {
    currentTools?: t.GraphTools;
    currentToolMap?: t.ToolMap;
  }): CustomToolNode<t.BaseGraphState> | ToolNode<t.BaseGraphState> {
    return new CustomToolNode<t.BaseGraphState>({
      tools: (currentTools as t.GenericTool[] | undefined) ?? [],
      toolMap: currentToolMap,
      toolCallStepIds: this.toolCallStepIds,
      errorHandler: (data, metadata) =>
        StandardGraph.handleToolCallErrorStatic(this, data, metadata),
    });
  }

  initializeModel({
    provider,
    tools,
    clientOptions,
  }: {
    provider: Providers;
    tools?: t.GraphTools;
    clientOptions?: t.ClientOptions;
  }): Runnable {
    const ChatModelClass = getChatModelClass(provider);
    const model = new ChatModelClass(clientOptions ?? {});

    if (
      isOpenAILike(provider) &&
      (model instanceof ChatOpenAI || model instanceof AzureChatOpenAI)
    ) {
      model.temperature = (clientOptions as t.OpenAIClientOptions)
        .temperature as number;
      model.topP = (clientOptions as t.OpenAIClientOptions).topP as number;
      model.frequencyPenalty = (clientOptions as t.OpenAIClientOptions)
        .frequencyPenalty as number;
      model.presencePenalty = (clientOptions as t.OpenAIClientOptions)
        .presencePenalty as number;
      model.n = (clientOptions as t.OpenAIClientOptions).n as number;
    } else if (
      provider === Providers.VERTEXAI &&
      model instanceof ChatVertexAI
    ) {
      model.temperature = (clientOptions as t.VertexAIClientOptions)
        .temperature as number;
      model.topP = (clientOptions as t.VertexAIClientOptions).topP as number;
      model.topK = (clientOptions as t.VertexAIClientOptions).topK as number;
      model.topLogprobs = (clientOptions as t.VertexAIClientOptions)
        .topLogprobs as number;
      model.frequencyPenalty = (clientOptions as t.VertexAIClientOptions)
        .frequencyPenalty as number;
      model.presencePenalty = (clientOptions as t.VertexAIClientOptions)
        .presencePenalty as number;
      model.maxOutputTokens = (clientOptions as t.VertexAIClientOptions)
        .maxOutputTokens as number;
    }

    if (!tools || tools.length === 0) {
      return model as unknown as Runnable;
    }

    return (model as t.ModelWithTools).bindTools(tools);
  }

  overrideTestModel(
    responses: string[],
    sleep?: number,
    toolCalls?: ToolCall[]
  ): void {
    this.overrideModel = createFakeStreamingLLM({
      responses,
      sleep,
      toolCalls,
    });
  }

  getNewModel({
    provider,
    clientOptions,
  }: {
    provider: Providers;
    clientOptions?: t.ClientOptions;
  }): t.ChatModelInstance {
    const ChatModelClass = getChatModelClass(provider);
    return new ChatModelClass(clientOptions ?? {});
  }

  getUsageMetadata(
    finalMessage?: BaseMessage
  ): Partial<UsageMetadata> | undefined {
    if (
      finalMessage &&
      'usage_metadata' in finalMessage &&
      finalMessage.usage_metadata != null
    ) {
      return finalMessage.usage_metadata as Partial<UsageMetadata>;
    }
  }

  /** Execute model invocation with streaming support */
  private async attemptInvoke(
    {
      currentModel,
      finalMessages,
      provider,
      tools,
    }: {
      currentModel?: t.ChatModel;
      finalMessages: BaseMessage[];
      provider: Providers;
      tools?: t.GraphTools;
    },
    config?: RunnableConfig
  ): Promise<Partial<t.BaseGraphState>> {
    const model = this.overrideModel ?? currentModel;
    if (!model) {
      throw new Error('No model found');
    }

    if ((tools?.length ?? 0) > 0 && manualToolStreamProviders.has(provider)) {
      if (!model.stream) {
        throw new Error('Model does not support stream');
      }
      const stream = await model.stream(finalMessages, config);
      let finalChunk: AIMessageChunk | undefined;
      for await (const chunk of stream) {
        await safeDispatchCustomEvent(
          GraphEvents.CHAT_MODEL_STREAM,
          { chunk, emitted: true },
          config
        );
        finalChunk = finalChunk ? concat(finalChunk, chunk) : chunk;
      }
      finalChunk = modifyDeltaProperties(provider, finalChunk);
      return { messages: [finalChunk as AIMessageChunk] };
    } else {
      const finalMessage = await model.invoke(finalMessages, config);
      if ((finalMessage.tool_calls?.length ?? 0) > 0) {
        finalMessage.tool_calls = finalMessage.tool_calls?.filter(
          (tool_call: ToolCall) => !!tool_call.name
        );
      }
      return { messages: [finalMessage] };
    }
  }

  cleanupSignalListener(currentModel?: t.ChatModel): void {
    if (!this.signal) {
      return;
    }
    const model = this.overrideModel ?? currentModel;
    if (!model) {
      return;
    }
    const client = (model as ChatOpenAI | undefined)?.exposedClient;
    if (!client?.abortHandler) {
      return;
    }
    this.signal.removeEventListener('abort', client.abortHandler);
    client.abortHandler = undefined;
  }

  createCallModel(agentId = 'default', currentModel?: t.ChatModel) {
    return async (
      state: t.BaseGraphState,
      config?: RunnableConfig
    ): Promise<Partial<t.BaseGraphState>> => {
      /**
       * Get agent context - it must exist by this point
       */
      const agentContext = this.agentContexts.get(agentId);
      if (!agentContext) {
        throw new Error(`Agent context not found for agentId: ${agentId}`);
      }

      const model = this.overrideModel ?? currentModel;
      if (!model) {
        throw new Error('No Graph model found');
      }
      if (!config) {
        throw new Error('No config provided');
      }

      // Ensure token calculations are complete before proceeding
      if (agentContext.tokenCalculationPromise) {
        await agentContext.tokenCalculationPromise;
      }
      if (!config.signal) {
        config.signal = this.signal;
      }
      this.config = config;
      const { messages } = state;

      let messagesToUse = messages;
      if (
        !agentContext.pruneMessages &&
        agentContext.tokenCounter &&
        agentContext.maxContextTokens != null &&
        agentContext.indexTokenCountMap[0] != null
      ) {
        const isAnthropicWithThinking =
          (agentContext.provider === Providers.ANTHROPIC &&
            (agentContext.clientOptions as t.AnthropicClientOptions).thinking !=
              null) ||
          (agentContext.provider === Providers.BEDROCK &&
            (agentContext.clientOptions as t.BedrockAnthropicInput)
              .additionalModelRequestFields?.['thinking'] != null) ||
          (agentContext.provider === Providers.OPENAI &&
            (
              (agentContext.clientOptions as t.OpenAIClientOptions).modelKwargs
                ?.thinking as t.AnthropicClientOptions['thinking']
            )?.type === 'enabled');

        agentContext.pruneMessages = createPruneMessages({
          startIndex: this.startIndex,
          provider: agentContext.provider,
          tokenCounter: agentContext.tokenCounter,
          maxTokens: agentContext.maxContextTokens,
          thinkingEnabled: isAnthropicWithThinking,
          indexTokenCountMap: agentContext.indexTokenCountMap,
        });
      }
      if (agentContext.pruneMessages) {
        const { context, indexTokenCountMap } = agentContext.pruneMessages({
          messages,
          usageMetadata: agentContext.currentUsage,
          // startOnMessageType: 'human',
        });
        agentContext.indexTokenCountMap = indexTokenCountMap;
        messagesToUse = context;
      }

      let finalMessages = messagesToUse;
      if (agentContext.useLegacyContent) {
        finalMessages = formatContentStrings(finalMessages);
      }

      const lastMessageX =
        finalMessages.length >= 2
          ? finalMessages[finalMessages.length - 2]
          : null;
      const lastMessageY =
        finalMessages.length >= 1
          ? finalMessages[finalMessages.length - 1]
          : null;

      if (
        agentContext.provider === Providers.BEDROCK &&
        lastMessageX instanceof AIMessageChunk &&
        lastMessageY instanceof ToolMessage &&
        typeof lastMessageX.content === 'string'
      ) {
        finalMessages[finalMessages.length - 2].content = '';
      }

      const isLatestToolMessage = lastMessageY instanceof ToolMessage;

      if (
        isLatestToolMessage &&
        agentContext.provider === Providers.ANTHROPIC
      ) {
        formatAnthropicArtifactContent(finalMessages);
      } else if (
        isLatestToolMessage &&
        ((isOpenAILike(agentContext.provider) &&
          agentContext.provider !== Providers.DEEPSEEK) ||
          isGoogleLike(agentContext.provider))
      ) {
        formatArtifactPayload(finalMessages);
      }

      if (agentContext.provider === Providers.ANTHROPIC) {
        const anthropicOptions = agentContext.clientOptions as
          | t.AnthropicClientOptions
          | undefined;
        const defaultHeaders = anthropicOptions?.clientOptions
          ?.defaultHeaders as Record<string, string> | undefined;
        const anthropicBeta = defaultHeaders?.['anthropic-beta'];

        if (
          typeof anthropicBeta === 'string' &&
          anthropicBeta.includes('prompt-caching')
        ) {
          finalMessages = addCacheControl<BaseMessage>(finalMessages);
        }
      } else if (agentContext.provider === Providers.BEDROCK) {
        const bedrockOptions = agentContext.clientOptions as
          | t.BedrockAnthropicClientOptions
          | undefined;
        if (bedrockOptions?.promptCache === true) {
          finalMessages = addBedrockCacheControl<BaseMessage>(finalMessages);
        }
      }

      /**
       * Handle edge case: when switching from a non-thinking agent to a thinking-enabled agent,
       * convert AI messages with tool calls to HumanMessages to avoid thinking block requirements.
       * This is required by Anthropic/Bedrock when thinking is enabled.
       */
      const isAnthropicWithThinking =
        (agentContext.provider === Providers.ANTHROPIC &&
          (agentContext.clientOptions as t.AnthropicClientOptions).thinking !=
            null) ||
        (agentContext.provider === Providers.BEDROCK &&
          (agentContext.clientOptions as t.BedrockAnthropicInput)
            .additionalModelRequestFields?.['thinking'] != null);

      if (isAnthropicWithThinking) {
        finalMessages = ensureThinkingBlockInMessages(
          finalMessages,
          agentContext.provider
        );
      }

      if (
        agentContext.lastStreamCall != null &&
        agentContext.streamBuffer != null
      ) {
        const timeSinceLastCall = Date.now() - agentContext.lastStreamCall;
        if (timeSinceLastCall < agentContext.streamBuffer) {
          const timeToWait =
            Math.ceil((agentContext.streamBuffer - timeSinceLastCall) / 1000) *
            1000;
          await sleep(timeToWait);
        }
      }

      agentContext.lastStreamCall = Date.now();

      let result: Partial<t.BaseGraphState> | undefined;
      const fallbacks =
        (agentContext.clientOptions as t.LLMConfig | undefined)?.fallbacks ??
        [];

      if (finalMessages.length === 0) {
        throw new Error(
          JSON.stringify({
            type: 'empty_messages',
            info: 'Message pruning removed all messages as none fit in the context window. Please increase the context window size or make your message shorter.',
          })
        );
      }

      try {
        result = await this.attemptInvoke(
          {
            currentModel: model,
            finalMessages,
            provider: agentContext.provider,
            tools: agentContext.tools,
          },
          config
        );
      } catch (primaryError) {
        let lastError: unknown = primaryError;
        for (const fb of fallbacks) {
          try {
            let model = this.getNewModel({
              provider: fb.provider,
              clientOptions: fb.clientOptions,
            });
            const bindableTools = agentContext.tools;
            model = (
              !bindableTools || bindableTools.length === 0
                ? model
                : model.bindTools(bindableTools)
            ) as t.ChatModelInstance;
            result = await this.attemptInvoke(
              {
                currentModel: model,
                finalMessages,
                provider: fb.provider,
                tools: agentContext.tools,
              },
              config
            );
            lastError = undefined;
            break;
          } catch (e) {
            lastError = e;
            continue;
          }
        }
        if (lastError !== undefined) {
          throw lastError;
        }
      }

      if (!result) {
        throw new Error('No result after model invocation');
      }
      agentContext.currentUsage = this.getUsageMetadata(result.messages?.[0]);
      this.cleanupSignalListener();
      return result;
    };
  }

  createAgentNode(agentId: string): t.CompiledAgentWorfklow {
    const agentContext = this.agentContexts.get(agentId);
    if (!agentContext) {
      throw new Error(`Agent context not found for agentId: ${agentId}`);
    }

    let currentModel = this.initializeModel({
      tools: agentContext.tools,
      provider: agentContext.provider,
      clientOptions: agentContext.clientOptions,
    });

    if (agentContext.systemRunnable) {
      currentModel = agentContext.systemRunnable.pipe(currentModel);
    }

    const agentNode = `${AGENT}${agentId}` as const;
    const toolNode = `${TOOLS}${agentId}` as const;

    const routeMessage = (
      state: t.BaseGraphState,
      config?: RunnableConfig
    ): string => {
      this.config = config;
      return toolsCondition(state, toolNode, this.invokedToolIds);
    };

    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
      }),
    });

    const workflow = new StateGraph(StateAnnotation)
      .addNode(agentNode, this.createCallModel(agentId, currentModel))
      .addNode(
        toolNode,
        this.initializeTools({
          currentTools: agentContext.tools,
          currentToolMap: agentContext.toolMap,
        })
      )
      .addEdge(START, agentNode)
      .addConditionalEdges(agentNode, routeMessage)
      .addEdge(toolNode, agentContext.toolEnd ? END : agentNode);

    // Cast to unknown to avoid tight coupling to external types; options are opt-in
    return workflow.compile(this.compileOptions as unknown as never);
  }

  createWorkflow(): t.CompiledStateWorkflow {
    /** Use the default (first) agent for now */
    const agentNode = this.createAgentNode(this.defaultAgentId);
    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => {
          if (!a.length) {
            this.startIndex = a.length + b.length;
          }
          const result = messagesStateReducer(a, b);
          this.messages = result;
          return result;
        },
        default: () => [],
      }),
    });
    const workflow = new StateGraph(StateAnnotation)
      .addNode(this.defaultAgentId, agentNode, { ends: [END] })
      .addEdge(START, this.defaultAgentId)
      .compile();

    return workflow;
  }

  /* Dispatchers */

  /**
   * Dispatches a run step to the client, returns the step ID
   */
  async dispatchRunStep(
    stepKey: string,
    stepDetails: t.StepDetails,
    metadata?: Record<string, unknown>
  ): Promise<string> {
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

    /**
     * Extract and store agentId from metadata
     */
    if (metadata) {
      try {
        const agentContext = this.getAgentContext(metadata);
        if (agentContext.agentId) {
          runStep.agentId = agentContext.agentId;
        }
      } catch (_e) {
        /** If we can't get agent context, that's okay - agentId remains undefined */
      }
    }

    this.contentData.push(runStep);
    this.contentIndexMap.set(stepId, runStep.index);
    await safeDispatchCustomEvent(
      GraphEvents.ON_RUN_STEP,
      runStep,
      this.config
    );
    return stepId;
  }

  async handleToolCallCompleted(
    data: t.ToolEndData,
    metadata?: Record<string, unknown>,
    omitOutput?: boolean
  ): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    }

    if (!data.output) {
      return;
    }

    const { input, output: _output } = data;
    if ((_output as Command | undefined)?.lg_name === 'Command') {
      return;
    }
    const output = _output as ToolMessage;
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

    await this.handlerRegistry
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
        this
      );
  }
  /**
   * Static version of handleToolCallError to avoid creating strong references
   * that prevent garbage collection
   */
  static async handleToolCallErrorStatic(
    graph: StandardGraph,
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): Promise<void> {
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

    await graph.handlerRegistry
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
  async handleToolCallError(
    data: t.ToolErrorData,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    await StandardGraph.handleToolCallErrorStatic(this, data, metadata);
  }

  async dispatchRunStepDelta(
    id: string,
    delta: t.ToolCallDelta
  ): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    } else if (!id) {
      throw new Error('No step ID found');
    }
    const runStepDelta: t.RunStepDeltaEvent = {
      id,
      delta,
    };
    await safeDispatchCustomEvent(
      GraphEvents.ON_RUN_STEP_DELTA,
      runStepDelta,
      this.config
    );
  }

  async dispatchMessageDelta(id: string, delta: t.MessageDelta): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const messageDelta: t.MessageDeltaEvent = {
      id,
      delta,
    };
    await safeDispatchCustomEvent(
      GraphEvents.ON_MESSAGE_DELTA,
      messageDelta,
      this.config
    );
  }

  dispatchReasoningDelta = async (
    stepId: string,
    delta: t.ReasoningDelta
  ): Promise<void> => {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const reasoningDelta: t.ReasoningDeltaEvent = {
      id: stepId,
      delta,
    };
    await safeDispatchCustomEvent(
      GraphEvents.ON_REASONING_DELTA,
      reasoningDelta,
      this.config
    );
  };
}
