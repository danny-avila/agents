/* eslint-disable no-console */
import { nanoid } from 'nanoid';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { Runnable, RunnableConfig } from '@langchain/core/runnables';
import { ToolMessage, AIMessageChunk } from '@langchain/core/messages';
import { START, END, StateGraph, Annotation } from '@langchain/langgraph';
import type {
  UsageMetadata,
  BaseMessage,
  MessageContent,
} from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  formatAnthropicArtifactContent,
  ensureThinkingBlockInMessages,
  convertMessagesToContent,
  sanitizeOrphanToolBlocks,
  extractToolDiscoveries,
  addBedrockCacheControl,
  formatArtifactPayload,
  formatContentStrings,
  createPruneMessages,
  addCacheControl,
  getMessageId,
} from '@/messages';
import {
  GraphNodeKeys,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
} from '@/common';
import {
  resetIfNotEmpty,
  isAnthropicLike,
  isOpenAILike,
  isGoogleLike,
  joinKeys,
  sleep,
} from '@/utils';
import { ToolNode as CustomToolNode, toolsCondition } from '@/tools/ToolNode';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { shouldTriggerSummarization } from '@/summarization';
import { createSummarizeNode } from '@/summarization/node';
import { messagesStateReducer } from '@/messages/reducer';
import { createSchemaOnlyTools } from '@/tools/schema';
import { AgentContext } from '@/agents/AgentContext';
import { createFakeStreamingLLM } from '@/llm/fake';
import { handleToolCalls } from '@/tools/handlers';
import { isThinkingEnabled } from '@/llm/request';
import { initializeModel } from '@/llm/init';
import { HandlerRegistry } from '@/events';
import { ChatOpenAI } from '@/llm/openai';

const { AGENT, TOOLS, SUMMARIZE } = GraphNodeKeys;

/** Minimum relative variance before calibrated toolSchemaTokens overrides current value. */
const CALIBRATION_VARIANCE_THRESHOLD = 0.15;

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
  abstract createCallModel(
    agentId?: string,
    currentModel?: t.ChatModel
  ): (
    state: t.AgentSubgraphState,
    config?: RunnableConfig
  ) => Promise<Partial<t.AgentSubgraphState>>;
  messageStepHasToolCalls: Map<string, boolean> = new Map();
  messageIdsByStepKey: Map<string, string> = new Map();
  prelimMessageIdsByStepKey: Map<string, string> = new Map();
  config: RunnableConfig | undefined;
  contentData: t.RunStep[] = [];
  stepKeyIds: Map<string, string[]> = new Map<string, string[]>();
  contentIndexMap: Map<string, number> = new Map();
  toolCallStepIds: Map<string, string> = new Map();
  /**
   * Step IDs that have been dispatched via handler registry directly
   * (in dispatchRunStep).  Used by the custom event callback to skip
   * duplicate dispatch through the LangGraph callback chain.
   */
  handlerDispatchedStepIds: Set<string> = new Set();
  signal?: AbortSignal;
  /** Set of invoked tool call IDs from non-message run steps completed mid-run, if any */
  invokedToolIds?: Set<string>;
  handlerRegistry: HandlerRegistry | undefined;
  /**
   * Tool session contexts for automatic state persistence across tool invocations.
   * Keyed by tool name (e.g., Constants.EXECUTE_CODE).
   * Currently supports code execution session tracking (session_id, files).
   */
  sessions: t.ToolSessionMap = new Map();

  /**
   * Clears heavy references to allow GC to reclaim memory held by
   * LangGraph's internal config / AsyncLocalStorage RunTree chain.
   * Call after a run completes and content has been extracted.
   */
  clearHeavyState(): void {
    this.config = undefined;
    this.signal = undefined;
    this.contentData = [];
    this.contentIndexMap = new Map();
    this.stepKeyIds = new Map();
    this.toolCallStepIds.clear();
    this.messageIdsByStepKey = new Map();
    this.messageStepHasToolCalls = new Map();
    this.prelimMessageIdsByStepKey = new Map();
    this.invokedToolIds = undefined;
    this.handlerRegistry = undefined;
    this.sessions.clear();
  }
}

export class StandardGraph extends Graph<t.BaseGraphState, t.GraphNode> {
  overrideModel?: t.ChatModel;
  /** Optional compile options passed into workflow.compile() */
  compileOptions?: t.CompileOptions | undefined;
  messages: BaseMessage[] = [];
  /** Cached run messages preserved before clearHeavyState() so getRunMessages() works after cleanup. */
  private cachedRunMessages?: BaseMessage[];
  runId: string | undefined;
  /**
   * Boundary between historical messages (loaded from conversation state)
   * and messages produced during the current run.  Set once in the state
   * reducer when messages first arrive.  Used by `getRunMessages()` and
   * multi-agent message filtering — NOT for pruner token counting (the
   * pruner maintains its own `lastTurnStartIndex` in its closure).
   */
  startIndex: number = 0;
  signal?: AbortSignal;
  /** Map of agent contexts by agent ID */
  agentContexts: Map<string, AgentContext> = new Map();
  /** Default agent ID to use */
  defaultAgentId: string;

  constructor({
    runId,
    signal,
    agents,
    tokenCounter,
    indexTokenCountMap,
    calibrationRatio,
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
      if (calibrationRatio != null && calibrationRatio > 0) {
        agentContext.calibrationRatio = calibrationRatio;
      }

      this.agentContexts.set(agentConfig.agentId, agentContext);
    }

    this.defaultAgentId = agents[0].agentId;
  }

  /* Init */

  resetValues(keepContent?: boolean): void {
    this.messages = [];
    this.cachedRunMessages = undefined;
    this.config = resetIfNotEmpty(this.config, undefined);
    if (keepContent !== true) {
      this.contentData = resetIfNotEmpty(this.contentData, []);
      this.contentIndexMap = resetIfNotEmpty(this.contentIndexMap, new Map());
    }
    this.stepKeyIds = resetIfNotEmpty(this.stepKeyIds, new Map());
    /**
     * Clear in-place instead of replacing with a new Map to preserve the
     * shared reference held by ToolNode (passed at construction time).
     * Using resetIfNotEmpty would create a new Map, leaving ToolNode with
     * a stale reference on 2nd+ processStream calls.
     */
    this.toolCallStepIds.clear();
    this.handlerDispatchedStepIds = resetIfNotEmpty(
      this.handlerDispatchedStepIds,
      new Set()
    );
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

  override clearHeavyState(): void {
    this.cachedRunMessages = this.messages.slice(this.startIndex);
    super.clearHeavyState();
    this.messages = [];
    this.overrideModel = undefined;
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
    } else if (currentNode.startsWith(SUMMARIZE)) {
      agentId = currentNode.substring(SUMMARIZE.length);
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
      keyList.push(`post-reasoning-${agentContext.reasoningTransitionCount}`);
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
    if (this.messages.length === 0 && this.cachedRunMessages != null) {
      return this.cachedRunMessages;
    }
    return this.messages.slice(this.startIndex);
  }

  getContentParts(): t.MessageContentComplex[] | undefined {
    return convertMessagesToContent(this.messages.slice(this.startIndex));
  }

  getCalibrationRatio(): number {
    const context = this.agentContexts.get(this.defaultAgentId);
    return context?.calibrationRatio ?? 1;
  }

  getResolvedInstructionOverhead(): number | undefined {
    const context = this.agentContexts.get(this.defaultAgentId);
    return context?.resolvedInstructionOverhead;
  }

  getToolCount(): number {
    const context = this.agentContexts.get(this.defaultAgentId);
    return (
      (context?.tools?.length ?? 0) + (context?.toolDefinitions?.length ?? 0)
    );
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

  initializeTools({
    currentTools,
    currentToolMap,
    agentContext,
  }: {
    currentTools?: t.GraphTools;
    currentToolMap?: t.ToolMap;
    agentContext?: AgentContext;
  }): CustomToolNode<t.BaseGraphState> | ToolNode<t.BaseGraphState> {
    const toolDefinitions = agentContext?.toolDefinitions;
    const eventDrivenMode =
      toolDefinitions != null && toolDefinitions.length > 0;

    if (eventDrivenMode) {
      const schemaTools = createSchemaOnlyTools(toolDefinitions);
      const toolDefMap = new Map(toolDefinitions.map((def) => [def.name, def]));
      const graphTools = agentContext?.graphTools as
        | t.GenericTool[]
        | undefined;

      const directToolNames = new Set<string>();
      const allTools = [...schemaTools] as t.GenericTool[];
      const allToolMap: t.ToolMap = new Map(
        schemaTools.map((tool) => [tool.name, tool])
      );

      if (graphTools && graphTools.length > 0) {
        for (const tool of graphTools) {
          if ('name' in tool) {
            allTools.push(tool);
            allToolMap.set(tool.name, tool);
            directToolNames.add(tool.name);
          }
        }
      }

      return new CustomToolNode<t.BaseGraphState>({
        tools: allTools,
        toolMap: allToolMap,
        eventDrivenMode: true,
        sessions: this.sessions,
        toolDefinitions: toolDefMap,
        agentId: agentContext?.agentId,
        toolCallStepIds: this.toolCallStepIds,
        toolRegistry: agentContext?.toolRegistry,
        directToolNames: directToolNames.size > 0 ? directToolNames : undefined,
        maxContextTokens: agentContext?.maxContextTokens,
        maxToolResultChars: agentContext?.maxToolResultChars,
        errorHandler: (data, metadata) =>
          StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      });
    }

    const graphTools = agentContext?.graphTools as t.GenericTool[] | undefined;
    const baseTools = (currentTools as t.GenericTool[] | undefined) ?? [];
    const allTraditionalTools =
      graphTools && graphTools.length > 0
        ? [...baseTools, ...graphTools]
        : baseTools;
    const traditionalToolMap =
      graphTools && graphTools.length > 0
        ? new Map([
          ...(currentToolMap ?? new Map()),
          ...graphTools
            .filter((t): t is t.GenericTool & { name: string } => 'name' in t)
            .map((t) => [t.name, t] as [string, t.GenericTool]),
        ])
        : currentToolMap;

    return new CustomToolNode<t.BaseGraphState>({
      tools: allTraditionalTools,
      toolMap: traditionalToolMap,
      toolCallStepIds: this.toolCallStepIds,
      errorHandler: (data, metadata) =>
        StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      toolRegistry: agentContext?.toolRegistry,
      sessions: this.sessions,
      maxContextTokens: agentContext?.maxContextTokens,
      maxToolResultChars: agentContext?.maxToolResultChars,
    });
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

  createCallModel(agentId = 'default') {
    return async (
      state: t.AgentSubgraphState,
      config?: RunnableConfig
    ): Promise<Partial<t.AgentSubgraphState>> => {
      const agentContext = this.agentContexts.get(agentId);
      if (!agentContext) {
        throw new Error(`Agent context not found for agentId: ${agentId}`);
      }

      if (!config) {
        throw new Error('No config provided');
      }

      const { messages } = state;

      const discoveredNames = extractToolDiscoveries(messages);
      if (discoveredNames.length > 0) {
        agentContext.markToolsAsDiscovered(discoveredNames);
      }

      const toolsForBinding = agentContext.getToolsForBinding();
      let model =
        this.overrideModel ??
        initializeModel({
          tools: toolsForBinding,
          provider: agentContext.provider,
          clientOptions: agentContext.clientOptions,
        });

      if (agentContext.systemRunnable) {
        model = agentContext.systemRunnable.pipe(model as Runnable);
      }

      if (agentContext.tokenCalculationPromise) {
        await agentContext.tokenCalculationPromise;
      }
      if (!config.signal) {
        config.signal = this.signal;
      }
      this.config = config;

      let messagesToUse = messages;
      if (
        !agentContext.pruneMessages &&
        agentContext.tokenCounter &&
        agentContext.maxContextTokens != null
      ) {
        agentContext.pruneMessages = createPruneMessages({
          startIndex:
            agentContext.indexTokenCountMap[0] != null ? this.startIndex : 0,
          provider: agentContext.provider,
          tokenCounter: agentContext.tokenCounter,
          maxTokens: agentContext.maxContextTokens,
          thinkingEnabled: isThinkingEnabled(
            agentContext.provider,
            agentContext.clientOptions
          ),
          indexTokenCountMap: agentContext.indexTokenCountMap,
          contextPruningConfig: agentContext.contextPruningConfig,
          summarizationEnabled: agentContext.summarizationEnabled,
          reserveRatio: agentContext.summarizationConfig?.reserveRatio,
          calibrationRatio: agentContext.calibrationRatio,
          getInstructionTokens: () => agentContext.instructionTokens,
          log: (level, message, data) => {
            emitAgentLog(config, level, 'prune', message, data, {
              runId: this.runId,
              agentId,
            });
          },
        });
      }
      if (agentContext.pruneMessages) {
        const {
          context,
          indexTokenCountMap,
          messagesToRefine,
          prePruneContextTokens,
          remainingContextTokens,
          originalToolContent,
          calibrationRatio,
          resolvedInstructionOverhead,
        } = agentContext.pruneMessages({
          messages,
          usageMetadata: agentContext.currentUsage,
          lastCallUsage: agentContext.lastCallUsage,
          totalTokensFresh: agentContext.totalTokensFresh,
        });
        agentContext.indexTokenCountMap = indexTokenCountMap;
        if (calibrationRatio != null && calibrationRatio > 0) {
          agentContext.calibrationRatio = calibrationRatio;
        }
        if (resolvedInstructionOverhead != null) {
          agentContext.resolvedInstructionOverhead =
            resolvedInstructionOverhead;
          const nonToolOverhead =
            agentContext.instructionTokens - agentContext.toolSchemaTokens;
          const calibratedToolTokens = Math.max(
            0,
            resolvedInstructionOverhead - nonToolOverhead
          );
          const currentToolTokens = agentContext.toolSchemaTokens;
          const variance =
            currentToolTokens > 0
              ? Math.abs(calibratedToolTokens - currentToolTokens) /
                currentToolTokens
              : 1;
          if (variance > CALIBRATION_VARIANCE_THRESHOLD) {
            agentContext.toolSchemaTokens = calibratedToolTokens;
          }
        }
        messagesToUse = context;

        const hasPrunedMessages =
          agentContext.summarizationEnabled === true &&
          Array.isArray(messagesToRefine) &&
          messagesToRefine.length > 0;

        if (hasPrunedMessages) {
          const shouldSkip = agentContext.shouldSkipSummarization(
            messages.length
          );
          const triggerResult =
            !shouldSkip &&
            shouldTriggerSummarization({
              trigger: agentContext.summarizationConfig?.trigger,
              maxContextTokens: agentContext.maxContextTokens,
              prePruneContextTokens:
                prePruneContextTokens != null
                  ? prePruneContextTokens + agentContext.instructionTokens
                  : undefined,
              remainingContextTokens,
              messagesToRefineCount: messagesToRefine.length,
            });

          if (triggerResult) {
            if (originalToolContent != null && originalToolContent.size > 0) {
              agentContext.pendingOriginalToolContent = originalToolContent;
            }

            emitAgentLog(
              config,
              'info',
              'graph',
              'Summarization triggered',
              undefined,
              { runId: this.runId, agentId }
            );
            emitAgentLog(
              config,
              'debug',
              'graph',
              'Summarization trigger details',
              {
                totalMessages: messages.length,
                remainingContextTokens: remainingContextTokens ?? 0,
                summaryVersion: agentContext.summaryVersion + 1,
                toolSchemaTokens: agentContext.toolSchemaTokens,
                instructionTokens: agentContext.instructionTokens,
                systemMessageTokens: agentContext.systemMessageTokens,
              },
              { runId: this.runId, agentId }
            );
            agentContext.markSummarizationTriggered(messages.length);
            return {
              summarizationRequest: {
                remainingContextTokens: remainingContextTokens ?? 0,
                agentId: agentId || agentContext.agentId,
              },
            };
          }

          if (shouldSkip) {
            emitAgentLog(
              config,
              'debug',
              'graph',
              'Summarization skipped — no new messages or per-run cap reached',
              {
                messageCount: messages.length,
                messagesToRefineCount: messagesToRefine.length,
                contextLength: context.length,
              },
              { runId: this.runId, agentId }
            );
          }
        }
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

      const anthropicLike = isAnthropicLike(
        agentContext.provider,
        agentContext.clientOptions as { model?: string }
      );

      if (
        agentContext.provider === Providers.BEDROCK &&
        lastMessageX instanceof AIMessageChunk &&
        lastMessageY instanceof ToolMessage &&
        typeof lastMessageX.content === 'string'
      ) {
        const trimmed = lastMessageX.content.trim();
        finalMessages[finalMessages.length - 2].content =
          trimmed.length > 0 ? [{ type: 'text' as const, text: trimmed }] : '';
      }

      if (lastMessageY instanceof ToolMessage) {
        if (anthropicLike) {
          formatAnthropicArtifactContent(finalMessages);
        } else if (
          (isOpenAILike(agentContext.provider) &&
            agentContext.provider !== Providers.DEEPSEEK) ||
          isGoogleLike(agentContext.provider)
        ) {
          formatArtifactPayload(finalMessages);
        }
      }

      if (agentContext.provider === Providers.ANTHROPIC) {
        const anthropicOptions = agentContext.clientOptions as
          | t.AnthropicClientOptions
          | undefined;
        if (
          anthropicOptions?.promptCache === true &&
          !agentContext.systemRunnable
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

      if (
        isThinkingEnabled(agentContext.provider, agentContext.clientOptions)
      ) {
        finalMessages = ensureThinkingBlockInMessages(
          finalMessages,
          agentContext.provider,
          config
        );
      }

      // Intentionally broad: runs when the pruner wasn't used OR any post-pruning
      // transform (addCacheControl, ensureThinkingBlock, etc.) reassigned finalMessages.
      // sanitizeOrphanToolBlocks fast-paths to a Set diff check when no orphans exist,
      // so the cost is negligible and this acts as a safety net for Anthropic/Bedrock.
      const needsOrphanSanitize =
        anthropicLike &&
        (!agentContext.pruneMessages || finalMessages !== messagesToUse);
      if (needsOrphanSanitize) {
        const beforeSanitize = finalMessages.length;
        finalMessages = sanitizeOrphanToolBlocks(finalMessages);
        if (finalMessages.length !== beforeSanitize) {
          emitAgentLog(
            config,
            'warn',
            'sanitize',
            'Orphan tool blocks removed',
            {
              before: beforeSanitize,
              after: finalMessages.length,
              dropped: beforeSanitize - finalMessages.length,
            },
            { runId: this.runId, agentId }
          );
        }
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
      agentContext.markTokensStale();

      let result: Partial<t.BaseGraphState> | undefined;
      const fallbacks =
        (agentContext.clientOptions as t.LLMConfig | undefined)?.fallbacks ??
        [];

      if (
        finalMessages.length === 0 &&
        !agentContext.hasPendingCompactionSummary()
      ) {
        const budgetBreakdown = agentContext.getTokenBudgetBreakdown(messages);
        const breakdown = agentContext.formatTokenBudgetBreakdown(messages);
        const instructionsExceedBudget =
          budgetBreakdown.instructionTokens > budgetBreakdown.maxContextTokens;

        let guidance: string;
        if (instructionsExceedBudget) {
          const toolPct =
            budgetBreakdown.toolSchemaTokens > 0
              ? Math.round(
                (budgetBreakdown.toolSchemaTokens /
                    budgetBreakdown.instructionTokens) *
                    100
              )
              : 0;
          guidance =
            toolPct > 50
              ? `Tool definitions consume ${budgetBreakdown.toolSchemaTokens} tokens (${toolPct}% of instructions) across ${budgetBreakdown.toolCount} tools, exceeding maxContextTokens (${budgetBreakdown.maxContextTokens}). Reduce the number of tools or increase maxContextTokens.`
              : `Instructions (${budgetBreakdown.instructionTokens} tokens) exceed maxContextTokens (${budgetBreakdown.maxContextTokens}). Increase maxContextTokens or shorten the system prompt.`;
          if (agentContext.summarizationEnabled === true) {
            guidance +=
              ' Summarization was skipped because the summary would further increase the instruction overhead.';
          }
        } else {
          guidance =
            'Please increase the context window size or make your message shorter.';
        }

        emitAgentLog(
          config,
          'error',
          'graph',
          'Empty messages after pruning',
          {
            messageCount: messages.length,
            instructionsExceedBudget,
            breakdown,
          },
          { runId: this.runId, agentId }
        );
        throw new Error(
          JSON.stringify({
            type: 'empty_messages',
            info: `Message pruning removed all messages as none fit in the context window. ${guidance}\n${breakdown}`,
          })
        );
      }

      const invokeStart = Date.now();
      const invokeMeta = { runId: this.runId, agentId };
      emitAgentLog(
        config,
        'debug',
        'graph',
        'Invoking LLM',
        {
          messageCount: finalMessages.length,
          provider: agentContext.provider,
        },
        invokeMeta,
        { force: true }
      );

      try {
        result = await attemptInvoke(
          {
            model: (this.overrideModel ?? model) as t.ChatModel,
            messages: finalMessages,
            provider: agentContext.provider,
            context: this,
          },
          config
        );
      } catch (primaryError) {
        result = await tryFallbackProviders({
          fallbacks,
          tools: agentContext.tools,
          messages: finalMessages,
          config,
          primaryError,
          context: this,
        });
      }

      if (!result) {
        throw new Error('No result after model invocation');
      }

      /**
       * Fallback: populate toolCallStepIds in the graph execution context.
       *
       * When model.stream() is available (the common case), attemptInvoke
       * processes all chunks through a local ChatModelStreamHandler which
       * creates run steps and populates toolCallStepIds before returning.
       * The code below is a fallback for the rare case where model.stream
       * is unavailable and model.invoke() was used instead.
       *
       * Text content is dispatched FIRST so that MESSAGE_CREATION is the
       * current step when handleToolCalls runs. handleToolCalls then creates
       * TOOL_CALLS on top of it. The dedup in getMessageId and
       * toolCallStepIds.has makes this safe when attemptInvoke already
       * handled everything — both paths become no-ops.
       */
      const responseMessage = result.messages?.[0];
      const toolCalls = (responseMessage as AIMessageChunk | undefined)
        ?.tool_calls;
      const hasToolCalls = Array.isArray(toolCalls) && toolCalls.length > 0;

      if (hasToolCalls) {
        const metadata = config.metadata as Record<string, unknown>;
        const stepKey = this.getStepKey(metadata);
        const content = responseMessage?.content as MessageContent | undefined;
        const hasTextContent =
          content != null &&
          (typeof content === 'string'
            ? content !== ''
            : Array.isArray(content) && content.length > 0);

        /**
         * Dispatch text content BEFORE creating TOOL_CALLS steps.
         * getMessageId returns a new ID only on the first call for a step key;
         * if the for-await consumer already claimed it, this is a no-op.
         */
        if (hasTextContent) {
          const messageId = getMessageId(stepKey, this) ?? '';
          if (messageId) {
            await this.dispatchRunStep(
              stepKey,
              {
                type: StepTypes.MESSAGE_CREATION,
                message_creation: { message_id: messageId },
              },
              metadata
            );
            const stepId = this.getStepIdByKey(stepKey);
            if (typeof content === 'string') {
              await this.dispatchMessageDelta(stepId, {
                content: [{ type: ContentTypes.TEXT, text: content }],
              });
            } else if (
              Array.isArray(content) &&
              content.every(
                (c) =>
                  typeof c === 'object' &&
                  'type' in c &&
                  typeof c.type === 'string' &&
                  c.type.startsWith('text')
              )
            ) {
              await this.dispatchMessageDelta(stepId, {
                content: content as t.MessageDelta['content'],
              });
            }
          }
        }

        await handleToolCalls(toolCalls as ToolCall[], metadata, this);
      }

      /**
       * When streaming is disabled, on_chat_model_stream events are never
       * emitted so ChatModelStreamHandler never fires. Dispatch the text
       * content as MESSAGE_CREATION + MESSAGE_DELTA here.
       */
      const disableStreaming =
        (agentContext.clientOptions as t.OpenAIClientOptions | undefined)
          ?.disableStreaming === true;

      if (
        disableStreaming &&
        !hasToolCalls &&
        responseMessage != null &&
        (responseMessage.content as MessageContent | undefined) != null
      ) {
        const metadata = config.metadata as Record<string, unknown>;
        const stepKey = this.getStepKey(metadata);
        const messageId = getMessageId(stepKey, this) ?? '';
        if (messageId) {
          await this.dispatchRunStep(
            stepKey,
            {
              type: StepTypes.MESSAGE_CREATION,
              message_creation: { message_id: messageId },
            },
            metadata
          );
          const stepId = this.getStepIdByKey(stepKey);
          const content = responseMessage.content;
          if (typeof content === 'string') {
            await this.dispatchMessageDelta(stepId, {
              content: [{ type: ContentTypes.TEXT, text: content }],
            });
          } else if (
            Array.isArray(content) &&
            content.every(
              (c) =>
                typeof c === 'object' &&
                'type' in c &&
                typeof c.type === 'string' &&
                c.type.startsWith('text')
            )
          ) {
            await this.dispatchMessageDelta(stepId, {
              content: content as t.MessageDelta['content'],
            });
          }
        }
      }

      const invokeElapsed = ((Date.now() - invokeStart) / 1000).toFixed(2);
      agentContext.currentUsage = this.getUsageMetadata(result.messages?.[0]);
      if (agentContext.currentUsage) {
        agentContext.updateLastCallUsage(agentContext.currentUsage);
        emitAgentLog(
          config,
          'debug',
          'graph',
          `LLM call complete (${invokeElapsed}s)`,
          {
            ...agentContext.currentUsage,
            elapsedSeconds: Number(invokeElapsed),
            instructionTokens: agentContext.instructionTokens,
            toolSchemaTokens: agentContext.toolSchemaTokens,
            messageCount: finalMessages.length,
          },
          invokeMeta,
          { force: true }
        );
      } else {
        emitAgentLog(
          config,
          'debug',
          'graph',
          `LLM call complete (${invokeElapsed}s)`,
          {
            elapsedSeconds: Number(invokeElapsed),
            messageCount: finalMessages.length,
          },
          invokeMeta,
          { force: true }
        );
      }
      this.cleanupSignalListener();
      return result;
    };
  }

  createAgentNode(agentId: string): t.CompiledAgentWorfklow {
    const getConfig = (): RunnableConfig | undefined => this.config;
    const agentContext = this.agentContexts.get(agentId);
    if (!agentContext) {
      throw new Error(`Agent context not found for agentId: ${agentId}`);
    }

    const agentNode = `${AGENT}${agentId}` as const;
    const toolNode = `${TOOLS}${agentId}` as const;
    const summarizeNode = `${SUMMARIZE}${agentId}` as const;

    const routeMessage = (
      state: t.AgentSubgraphState,
      config?: RunnableConfig
    ): string => {
      this.config = config;
      if (state.summarizationRequest != null) {
        return summarizeNode;
      }
      return toolsCondition(
        state as t.BaseGraphState,
        toolNode,
        this.invokedToolIds
      );
    };

    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
      }),
      summarizationRequest: Annotation<t.SummarizationNodeInput | undefined>({
        reducer: (
          _: t.SummarizationNodeInput | undefined,
          b: t.SummarizationNodeInput | undefined
        ) => b,
        default: () => undefined,
      }),
    });

    const workflow = new StateGraph(StateAnnotation)
      .addNode(agentNode, this.createCallModel(agentId))
      .addNode(
        toolNode,
        this.initializeTools({
          currentTools: agentContext.tools,
          currentToolMap: agentContext.toolMap,
          agentContext,
        })
      )
      .addNode(
        summarizeNode,
        createSummarizeNode({
          agentContext,
          graph: {
            contentData: this.contentData,
            contentIndexMap: this.contentIndexMap,
            get config() {
              return getConfig();
            },
            runId: this.runId,
            isMultiAgent: this.isMultiAgentGraph(),
            dispatchRunStep: async (runStep, nodeConfig) => {
              this.contentData.push(runStep);
              this.contentIndexMap.set(runStep.id, runStep.index);

              const resolvedConfig = nodeConfig ?? this.config;
              const handler = this.handlerRegistry?.getHandler(
                GraphEvents.ON_RUN_STEP
              );
              if (handler) {
                await handler.handle(
                  GraphEvents.ON_RUN_STEP,
                  runStep,
                  resolvedConfig?.configurable,
                  this
                );
                this.handlerDispatchedStepIds.add(runStep.id);
              }

              if (resolvedConfig) {
                await safeDispatchCustomEvent(
                  GraphEvents.ON_RUN_STEP,
                  runStep,
                  resolvedConfig
                );
              }
            },
            dispatchRunStepCompleted: async (
              stepId: string,
              result: t.StepCompleted,
              nodeConfig?: RunnableConfig
            ) => {
              const resolvedConfig = nodeConfig ?? this.config;
              const runStep = this.contentData.find((s) => s.id === stepId);
              const handler = this.handlerRegistry?.getHandler(
                GraphEvents.ON_RUN_STEP_COMPLETED
              );
              if (handler) {
                await handler.handle(
                  GraphEvents.ON_RUN_STEP_COMPLETED,
                  {
                    result: {
                      ...result,
                      id: stepId,
                      index: runStep?.index ?? 0,
                    },
                  },
                  resolvedConfig?.configurable,
                  this
                );
              }
            },
          },
          generateStepId: (stepKey: string) => this.generateStepId(stepKey),
        })
      )
      .addEdge(START, agentNode)
      .addConditionalEdges(agentNode, routeMessage)
      .addEdge(summarizeNode, agentNode)
      .addEdge(toolNode, agentContext.toolEnd ? END : agentNode);

    return workflow.compile();
  }

  createWorkflow(): t.CompiledStateWorkflow {
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
      // LangGraph compile() types are overly strict for opt-in options
      .compile(this.compileOptions as unknown as never);

    return workflow;
  }

  /**
   * Indicates if this is a multi-agent graph.
   * Override in MultiAgentGraph to return true.
   * Used to conditionally include agentId in RunStep for frontend rendering.
   */
  protected isMultiAgentGraph(): boolean {
    return false;
  }

  /**
   * Get the parallel group ID for an agent, if any.
   * Override in MultiAgentGraph to provide actual group IDs.
   * Group IDs are incrementing numbers (1, 2, 3...) reflecting execution order.
   * @param _agentId - The agent ID to look up
   * @returns undefined for StandardGraph (no parallel groups), or group number for MultiAgentGraph
   */
  protected getParallelGroupIdForAgent(_agentId: string): number | undefined {
    return undefined;
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

    if (metadata) {
      try {
        const agentContext = this.getAgentContext(metadata);
        if (this.isMultiAgentGraph() && agentContext.agentId) {
          runStep.agentId = agentContext.agentId;
          const groupId = this.getParallelGroupIdForAgent(agentContext.agentId);
          if (groupId != null) {
            runStep.groupId = groupId;
          }
        }
      } catch (_e) {
        /** If we can't get agent context, that's okay - agentId remains undefined */
      }
    }

    this.contentData.push(runStep);
    this.contentIndexMap.set(stepId, runStep.index);

    // Primary dispatch: handler registry (reliable, always works).
    // This mirrors how handleToolCallCompleted dispatches ON_RUN_STEP_COMPLETED
    // via the handler registry, ensuring the event always reaches the handler
    // even when LangGraph's callback system drops the custom event.
    const handler = this.handlerRegistry?.getHandler(GraphEvents.ON_RUN_STEP);
    if (handler) {
      await handler.handle(GraphEvents.ON_RUN_STEP, runStep, metadata, this);
      this.handlerDispatchedStepIds.add(stepId);
    }

    // Secondary dispatch: custom event for LangGraph callback chain
    // (tracing, Langfuse, external consumers).  May be silently dropped
    // in some scenarios (stale run ID, subgraph callback propagation issues),
    // but the primary dispatch above guarantees the event reaches the handler.
    // The customEventCallback in run.ts skips events already dispatched above
    // to prevent double handling.
    await safeDispatchCustomEvent(
      GraphEvents.ON_RUN_STEP,
      runStep,
      this.config
    );
    return stepId;
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
