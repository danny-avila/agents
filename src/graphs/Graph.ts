/* eslint-disable no-console */
import { nanoid } from 'nanoid';
import { tool } from '@langchain/core/tools';
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
  enforceOriginalContentCap,
  formatContentStrings,
  createPruneMessages,
  addCacheControl,
  getMessageId,
  makeIsDeferred,
  partitionAndMarkAnthropicToolCache,
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
import { SubagentExecutor, resolveSubagentConfigs } from '@/tools/subagent';
import { buildSubagentToolParams } from '@/tools/SubagentTool';
import { ToolNode as CustomToolNode, toolsCondition } from '@/tools/ToolNode';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { shouldTriggerSummarization } from '@/summarization';
import { createSummarizeNode } from '@/summarization/node';
import { messagesStateReducer } from '@/messages/reducer';
import {
  appendCallbacks,
  findCallback,
  type CallbackEntry,
} from '@/utils/callbacks';
import { createSchemaOnlyTools } from '@/tools/schema';
import { AgentContext } from '@/agents/AgentContext';
import { createFakeStreamingLLM } from '@/llm/fake';
import { handleToolCalls } from '@/tools/handlers';
import { resolveLocalToolsForBinding } from '@/tools/local';
import { createLocalCodingToolBundle } from '@/tools/local/LocalCodingTools';
import { createCloudflareCodingToolBundle } from '@/tools/cloudflare';
import { isThinkingEnabled } from '@/llm/request';
import { initializeModel } from '@/llm/init';
import {
  createLangfuseHandler,
  createLangfuseTraceMetadata,
  disposeLangfuseHandler,
  isLangfuseCallbackHandler,
} from '@/langfuse';
import { initializeLangfuseTracing } from '@/instrumentation';
import {
  resolveLangfuseConfig,
  shouldTraceToolNodeForLangfuse,
  withLangfuseToolOutputTracingConfig,
} from '@/langfuseToolOutputTracing';
import { HandlerRegistry } from '@/events';
import { ChatOpenAI } from '@/llm/openai';
import { partitionAndMarkOpenRouterToolCache } from '@/llm/openrouter/toolCache';
import { partitionAndMarkBedrockToolCache } from '@/llm/bedrock/toolCache';
import type { HookRegistry } from '@/hooks';

const { AGENT, TOOLS, SUMMARIZE } = GraphNodeKeys;

/** Minimum relative variance before calibrated toolSchemaTokens overrides current value. */
const CALIBRATION_VARIANCE_THRESHOLD = 0.15;

type ReasoningKey = 'reasoning_content' | 'reasoning';
type ReasoningSummary = { summary?: Array<{ text?: string }> };
type ReasoningDetail = { type?: string; text?: string };

function getHandlerDispatchedEventKey(
  eventName: string,
  stepId: string
): string {
  return `${eventName}:${stepId}`;
}

function getReasoningText(
  value: string | Partial<ReasoningSummary> | null | undefined
): string | undefined {
  if (typeof value === 'string') {
    return value !== '' ? value : undefined;
  }
  const summaryText = value?.summary
    ?.map((summary) => summary.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return summaryText != null && summaryText !== '' ? summaryText : undefined;
}

function getReasoningDetailsText(
  value: ReasoningDetail[] | null | undefined
): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const reasoningText = value
    .filter((detail) => detail.type === 'reasoning.text')
    .map((detail) => detail.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return reasoningText !== '' ? reasoningText : undefined;
}

function getResponseReasoningContent({
  responseMessage,
  reasoningKey,
}: {
  responseMessage?: Partial<AIMessageChunk>;
  reasoningKey: ReasoningKey;
}): string | undefined {
  const additionalKwargs = responseMessage?.additional_kwargs;
  if (additionalKwargs == null) {
    return undefined;
  }

  const keyedReasoning = getReasoningText(
    additionalKwargs[reasoningKey] as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (keyedReasoning != null) {
    return keyedReasoning;
  }

  const reasoningContent = getReasoningText(
    additionalKwargs.reasoning_content as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (reasoningContent != null) {
    return reasoningContent;
  }

  const reasoning = getReasoningText(
    additionalKwargs.reasoning as
      | string
      | Partial<ReasoningSummary>
      | null
      | undefined
  );
  if (reasoning != null) {
    return reasoning;
  }

  return getReasoningDetailsText(
    additionalKwargs.reasoning_details as ReasoningDetail[] | null | undefined
  );
}

function getTextMessageDeltaContent(
  content: MessageContent | undefined
): t.MessageDelta['content'] | undefined {
  if (content == null) {
    return undefined;
  }
  if (typeof content === 'string') {
    return content !== ''
      ? [{ type: ContentTypes.TEXT, text: content }]
      : undefined;
  }
  if (content.length === 0) {
    return undefined;
  }
  if (
    !content.every(
      (contentPart) =>
        typeof contentPart === 'object' &&
        'type' in contentPart &&
        typeof contentPart.type === 'string' &&
        contentPart.type.startsWith('text')
    )
  ) {
    return undefined;
  }
  return content as t.MessageDelta['content'];
}

async function dispatchTextMessageContent({
  graph,
  stepKey,
  content,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  stepKey: string;
  content: t.MessageDelta['content'];
  metadata: Record<string, unknown>;
}): Promise<boolean> {
  const messageId = getMessageId(stepKey, graph) ?? '';
  if (!messageId) {
    return false;
  }
  await graph.dispatchRunStep(
    stepKey,
    {
      type: StepTypes.MESSAGE_CREATION,
      message_creation: { message_id: messageId },
    },
    metadata
  );
  const stepId = graph.getStepIdByKey(stepKey);
  await graph.dispatchMessageDelta(stepId, { content }, metadata);
  return true;
}

async function dispatchReasoningContent({
  graph,
  agentContext,
  reasoningContent,
  metadata,
}: {
  graph: Graph<t.BaseGraphState>;
  agentContext: AgentContext;
  reasoningContent: string;
  metadata: Record<string, unknown>;
}): Promise<boolean> {
  const previousTokenType = agentContext.currentTokenType;
  const previousTokenTypeSwitch = agentContext.tokenTypeSwitch;
  const previousTransitionCount = agentContext.reasoningTransitionCount;

  agentContext.currentTokenType = ContentTypes.THINK;
  agentContext.tokenTypeSwitch = 'reasoning';

  const stepKey = graph.getStepKey(metadata);
  const messageId = getMessageId(stepKey, graph) ?? '';
  if (!messageId) {
    agentContext.currentTokenType = previousTokenType;
    agentContext.tokenTypeSwitch = previousTokenTypeSwitch;
    agentContext.reasoningTransitionCount = previousTransitionCount;
    return false;
  }

  await graph.dispatchRunStep(
    stepKey,
    {
      type: StepTypes.MESSAGE_CREATION,
      message_creation: { message_id: messageId },
    },
    metadata
  );
  const stepId = graph.getStepIdByKey(stepKey);
  await graph.dispatchReasoningDelta(
    stepId,
    {
      content: [{ type: ContentTypes.THINK, think: reasoningContent }],
    },
    metadata
  );
  return true;
}

function markPostReasoningContent(agentContext: AgentContext): void {
  if (
    agentContext.tokenTypeSwitch !== 'reasoning' ||
    agentContext.currentTokenType === ContentTypes.TEXT
  ) {
    return;
  }
  agentContext.currentTokenType = ContentTypes.TEXT;
  agentContext.tokenTypeSwitch = 'content';
  agentContext.reasoningTransitionCount++;
}

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
    delta: t.ToolCallDelta,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  abstract dispatchMessageDelta(
    id: string,
    delta: t.MessageDelta,
    metadata?: Record<string, unknown>
  ): Promise<void>;
  abstract dispatchReasoningDelta(
    stepId: string,
    delta: t.ReasoningDelta,
    metadata?: Record<string, unknown>
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
   * Step IDs dispatched through the handler registry during this run.
   * Event echo suppression is tracked separately so repeated deltas for
   * the same step are scoped to the active custom event dispatch.
   */
  handlerDispatchedStepIds: Set<string> = new Set();
  protected handlerDispatchedEventCounts: Map<string, number> = new Map();
  signal?: AbortSignal;
  /** Set of invoked tool call IDs from non-message run steps completed mid-run, if any */
  invokedToolIds?: Set<string>;
  handlerRegistry: HandlerRegistry | undefined;
  /**
   * True when event-driven tool execution can be routed through callbacks even
   * though this graph intentionally does not own the full handler registry.
   * Self-spawned subagent graphs use this shape: their callback forwarder sends
   * `ON_TOOL_EXECUTE` to the parent's handler, while child run-step events stay
   * wrapped as `ON_SUBAGENT_UPDATE` instead of leaking as parent events.
   */
  eventToolExecutionAvailable: boolean = false;
  hookRegistry: HookRegistry | undefined;
  /**
   * Run-scoped HITL configuration. When `humanInTheLoop?.enabled` is
   * `true`, `ToolNode` raises a real `interrupt()` for `PreToolUse`
   * `ask` decisions instead of treating them as a synchronous deny.
   * Threaded from `RunConfig.humanInTheLoop`.
   */
  humanInTheLoop: t.HumanInTheLoopConfig | undefined;
  /**
   * Run-scoped config for the tool output reference registry. Threaded
   * from `RunConfig.toolOutputReferences` down into every ToolNode this
   * graph compiles.
   */
  toolOutputReferences: t.ToolOutputReferencesConfig | undefined;
  /**
   * Run-scoped Langfuse defaults. Per-agent config wins when present.
   */
  langfuse: t.LangfuseConfig | undefined;
  /**
   * Run-scoped opt-in for eager event-driven tool execution. The stream
   * handler may prestart eligible event-driven tools; ToolNode later
   * consumes the settled promises while preserving final ToolMessage order.
   */
  eagerEventToolExecution: t.EagerEventToolExecutionConfig | undefined;
  eagerEventToolExecutions: Map<string, t.EagerEventToolExecution> = new Map();
  eagerEventToolUsageCount: Map<string, number> = new Map();
  private eagerEventToolUsageCountsByAgentId: Map<string, Map<string, number>> =
    new Map();
  eagerEventToolCallChunks: Map<string, t.EagerEventToolCallChunkState> =
    new Map();
  /**
   * Run-scoped execution backend for built-in code tools. Defaults to the
   * remote Code API sandbox when unset.
   */
  toolExecution: t.ToolExecutionConfig | undefined;
  /**
   * Shared registry instance used by every ToolNode compiled from this
   * graph. Lazily constructed on first access so multi-agent graphs
   * produce one registry per run (not one per agent), letting cross-
   * agent `{{tool<i>turn<n>}}` substitutions resolve.
   */
  private _toolOutputRegistry?: ToolOutputReferenceRegistry;
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
    this.hookRegistry = undefined;
    this.humanInTheLoop = undefined;
    this.toolOutputReferences = undefined;
    this.eagerEventToolExecution = undefined;
    this.eagerEventToolExecutions.clear();
    this.clearEagerEventToolUsageCounts();
    this.eagerEventToolCallChunks.clear();
    this.toolExecution = undefined;
    this.handlerDispatchedEventCounts.clear();
    /**
     * ToolNodes compiled from this graph captured the registry
     * instance at construction time, so simply dropping the Graph's
     * own reference would leave their captured reference — and every
     * stored `tool<i>turn<n>` entry, plus up to `maxTotalSize` of raw
     * output — alive across subsequent `processStream()` calls. Wipe
     * the registry's contents first so subsequent runs start fresh.
     */
    this._toolOutputRegistry?.clear();
    this._toolOutputRegistry = undefined;
    // NB: `_fileCheckpointer` is intentionally NOT cleared here.
    // `Run.processStream()` calls `clearHeavyState()` in its
    // finally block on natural-completion / error paths — exactly
    // when the host is most likely to want `Run.rewindFiles()` (for
    // rollback after a failed batch). Per-Run isolation is already
    // automatic because each `Run.create()` constructs a brand-new
    // Graph instance, so the next Run gets its own checkpointer
    // without us needing to reset this field. Codex P1 #32: pre-fix
    // the checkpointer was nulled before the caller could reach it.
    // Flush each compiled ToolNode's direct-path turn cache so it
    // doesn't leak across Runs (Codex P2 #33). The cache survives
    // `run()` re-entry by design (resume-stable), but end-of-Run
    // is the right point to reset it.
    for (const node of this._compiledToolNodes) {
      node.clearDirectPathTurns();
    }
    this._compiledToolNodes.clear();
    this.sessions.clear();
  }

  getEagerEventToolUsageCount(agentId?: string): Map<string, number> {
    if (agentId == null || agentId === '') {
      return this.eagerEventToolUsageCount;
    }
    let usageCount = this.eagerEventToolUsageCountsByAgentId.get(agentId);
    if (usageCount == null) {
      usageCount = new Map<string, number>();
      this.eagerEventToolUsageCountsByAgentId.set(agentId, usageCount);
    }
    return usageCount;
  }

  protected clearEagerEventToolUsageCounts(): void {
    this.eagerEventToolUsageCount.clear();
    for (const usageCount of this.eagerEventToolUsageCountsByAgentId.values()) {
      usageCount.clear();
    }
  }

  markHandlerDispatchedEvent(eventName: string, stepId: string): () => void {
    const key = getHandlerDispatchedEventKey(eventName, stepId);
    this.handlerDispatchedEventCounts.set(
      key,
      (this.handlerDispatchedEventCounts.get(key) ?? 0) + 1
    );
    return () => {
      const count = this.handlerDispatchedEventCounts.get(key) ?? 0;
      if (count <= 1) {
        this.handlerDispatchedEventCounts.delete(key);
        return;
      }
      this.handlerDispatchedEventCounts.set(key, count - 1);
    };
  }

  hasHandlerDispatchedEvent(eventName: string, stepId: string): boolean {
    const key = getHandlerDispatchedEventKey(eventName, stepId);
    return (this.handlerDispatchedEventCounts.get(key) ?? 0) > 0;
  }

  /**
   * Subclass hook to register a freshly compiled ToolNode so
   * `clearHeavyState` can flush its per-Run direct-path turn cache
   * at end-of-Run. Internal — called from `initializeTools` in the
   * concrete graph subclasses.
   */
  protected registerCompiledToolNode(node: {
    clearDirectPathTurns(): void;
  }): void {
    this._compiledToolNodes.add(node);
  }

  /**
   * Returns the shared `ToolOutputReferenceRegistry` for this run,
   * constructing it on first access. Returns `undefined` when the
   * feature is disabled. All ToolNodes compiled from this graph share
   * this single instance so cross-agent `{{…}}` references resolve.
   *
   * @internal Public so `attemptInvoke` can read it through the typed
   * `InvokeContext` and project ToolMessages into LLM-facing annotated
   * copies right before each provider call (see
   * `annotateMessagesForLLM`). Host code should not call this directly
   * — registry mutations outside the ToolNode lifecycle break the
   * partitioning, eviction, and turn-counter invariants.
   */
  public getOrCreateToolOutputRegistry():
    | ToolOutputReferenceRegistry
    | undefined {
    if (this.toolOutputReferences?.enabled !== true) {
      return undefined;
    }
    if (this._toolOutputRegistry == null) {
      this._toolOutputRegistry = new ToolOutputReferenceRegistry({
        maxOutputSize: this.toolOutputReferences.maxOutputSize,
        maxTotalSize: this.toolOutputReferences.maxTotalSize,
      });
    }
    return this._toolOutputRegistry;
  }

  /**
   * Single per-Run file checkpointer shared across every ToolNode the
   * graph compiles. Lazily constructed when
   * `toolExecution.local.fileCheckpointing === true` or
   * `toolExecution.cloudflare.fileCheckpointing === true` so
   * multi-agent graphs see ONE snapshot store, not one-per-agent.
   * Returns undefined when checkpointing is disabled or a supported
   * coding-tool engine isn't selected. Exposed via
   * `Run.getFileCheckpointer()` / `Run.rewindFiles()`.
   */
  private _fileCheckpointer?: t.LocalFileCheckpointer;
  /**
   * ToolNodes compiled into this Graph's workflow. Tracked so
   * `clearHeavyState()` can flush their per-Run direct-path turn
   * cache (`directPathTurns`) at end-of-Run — that map intentionally
   * survives `run()` re-entry (resume-stable per Codex P2 #30) but
   * would otherwise grow linearly with tool calls and could collide
   * across Runs if a provider reuses call ids (Codex P2 #33).
   */
  private _compiledToolNodes: Set<{
    clearDirectPathTurns(): void;
  }> = new Set();
  public getOrCreateFileCheckpointer(): t.LocalFileCheckpointer | undefined {
    // Return the cached instance unconditionally if one exists. The
    // toolExecution check below decides whether to *create* a new
    // one — `clearHeavyState` nulls `this.toolExecution` at end-of-
    // Run, but we want post-Run `Run.rewindFiles()` to still resolve
    // to the checkpointer that captured the writes. Codex P1 #32.
    if (this._fileCheckpointer != null) {
      return this._fileCheckpointer;
    }
    // Eagerly create via the bundle factory so the construction path
    // matches the bundle-only callers (and future bundle-internal
    // cleanup hooks fire). The bundle factory itself accepts a pre-
    // supplied checkpointer when present, so re-injecting this one
    // into every ToolNode is idempotent.
    if (
      this.toolExecution?.engine === 'local' &&
      this.toolExecution.local?.fileCheckpointing === true
    ) {
      const bundle = createLocalCodingToolBundle(
        this.toolExecution.local ?? {}
      );
      this._fileCheckpointer = bundle.checkpointer;
      return this._fileCheckpointer;
    }
    if (
      this.toolExecution?.engine === 'cloudflare-sandbox' &&
      this.toolExecution.cloudflare?.fileCheckpointing === true
    ) {
      const bundle = createCloudflareCodingToolBundle(
        this.toolExecution.cloudflare
      );
      this._fileCheckpointer = bundle.checkpointer;
      return this._fileCheckpointer;
    }
    return undefined;
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
    langfuse,
    tokenCounter,
    indexTokenCountMap,
    calibrationRatio,
  }: t.StandardGraphInput) {
    super();
    this.runId = runId;
    this.signal = signal;
    this.langfuse = langfuse;

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
    this.eagerEventToolExecutions.clear();
    this.clearEagerEventToolUsageCounts();
    this.eagerEventToolCallChunks.clear();
    this.handlerDispatchedStepIds = resetIfNotEmpty(
      this.handlerDispatchedStepIds,
      new Set()
    );
    this.handlerDispatchedEventCounts = resetIfNotEmpty(
      this.handlerDispatchedEventCounts,
      new Map()
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

    const configurable = this.config?.configurable;
    const runId =
      (metadata.run_id as string | undefined) ??
      (configurable?.run_id as string | undefined) ??
      this.runId;
    const threadId =
      (metadata.thread_id as string | undefined) ??
      (configurable?.thread_id as string | undefined) ??
      runId;
    const checkpointNs =
      (metadata.checkpoint_ns as string | undefined) ??
      (metadata.langgraph_checkpoint_ns as string | undefined) ??
      '';
    const keyList = [
      runId,
      threadId,
      metadata.langgraph_node as string,
      metadata.langgraph_step as number,
      checkpointNs,
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
    const traceToolNode = shouldTraceToolNodeForLangfuse({
      runLangfuse: this.langfuse,
      agentLangfuse: agentContext?.langfuse,
    });

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

      const node = new CustomToolNode<t.BaseGraphState>({
        tools: allTools,
        toolMap: allToolMap,
        trace: traceToolNode,
        runLangfuse: this.langfuse,
        agentLangfuse: agentContext?.langfuse,
        eventDrivenMode: true,
        sessions: this.sessions,
        toolDefinitions: toolDefMap,
        agentId: agentContext?.agentId,
        toolCallStepIds: this.toolCallStepIds,
        toolRegistry: agentContext?.toolRegistry,
        hookRegistry: this.hookRegistry,
        humanInTheLoop: this.humanInTheLoop,
        eagerEventToolExecution: this.eagerEventToolExecution,
        eagerEventToolExecutions: this.eagerEventToolExecutions,
        eagerEventToolUsageCount: this.getEagerEventToolUsageCount(
          agentContext?.agentId
        ),
        toolExecution: this.toolExecution,
        directToolNames: directToolNames.size > 0 ? directToolNames : undefined,
        maxContextTokens: agentContext?.maxContextTokens,
        maxToolResultChars: agentContext?.maxToolResultChars,
        toolOutputRegistry: this.getOrCreateToolOutputRegistry(),
        fileCheckpointer: this.getOrCreateFileCheckpointer(),
        errorHandler: (data, metadata): Promise<void> =>
          StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      });
      this.registerCompiledToolNode(node);
      return node;
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

    const node = new CustomToolNode<t.BaseGraphState>({
      tools: allTraditionalTools,
      toolMap: traditionalToolMap,
      trace: traceToolNode,
      runLangfuse: this.langfuse,
      agentLangfuse: agentContext?.langfuse,
      toolCallStepIds: this.toolCallStepIds,
      errorHandler: (data, metadata): Promise<void> =>
        StandardGraph.handleToolCallErrorStatic(this, data, metadata),
      toolRegistry: agentContext?.toolRegistry,
      sessions: this.sessions,
      toolExecution: this.toolExecution,
      hookRegistry: this.hookRegistry,
      humanInTheLoop: this.humanInTheLoop,
      maxContextTokens: agentContext?.maxContextTokens,
      maxToolResultChars: agentContext?.maxToolResultChars,
      toolOutputRegistry: this.getOrCreateToolOutputRegistry(),
      fileCheckpointer: this.getOrCreateFileCheckpointer(),
    });
    this.registerCompiledToolNode(node);
    return node;
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

      const rawToolsForBinding = resolveLocalToolsForBinding({
        tools: agentContext.getToolsForBinding(),
        toolExecution: this.toolExecution,
      });

      /**
       * Anthropic prompt-cache breakpoint on the tool definitions.
       *
       * Without this, the (often static) tool inventory shows up as
       * fresh input on every turn — measured at ~28k tokens/turn for
       * the local engine's coding-tool bundle, dominating per-turn
       * cost even when message-level caching is on.
       *
       * Strategy: partition tools into [static, deferred] and stamp
       * `cache_control: ephemeral` on the last static tool.
       * Discovered deferred tools that arrive across turns sit *after*
       * the breakpoint and don't invalidate the prefix.
       */
      let toolsForBinding = rawToolsForBinding;
      if (
        agentContext.provider === Providers.ANTHROPIC &&
        (agentContext.clientOptions as t.AnthropicClientOptions | undefined)
          ?.promptCache === true
      ) {
        toolsForBinding =
          partitionAndMarkAnthropicToolCache(
            rawToolsForBinding,
            makeIsDeferred(agentContext.toolDefinitions)
          ) ?? rawToolsForBinding;
      } else if (
        agentContext.provider === Providers.OPENROUTER &&
        (
          agentContext.clientOptions as
            | t.ProviderOptionsMap[Providers.OPENROUTER]
            | undefined
        )?.promptCache === true
      ) {
        toolsForBinding =
          partitionAndMarkOpenRouterToolCache(
            rawToolsForBinding,
            makeIsDeferred(agentContext.toolDefinitions)
          ) ?? rawToolsForBinding;
      } else if (
        agentContext.provider === Providers.BEDROCK &&
        (
          agentContext.clientOptions as
            | t.BedrockAnthropicClientOptions
            | undefined
        )?.promptCache === true
      ) {
        toolsForBinding =
          partitionAndMarkBedrockToolCache(
            rawToolsForBinding,
            makeIsDeferred(agentContext.toolDefinitions)
          ) ?? rawToolsForBinding;
      }

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
              /**
               * Merge — never overwrite — the pruner's masking record
               * into pendingOriginalToolContent.  Carry-over entries
               * from a prior summarize (preserved by the recency
               * window for masked tool messages still in the tail) and
               * the current pruner's new entries are both keyed by
               * indices in the current `state.messages`, so a key-wise
               * union is correct.  Overwriting would discard the
               * carry-over and reduce summary fidelity when those
               * masked tail messages eventually move into the head.
               */
              if (agentContext.pendingOriginalToolContent == null) {
                agentContext.pendingOriginalToolContent = originalToolContent;
              } else {
                for (const [idx, content] of originalToolContent) {
                  agentContext.pendingOriginalToolContent.set(idx, content);
                }
                /**
                 * Re-apply the per-store char cap after the union.  The
                 * pruner enforces ORIGINAL_CONTENT_MAX_CHARS inside its
                 * own map via the onContentStored callback, but a
                 * key-wise merge with recency carry-over bypasses that
                 * accounting and could let the merged map grow without
                 * bound across long sessions.
                 */
                enforceOriginalContentCap(
                  agentContext.pendingOriginalToolContent
                );
              }
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
      } else if (agentContext.provider === Providers.OPENROUTER) {
        const openRouterOptions = agentContext.clientOptions as
          | t.ProviderOptionsMap[Providers.OPENROUTER]
          | undefined;
        if (
          openRouterOptions?.promptCache === true &&
          !agentContext.systemRunnable
        ) {
          finalMessages = addCacheControl<BaseMessage>(finalMessages);
        }
      }

      if (
        isThinkingEnabled(agentContext.provider, agentContext.clientOptions)
      ) {
        /**
         * Pass `this.startIndex` so the function can distinguish CURRENT-run
         * AI messages (the agent's own iterations — possibly without a
         * leading thinking block, which Claude is allowed to skip) from
         * historical context that genuinely needs the
         * `[Previous agent context]` placeholder. Without this signal the
         * function would convert the agent's own in-run tool_use messages,
         * polluting the next iteration's prompt with a placeholder the
         * model treats as suspicious injected content.
         */
        finalMessages = ensureThinkingBlockInMessages(
          finalMessages,
          agentContext.provider,
          config,
          this.startIndex
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

      const langfuse = resolveLangfuseConfig(
        this.langfuse,
        agentContext.langfuse
      );
      const traceMetadata = createLangfuseTraceMetadata({
        messageId: this.runId,
        parentMessageId: config.configurable?.requestBody?.parentMessageId,
        agentId,
        agentName: agentContext.name,
      });
      let langfuseHandler: CallbackEntry | undefined;
      let invokeConfig = {
        ...config,
        metadata: {
          ...(config.metadata ?? {}),
          ...traceMetadata,
        },
      };
      initializeLangfuseTracing(langfuse);
      if (findCallback(config.callbacks, isLangfuseCallbackHandler) == null) {
        langfuseHandler = createLangfuseHandler({
          langfuse,
          userId: config.configurable?.user_id as string | undefined,
          sessionId: config.configurable?.thread_id as string | undefined,
          traceMetadata,
          tags: ['librechat', 'agent'],
        });
        if (langfuseHandler != null) {
          invokeConfig = {
            ...invokeConfig,
            callbacks: appendCallbacks(invokeConfig.callbacks, [
              langfuseHandler,
            ]),
          };
        }
      }

      try {
        result = await withLangfuseToolOutputTracingConfig(
          this.langfuse,
          () =>
            attemptInvoke(
              {
                model: (this.overrideModel ?? model) as t.ChatModel,
                messages: finalMessages,
                provider: agentContext.provider,
                context: this,
              },
              invokeConfig
            ),
          agentContext.langfuse
        );
      } catch (primaryError) {
        result = await withLangfuseToolOutputTracingConfig(
          this.langfuse,
          () =>
            tryFallbackProviders({
              fallbacks,
              tools: agentContext.tools,
              messages: finalMessages,
              config: invokeConfig,
              primaryError,
              context: this,
            }),
          agentContext.langfuse
        );
      } finally {
        await disposeLangfuseHandler(langfuseHandler);
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
      const metadata = config.metadata as Record<string, unknown>;
      const responseReasoningContent = getResponseReasoningContent({
        responseMessage: responseMessage as Partial<AIMessageChunk> | undefined,
        reasoningKey: agentContext.reasoningKey,
      });
      const textMessageContent = getTextMessageDeltaContent(
        responseMessage?.content as MessageContent | undefined
      );

      if (hasToolCalls) {
        const dispatchedReasoning =
          responseReasoningContent != null &&
          (await dispatchReasoningContent({
            graph: this,
            agentContext,
            reasoningContent: responseReasoningContent,
            metadata,
          }));
        if (dispatchedReasoning) {
          markPostReasoningContent(agentContext);
        }
        if (textMessageContent != null) {
          const stepKey = this.getStepKey(metadata);
          const dispatchedText = await dispatchTextMessageContent({
            graph: this,
            stepKey,
            content: textMessageContent,
            metadata,
          });
          if (dispatchedText) {
            markPostReasoningContent(agentContext);
          }
        }

        await handleToolCalls(toolCalls as ToolCall[], metadata, this);
      }

      /**
       * When streaming events are unavailable, ChatModelStreamHandler never
       * fires. Dispatch final reasoning/text content here. getMessageId makes
       * this a no-op when the streaming path already handled the same step.
       */
      if (!hasToolCalls && responseMessage != null) {
        const dispatchedReasoning =
          responseReasoningContent != null &&
          (await dispatchReasoningContent({
            graph: this,
            agentContext,
            reasoningContent: responseReasoningContent,
            metadata,
          }));
        if (dispatchedReasoning && textMessageContent != null) {
          markPostReasoningContent(agentContext);
        }
        if (textMessageContent != null) {
          const stepKey = this.getStepKey(metadata);
          await dispatchTextMessageContent({
            graph: this,
            stepKey,
            content: textMessageContent,
            metadata,
          });
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

    /**
     * Depth countdown across graph boundaries: the parent's `maxSubagentDepth`
     * becomes this executor's `maxDepth`. When the child graph is constructed,
     * `buildChildInputs()` decrements `maxSubagentDepth` on the child's
     * `AgentInputs` (only when `allowNested: true`; otherwise subagentConfigs
     * are stripped entirely). The child graph's own `createAgentNode()` then
     * reads the decremented value here and creates a narrower executor —
     * recursion is bounded even though each graph has its own separate
     * executor instance.
     */
    const effectiveSubagentDepth = agentContext.maxSubagentDepth ?? 1;
    if (
      agentContext.subagentConfigs != null &&
      agentContext.subagentConfigs.length > 0 &&
      effectiveSubagentDepth > 0
    ) {
      const resolvedConfigs = resolveSubagentConfigs(
        agentContext.subagentConfigs,
        agentContext
      );
      if (resolvedConfigs.length > 0) {
        const getParentHandlerRegistry = (): HandlerRegistry | undefined =>
          this.handlerRegistry;
        const executor = new SubagentExecutor({
          configs: new Map(resolvedConfigs.map((c) => [c.type, c])),
          parentSignal: this.signal,
          hookRegistry: this.hookRegistry,
          /** Lazy — Run wires the registry onto the graph AFTER
           *  `createWorkflow()` runs, so a direct capture here would be
           *  `undefined` at construction time. */
          parentHandlerRegistry: getParentHandlerRegistry,
          parentRunId: this.runId ?? '',
          parentAgentId: agentContext.agentId,
          langfuse: this.langfuse,
          tokenCounter: agentContext.tokenCounter,
          maxDepth: effectiveSubagentDepth,
          createChildGraph: (input): StandardGraph => {
            const childGraph = new StandardGraph(input);
            childGraph.hookRegistry = this.hookRegistry;
            /**
             * Do not propagate `humanInTheLoop` into the child graph yet:
             * nested subagent interrupts need a stable child checkpoint and
             * resume bridge. Child hooks still fire; `ask` decisions fail
             * closed inside the subagent until that flow is implemented.
             */
            childGraph.toolOutputReferences = this.toolOutputReferences;
            childGraph.eagerEventToolExecution = this.eagerEventToolExecution;
            childGraph.toolExecution = this.toolExecution;
            childGraph.eventToolExecutionAvailable =
              this.handlerRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) !=
              null;
            return childGraph;
          },
        });

        const subagentTool = tool(async (rawInput, config) => {
          const input = rawInput as {
            description?: string;
            subagent_type?: string;
          };
          const description =
            typeof input.description === 'string' &&
            input.description.trim().length > 0
              ? input.description
              : 'No task description provided';
          const subagentType =
            typeof input.subagent_type === 'string' ? input.subagent_type : '';
          const threadId = config.configurable?.thread_id as string | undefined;
          /**
           * When the tool is dispatched from an LLM's `tool_call`, LangChain
           * threads the originating `ToolCall` onto the RunnableConfig as
           * `config.toolCall` (see `ToolRunnableConfig` in
           * `@langchain/core/tools` — internal but stable since ≥0.3.x).
           * Surfacing its id lets hosts correlate `SubagentUpdateEvent`s
           * back to the parent's `tool_call_id` deterministically — no
           * temporal heuristics needed. If a future LangChain version
           * changes the threading, the type-guarded read falls back to
           * `undefined` and the correlation degrades gracefully.
           */
          const toolCall = (config as { toolCall?: { id?: string } }).toolCall;
          const parentToolCallId =
            typeof toolCall?.id === 'string' ? toolCall.id : undefined;
          const result = await executor.execute({
            description,
            subagentType,
            threadId,
            parentToolCallId,
            /**
             * Forward the parent's `configurable` so host-set fields
             * (`requestBody`, `user`, etc.) propagate into the child
             * workflow. The executor scrubs run-identity fields before
             * forwarding — see `SubagentExecuteParams.parentConfigurable`.
             */
            parentConfigurable: config.configurable as
              | Record<string, unknown>
              | undefined,
          });
          return result.content;
        }, buildSubagentToolParams(resolvedConfigs));

        if (!agentContext.graphTools) {
          agentContext.graphTools = [];
        }
        (agentContext.graphTools as t.GenericTool[]).push(subagentTool);

        /**
         * Refresh toolSchemaTokens to include the subagent tool's schema.
         * `calculateInstructionTokens()` was kicked off in `fromConfig()`
         * before graphTools was populated, so its result did not count this
         * tool. Without this retrigger, token-budget/pruning logic
         * underestimates prompt overhead.
         */
        if (agentContext.tokenCounter) {
          const { tokenCounter, baseIndexTokenCountMap } = agentContext;
          agentContext.tokenCalculationPromise = agentContext
            .calculateInstructionTokens(tokenCounter)
            .then(() => {
              agentContext.updateTokenMapWithInstructions(
                baseIndexTokenCountMap
              );
            })
            .catch((err) => {
              console.error(
                'Error recalculating instruction tokens after subagent tool injection:',
                err
              );
            });
        }
      }
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
            hookRegistry: this.hookRegistry,
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

              const unmarkHandlerDispatchedEvent = handler
                ? this.markHandlerDispatchedEvent(
                  GraphEvents.ON_RUN_STEP,
                  runStep.id
                )
                : undefined;
              try {
                if (resolvedConfig) {
                  await safeDispatchCustomEvent(
                    GraphEvents.ON_RUN_STEP,
                    runStep,
                    resolvedConfig
                  );
                }
              } finally {
                unmarkHandlerDispatchedEvent?.();
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
          if (!this.messages.length) {
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
      .addNode(
        this.defaultAgentId,
        agentNode as Runnable<
          t.AgentSubgraphState,
          Partial<t.AgentSubgraphState>
        >,
        { ends: [END] }
      )
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
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_RUN_STEP, stepId)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP,
        runStep,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
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
    delta: t.ToolCallDelta,
    metadata?: Record<string, unknown>
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
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_RUN_STEP_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_RUN_STEP_DELTA,
        runStepDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(id);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_RUN_STEP_DELTA, id)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_DELTA,
        runStepDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  }

  async dispatchMessageDelta(
    id: string,
    delta: t.MessageDelta,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const messageDelta: t.MessageDeltaEvent = {
      id,
      delta,
    };
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_MESSAGE_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_MESSAGE_DELTA,
        messageDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(id);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_MESSAGE_DELTA, id)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_MESSAGE_DELTA,
        messageDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  }

  dispatchReasoningDelta = async (
    stepId: string,
    delta: t.ReasoningDelta,
    metadata?: Record<string, unknown>
  ): Promise<void> => {
    if (!this.config) {
      throw new Error('No config provided');
    }
    const reasoningDelta: t.ReasoningDeltaEvent = {
      id: stepId,
      delta,
    };
    const handler = this.handlerRegistry?.getHandler(
      GraphEvents.ON_REASONING_DELTA
    );
    if (handler) {
      await handler.handle(
        GraphEvents.ON_REASONING_DELTA,
        reasoningDelta,
        metadata,
        this
      );
      this.handlerDispatchedStepIds.add(stepId);
    }
    const unmarkHandlerDispatchedEvent = handler
      ? this.markHandlerDispatchedEvent(GraphEvents.ON_REASONING_DELTA, stepId)
      : undefined;
    try {
      await safeDispatchCustomEvent(
        GraphEvents.ON_REASONING_DELTA,
        reasoningDelta,
        this.config
      );
    } finally {
      unmarkHandlerDispatchedEvent?.();
    }
  };
}
