import { nanoid } from 'nanoid';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import {
  Command,
  END,
  GraphInterrupt,
  INTERRUPT,
  MessagesAnnotation,
  START,
  StateGraph,
  isGraphInterrupt,
  isInterrupted,
} from '@langchain/langgraph';
import type {
  Interrupt,
  StateSnapshot,
  BaseCheckpointSaver,
} from '@langchain/langgraph';
import type { ChatGeneration, LLMResult } from '@langchain/core/outputs';
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  AgentInputs,
  BaseGraphState,
  CompiledStateWorkflow,
  HumanInTheLoopConfig,
  InjectedMessage,
  MessageDeltaEvent,
  ProcessedToolCall,
  ReasoningDeltaEvent,
  RunStep,
  RunStepDeltaEvent,
  StandardGraphInput,
  ResolvedSubagentConfig,
  StepCompleted,
  SubagentConfig,
  SubagentUpdateEvent,
  SubagentUpdatePhase,
  SubagentUsageSink,
  ToolApprovalInterruptPayload,
  ToolExecuteBatchRequest,
  ToolCallDelta,
  TokenCounter,
  ToolApprovalDecision,
  ToolApprovalDecisionMap,
} from '@/types';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type { SettledSubagentToolOutput } from './SubagentReplay';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs/Graph';
import type { HandlerRegistry } from '@/events';
import {
  StreamLimitExceededError,
  RUN_BREAKER_SCOPE_CONFIG_KEY,
} from '@/llm/streamLimits';
import {
  ContentTypes,
  Constants,
  GraphEvents,
  Callback,
  StepTypes,
} from '@/common';
import { executeHooks } from '@/hooks';

const DEFAULT_MAX_TURNS = 25;
const RECURSION_MULTIPLIER = 3;
const ERROR_MESSAGE_MAX_CHARS = 200;
const MAX_PENDING_SUBAGENT_UPDATES = 64;
const TEXT_DELTA_CONTENT_TYPE = `${ContentTypes.TEXT}_delta`;

const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
  additionalContexts: [] as string[],
  injectedMessages: [] as InjectedMessage[],
  errors: [] as string[],
});

type SanitizedSubagentToolCall = {
  id: string;
  name: string;
  args?: ToolExecuteBatchRequest['toolCalls'][number]['args'];
};

type SanitizedSubagentToolExecuteData = {
  toolCalls: SanitizedSubagentToolCall[];
  agentId?: string;
};

type SanitizedRunStep = Partial<
  Pick<
    RunStep,
    | 'agentId'
    | 'groupId'
    | 'id'
    | 'index'
    | 'runId'
    | 'stepIndex'
    | 'summary'
    | 'type'
    | 'usage'
  >
> & {
  stepDetails?: SanitizedStepDetails;
};

type SanitizedStepDetails =
  | {
      type: StepTypes.MESSAGE_CREATION;
      message_creation?: {
        message_id?: string;
      };
    }
  | {
      type: StepTypes.TOOL_CALLS;
      tool_calls?: SanitizedAgentToolCall[];
    };

type SanitizedAgentToolCall = {
  id?: string;
  name?: string;
  args?: string | object;
  type?: string;
  function?: {
    name?: string;
    arguments?: string | object;
  };
};

type SanitizedRunStepDelta = Partial<Pick<RunStepDeltaEvent, 'id'>> & {
  delta?: SanitizedToolCallDelta;
};

type SanitizedToolCallDelta = Partial<
  Pick<ToolCallDelta, 'auth' | 'expires_at' | 'summary' | 'type'>
> & {
  tool_calls?: SanitizedAgentToolCall[];
};

type SanitizedStepCompleted =
  | {
      id?: string;
      index?: number;
      type: 'tool_call';
      tool_call?: SanitizedProcessedToolCall;
    }
  | {
      type: 'summary';
      summary?: Extract<StepCompleted, { type: 'summary' }>['summary'];
    };

type SanitizedProcessedToolCall = Partial<
  Pick<
    ProcessedToolCall,
    'args' | 'id' | 'name' | 'output' | 'progress' | 'outcome'
  >
>;

type SanitizedRunStepCompleted = {
  result?: SanitizedStepCompleted;
};

type SanitizedMessageDelta = Partial<Pick<MessageDeltaEvent, 'id'>> & {
  delta?: {
    content?: MessageDeltaEvent['delta']['content'];
    tool_call_ids?: MessageDeltaEvent['delta']['tool_call_ids'];
  };
};

type SanitizedReasoningDelta = Partial<Pick<ReasoningDeltaEvent, 'id'>> & {
  delta?: {
    content?: ReasoningDeltaEvent['delta']['content'];
  };
};

type QueuedSubagentUpdate = {
  eventName: string;
  phase: SubagentUpdatePhase;
  data: unknown;
};

type ForwarderCallback = {
  handler: BaseCallbackHandler;
  drain: () => Promise<void>;
};

type StatefulCompiledWorkflow = Omit<CompiledStateWorkflow, 'invoke'> & {
  invoke(
    input: BaseGraphState | Command | null,
    config?: RunnableConfig
  ): Promise<BaseGraphState>;
  getState(config: RunnableConfig): Promise<StateSnapshot>;
  updateState?(
    config: RunnableConfig,
    values: Record<string, unknown>,
    asNode?: string
  ): Promise<RunnableConfig>;
};

type ReplayCheckpointWorkflow = {
  updateState(
    config: RunnableConfig,
    values: { messages: BaseMessage[] },
    asNode: string
  ): Promise<RunnableConfig>;
};

type ActiveChildRun = {
  graph: StandardGraph;
  workflow: StatefulCompiledWorkflow;
  pendingInterrupts: Interrupt[];
  invokeConfig?: RunnableConfig;
  childAgentId: string;
};

type PersistedToolOutput = {
  content: ToolMessage['content'];
  toolCallId: string;
  id?: string;
  name?: string;
  status?: 'success' | 'error';
  additionalKwargs: ToolMessage['additional_kwargs'];
  responseMetadata: ToolMessage['response_metadata'];
  metadata?: Record<string, unknown>;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  referenceContent?: string;
};

type SubagentCheckpointMarker = {
  version: 1;
  parentToolCallId: string;
  lifecycleComplete: true;
  hookSessionId?: string;
  settledOutput?: PersistedToolOutput;
};

const LANGGRAPH_RUNTIME_CONFIG_PREFIX = '__pregel_';
const LANGGRAPH_RESUME_MAP_CONFIG_KEY = '__pregel_resume_map';
const LANGGRAPH_CHECKPOINT_CONFIG_KEYS = new Set([
  'checkpoint_id',
  'checkpoint_map',
  'checkpoint_ns',
]);
const SUBAGENT_CHECKPOINT_MARKER_KEY = '__librechat_subagent_checkpoint';
const SUBAGENT_HOOK_SESSION_KEY = '__librechat_subagent_hook_session';
const SUBAGENT_REPLAY_NODE = 'subagent-replay';

function isCheckpointSaver(value: unknown): value is BaseCheckpointSaver {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<BaseCheckpointSaver>;
  return (
    typeof candidate.getTuple === 'function' &&
    typeof candidate.list === 'function' &&
    typeof candidate.put === 'function' &&
    typeof candidate.putWrites === 'function' &&
    typeof candidate.deleteThread === 'function'
  );
}

function isSubagentCheckpointMarker(
  value: unknown
): value is SubagentCheckpointMarker {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const marker = value as Partial<SubagentCheckpointMarker>;
  return (
    marker.version === 1 &&
    marker.lifecycleComplete === true &&
    typeof marker.parentToolCallId === 'string' &&
    (marker.hookSessionId == null ||
      typeof marker.hookSessionId === 'string') &&
    (marker.settledOutput == null ||
      isPersistedToolOutput(marker.settledOutput))
  );
}

function isPersistedToolOutput(value: unknown): value is PersistedToolOutput {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const output = value as {
    content?: unknown;
    toolCallId?: unknown;
    status?: unknown;
    additionalKwargs?: unknown;
    responseMetadata?: unknown;
    additionalContexts?: unknown;
    resolvedArgs?: unknown;
    referenceContent?: unknown;
  };
  const contentIsValid =
    typeof output.content === 'string' || Array.isArray(output.content);
  const statusIsValid =
    output.status == null ||
    output.status === 'success' ||
    output.status === 'error';
  const contextsAreValid =
    Array.isArray(output.additionalContexts) &&
    output.additionalContexts.every((context) => typeof context === 'string');
  const resolvedArgsAreValid =
    output.resolvedArgs == null ||
    (typeof output.resolvedArgs === 'object' &&
      !Array.isArray(output.resolvedArgs));
  return (
    contentIsValid &&
    typeof output.toolCallId === 'string' &&
    output.additionalKwargs != null &&
    typeof output.additionalKwargs === 'object' &&
    output.responseMetadata != null &&
    typeof output.responseMetadata === 'object' &&
    statusIsValid &&
    contextsAreValid &&
    resolvedArgsAreValid &&
    (output.referenceContent == null ||
      typeof output.referenceContent === 'string')
  );
}

function getSubagentCheckpointMarker(
  messages: BaseMessage[],
  parentToolCallId: string
): SubagentCheckpointMarker | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const marker =
      messages[i].additional_kwargs[SUBAGENT_CHECKPOINT_MARKER_KEY];
    if (
      isSubagentCheckpointMarker(marker) &&
      marker.parentToolCallId === parentToolCallId
    ) {
      return marker;
    }
  }
  return undefined;
}

function getSubagentHookSessionId(messages: BaseMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const sessionId = messages[i].additional_kwargs[SUBAGENT_HOOK_SESSION_KEY];
    if (typeof sessionId === 'string' && sessionId.length > 0) {
      return sessionId;
    }
  }
  return undefined;
}

function isSubagentCheckpointMarkerMessage(message: BaseMessage): boolean {
  return isSubagentCheckpointMarker(
    message.additional_kwargs[SUBAGENT_CHECKPOINT_MARKER_KEY]
  );
}

function createSubagentCheckpointMarkerMessage(
  marker: SubagentCheckpointMarker
): AIMessage {
  return new AIMessage({
    content: '',
    additional_kwargs: { [SUBAGENT_CHECKPOINT_MARKER_KEY]: marker },
  });
}

function getCheckpointMessages(value: unknown): BaseMessage[] {
  return Array.isArray(value) ? value.filter(BaseMessage.isInstance) : [];
}

function serializeToolOutput(
  settled: SettledSubagentToolOutput
): PersistedToolOutput {
  const { output } = settled;
  return {
    content: output.content,
    toolCallId: output.tool_call_id,
    ...(output.id == null ? {} : { id: output.id }),
    ...(output.name == null ? {} : { name: output.name }),
    ...(output.status == null ? {} : { status: output.status }),
    additionalKwargs: output.additional_kwargs,
    responseMetadata: output.response_metadata,
    ...(output.metadata == null ? {} : { metadata: output.metadata }),
    additionalContexts: settled.additionalContexts,
    ...(settled.resolvedArgs == null
      ? {}
      : { resolvedArgs: settled.resolvedArgs }),
    ...(settled.referenceContent == null
      ? {}
      : { referenceContent: settled.referenceContent }),
  };
}

function deserializeToolOutput(
  output: PersistedToolOutput
): SettledSubagentToolOutput {
  return {
    output: new ToolMessage({
      content: output.content,
      tool_call_id: output.toolCallId,
      ...(output.id == null ? {} : { id: output.id }),
      ...(output.name == null ? {} : { name: output.name }),
      ...(output.status == null ? {} : { status: output.status }),
      additional_kwargs: output.additionalKwargs,
      response_metadata: output.responseMetadata,
      ...(output.metadata == null ? {} : { metadata: output.metadata }),
    }),
    additionalContexts: output.additionalContexts,
    ...(output.resolvedArgs == null
      ? {}
      : { resolvedArgs: output.resolvedArgs }),
    ...(output.referenceContent == null
      ? {}
      : { referenceContent: output.referenceContent }),
  };
}

function getParentCheckpointFork(
  configurable: Record<string, unknown> | undefined
): string {
  const checkpointId = configurable?.checkpoint_id;
  return typeof checkpointId === 'string' && checkpointId.length > 0
    ? checkpointId
    : 'root';
}

function getChildThreadId(args: {
  parentRunId: string;
  parentAgentId?: string;
  threadId?: string;
  parentToolCallId: string;
  parentConfigurable?: Record<string, unknown>;
}): string {
  const durableParentId = args.threadId ?? args.parentRunId;
  const parentFork = getParentCheckpointFork(args.parentConfigurable);
  const identity = JSON.stringify([
    durableParentId,
    parentFork,
    args.parentAgentId ?? 'agent',
    args.parentToolCallId,
  ]);
  return `subagent:${Buffer.from(identity).toString('base64url')}`;
}

function isToolApprovalPayload(
  value: unknown
): value is ToolApprovalInterruptPayload {
  return (
    value != null &&
    typeof value === 'object' &&
    (value as { type?: unknown }).type === 'tool_approval'
  );
}

function addSubagentScope(
  interrupts: Interrupt[],
  scope: NonNullable<ToolApprovalInterruptPayload['subagent']>
): Interrupt[] {
  return interrupts.map((childInterrupt) => ({
    ...childInterrupt,
    value: isToolApprovalPayload(childInterrupt.value)
      ? {
        ...childInterrupt.value,
        subagent: childInterrupt.value.subagent ?? scope,
      }
      : childInterrupt.value,
  }));
}

type ToolApprovalResumeValue = ToolApprovalDecision[] | ToolApprovalDecisionMap;

function getChildResumeMap(
  pendingInterrupts: Interrupt[],
  parentConfigurable: Record<string, unknown> | undefined
): Record<string, ToolApprovalResumeValue> | undefined {
  const resumeMap = parentConfigurable?.[LANGGRAPH_RESUME_MAP_CONFIG_KEY];
  if (resumeMap == null || typeof resumeMap !== 'object') {
    return undefined;
  }

  const parentResumeMap = resumeMap as Record<string, ToolApprovalResumeValue>;
  const childResumeMap: Record<string, ToolApprovalResumeValue> = {};
  for (const childInterrupt of pendingInterrupts) {
    const interruptId = childInterrupt.id;
    if (
      typeof interruptId === 'string' &&
      Object.prototype.hasOwnProperty.call(parentResumeMap, interruptId)
    ) {
      childResumeMap[interruptId] = parentResumeMap[interruptId];
    }
  }
  return Object.keys(childResumeMap).length > 0 ? childResumeMap : undefined;
}

function getPersistedInterrupts(snapshot: StateSnapshot): Interrupt[] {
  const interrupts: Interrupt[] = [];
  for (const task of snapshot.tasks) {
    for (const pendingInterrupt of task.interrupts) {
      interrupts.push(pendingInterrupt);
    }
  }
  return interrupts;
}

function getPersistedMessages(
  snapshot: StateSnapshot
): BaseMessage[] | undefined {
  if (snapshot.values == null || typeof snapshot.values !== 'object') {
    return undefined;
  }
  const values = snapshot.values as { messages?: BaseMessage[] };
  if (!Array.isArray(values.messages) || values.messages.length === 0) {
    return undefined;
  }
  const messages = values.messages.filter(
    (message) => !isSubagentCheckpointMarkerMessage(message)
  );
  return messages.length > 0 ? messages : undefined;
}

function createReplayCheckpointWorkflow(
  checkpointer: BaseCheckpointSaver
): ReplayCheckpointWorkflow {
  return new StateGraph(MessagesAnnotation)
    .addNode(SUBAGENT_REPLAY_NODE, (state) => state)
    .addEdge(START, SUBAGENT_REPLAY_NODE)
    .addEdge(SUBAGENT_REPLAY_NODE, END)
    .compile({ checkpointer }) as ReplayCheckpointWorkflow;
}

export type SubagentExecuteParams = {
  description: string;
  subagentType: string;
  threadId?: string;
  /**
   * Breaker controller captured at the parent TOOL BATCH's entry, before
   * PreToolUse hooks. Preferred over the live scope accessor: a graph reset
   * during a hook would otherwise bind this child to the NEW run's
   * controller — reviving it on a fresh signal and letting its trips cancel
   * unrelated work.
   */
  breaker?: AbortController;
  /**
   * Parent-side `tool_call_id` of the `subagent` tool invocation that
   * triggered this execution. Surfaced on {@link SubagentUpdateEvent} so
   * hosts can correlate child updates back to the originating tool call
   * without relying on event ordering heuristics.
   */
  parentToolCallId?: string;
  /**
   * Snapshot of the parent invocation's host `config.configurable` at
   * the spawn-tool call site. Host-set fields (`requestBody`, `user`,
   * `userMCPAuthMap`, etc.) propagate into the child workflow's
   * `configurable` — fixing MCP body-placeholder substitution and
   * per-user lookups for subagent tool calls. LangGraph runtime keys
   * (`__pregel_*`, checkpoint bookkeeping) are intentionally not
   * inherited; the child graph recreates its own runtime config.
   *
   * Inheritance details (verified empirically against LangGraph):
   *   - host-set keys propagate as-is into the child's tool dispatches;
   *   - with nested HITL enabled, `thread_id` is replaced with a stable
   *     child checkpoint id derived from the parent's durable thread id,
   *     checkpoint fork, parent agent id, and spawning tool call id so parent
   *     and child checkpoints cannot collide, sibling parent forks stay
   *     isolated, and reconstruction returns to the same child checkpoint;
   *     parent-scoped hook lookup remains keyed by the inherited `run_id`;
   *   - `parent_run_id` propagates when the host put it on parent's
   *     configurable;
   *   - `run_id` is *overwritten by the LangGraph runtime* at child
   *     invoke time regardless of what we forward — child's tool
   *     dispatches see the child graph's runtime runId in
   *     `configurable.run_id`, not the parent's. Hosts that need
   *     parent-scoped run identity for downstream consumers should
   *     plumb it via a host-defined key (e.g. `requestBody.messageId`),
   *     not `run_id`.
   *
   * A future revision will likely make this inheritance configurable
   * per spawn type — background / async subagents may want isolation
   * rather than sharing parent's host context.
   */
  parentConfigurable?: Record<string, unknown>;
};

export type SubagentExecuteResult = {
  content: string;
  messages: BaseMessage[];
};

/**
 * Factory that constructs a child graph for subagent execution. Injected
 * rather than imported so that `SubagentExecutor` does not have a runtime
 * dependency on `StandardGraph` — this avoids a circular dependency between
 * `src/graphs/Graph.ts` and `src/tools/subagent/` that would otherwise break
 * Rollup's chunking under `preserveModules`.
 */
export type ChildGraphFactory = (input: StandardGraphInput) => StandardGraph;

export type SubagentExecutorOptions = {
  configs: Map<string, ResolvedSubagentConfig>;
  parentSignal?: AbortSignal;
  /** Run-scoped breaker abort shared by every executor of one graph, so a
   * child tripping a stream limit stops subagents running under OTHER
   * parallel agent nodes too. An accessor rather than a captured controller:
   * the graph recreates its controller per run, and each execution must
   * bind to the controller current when it STARTS — signal and trip target
   * together, so a straggler from a failed run can neither revive on nor
   * circuit-break a later run's controller. Absent (tests, minimal hosts),
   * the executor falls back to its own private controller. */
  breakerScope?: {
    controller: () => AbortController;
  };
  hookRegistry?: HookRegistry;
  parentRunId: string;
  parentAgentId?: string;
  langfuse?: StandardGraphInput['langfuse'];
  tokenCounter?: TokenCounter;
  /**
   * Run-level stream circuit breakers, forwarded into every child graph so
   * a host raising, lowering, or disabling the limits governs subagents too.
   * Child model calls run through `attemptInvoke`'s local stream handler
   * (children have no registered dispatcher), which enforces the child
   * graph's own resolved limits; without this the child would silently
   * revert to the defaults.
   */
  streamLimits?: StandardGraphInput['streamLimits'];
  humanInTheLoop?: HumanInTheLoopConfig;
  /** Shared durable saver used to recover outer tool lifecycle results before
   * parent hooks re-enter after a process rebuild. Narrowed structurally at
   * construction because graph compile options also permit framework flags. */
  checkpointer?: unknown;
  /** Remaining nesting budget. 0 or negative blocks execution. */
  maxDepth?: number;
  /**
   * Factory for constructing the isolated child graph. Callers pass
   * `(input) => new StandardGraph(input)` — injected to break a circular
   * module dependency.
   */
  createChildGraph: ChildGraphFactory;
  /**
   * Parent's event handler registry. When provided, child-graph events are
   * forwarded through this registry so hosts can:
   *   (a) execute event-driven tools (`ON_TOOL_EXECUTE` routed to parent's handler),
   *   (b) surface child activity to a UI via wrapped {@link GraphEvents.ON_SUBAGENT_UPDATE}.
   * When omitted, the child runs fully isolated (legacy behavior).
   *
   * Can be a direct `HandlerRegistry` or a zero-arg getter — use the getter
   * form when the registry is assigned to the graph AFTER the executor is
   * constructed (the current `Run.create` flow sets `handlerRegistry`
   * post-`createWorkflow`, so `createAgentNode` must capture lazily).
   */
  parentHandlerRegistry?: HandlerRegistry | (() => HandlerRegistry | undefined);
  /**
   * Receives a usage event for every model call the child run makes. The
   * child workflow executes via `invoke()` with a detached callbacks array,
   * so its `on_chat_model_end` events never reach the parent's handler
   * registry — without this sink, child token usage is invisible to the
   * host (unbilled model calls). Forwarded into the child graph's input so
   * nested subagents report through the same sink.
   */
  usageSink?: SubagentUsageSink;
};

export class SubagentExecutor {
  private readonly configs: Map<string, ResolvedSubagentConfig>;
  private readonly parentSignal?: AbortSignal;
  /** Aborted when a child trips a stream circuit breaker: parallel sibling
   * subagents run concurrently on the parent's signal, and rejecting the
   * batch alone would leave their provider requests streaming after the
   * safety abort. One-way by design — a tripped breaker ends the run, so no
   * later child of this executor should start either. Fallback for hosts
   * that do not supply the graph's shared `breakerScope`. */
  private readonly childRunAbort = new AbortController();
  private readonly breakerScope?: SubagentExecutorOptions['breakerScope'];
  private readonly hookRegistry?: HookRegistry;
  private readonly parentRunId: string;
  private readonly parentAgentId?: string;
  private readonly langfuse?: StandardGraphInput['langfuse'];
  private readonly tokenCounter?: TokenCounter;
  private readonly streamLimits?: StandardGraphInput['streamLimits'];
  private readonly humanInTheLoop?: HumanInTheLoopConfig;
  private readonly checkpointer?: BaseCheckpointSaver;
  private readonly maxDepth: number;
  private readonly createChildGraph: ChildGraphFactory;
  private readonly usageSink?: SubagentUsageSink;
  private readonly checkpointThreadIds = new Set<string>();
  private readonly startedChildRuns = new Set<string>();
  private readonly completedChildRuns = new Set<string>();
  private readonly completedChildResults = new Map<
    string,
    SubagentExecuteResult
  >();
  private readonly activeChildRuns = new Map<string, ActiveChildRun>();
  private replayCheckpointWorkflow?: ReplayCheckpointWorkflow;
  private readonly resolveParentHandlerRegistry?: () =>
    | HandlerRegistry
    | undefined;

  constructor(options: SubagentExecutorOptions) {
    this.configs = options.configs;
    this.parentSignal = options.parentSignal;
    this.breakerScope = options.breakerScope;
    this.hookRegistry = options.hookRegistry;
    this.parentRunId = options.parentRunId;
    this.parentAgentId = options.parentAgentId;
    this.langfuse = options.langfuse;
    this.tokenCounter = options.tokenCounter;
    this.streamLimits = options.streamLimits;
    this.humanInTheLoop = options.humanInTheLoop;
    this.checkpointer = isCheckpointSaver(options.checkpointer)
      ? options.checkpointer
      : undefined;
    this.maxDepth = options.maxDepth ?? 1;
    this.createChildGraph = options.createChildGraph;
    this.usageSink = options.usageSink;
    const rawRegistry = options.parentHandlerRegistry;
    if (typeof rawRegistry === 'function') {
      this.resolveParentHandlerRegistry = rawRegistry;
    } else if (rawRegistry != null) {
      this.resolveParentHandlerRegistry = (): HandlerRegistry => rawRegistry;
    }
  }

  /** The breaker controller current for this execution — read per spawn
   * because the graph recreates its controller each run. */
  private resolveBreakerController(): AbortController {
    return this.breakerScope?.controller() ?? this.childRunAbort;
  }

  /** One signal that fires on the parent's abort or the breaker abort,
   * collapsed to a single signal when possible (mirrors
   * `composeAbortSignals` in Graph.ts). */
  private composeChildSignal(breaker: AbortController): AbortSignal {
    const child = breaker.signal;
    if (this.parentSignal == null || this.parentSignal === child) {
      return child;
    }
    return AbortSignal.any([this.parentSignal, child]);
  }

  /** Snapshot of the parent's registry at the moment a subagent is dispatched. */
  private getParentHandlerRegistry(): HandlerRegistry | undefined {
    return this.resolveParentHandlerRegistry?.();
  }

  getChildCheckpointThreadIds(): string[] {
    const threadIds = new Set(this.checkpointThreadIds);
    for (const activeChildRun of this.activeChildRuns.values()) {
      for (const threadId of this.getGraphChildCheckpointThreadIds(
        activeChildRun.graph
      )) {
        threadIds.add(threadId);
      }
    }
    return [...threadIds];
  }

  private getGraphChildCheckpointThreadIds(graph: StandardGraph): string[] {
    const checkpointGraph = graph as {
      getChildCheckpointThreadIds?: () => string[];
    };
    return checkpointGraph.getChildCheckpointThreadIds?.() ?? [];
  }

  private clearChildGraph(graph: StandardGraph): void {
    for (const threadId of this.getGraphChildCheckpointThreadIds(graph)) {
      this.checkpointThreadIds.add(threadId);
    }
    graph.clearHeavyState();
  }

  clearHeavyState(): void {
    for (const activeChildRun of this.activeChildRuns.values()) {
      this.clearChildGraph(activeChildRun.graph);
    }
    this.activeChildRuns.clear();
    this.completedChildResults.clear();
    this.startedChildRuns.clear();
    this.completedChildRuns.clear();
    this.replayCheckpointWorkflow = undefined;
  }

  async getSettledToolOutput(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<SettledSubagentToolOutput | undefined> {
    const parentToolCallId = call.id;
    if (
      this.humanInTheLoop?.enabled !== true ||
      this.checkpointer == null ||
      parentToolCallId == null ||
      parentToolCallId === ''
    ) {
      return undefined;
    }
    const parentConfigurable = config.configurable as
      | Record<string, unknown>
      | undefined;
    const threadId = parentConfigurable?.thread_id;
    const childThreadId = getChildThreadId({
      parentRunId: this.parentRunId,
      parentAgentId: this.parentAgentId,
      threadId: typeof threadId === 'string' ? threadId : undefined,
      parentToolCallId,
      parentConfigurable,
    });
    this.checkpointThreadIds.add(childThreadId);
    const checkpoint = await this.checkpointer.getTuple({
      configurable: { thread_id: childThreadId },
    });
    const messages = getCheckpointMessages(
      checkpoint?.checkpoint.channel_values.messages
    );
    const marker = getSubagentCheckpointMarker(messages, parentToolCallId);
    const persistedHookSessionId =
      marker?.hookSessionId ?? getSubagentHookSessionId(messages);
    const currentHookSessionId = parentConfigurable?.run_id;
    if (
      persistedHookSessionId != null &&
      typeof currentHookSessionId === 'string' &&
      currentHookSessionId.length > 0
    ) {
      this.hookRegistry?.copySession(
        persistedHookSessionId,
        currentHookSessionId
      );
    }
    return marker?.settledOutput == null
      ? undefined
      : deserializeToolOutput(marker.settledOutput);
  }

  async persistSettledToolOutput(
    call: ToolCall,
    config: RunnableConfig,
    settled: SettledSubagentToolOutput
  ): Promise<void> {
    const parentToolCallId = call.id;
    if (
      this.humanInTheLoop?.enabled !== true ||
      this.checkpointer == null ||
      parentToolCallId == null ||
      parentToolCallId === ''
    ) {
      return;
    }
    const parentConfigurable = config.configurable as
      | Record<string, unknown>
      | undefined;
    const threadId = parentConfigurable?.thread_id;
    const childThreadId = getChildThreadId({
      parentRunId: this.parentRunId,
      parentAgentId: this.parentAgentId,
      threadId: typeof threadId === 'string' ? threadId : undefined,
      parentToolCallId,
      parentConfigurable,
    });
    this.checkpointThreadIds.add(childThreadId);
    const activeChildRun = this.activeChildRuns.get(childThreadId);
    const persistedOutput = serializeToolOutput(settled);
    if (activeChildRun != null) {
      await this.persistChildCheckpointMarker(
        activeChildRun,
        parentToolCallId,
        persistedOutput
      );
      this.clearChildGraph(activeChildRun.graph);
      this.activeChildRuns.delete(childThreadId);
      return;
    }
    this.replayCheckpointWorkflow ??= createReplayCheckpointWorkflow(
      this.checkpointer
    );
    await this.replayCheckpointWorkflow.updateState(
      { configurable: { thread_id: childThreadId } },
      {
        messages: [
          createSubagentCheckpointMarkerMessage({
            version: 1,
            parentToolCallId,
            lifecycleComplete: true,
            hookSessionId:
              typeof parentConfigurable?.run_id === 'string'
                ? parentConfigurable.run_id
                : this.parentRunId,
            settledOutput: persistedOutput,
          }),
        ],
      },
      SUBAGENT_REPLAY_NODE
    );
  }

  private async persistChildCheckpointMarker(
    activeChildRun: ActiveChildRun,
    parentToolCallId: string,
    settledOutput?: PersistedToolOutput
  ): Promise<void> {
    if (
      this.humanInTheLoop?.enabled !== true ||
      activeChildRun.workflow.updateState == null ||
      activeChildRun.invokeConfig == null
    ) {
      return;
    }
    await activeChildRun.workflow.updateState(
      activeChildRun.invokeConfig,
      {
        messages: [
          createSubagentCheckpointMarkerMessage({
            version: 1,
            parentToolCallId,
            lifecycleComplete: true,
            hookSessionId:
              typeof activeChildRun.invokeConfig.configurable?.run_id ===
              'string'
                ? activeChildRun.invokeConfig.configurable.run_id
                : this.parentRunId,
            ...(settledOutput == null ? {} : { settledOutput }),
          }),
        ],
      },
      activeChildRun.childAgentId
    );
  }

  async execute(params: SubagentExecuteParams): Promise<SubagentExecuteResult> {
    const { description, subagentType, threadId, parentToolCallId } = params;
    /** Captured ONCE per execution, preferring the controller the parent
     * tool batch captured at ITS entry (before PreToolUse hooks): a failed
     * run's graph reset replaces the live controller, and resolving it here
     * — after the hook awaits — would bind this child to the NEW run's
     * un-aborted controller: reviving old-run work, or worse, tripping the
     * new run's breaker from an old child's stream-limit breach. Signal and
     * trip target both bind to this capture. */
    const childBreaker = params.breaker ?? this.resolveBreakerController();
    const childSignal = this.composeChildSignal(childBreaker);
    const config = this.configs.get(subagentType);

    if (!config) {
      const available = [...this.configs.keys()].join(', ');
      return {
        content: `Error: Unknown subagent type "${subagentType}". Available types: ${available}`,
        messages: [],
      };
    }

    if (this.maxDepth <= 0) {
      return {
        content: 'Error: Maximum subagent nesting depth exceeded.',
        messages: [],
      };
    }

    if (
      this.humanInTheLoop?.enabled === true &&
      (parentToolCallId == null || parentToolCallId === '')
    ) {
      return {
        content:
          'Error: Resumable subagent execution requires a parent tool call ID.',
        messages: [],
      };
    }

    const executionSuffix = parentToolCallId ?? nanoid(8);
    const childRunId = `${this.parentRunId}_sub_${executionSuffix}`;
    const childThreadId = getChildThreadId({
      parentRunId: this.parentRunId,
      parentAgentId: this.parentAgentId,
      threadId,
      parentToolCallId: executionSuffix,
      parentConfigurable: params.parentConfigurable,
    });
    if (this.humanInTheLoop?.enabled === true && this.checkpointer != null) {
      this.checkpointThreadIds.add(childThreadId);
    }
    const childExecutionKey = childThreadId;
    const childAgentId =
      config.agentInputs.agentId ||
      `${this.parentAgentId ?? 'agent'}_sub_${executionSuffix}`;
    const completedChildResult =
      this.completedChildResults.get(childExecutionKey);
    if (completedChildResult != null) {
      return completedChildResult;
    }

    const parentRegistry = this.getParentHandlerRegistry();
    const forwardingEnabled = parentRegistry != null;
    /**
     * Keep `toolDefinitions` only when the host has actually wired an
     * `ON_TOOL_EXECUTE` handler. `Run` always constructs a `HandlerRegistry`,
     * so treating any registry as "forwarding enabled" would leak
     * `toolDefinitions` into children whose hosts cannot execute them — the
     * child's `ToolNode` batch promise would hang forever with no handler to
     * resolve/reject. Gating on the tool-execute handler preserves the
     * recoverable "no tools" path for registry-but-no-handler configs.
     */
    const hasToolExecuteHandler =
      parentRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) != null;
    const childInputs = buildChildInputs(
      config,
      childAgentId,
      this.maxDepth,
      /* keepToolDefinitions */ hasToolExecuteHandler
    );
    const maxTurns = config.maxTurns ?? DEFAULT_MAX_TURNS;

    const hostUsageSink = this.usageSink;
    const cachedChildRun = this.activeChildRuns.get(childExecutionKey);
    const childGraph =
      cachedChildRun?.graph ??
      this.createChildGraph({
        runId: childRunId,
        signal: childSignal,
        agents: [childInputs],
        langfuse: this.langfuse,
        tokenCounter: this.tokenCounter,
        streamLimits: this.streamLimits,
        subagentScope: true,
        /**
         * Forwarded so the child graph's own `SubagentExecutor` (created in
         * its `createAgentNode` when `allowNested` keeps subagentConfigs)
         * reports nested-child usage through the same host sink. Each nesting
         * level attaches its own capture callback — `workflow.invoke` replaces
         * the inherited callback chain, so a single top-level handler would
         * never see grandchild model calls.
         *
         * The wrapper rewrites `runId` to THIS executor's parent run: nested
         * executors emit with their own `parentRunId` (a `*_sub_*` child id),
         * and each wrapper layer rewrites upward, so by the time an event
         * reaches the host sink its `runId` is the ROOT run — hosts keying
         * billing by run id never see intermediate child run ids there
         * (`subagentRunId` still identifies the emitting child).
         */
        subagentUsageSink:
          hostUsageSink == null
            ? undefined
            : /** Returns the host sink's result so async sinks stay awaited
               *  through every wrapper layer. */
            (event): void | Promise<void> =>
              hostUsageSink({ ...event, runId: this.parentRunId }),
      });

    let forwarding: ForwarderCallback | undefined;
    if (forwardingEnabled) {
      forwarding = this.createForwarderCallback({
        parentRegistry: parentRegistry!,
        subagentType,
        subagentAgentId: childAgentId,
        childRunId,
        parentToolCallId,
      });
    }
    const forwarder = forwarding?.handler;
    let childAlreadyStarted = this.startedChildRuns.has(childExecutionKey);
    let childAlreadyCompleted = this.completedChildRuns.has(childExecutionKey);

    let result: { messages: BaseMessage[] } | undefined;
    let recoveredComplete = false;
    let recoveredInProgress = false;
    try {
      const workflow = (cachedChildRun?.workflow ??
        childGraph.createWorkflow()) as StatefulCompiledWorkflow;
      const activeChildRun = cachedChildRun ?? {
        graph: childGraph,
        workflow,
        pendingInterrupts: [],
        childAgentId,
      };
      if (cachedChildRun == null) {
        this.activeChildRuns.set(childExecutionKey, activeChildRun);
      }
      /**
       * When `parentHandlerRegistry` is provided (forwarding mode), attach a
       * lightweight callback that intercepts the child's `on_custom_event`
       * dispatches and routes them to the parent's registry — either as
       * operational events (ON_TOOL_EXECUTE) or wrapped ON_SUBAGENT_UPDATE
       * envelopes. Native LangChain streaming events (on_chat_model_stream,
       * etc.) still do NOT propagate to the parent's outer streamEvents
       * iterator — the `callbacks` array REPLACES the inherited chain, so
       * parent handlers won't receive child stream chunks and raise "No
       * agent context found" lookups on the parent's agentContexts map.
       *
       * When no registry is provided (legacy isolation), `callbacks: []`
       * fully detaches the child.
       *
       * `runName` gives the child a distinct LangSmith trace root (avoids
       * nested trace pollution).
       */
      const callbackHandlers: BaseCallbackHandler[] = [];
      if (forwarder) {
        callbackHandlers.push(forwarder);
      }
      /**
       * Usage capture rides the same detached callbacks array. Because
       * `callbacks` REPLACES the inherited chain (see above), the host's
       * `CHAT_MODEL_END` handler never observes the child's model calls —
       * this handler is the child-side equivalent of `ModelEndHandler`,
       * reporting per-call usage to the host's sink for billing.
       */
      if (this.usageSink) {
        callbackHandlers.push(
          createUsageCaptureHandler({
            sink: this.usageSink,
            subagentType,
            subagentRunId: childRunId,
            subagentAgentId: childAgentId,
            parentRunId: this.parentRunId,
            provider: config.agentInputs.provider,
            fallbackModel: extractConfiguredModel(config.agentInputs),
          })
        );
      }
      const callbacks: Callbacks = callbackHandlers;
      /**
       * Inherit the parent's host `configurable` while binding LangGraph's
       * checkpoint identity to a stable child id derived from the durable
       * parent thread and checkpoint fork. The parent thread id cannot be
       * reused here: parent and child share one checkpointer when nested HITL
       * is enabled, and root checkpoint namespaces are normalized by
       * LangGraph, so a shared `thread_id` would collide with the parent.
       *
       * `run_id` still propagates as the parent run id, which is the key used
       * for session-scoped hook lookup. Child hook inputs therefore retain
       * the parent policy scope while their `threadId` truthfully identifies
       * the independently checkpointed child execution.
       */
      const inheritedConfigurable: Record<string, unknown> =
        sanitizeChildConfigurable(params.parentConfigurable);
      const currentHookSessionId =
        typeof inheritedConfigurable.run_id === 'string' &&
        inheritedConfigurable.run_id.length > 0
          ? inheritedConfigurable.run_id
          : this.parentRunId;
      const childInvokeConfig = {
        recursionLimit: maxTurns * RECURSION_MULTIPLIER,
        signal: childSignal,
        callbacks,
        runName: `subagent:${subagentType}`,
        configurable: {
          ...inheritedConfigurable,
          thread_id:
            this.humanInTheLoop?.enabled === true
              ? childThreadId
              : (inheritedConfigurable.thread_id ?? childRunId),
        },
      };
      activeChildRun.invokeConfig = childInvokeConfig;
      if (cachedChildRun == null && this.humanInTheLoop?.enabled === true) {
        /** Rehydrate child-owned interrupt state when a host rebuilds Run
         * around the same durable checkpointer after a process boundary. */
        const persistedState = await workflow.getState(childInvokeConfig);
        const checkpointMessages = getCheckpointMessages(
          (persistedState.values as { messages?: unknown } | undefined)
            ?.messages
        );
        const persistedHookSessionId =
          getSubagentHookSessionId(checkpointMessages);
        if (persistedHookSessionId != null) {
          this.hookRegistry?.copySession(
            persistedHookSessionId,
            currentHookSessionId
          );
        }
        const persistedInterrupts = getPersistedInterrupts(persistedState);
        if (persistedInterrupts.length > 0) {
          activeChildRun.pendingInterrupts = persistedInterrupts;
          this.startedChildRuns.add(childExecutionKey);
          childAlreadyStarted = true;
        } else if (persistedState.next.length > 0) {
          recoveredInProgress = true;
          childAlreadyStarted = true;
          this.startedChildRuns.add(childExecutionKey);
        } else if (persistedState.next.length === 0) {
          const persistedMessages = getPersistedMessages(persistedState);
          if (persistedMessages != null) {
            const marker = getSubagentCheckpointMarker(
              checkpointMessages,
              executionSuffix
            );
            result = { messages: persistedMessages };
            recoveredComplete = true;
            childAlreadyStarted = true;
            childAlreadyCompleted = marker?.lifecycleComplete === true;
            this.startedChildRuns.add(childExecutionKey);
          }
        }
      }
      if (!recoveredComplete) {
        const childResumeMap = getChildResumeMap(
          activeChildRun.pendingInterrupts,
          params.parentConfigurable
        );
        if (
          activeChildRun.pendingInterrupts.length > 0 &&
          childResumeMap == null
        ) {
          throw new GraphInterrupt(activeChildRun.pendingInterrupts);
        }
        let childInput: BaseGraphState | Command | null;
        if (childResumeMap != null) {
          childInput = new Command({ resume: childResumeMap });
        } else if (recoveredInProgress) {
          childInput = null;
        } else {
          childInput = {
            messages: [
              new HumanMessage({
                content: description,
                additional_kwargs: {
                  [SUBAGENT_HOOK_SESSION_KEY]: currentHookSessionId,
                },
              }),
            ],
          };
        }

        if (
          !childAlreadyStarted &&
          this.hookRegistry?.hasHookFor('SubagentStart', this.parentRunId) ===
            true
        ) {
          const hookResult = await executeHooks({
            registry: this.hookRegistry,
            input: {
              hook_event_name: 'SubagentStart',
              runId: this.parentRunId,
              threadId,
              parentAgentId: this.parentAgentId,
              agentId: childAgentId,
              agentType: subagentType,
              inputs: [new HumanMessage(description)],
            },
            sessionId: this.parentRunId,
            matchQuery: subagentType,
          }).catch((): AggregatedHookResult => HOOK_FALLBACK);

          if (hookResult.decision === 'deny' || hookResult.decision === 'ask') {
            this.clearChildGraph(childGraph);
            this.activeChildRuns.delete(childExecutionKey);
            return {
              content: `Blocked: ${hookResult.reason ?? 'Blocked by hook'}`,
              messages: [],
            };
          }
        }
        this.startedChildRuns.add(childExecutionKey);

        if (forwarder && !childAlreadyStarted) {
          await this.emitSubagentUpdate(parentRegistry!, {
            childRunId,
            subagentType,
            subagentAgentId: childAgentId,
            parentToolCallId,
            phase: 'start',
            label: `Subagent "${subagentType}" started`,
          });
        }

        let childResult: BaseGraphState;
        if (this.humanInTheLoop?.enabled === true) {
          /** Execute as an independently checkpointed root instead of inheriting
           * the parent's Pregel namespace. Parent decisions are routed explicitly
           * by interrupt id, so concurrent children keep isolated resume state. */
          childResult = await AsyncLocalStorageProviderSingleton.runWithConfig(
            childInvokeConfig,
            (): Promise<BaseGraphState> =>
              workflow.invoke(childInput, childInvokeConfig)
          );
        } else {
          childResult = await workflow.invoke(childInput, childInvokeConfig);
        }
        const childInterrupts = isInterrupted(childResult)
          ? childResult[INTERRUPT]
          : undefined;
        if (childInterrupts != null && childInterrupts.length > 0) {
          throw new GraphInterrupt(childInterrupts);
        }
        result = { messages: childResult.messages };
      }
    } catch (error) {
      if (isGraphInterrupt(error)) {
        const activeChildRun = this.activeChildRuns.get(childExecutionKey);
        if (activeChildRun != null) {
          activeChildRun.pendingInterrupts = error.interrupts;
        }
        await forwarding?.drain();
        throw new GraphInterrupt(
          addSubagentScope(error.interrupts, {
            run_id: childRunId,
            agent_id: childAgentId,
            subagent_type: subagentType,
            parent_tool_call_id: parentToolCallId,
          })
        );
      }
      /** Aborted before any observational work below: parallel siblings — in
       * this executor and, via the graph-scoped breaker, under other
       * parallel agent nodes — stream on the composed child signal, and
       * awaiting forwarding.drain() first would let them consume provider
       * quota for that entire interval. Trips the ENTRY-captured controller:
       * after a reset, a straggler must break its own dead run, not the
       * current one. */
      if (error instanceof StreamLimitExceededError) {
        childBreaker.abort(error);
      }
      const errorMessage = truncateErrorMessage(error);
      if (forwarding) {
        await forwarding.drain();
        await this.emitSubagentUpdate(parentRegistry!, {
          childRunId,
          subagentType,
          subagentAgentId: childAgentId,
          parentToolCallId,
          phase: 'error',
          label: `Subagent "${subagentType}" errored: ${errorMessage}`,
          data: { message: errorMessage },
        });
      }
      this.clearChildGraph(childGraph);
      this.activeChildRuns.delete(childExecutionKey);
      /**
       * A tripped stream circuit breaker is a safety abort, not a recoverable
       * subagent failure: converting it into a tool result would let the
       * parent keep generating (or spawn another child) after the limit
       * fired. Rethrown here and passed through ToolNode's error conversion,
       * so the parent run rejects with the child's limit error.
       */
      if (error instanceof StreamLimitExceededError) {
        throw error;
      }
      return {
        content: `Subagent error: ${errorMessage}`,
        messages: [],
      };
    }

    if (result == null) {
      throw new Error('Subagent completed without producing graph state.');
    }
    const filteredContent = filterSubagentResult(result.messages);

    if (
      !childAlreadyCompleted &&
      this.hookRegistry?.hasHookFor('SubagentStop', this.parentRunId) === true
    ) {
      /**
       * Awaited (not fire-and-forget) for deterministic test synchronization
       * and consistency with PostCompact. The parent is already waiting on the
       * tool result, so the small extra latency is acceptable. Errors are
       * swallowed — SubagentStop is observational.
       */
      await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'SubagentStop',
          runId: this.parentRunId,
          threadId,
          agentId: childAgentId,
          agentType: subagentType,
          messages: result.messages,
        },
        sessionId: this.parentRunId,
        matchQuery: subagentType,
      }).catch(() => {
        /* SubagentStop is observational — swallow errors */
      });
    }

    if (forwarding && !childAlreadyCompleted) {
      await forwarding.drain();
      await this.emitSubagentUpdate(parentRegistry!, {
        childRunId,
        subagentType,
        subagentAgentId: childAgentId,
        parentToolCallId,
        phase: 'stop',
        label: `Subagent "${subagentType}" finished`,
      });
    }
    if (!childAlreadyCompleted) {
      const activeChildRun = this.activeChildRuns.get(childExecutionKey);
      if (activeChildRun != null && parentToolCallId != null) {
        await this.persistChildCheckpointMarker(
          activeChildRun,
          parentToolCallId
        );
      }
    }
    this.completedChildRuns.add(childExecutionKey);

    this.clearChildGraph(childGraph);

    const completedResult = {
      content: filteredContent,
      messages: result.messages,
    };
    this.completedChildResults.set(childExecutionKey, completedResult);
    return completedResult;
  }

  /**
   * Emits a single {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope through the
   * parent's handler registry. Silent no-op when no parent registry is set.
   * Errors are swallowed — update events are observational.
   */
  private async emitSubagentUpdate(
    parentRegistry: HandlerRegistry,
    args: {
      childRunId: string;
      subagentType: string;
      subagentAgentId: string;
      parentToolCallId?: string;
      phase: SubagentUpdatePhase;
      data?: unknown;
      label?: string;
    }
  ): Promise<void> {
    const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
    if (!handler) {
      return;
    }
    const event: SubagentUpdateEvent = {
      runId: this.parentRunId,
      subagentRunId: args.childRunId,
      subagentType: args.subagentType,
      subagentAgentId: args.subagentAgentId,
      parentAgentId: this.parentAgentId,
      parentToolCallId: args.parentToolCallId,
      phase: args.phase,
      data: args.data,
      label: args.label,
      timestamp: new Date().toISOString(),
    };
    try {
      await handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event);
    } catch {
      /* observational — swallow */
    }
  }

  /**
   * Builds a BaseCallbackHandler that intercepts the child graph's custom
   * events. Routing rules:
   *   - `ON_TOOL_EXECUTE` → forwarded as-is to the parent's ON_TOOL_EXECUTE
   *     handler (so event-driven tools work identically for child and parent).
   *   - `ON_RUN_STEP` / `ON_RUN_STEP_DELTA` / `ON_RUN_STEP_COMPLETED` /
   *     `ON_MESSAGE_DELTA` / `ON_REASONING_DELTA` → wrapped in a
   *     {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope with a human-readable
   *     label, delivered to the parent's subagent-update handler.
   *   - Everything else → ignored (keeps parent's UI scoped to the events it
   *     cares about; host apps can extend by registering more phases).
   */
  private createForwarderCallback(args: {
    parentRegistry: HandlerRegistry;
    subagentType: string;
    subagentAgentId: string;
    childRunId: string;
    parentToolCallId?: string;
  }): ForwarderCallback {
    const {
      parentRegistry,
      subagentType,
      subagentAgentId,
      childRunId,
      parentToolCallId,
    } = args;
    const parentRunId = this.parentRunId;
    const parentAgentId = this.parentAgentId;

    const wrap = async (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown
    ): Promise<void> => {
      const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
      if (!handler) {
        return;
      }
      try {
        const event: SubagentUpdateEvent = {
          runId: parentRunId,
          subagentRunId: childRunId,
          subagentType,
          subagentAgentId,
          parentAgentId,
          parentToolCallId,
          phase,
          data: sanitizeForwardedSubagentUpdateData(eventName, data),
          label: summarizeEvent(eventName, data),
          timestamp: new Date().toISOString(),
        };
        await handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event);
      } catch {
        /* observational — swallow */
      }
    };

    const queuedUpdates: QueuedSubagentUpdate[] = [];
    let drainPromise: Promise<void> | undefined;

    const enqueue = (update: QueuedSubagentUpdate): void => {
      if (queuedUpdates.length >= MAX_PENDING_SUBAGENT_UPDATES) {
        const dropIndex = queuedUpdates.findIndex((queued) =>
          isDroppableSubagentUpdatePhase(queued.phase)
        );
        if (dropIndex >= 0) {
          queuedUpdates.splice(dropIndex, 1);
        } else if (isDroppableSubagentUpdatePhase(update.phase)) {
          return;
        }
      }
      queuedUpdates.push(update);
    };

    const drain = async (): Promise<void> => {
      if (drainPromise != null) {
        await drainPromise;
        return;
      }
      drainPromise = (async (): Promise<void> => {
        while (queuedUpdates.length > 0) {
          const update = queuedUpdates.shift();
          if (update == null) {
            continue;
          }
          await wrap(update.eventName, update.phase, update.data);
        }
      })();
      try {
        await drainPromise;
      } finally {
        drainPromise = undefined;
        if (queuedUpdates.length > 0) {
          await drain();
        }
      }
    };

    const scheduleWrap = (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown
    ): void => {
      enqueue({ eventName, phase, data });
      void drain();
    };

    const handler = BaseCallbackHandler.fromMethods({
      [Callback.CUSTOM_EVENT]: async (
        eventName: string,
        data: unknown
      ): Promise<void> => {
        if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
          const toolHandler = parentRegistry.getHandler(
            GraphEvents.ON_TOOL_EXECUTE
          );
          if (toolHandler) {
            await toolHandler.handle(
              GraphEvents.ON_TOOL_EXECUTE,
              data as ToolExecuteBatchRequest
            );
          }
          /**
           * We also surface a short notice in the subagent-update stream so
           * the UI can show "calling <tool>" for each tool the child spawns.
           */
          scheduleWrap(eventName, 'run_step', data);
          return;
        }

        if (eventName === GraphEvents.ON_RUN_STEP) {
          scheduleWrap(eventName, 'run_step', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
          scheduleWrap(eventName, 'run_step_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
          scheduleWrap(eventName, 'run_step_completed', data);
          return;
        }
        if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
          scheduleWrap(eventName, 'message_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_REASONING_DELTA) {
          scheduleWrap(eventName, 'reasoning_delta', data);
          return;
        }
      },
    });
    /**
     * `awaitHandlers = true` is required so the child's `ToolNode` actually
     * blocks on the parent's `ON_TOOL_EXECUTE` handler until it resolves
     * the batch request. Observational `ON_SUBAGENT_UPDATE` calls are queued
     * behind a bounded sequential dispatcher so host UI publication cannot
     * backpressure each child emission or run unbounded concurrent publishes.
     * The executor drains this queue before terminal stop/error envelopes to
     * preserve phase ordering.
     */
    handler.awaitHandlers = true;
    return { handler, drain };
  }
}

/**
 * Builds the child-run equivalent of a host `CHAT_MODEL_END` handler: a
 * callback that joins per-call model identity (captured from
 * `ls_model_name` at chat-model start) with the usage metadata reported at
 * LLM end, and emits a {@link SubagentUsageEvent} through the host's sink.
 *
 * Attached to the child `workflow.invoke` callbacks array, so it observes
 * every model call inside the child graph — the agent loop and any
 * auxiliary calls (e.g. child-side summarization). It does NOT observe
 * deeper subagent levels: each nesting level replaces the callback chain
 * and attaches its own capture handler via the forwarded
 * `subagentUsageSink` on the child graph's input.
 */
function createUsageCaptureHandler(args: {
  sink: SubagentUsageSink;
  subagentType: string;
  subagentRunId: string;
  subagentAgentId: string;
  parentRunId: string;
  /**
   * Child config's provider enum — the default tag when a call carries no
   * `INVOKED_PROVIDER` metadata (hosts key pricing/cache semantics off it).
   */
  provider?: string;
  /**
   * Child config's model, used when a call carries neither `ls_model_name`
   * nor `INVOKED_MODEL` metadata.
   */
  fallbackModel?: string;
}): BaseCallbackHandler {
  const {
    sink,
    subagentType,
    subagentRunId,
    subagentAgentId,
    parentRunId,
    provider,
    fallbackModel,
  } = args;
  /**
   * Per-call attribution keyed by LangChain callback runId. `model` joins
   * `ls_model_name` (provider-reported) with `INVOKED_MODEL` (stamped by
   * `tryFallbackProviders` from the fallback's client options); `provider`
   * is `INVOKED_PROVIDER`, stamped by `attemptInvoke` with the SDK enum of
   * the provider that ACTUALLY served the call — correct for
   * fallback-served calls, where the static config provider would mis-tag
   * pricing/cache semantics.
   */
  const callInfoByCallId = new Map<
    string,
    { model?: string; provider?: string }
  >();
  const handler = BaseCallbackHandler.fromMethods({
    handleChatModelStart: (
      _llm: unknown,
      _messages: unknown,
      runId: string,
      _parentRunId?: string,
      _extraParams?: Record<string, unknown>,
      _tags?: string[],
      metadata?: Record<string, unknown>
    ): void => {
      const callModel =
        asNonEmptyString(metadata?.ls_model_name) ??
        asNonEmptyString(metadata?.[Constants.INVOKED_MODEL]);
      const callProvider = asNonEmptyString(
        metadata?.[Constants.INVOKED_PROVIDER]
      );
      if (callModel != null || callProvider != null) {
        callInfoByCallId.set(runId, {
          model: callModel,
          provider: callProvider,
        });
      }
    },
    handleLLMEnd: async (output: LLMResult, runId: string): Promise<void> => {
      const callInfo = callInfoByCallId.get(runId);
      callInfoByCallId.delete(runId);
      const model = callInfo?.model ?? fallbackModel;
      const callProvider = callInfo?.provider ?? provider;
      for (const generationGroup of output.generations) {
        /**
         * At most ONE event per generation group: each group is one
         * provider request (the outer array is per-prompt for batched
         * calls), and with multiple completions (`n > 1`) every choice in
         * a group repeats the request-level `usage_metadata` — emitting
         * per choice would multiply billed tokens.
         */
        for (const generation of generationGroup) {
          const message = (generation as ChatGeneration | undefined)?.message;
          const usage = (
            message as { usage_metadata?: UsageMetadata } | undefined
          )?.usage_metadata;
          if (usage == null) {
            continue;
          }
          /**
           * Awaited so async host sinks (billing/persistence) complete
           * before the model call resolves — `awaitHandlers` only waits on
           * `handleLLMEnd` itself, so a dropped promise here would let the
           * parent run finish before usage is recorded and would turn sink
           * rejections into unhandled rejections.
           */
          try {
            await sink({
              usage,
              model,
              provider: callProvider,
              subagentType,
              subagentRunId,
              subagentAgentId,
              runId: parentRunId,
            });
          } catch {
            /* observational — a throwing/rejecting host sink must not break the child run */
          }
          break;
        }
      }
    },
    handleLLMError: (_err: unknown, runId: string): void => {
      callInfoByCallId.delete(runId);
    },
  });
  /**
   * Dispatch usage synchronously with each model call so all entries are
   * sunk before `workflow.invoke` resolves — hosts read their accumulator
   * right after the parent run completes.
   */
  handler.awaitHandlers = true;
  return handler;
}

function asNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined;
}

/**
 * Best-effort read of the configured model from a subagent's client
 * options. Providers disagree on the key (`model` vs `modelName`), and the
 * value is only a fallback for calls that carry no `ls_model_name`.
 */
function extractConfiguredModel(agentInputs: AgentInputs): string | undefined {
  const clientOptions = agentInputs.clientOptions as
    | { model?: unknown; modelName?: unknown }
    | undefined;
  if (typeof clientOptions?.model === 'string' && clientOptions.model !== '') {
    return clientOptions.model;
  }
  if (
    typeof clientOptions?.modelName === 'string' &&
    clientOptions.modelName !== ''
  ) {
    return clientOptions.modelName;
  }
  return undefined;
}

function sanitizeChildConfigurable(
  parentConfigurable: Record<string, unknown> | undefined
): Record<string, unknown> {
  if (parentConfigurable == null) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(parentConfigurable).filter(
      ([key]) => !isLangGraphRuntimeConfigKey(key)
    )
  );
}

function isLangGraphRuntimeConfigKey(key: string): boolean {
  return (
    key.startsWith(LANGGRAPH_RUNTIME_CONFIG_PREFIX) ||
    LANGGRAPH_CHECKPOINT_CONFIG_KEYS.has(key) ||
    /** The parent batch's breaker scope must not leak into the child
     * workflow's configurable — children own separate controllers. */
    key === RUN_BREAKER_SCOPE_CONFIG_KEY
  );
}

export function sanitizeForwardedSubagentUpdateData(
  eventName: string,
  data: unknown
): unknown {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    return sanitizeToolExecuteUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    return sanitizeRunStepUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
    return sanitizeRunStepDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    return sanitizeRunStepCompletedUpdateData(data);
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return sanitizeMessageDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_REASONING_DELTA) {
    return sanitizeReasoningDeltaUpdateData(data);
  }
  return undefined;
}

function isDroppableSubagentUpdatePhase(phase: SubagentUpdatePhase): boolean {
  return (
    phase === 'message_delta' ||
    phase === 'reasoning_delta' ||
    phase === 'run_step_delta'
  );
}

function sanitizeToolExecuteUpdateData(
  data: unknown
): SanitizedSubagentToolExecuteData {
  const request = data as Partial<ToolExecuteBatchRequest>;
  const toolCalls = Array.isArray(request.toolCalls)
    ? request.toolCalls.map(sanitizeToolCallForUpdate)
    : [];
  const sanitized: SanitizedSubagentToolExecuteData = { toolCalls };
  if (typeof request.agentId === 'string') {
    sanitized.agentId = request.agentId;
  }
  return sanitized;
}

function sanitizeToolCallForUpdate(
  call: ToolExecuteBatchRequest['toolCalls'][number]
): SanitizedSubagentToolCall {
  const sanitized: SanitizedSubagentToolCall = {
    id: call.id,
    name: call.name,
    args: call.args,
  };
  return sanitized;
}

function sanitizeRunStepUpdateData(
  data: unknown
): SanitizedRunStep | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const step = data as Partial<RunStep>;
  const sanitized: SanitizedRunStep = {};
  assignString(sanitized, 'agentId', step.agentId);
  assignNumber(sanitized, 'groupId', step.groupId);
  assignString(sanitized, 'id', step.id);
  assignNumber(sanitized, 'index', step.index);
  assignString(sanitized, 'runId', step.runId);
  assignNumber(sanitized, 'stepIndex', step.stepIndex);
  assignString(sanitized, 'type', step.type);
  if (step.summary !== undefined) {
    sanitized.summary = step.summary;
  }
  if (step.usage !== undefined) {
    sanitized.usage = step.usage;
  }
  sanitized.stepDetails = sanitizeStepDetails(step.stepDetails);
  return sanitized;
}

function sanitizeRunStepDeltaUpdateData(
  data: unknown
): SanitizedRunStepDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<RunStepDeltaEvent>;
  const sanitized: SanitizedRunStepDelta = {};
  assignString(sanitized, 'id', event.id);
  sanitized.delta = sanitizeToolCallDelta(event.delta);
  return sanitized;
}

function sanitizeRunStepCompletedUpdateData(
  data: unknown
): SanitizedRunStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as { result?: unknown };
  return { result: sanitizeStepCompleted(event.result) };
}

function sanitizeMessageDeltaUpdateData(
  data: unknown
): SanitizedMessageDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<MessageDeltaEvent>;
  const sanitized: SanitizedMessageDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta != null) {
    sanitized.delta = {};
    if (event.delta.content !== undefined) {
      sanitized.delta.content = event.delta.content;
    }
    if (event.delta.tool_call_ids !== undefined) {
      sanitized.delta.tool_call_ids = event.delta.tool_call_ids;
    }
  }
  return sanitized;
}

function sanitizeReasoningDeltaUpdateData(
  data: unknown
): SanitizedReasoningDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<ReasoningDeltaEvent>;
  const sanitized: SanitizedReasoningDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta?.content !== undefined) {
    sanitized.delta = { content: event.delta.content };
  }
  return sanitized;
}

function sanitizeStepDetails(
  stepDetails: unknown
): SanitizedStepDetails | undefined {
  if (!isObjectLike(stepDetails)) {
    return undefined;
  }
  const rawDetails = stepDetails as {
    message_creation?: { message_id?: unknown };
    tool_calls?: unknown[];
    type?: unknown;
  };
  if (rawDetails.type === StepTypes.MESSAGE_CREATION) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.MESSAGE_CREATION,
    };
    const messageId = rawDetails.message_creation?.message_id;
    if (typeof messageId === 'string') {
      sanitized.message_creation = { message_id: messageId };
    }
    return sanitized;
  }
  if (rawDetails.type === StepTypes.TOOL_CALLS) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.TOOL_CALLS,
    };
    if (Array.isArray(rawDetails.tool_calls)) {
      sanitized.tool_calls = rawDetails.tool_calls.map(sanitizeAgentToolCall);
    }
    return sanitized;
  }
  return undefined;
}

function sanitizeToolCallDelta(
  delta: ToolCallDelta | undefined
): SanitizedToolCallDelta | undefined {
  if (!isObjectLike(delta)) {
    return undefined;
  }
  const sanitized: SanitizedToolCallDelta = {};
  assignString(sanitized, 'auth', delta.auth);
  assignNumber(sanitized, 'expires_at', delta.expires_at);
  assignString(sanitized, 'type', delta.type);
  if (delta.summary !== undefined) {
    sanitized.summary = delta.summary;
  }
  if (Array.isArray(delta.tool_calls)) {
    sanitized.tool_calls = delta.tool_calls.map(sanitizeAgentToolCall);
  }
  return sanitized;
}

function sanitizeStepCompleted(
  data: unknown
): SanitizedStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const completed = data as Partial<StepCompleted> & {
    id?: unknown;
    index?: unknown;
    tool_call?: unknown;
  };
  if (completed.type === 'summary') {
    return {
      type: 'summary',
      summary: completed.summary,
    };
  }
  if (completed.type !== 'tool_call') {
    return undefined;
  }
  const sanitized: SanitizedStepCompleted = { type: 'tool_call' };
  assignString(sanitized, 'id', completed.id);
  assignNumber(sanitized, 'index', completed.index);
  sanitized.tool_call = sanitizeProcessedToolCall(completed.tool_call);
  return sanitized;
}

function sanitizeProcessedToolCall(
  toolCall: unknown
): SanitizedProcessedToolCall | undefined {
  if (!isObjectLike(toolCall)) {
    return undefined;
  }
  const call = toolCall as Partial<ProcessedToolCall>;
  const sanitized: SanitizedProcessedToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  assignString(sanitized, 'output', call.output);
  assignString(sanitized, 'outcome', call.outcome);
  assignNumber(sanitized, 'progress', call.progress);
  return sanitized;
}

function sanitizeAgentToolCall(toolCall: unknown): SanitizedAgentToolCall {
  if (!isObjectLike(toolCall)) {
    return {};
  }
  const call = toolCall as SanitizedAgentToolCall;
  const sanitized: SanitizedAgentToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  assignString(sanitized, 'type', call.type);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  if (isObjectLike(call.function)) {
    const fn: SanitizedAgentToolCall['function'] = {};
    assignString(fn, 'name', call.function.name);
    if (
      typeof call.function.arguments === 'string' ||
      isObjectLike(call.function.arguments)
    ) {
      fn.arguments = call.function.arguments;
    }
    sanitized.function = fn;
  }
  return sanitized;
}

function isObjectLike(value: unknown): value is object {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assignString<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'string') {
    target[key] = value as T[K];
  }
}

function assignNumber<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'number') {
    target[key] = value as T[K];
  }
}

/**
 * Produces a short single-line label for an arbitrary forwarded child event.
 * Used to populate {@link SubagentUpdateEvent.label} so the host UI can show
 * a compact status ticker without parsing the raw payload.
 */
export function summarizeEvent(eventName: string, data: unknown): string {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    const req = data as { toolCalls?: Array<{ name?: string }> };
    const names = (req.toolCalls ?? [])
      .map((c) => c.name)
      .filter((n): n is string => typeof n === 'string');
    return names.length > 0 ? `Calling ${names.join(', ')}` : 'Calling tool';
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    const step = data as {
      type?: string;
      stepDetails?: { type?: string; tool_calls?: Array<{ name?: string }> };
    };
    const detailType = step.stepDetails?.type ?? step.type ?? 'step';
    if (detailType === 'tool_calls') {
      const names = (step.stepDetails?.tool_calls ?? [])
        .map((c) => c.name)
        .filter((n): n is string => typeof n === 'string');
      return names.length > 0
        ? `Using tool: ${names.join(', ')}`
        : 'Planning tool call';
    }
    if (detailType === 'message_creation') {
      return 'Thinking…';
    }
    return `Step: ${detailType}`;
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    const step = data as {
      result?: {
        type?: string;
        tool_call?: { name?: string; output?: string };
      };
    };
    const tool = step.result?.tool_call;
    if (tool?.name != null && tool.name !== '') {
      return `Tool ${tool.name} complete`;
    }
    return 'Step complete';
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return 'Streaming…';
  }
  return eventName;
}

/**
 * Walk messages from last to first, returning the text content of the most
 * recent AIMessage that has any. Non-text blocks (tool_use, thinking,
 * redacted_thinking, tool_result) are stripped. If the last AIMessage is
 * pure tool_use (e.g. the subagent hit `maxTurns` mid-tool-call), the walk
 * continues to earlier AIMessages so partial progress is salvaged — this
 * matches Claude Code's behavior in `agentToolUtils.finalizeAgentTool`.
 * Consecutive streamed text-delta blocks with the same provider index are
 * coalesced without adding whitespace. Annotation-only text blocks are
 * ignored; complete text blocks and distinct delta indexes remain separated.
 * Returns "Task completed" only when no AIMessage in the history contains
 * any text.
 */
export function filterSubagentResult(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() !== 'ai') {
      continue;
    }

    const content = messages[i].content;

    if (typeof content === 'string') {
      if (content) return content;
      continue;
    }

    if (!Array.isArray(content)) {
      continue;
    }

    const textParts: string[] = [];
    let textDeltaParts: string[] = [];
    let textDeltaIndex: string | number | undefined;
    const flushTextDeltaParts = (): void => {
      if (textDeltaParts.length === 0) {
        return;
      }
      textParts.push(textDeltaParts.join(''));
      textDeltaParts = [];
      textDeltaIndex = undefined;
    };
    for (const block of content) {
      if (typeof block === 'string') {
        flushTextDeltaParts();
        if (block !== '') {
          textParts.push(block);
        }
        continue;
      }

      const type =
        'type' in block && typeof block.type === 'string' ? block.type : '';
      const isTextDelta = type === TEXT_DELTA_CONTENT_TYPE;
      const isText = type === ContentTypes.TEXT || isTextDelta;
      const text =
        isText && 'text' in block && typeof block.text === 'string'
          ? block.text
          : '';
      if (isTextDelta) {
        if (text === '') {
          continue;
        }
        const index =
          'index' in block &&
          (typeof block.index === 'string' || typeof block.index === 'number')
            ? block.index
            : undefined;
        if (
          textDeltaIndex != null &&
          index != null &&
          index !== textDeltaIndex
        ) {
          flushTextDeltaParts();
        }
        textDeltaIndex ??= index;
        textDeltaParts.push(text);
        continue;
      }

      if (type === ContentTypes.TEXT && text === '') {
        continue;
      }

      flushTextDeltaParts();
      if (text !== '') {
        textParts.push(text);
      }
    }
    flushTextDeltaParts();

    if (textParts.length > 0) {
      return textParts.join('\n');
    }
  }

  return 'Task completed';
}

/**
 * Resolve self-spawn configs by filling in agentInputs from the parent context.
 * Returns configs with agentInputs guaranteed present. Throws on duplicate
 * `type` values to prevent silent config shadowing.
 */
export function resolveSubagentConfigs(
  configs: SubagentConfig[],
  parentContext: AgentContext
): ResolvedSubagentConfig[] {
  const resolved = configs
    .map((config) => {
      if (config.agentInputs != null) {
        return config as ResolvedSubagentConfig;
      }
      if (config.self !== true || parentContext._sourceInputs == null) {
        return null;
      }
      return {
        ...config,
        agentInputs: { ...parentContext._sourceInputs },
      } as ResolvedSubagentConfig;
    })
    .filter((c): c is ResolvedSubagentConfig => c != null);

  const seenTypes = new Set<string>();
  for (const config of resolved) {
    if (seenTypes.has(config.type)) {
      throw new Error(
        `Duplicate subagent type "${config.type}". Each SubagentConfig must have a unique "type" field.`
      );
    }
    seenTypes.add(config.type);
  }

  return resolved;
}

/**
 * Build child AgentInputs from a resolved config, stripping nesting and
 * (optionally) event-driven fields. When `allowNested: true`, the child's
 * `maxSubagentDepth` is decremented so that depth is consumed as the call
 * chain deepens across graph boundaries — the parent's executor-level check
 * alone cannot see into the child graph's separate executor.
 *
 * When `keepToolDefinitions` is `true`, the child retains the parent's
 * `toolDefinitions` so event-driven tools remain usable. This is only safe
 * when the caller has wired a forwarder for `ON_TOOL_EXECUTE` to a
 * registered handler — otherwise the child will hang on tool dispatch.
 *
 * @remarks Advanced utility: exported primarily for testing and by
 * {@link SubagentExecutor}. Host applications configuring subagents should
 * not need to call this directly — it is invoked internally when a subagent
 * tool is dispatched. The depth-countdown contract (parent's `maxDepth` in,
 * child's decremented `maxSubagentDepth` on the returned inputs) is the
 * mechanism that bounds nesting across graph boundaries; callers must
 * respect it.
 */
export function buildChildInputs(
  config: ResolvedSubagentConfig,
  childAgentId: string,
  parentMaxDepth: number,
  keepToolDefinitions: boolean = false
): AgentInputs {
  const { agentInputs } = config;
  const childInputs: AgentInputs = {
    ...agentInputs,
    agentId: childAgentId,
    toolDefinitions: keepToolDefinitions
      ? agentInputs.toolDefinitions
      : undefined,
    /**
     * Subagents run in an isolated context by contract. Parent-run-scoped
     * fields that would otherwise survive the shallow-spread clone — the
     * cross-run conversation summary and the prior-turn tool-discovery
     * set — are cleared here so the child starts fresh. Host applications
     * that want a subagent to see parent context must thread it in
     * explicitly (e.g. via the `description` argument to the subagent
     * tool), not via inherited state.
     */
    initialSummary: undefined,
    discoveredTools: undefined,
    /**
     * Host-supplied direct tools are scrubbed from INHERITED configs only.
     * A self-spawn config's `agentInputs` is a shallow spread of the parent's
     * `_sourceInputs`, so without this a parent-scoped graph tool (e.g. an
     * interrupt-raising ask_user_question) would silently become available to
     * the child. An EXPLICIT child config that lists its own `graphTools` is a
     * deliberate host choice and keeps them (Codex #289 P2); with HITL enabled,
     * those tools use the shared checkpointer and can pause and resume safely.
     */
    graphTools: config.self === true ? undefined : agentInputs.graphTools,
  };

  if (config.allowNested === true) {
    childInputs.maxSubagentDepth = Math.max(0, parentMaxDepth - 1);
  } else {
    childInputs.subagentConfigs = undefined;
    childInputs.maxSubagentDepth = undefined;
  }

  return childInputs;
}

function truncateErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.length <= ERROR_MESSAGE_MAX_CHARS) {
    return message;
  }
  return `${message.slice(0, ERROR_MESSAGE_MAX_CHARS)}...`;
}
