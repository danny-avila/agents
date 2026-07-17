// src/types/graph.ts
import type {
  BaseMessage,
  AIMessageChunk,
  SystemMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { START, StateGraph, StateGraphArgs } from '@langchain/langgraph';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { GoogleAIToolType } from '@langchain/google-common';
import type {
  SummarizationNodeInput,
  SummarizeCompleteEvent,
  SummarizationConfig,
  SummarizeStartEvent,
  SummarizeDeltaEvent,
} from '@/types/summarize';
import type {
  ToolMap,
  ToolEndEvent,
  GenericTool,
  LCTool,
  ToolExecuteBatchRequest,
} from '@/types/tools';
import type {
  RunStep,
  RunStepDeltaEvent,
  MessageDeltaEvent,
  ReasoningDeltaEvent,
} from '@/types/stream';
import type { TokenCounter, TokenBudgetBreakdown } from '@/types/run';
import type { Providers, Callback, GraphNodeKeys } from '@/common';
import type { StandardGraph, MultiAgentGraph } from '@/graphs';
import type { ClientOptions } from '@/types/llm';

/** Interface for bound model with stream and invoke methods */
export interface ChatModel {
  stream?: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<AsyncIterable<AIMessageChunk>>;
  invoke: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<AIMessageChunk>;
}

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

export type BaseGraphState = {
  messages: BaseMessage[];
};

export type AgentSubgraphState = BaseGraphState & {
  summarizationRequest?: SummarizationNodeInput;
};

export type MultiAgentGraphState = BaseGraphState & {
  agentMessages?: BaseMessage[];
};

export type IState = BaseGraphState;

export interface AgentLogEvent {
  level: 'debug' | 'info' | 'warn' | 'error';
  scope: 'prune' | 'summarize' | 'graph' | 'sanitize' | (string & {});
  message: string;
  data?: Record<string, unknown>;
  runId?: string;
  agentId?: string;
}

/**
 * Per-model-call context window usage snapshot, dispatched after pruning and
 * before the model invocation. Dispatched once per `callModel` invocation:
 * fallback retries reuse the snapshot since the prompt is identical — budget
 * numbers reflect the primary provider's tokenizer, and the calibration
 * ratio self-corrects from whichever provider reports usage.
 */
export interface ContextUsageEvent {
  runId?: string;
  agentId?: string;
  /** Structural token budget snapshot from AgentContext.getTokenBudgetBreakdown */
  breakdown: TokenBudgetBreakdown;
  /** Usable budget this call: maxContextTokens minus output reserve */
  contextBudget?: number;
  /** Calibrated instruction overhead actually applied this call */
  effectiveInstructionTokens?: number;
  /** Calibrated message tokens before pruning (excluding instructions) */
  prePruneContextTokens?: number;
  /** Tokens still free after instructions + pruned messages */
  remainingContextTokens?: number;
  /** EMA ratio of provider-reported vs locally estimated token counts */
  calibrationRatio?: number;
}

export interface EventHandler {
  handle(
    event: string,
    data:
      | StreamEventData
      | ModelEndData
      | RunStep
      | RunStepDeltaEvent
      | MessageDeltaEvent
      | ReasoningDeltaEvent
      | SummarizeStartEvent
      | SummarizeDeltaEvent
      | SummarizeCompleteEvent
      | SubagentUpdateEvent
      | AgentLogEvent
      | ContextUsageEvent
      | ToolExecuteBatchRequest
      | { result: ToolEndEvent },
    metadata?: Record<string, unknown>,
    graph?: StandardGraph | MultiAgentGraph
  ): void | Promise<void>;
}

export type GraphStateChannels<T extends BaseGraphState> =
  StateGraphArgs<T>['channels'];

export type Workflow<
  T extends BaseGraphState = BaseGraphState,
  U extends Partial<T> = Partial<T>,
  N extends string = string,
> = StateGraph<T, U, N>;

type LangChainEventStreamCallbackHandlerInput = NonNullable<
  Parameters<Runnable['streamEvents']>[2]
>;

export type EventStreamCallbackHandlerInput =
  LangChainEventStreamCallbackHandlerInput & {
    autoClose?: boolean;
    raiseError?: boolean;
    ignoreCustomEvent?: boolean;
  };

export type WorkflowValuesStreamConfig = RunnableConfig & {
  streamMode: 'values';
};

/**
 * LangGraph stream output is mode-dependent (`values`, `updates`, SSE, etc.).
 * Keep the base Runnable stream output as unknown and narrow at callsites that
 * choose a concrete streamMode.
 */
export type CompiledWorkflow<
  TInput extends BaseGraphState = BaseGraphState,
  TOutput extends BaseGraphState = TInput,
> = Omit<Runnable<TInput, unknown>, 'invoke'> & {
  invoke(input: TInput, config?: RunnableConfig): Promise<TOutput>;
};

export type CompiledStateWorkflow = CompiledWorkflow;

export type CompiledMultiAgentWorkflow = CompiledWorkflow<MultiAgentGraphState>;

export type CompiledAgentWorfklow = CompiledWorkflow<
  AgentSubgraphState,
  AgentSubgraphState
>;

export type SystemRunnable =
  | Runnable<
      BaseMessage[],
      (BaseMessage | SystemMessage)[],
      RunnableConfig<Record<string, unknown>>
    >
  | undefined;

/**
 * Optional compile options passed to workflow.compile().
 * These are intentionally untyped to avoid coupling to library internals.
 */
export type CompileOptions = {
  checkpointer?: unknown;
  interruptBefore?: string[];
  interruptAfter?: string[];
};

export type StreamChunk =
  | (ChatGenerationChunk & {
      message: AIMessageChunk;
    })
  | AIMessageChunk;

/**
 * Data associated with a StreamEvent.
 */
export type StreamEventData = {
  /**
   * The input passed to the runnable that generated the event.
   * Inputs will sometimes be available at the *START* of the runnable, and
   * sometimes at the *END* of the runnable.
   * If a runnable is able to stream its inputs, then its input by definition
   * won't be known until the *END* of the runnable when it has finished streaming
   * its inputs.
   */
  input?: unknown;
  /**
   * The output of the runnable that generated the event.
   * Outputs will only be available at the *END* of the runnable.
   * For most runnables, this field can be inferred from the `chunk` field,
   * though there might be some exceptions for special cased runnables (e.g., like
   * chat models), which may return more information.
   */
  output?: unknown;
  /**
   * A streaming chunk from the output that generated the event.
   * chunks support addition in general, and adding them up should result
   * in the output of the runnable that generated the event.
   */
  chunk?: StreamChunk;
  /**
   * Runnable config for invoking other runnables within handlers.
   */
  config?: RunnableConfig;
  /**
   * Custom result from the runnable that generated the event.
   */
  result?: unknown;
  /**
   * Custom field to indicate the event was manually emitted, and may have been handled already
   */
  emitted?: boolean;
};

/**
 * A streaming event.
 *
 * Schema of a streaming event which is produced from the streamEvents method.
 */
export type StreamEvent = {
  /**
   * Event names are of the format: on_[runnable_type]_(start|stream|end).
   *
   * Runnable types are one of:
   * - llm - used by non chat models
   * - chat_model - used by chat models
   * - prompt --  e.g., ChatPromptTemplate
   * - tool -- LangChain tools
   * - chain - most Runnables are of this type
   *
   * Further, the events are categorized as one of:
   * - start - when the runnable starts
   * - stream - when the runnable is streaming
   * - end - when the runnable ends
   *
   * start, stream and end are associated with slightly different `data` payload.
   *
   * Please see the documentation for `EventData` for more details.
   */
  event: string;
  /** The name of the runnable that generated the event. */
  name: string;
  /**
   * An randomly generated ID to keep track of the execution of the given runnable.
   *
   * Each child runnable that gets invoked as part of the execution of a parent runnable
   * is assigned its own unique ID.
   */
  run_id: string;
  /**
   * Tags associated with the runnable that generated this event.
   * Tags are always inherited from parent runnables.
   */
  tags?: string[];
  /** Metadata associated with the runnable that generated this event. */
  metadata: Record<string, unknown>;
  /**
   * Event data.
   *
   * The contents of the event data depend on the event type.
   */
  data: StreamEventData;
};

export type GraphConfig = {
  provider: string;
  thread_id?: string;
  run_id?: string;
};

export type PartMetadata = {
  progress?: number;
  asset_pointer?: string;
  status?: string;
  action?: boolean;
  output?: string;
  auth?: string;
  expires_at?: number;
};

export type ModelEndData =
  | (StreamEventData & { output: AIMessageChunk | undefined })
  | undefined;
export type GraphTools = GenericTool[] | BindToolsInput[] | GoogleAIToolType[];
export type StandardGraphInput = {
  runId?: string;
  signal?: AbortSignal;
  agents: AgentInputs[];
  langfuse?: LangfuseConfig;
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  calibrationRatio?: number;
  /**
   * Receives a {@link SubagentUsageEvent} for every model call made inside
   * a subagent child run spawned from this graph (including nested
   * subagents and child-side summarization calls). Child graphs run via
   * `invoke()` outside the host's `streamEvents` loop, so their
   * `on_chat_model_end` events never reach the run's handler registry —
   * this sink is the only way hosts can observe child token usage for
   * billing/accounting. Parent-graph model calls are NOT reported here;
   * they already flow through the registry's `CHAT_MODEL_END` handler.
   */
  subagentUsageSink?: SubagentUsageSink;
  /**
   * True when this graph IS a subagent child run (set by `SubagentExecutor`
   * when it constructs the child graph). Drives the hook-input `agentId`
   * subagent-scope marker: hook dispatches from this graph's tool nodes
   * carry `agentId` so run-scoped host hooks — which fire for child scopes
   * too, because children inherit the parent's `run_id` — can tell child
   * scope from the top level. Top-level graphs leave this unset and their
   * hook inputs carry only `executingAgentId`.
   */
  subagentScope?: boolean;
};

export type GraphEdge = {
  /** Agent ID, use a list for multiple sources */
  from: string | string[];
  /** Agent ID, use a list for multiple destinations */
  to: string | string[];
  description?: string;
  /** Can return boolean or specific destination(s) */
  condition?: (state: BaseGraphState) => boolean | string | string[];
  /** 'handoff' creates tools for dynamic routing, 'direct' creates direct edges, which also allow parallel execution */
  edgeType?: 'handoff' | 'direct';
  /**
   * For direct edges: Optional prompt to add when transitioning through this edge.
   * String prompts can include variables like {results} which will be replaced with
   * messages from startIndex onwards. When {results} is used, excludeResults defaults to true.
   *
   * For handoff edges: Description for the input parameter that the handoff tool accepts,
   * allowing the supervisor to pass specific instructions/context to the transferred agent.
   */
  prompt?:
    | string
    | ((
        messages: BaseMessage[],
        runStartIndex: number
      ) => string | Promise<string> | undefined);
  /**
   * When true, excludes messages from startIndex when adding prompt.
   * Automatically set to true when {results} variable is used in prompt.
   */
  excludeResults?: boolean;
  /**
   * For handoff edges: Customizes the parameter name for the handoff input.
   * Defaults to "instructions" if not specified.
   * Only applies when prompt is provided for handoff edges.
   */
  promptKey?: string;
};

export type MultiAgentGraphInput = StandardGraphInput & {
  edges: GraphEdge[];
};

/** Configuration for a subagent type that can be spawned by a parent agent. */
export type SubagentConfig = {
  /** Identifier used in the tool's `subagent_type` enum (e.g. 'researcher', 'coder'). */
  type: string;
  /** Human-readable display name. */
  name: string;
  /** What this subagent specializes in — shown to the LLM. */
  description: string;
  /** Full agent config for the child graph. Omit when `self` is true. */
  agentInputs?: AgentInputs;
  /** When true, reuse the parent's AgentInputs (context isolation without separate config). */
  self?: boolean;
  /** Max AGENT→TOOLS cycles before forced stop (default: 25). */
  maxTurns?: number;
  /** Allow this subagent to spawn its own subagents (default: false). */
  allowNested?: boolean;
};

/** SubagentConfig with agentInputs guaranteed present (self-spawn resolved). */
export type ResolvedSubagentConfig = SubagentConfig & {
  agentInputs: AgentInputs;
};

/** Lifecycle phase carried on {@link SubagentUpdateEvent}. */
export type SubagentUpdatePhase =
  | 'start'
  | 'run_step'
  | 'run_step_delta'
  | 'run_step_completed'
  | 'message_delta'
  | 'reasoning_delta'
  | 'stop'
  | 'error';

/**
 * Wrapper event emitted when a subagent's child graph dispatches activity.
 * Lets hosts show subagent progress in a UI surface separate from the parent
 * conversation without having to untangle events by agent ID.
 */
export interface SubagentUpdateEvent {
  /** Parent run ID. */
  runId: string;
  /** Child run ID (unique per subagent execution). */
  subagentRunId: string;
  /**
   * Parent-side `tool_call_id` for the `subagent` tool invocation that
   * triggered this run. Stable for the duration of the child; lets hosts
   * correlate updates deterministically instead of inferring by ordering.
   * Omitted when the executor was invoked outside of a tool-call context.
   */
  parentToolCallId?: string;
  /** Subagent `type` identifier from the SubagentConfig. */
  subagentType: string;
  /** Child agent ID assigned to this subagent execution. */
  subagentAgentId: string;
  /** Parent agent ID that spawned this subagent. */
  parentAgentId?: string;
  /** Lifecycle phase carried by this update. */
  phase: SubagentUpdatePhase;
  /** Underlying event payload (shape depends on phase). */
  data?: unknown;
  /** Short human-readable description. Hosts can render this directly. */
  label?: string;
  /** ISO timestamp for ordering / display. */
  timestamp: string;
}

/**
 * Token usage for a single model call made inside a subagent child run.
 * Emitted through {@link SubagentUsageSink} as each call completes, so
 * hosts can bill child-run model usage that never reaches the parent
 * run's `CHAT_MODEL_END` handler (child graphs execute via `invoke()`
 * outside the host's `streamEvents` loop).
 */
export interface SubagentUsageEvent {
  /** Usage metadata reported by the child's model call. */
  usage: UsageMetadata;
  /**
   * Model that produced this usage. Per-call `ls_model_name` from the
   * model's callback metadata when available (covers child-side
   * summarization or any call that differs from the configured model),
   * then the fallback-invocation's configured model (`INVOKED_MODEL`
   * metadata), then the subagent config's `clientOptions` model.
   */
  model?: string;
  /**
   * Provider that actually served this call — the SDK `Providers` enum
   * value stamped per-invocation by `attemptInvoke` (`INVOKED_PROVIDER`
   * metadata), so fallback-served calls are attributed to the fallback
   * provider, not the configured primary. Falls back to the subagent
   * config's provider. Never LangSmith's `ls_provider` string — derived
   * providers inherit that from their base class, and hosts key
   * pricing/cache semantics off the enum.
   */
  provider?: string;
  /** Subagent `type` identifier from the SubagentConfig. */
  subagentType: string;
  /** Child run ID (unique per subagent execution). */
  subagentRunId: string;
  /** Child agent ID assigned to this subagent execution. */
  subagentAgentId: string;
  /**
   * ROOT run ID of the host run that owns billing. For nested subagents
   * each forwarding layer rewrites this upward, so events from any depth
   * surface with the outermost run's ID — never an intermediate
   * `*_sub_*` child id (use {@link subagentRunId} to identify the
   * emitting child).
   */
  runId: string;
}

/**
 * Host-provided callback receiving {@link SubagentUsageEvent}s. Invoked as
 * each child model call completes. May return a promise — the executor
 * awaits each dispatch (so all usage is recorded before the child's result
 * resolves to the parent) and swallows both synchronous throws and
 * rejections; implementations should still be cheap, as they sit on the
 * child's model-call path.
 */
export type SubagentUsageSink = (
  event: SubagentUsageEvent
) => void | Promise<void>;

export type LangfuseToolOutputTracingConfig = {
  /**
   * Whether tool outputs should be exported to Langfuse. Defaults to
   * `true`. Set to `false` to keep tool spans and redact their output.
   */
  enabled?: boolean;
  /**
   * Optional allowlist of tool names whose outputs should be redacted even
   * when `enabled` is true.
   */
  redactedToolNames?: string[];
  /**
   * Match strategy for `redactedToolNames`. Defaults to `exact`; use
   * `partial` to redact tools whose names contain a configured value.
   */
  redactedToolNameMatchMode?: 'exact' | 'partial';
  /** Replacement text used for redacted tool outputs. */
  redactionText?: string;
};

export type LangfuseToolNodeTracingConfig = {
  /**
   * Overrides ToolNode callback tracing. ToolNode spans are exported by the
   * env-backed Langfuse callback, so this only enables tracing when that
   * callback is configured.
   */
  enabled?: boolean;
};

export interface LangfuseConfig {
  enabled?: boolean;
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
  /**
   * Environment identifier attached to exported traces (Langfuse
   * `environment`). When unset, falls back to `LANGFUSE_TRACING_ENVIRONMENT`
   * then `NODE_ENV`, so production traces are not collapsed under the
   * `default` environment.
   */
  environment?: string;
  metadata?: Record<string, string | number | boolean | null | undefined>;
  /**
   * Internal OTLP span attributes to attach to Langfuse observations before
   * export. Intended for collector-side routing/filtering; strip these in the
   * collector before forwarding spans to Langfuse.
   */
  librechatTraceAttributes?: Record<
    string,
    string | number | boolean | null | undefined
  >;
  tags?: string[];
  toolNodeTracing?: LangfuseToolNodeTracingConfig;
  toolOutputTracing?: LangfuseToolOutputTracingConfig;
  /**
   * When true, derive the run's root Langfuse trace id deterministically from
   * its `runId` (`sha256(runId)` → 32 hex chars, matching `@langfuse/tracing`
   * `createTraceId`) instead of a random id. This lets external systems attach
   * scores or observations to the trace afterwards by regenerating the same id
   * from the run/message id, without a trace lookup. Default: random ids.
   */
  deterministicTraceId?: boolean;
}

export interface AgentInputs {
  agentId: string;
  /** Human-readable name for the agent (used in handoff context). Defaults to agentId if not provided. */
  name?: string;
  toolEnd?: boolean;
  toolMap?: ToolMap;
  tools?: GraphTools;
  provider: Providers;
  /** Stable/cacheable system instructions. */
  instructions?: string;
  streamBuffer?: number;
  maxContextTokens?: number;
  clientOptions?: ClientOptions;
  /** Per-agent Langfuse tracing configuration. */
  langfuse?: LangfuseConfig;
  /** Dynamic system tail appended after stable instructions without provider cache markers. */
  additional_instructions?: string;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  /** Format content blocks as strings (for legacy compatibility i.e. Ollama/Azure Serverless) */
  useLegacyContent?: boolean;
  /**
   * Tool definitions for all tools, including deferred and programmatic.
   * Used for tool search and programmatic tool calling.
   * Maps tool name to LCTool definition.
   */
  toolRegistry?: Map<string, LCTool>;
  /**
   * Serializable tool definitions for event-driven execution.
   * When provided, ToolNode operates in event-driven mode, dispatching
   * ON_TOOL_EXECUTE events instead of invoking tools directly.
   */
  toolDefinitions?: LCTool[];
  /**
   * Tool names discovered from previous conversation history.
   * These tools will be pre-marked as discovered so they're included
   * in tool binding without requiring tool_search.
   */
  discoveredTools?: string[];
  summarizationEnabled?: boolean;
  summarizationConfig?: SummarizationConfig;
  /** Cross-run summary from a previous run, forwarded from formatAgentMessages.
   *  Injected into the dynamic system tail via AgentContext. */
  initialSummary?: { text: string; tokenCount: number };
  contextPruningConfig?: ContextPruningConfig;
  maxToolResultChars?: number;
  /** Pre-computed tool schema token count (from cache). Skips recalculation when provided. */
  toolSchemaTokens?: number;
  /** Subagent configurations for hierarchical delegation. Each defines a child agent type. */
  subagentConfigs?: SubagentConfig[];
  /** Maximum subagent nesting depth. Default 1 means top-level agents can spawn subagents but subagents cannot nest further. */
  maxSubagentDepth?: number;
  /**
   * Host-supplied tool instances that must execute IN-PROCESS inside the graph's
   * ToolNode even when the run is event-driven (`toolDefinitions` non-empty). Each
   * instance is bound to the model alongside the schema-only event tools and its
   * name is marked direct, so calls bypass ON_TOOL_EXECUTE dispatch and run inside
   * the Pregel task frame. This is the only execution mode where a tool body may
   * raise a LangGraph `interrupt()` (e.g. a tool built on `askUserQuestion()`) —
   * the host-side event handler runs outside the graph task, where `interrupt()`
   * throws. Do NOT also list these tools in `toolDefinitions` (they would be bound
   * twice). NOT inherited by SELF-SPAWNED subagent children (their config is a
   * shallow spread of the parent's inputs, and child graphs compile without a
   * checkpointer, so an interrupt-capable tool could never pause there) —
   * `buildChildInputs` scrubs the inherited copy; an EXPLICIT child config that
   * lists its own `graphTools` keeps them.
   *
   * Deliberately `GenericTool[]`, not `GraphTools`: the wider union admits
   * schema-only shapes (OpenAI `BindToolsInput`, Google tool objects) that
   * `initializeTools` cannot register in the ToolNode direct map — the model
   * would bind a tool the SDK advertised as in-process but cannot execute.
   * Every entry must be a real executable tool instance with a `name`.
   */
  graphTools?: GenericTool[];
}

export interface ContextPruningConfig {
  enabled?: boolean;
  keepLastAssistants?: number;
  softTrimRatio?: number;
  hardClearRatio?: number;
  minPrunableToolChars?: number;
  softTrim?: {
    maxChars?: number;
    headChars?: number;
    tailChars?: number;
  };
  hardClear?: {
    enabled?: boolean;
    placeholder?: string;
  };
}
