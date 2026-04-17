// src/types/graph.ts
import type {
  START,
  StateType,
  UpdateType,
  StateGraph,
  StateGraphArgs,
  StateDefinition,
  CompiledStateGraph,
  BinaryOperatorAggregate,
} from '@langchain/langgraph';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type {
  BaseMessage,
  AIMessageChunk,
  SystemMessage,
} from '@langchain/core/messages';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { GoogleAIToolType } from '@langchain/google-common';
import type {
  ToolMap,
  ToolEndEvent,
  GenericTool,
  LCTool,
  ToolExecuteBatchRequest,
} from '@/types/tools';
import type { Providers, Callback, GraphNodeKeys } from '@/common';
import type { StandardGraph, MultiAgentGraph } from '@/graphs';
import type { ClientOptions } from '@/types/llm';
import type {
  SummarizationNodeInput,
  SummarizeCompleteEvent,
  SummarizationConfig,
  SummarizeStartEvent,
  SummarizeDeltaEvent,
} from '@/types/summarize';
import type {
  RunStep,
  RunStepDeltaEvent,
  MessageDeltaEvent,
  ReasoningDeltaEvent,
} from '@/types/stream';
import type { TokenCounter } from '@/types/run';

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

export type CompiledWorkflow<
  T extends BaseGraphState = BaseGraphState,
  U extends Partial<T> = Partial<T>,
  N extends string = string,
> = CompiledStateGraph<T, U, N>;

export type CompiledStateWorkflow = CompiledStateGraph<
  StateType<{
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  }>,
  UpdateType<{
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  }>,
  string,
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  StateDefinition
>;

export type CompiledMultiAgentWorkflow = CompiledStateGraph<
  StateType<{
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    agentMessages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  }>,
  UpdateType<{
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    agentMessages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  }>,
  string,
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    agentMessages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    agentMessages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  StateDefinition
>;

export type CompiledAgentWorfklow = CompiledStateGraph<
  AgentSubgraphState,
  Partial<AgentSubgraphState>,
  '__start__' | `agent=${string}` | `tools=${string}` | `summarize=${string}`,
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    summarizationRequest: BinaryOperatorAggregate<
      SummarizationNodeInput | undefined,
      SummarizationNodeInput | undefined
    >;
  },
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
    summarizationRequest: BinaryOperatorAggregate<
      SummarizationNodeInput | undefined,
      SummarizationNodeInput | undefined
    >;
  },
  StateDefinition,
  {
    [x: `agent=${string}`]: Partial<BaseGraphState>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [x: `tools=${string}`]: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [x: `summarize=${string}`]: any;
  }
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
  // A checkpointer instance (e.g., MemorySaver, SQL saver)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  checkpointer?: any;
  interruptBefore?: string[];
  interruptAfter?: string[];
};

export type EventStreamCallbackHandlerInput =
  Parameters<CompiledWorkflow['streamEvents']>[2] extends Omit<
    infer T,
    'autoClose'
  >
    ? T
    : never;

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
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  calibrationRatio?: number;
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

export interface AgentInputs {
  agentId: string;
  /** Human-readable name for the agent (used in handoff context). Defaults to agentId if not provided. */
  name?: string;
  toolEnd?: boolean;
  toolMap?: ToolMap;
  tools?: GraphTools;
  provider: Providers;
  instructions?: string;
  streamBuffer?: number;
  maxContextTokens?: number;
  clientOptions?: ClientOptions;
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
   *  Injected into the system message via AgentContext.buildInstructionsString(). */
  initialSummary?: { text: string; tokenCount: number };
  contextPruningConfig?: ContextPruningConfig;
  maxToolResultChars?: number;
  /** Pre-computed tool schema token count (from cache). Skips recalculation when provided. */
  toolSchemaTokens?: number;
  /** Subagent configurations for hierarchical delegation. Each defines a child agent type. */
  subagentConfigs?: SubagentConfig[];
  /** Maximum subagent nesting depth. Default 1 means top-level agents can spawn subagents but subagents cannot nest further. */
  maxSubagentDepth?: number;
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
