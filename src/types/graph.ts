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
import type { ToolMap, ToolEndEvent, GenericTool } from '@/types/tools';
import type { Providers, Callback, GraphNodeKeys } from '@/common';
import type { StandardGraph, MultiAgentGraph } from '@/graphs';
import type { ClientOptions } from '@/types/llm';
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

export type MultiAgentGraphState = BaseGraphState & {
  agentMessages?: BaseMessage[];
};

export type IState = BaseGraphState;

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
  {
    messages: BaseMessage[];
  },
  {
    messages?: BaseMessage[] | undefined;
  },
  '__start__' | `agent=${string}` | `tools=${string}`,
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  {
    messages: BinaryOperatorAggregate<BaseMessage[], BaseMessage[]>;
  },
  StateDefinition,
  {
    [x: `agent=${string}`]: Partial<BaseGraphState>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [x: `tools=${string}`]: any;
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
    | ((messages: BaseMessage[], runStartIndex: number) => string | undefined);
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

export interface AgentInputs {
  agentId: string;
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
}
