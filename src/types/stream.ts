// src/types/stream.ts
import type OpenAITypes from 'openai';
import type {
  MessageContentImageUrl,
  MessageContentText,
  ToolMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { LLMResult, Generation } from '@langchain/core/outputs';
import type { AnthropicContentBlock } from '@/llm/anthropic/types';
import type { Command } from '@langchain/langgraph';
import type { ToolEndEvent } from '@/types/tools';
import { StepTypes, ContentTypes, GraphEvents } from '@/common/enum';

export type HandleLLMEnd = (
  output: LLMResult,
  runId: string,
  parentRunId?: string,
  tags?: string[]
) => void;

export type MetadataAggregatorResult = {
  handleLLMEnd: HandleLLMEnd;
  collected: Record<string, unknown>[];
};

export type StreamGeneration = Generation & {
  text?: string;
  message?: BaseMessage;
};

/** Event names are of the format: on_[runnable_type]_(start|stream|end).

Runnable types are one of:

llm - used by non chat models
chat_model - used by chat models
prompt -- e.g., ChatPromptTemplate
tool -- LangChain tools
chain - most Runnables are of this type
Further, the events are categorized as one of:

start - when the runnable starts
stream - when the runnable is streaming
end - when the runnable ends
start, stream and end are associated with slightly different data payload.

Please see the documentation for EventData for more details. */
export type EventName = string;

export type RunStep = {
  // id: string;
  // object: 'thread.run.step'; // Updated from 'run.step' # missing
  // created_at: number;
  // run_id: string;
  // assistant_id: string;
  // thread_id: string;
  type: StepTypes;
  // status: 'in_progress' | 'completed' | 'failed' | 'cancelled'; // Add other possible status values if needed
  // cancelled_at: number | null;
  // completed_at: number | null;
  // expires_at: number;
  // failed_at: number | null;
  // last_error: string | null;
  id: string; // #new
  runId?: string; // #new
  agentId?: string; // #new - tracks which agent this step belongs to
  /**
   * Group ID - incrementing number (1, 2, 3...) reflecting execution order.
   * Agents with the same groupId run in parallel and should be rendered together.
   * undefined means the agent runs sequentially (not part of any parallel group).
   *
   * Example for: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: undefined (sequential)
   * - analyst1, analyst2, analyst3: 1 (first parallel group)
   * - summarizer: undefined (sequential)
   */
  groupId?: number; // #new
  index: number; // #new
  stepIndex?: number; // #new
  stepDetails: StepDetails;
  usage?: null | object;
  // {
  // Define usage structure if it's ever non-null
  // prompt_tokens: number; // #new
  // completion_tokens: number; // #new
  // total_tokens: number; // #new
  // };
};

/**
 * Represents a run step delta i.e. any changed fields on a run step during
 * streaming.
 */
export interface RunStepDeltaEvent {
  /**
   * The identifier of the run step, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The delta containing the fields that have changed on the run step.
   */
  delta: ToolCallDelta;
}

export type StepDetails = MessageCreationDetails | ToolCallsDetails;

export type StepCompleted = ToolCallCompleted;

export type MessageCreationDetails = {
  type: StepTypes.MESSAGE_CREATION;
  message_creation: {
    message_id: string;
  };
};

export type ToolEndData = {
  input: string | Record<string, unknown>;
  output?: ToolMessage | Command;
};
export type ToolErrorData = {
  id: string;
  name: string;
  error?: Error;
} & Pick<ToolEndData, 'input'>;
export type ToolEndCallback = (
  data: ToolEndData,
  metadata?: Record<string, unknown>
) => Promise<void>;

export type ProcessedToolCall = {
  name: string;
  args: string | Record<string, unknown>;
  id: string;
  output: string;
  progress: number;
};

export type ProcessedContent = {
  type: ContentType;
  text?: string;
  tool_call?: ProcessedToolCall;
};

export type ToolCallCompleted = {
  type: 'tool_call';
  tool_call: ProcessedToolCall;
};

export type ToolCompleteEvent = ToolCallCompleted & {
  /** The Step Id of the Tool Call */
  id: string;
  /** The content index of the tool call */
  index: number;
  type: 'tool_call';
};

export type ToolCallsDetails = {
  type: StepTypes.TOOL_CALLS;
  tool_calls?: AgentToolCall[]; // #new
};

export type ToolCallDelta = {
  type: StepTypes;
  tool_calls?: ToolCallChunk[]; // #new
};

export type AgentToolCall =
  | {
      id: string; // #new
      type: 'function'; // #new
      function: {
        name: string; // #new
        arguments: string | object; // JSON string // #new
      };
    }
  | ToolCall;

export interface ExtendedMessageContent {
  type?: string;
  text?: string;
  input?: string;
  index?: number;
  id?: string;
  name?: string;
}

export type AgentUpdate = {
  type: ContentTypes.AGENT_UPDATE;
  agent_update: {
    index: number;
    runId: string;
    agentId: string;
  };
};

/**
 * Represents a message delta i.e. any changed fields on a message during
 * streaming.
 */
export interface MessageDeltaEvent {
  /**
   * The identifier of the message, which can be referenced in API endpoints.
   */
  id: string;

  /**
   * The delta containing the fields that have changed on the Message.
   */
  delta: MessageDelta;
}

/**
 * The delta containing the fields that have changed on the Message.
 */
export interface MessageDelta {
  /**
   * The content of the message in array of text and/or images.
   */
  content?: MessageContentComplex[];
  /**
   * The tool call ids associated with the message.
   */
  tool_call_ids?: string[];
}

/**
 * Represents a reasoning delta i.e. any changed fields on a message during
 * streaming.
 */
export interface ReasoningDeltaEvent {
  /**
   * The identifier of the message, which can be referenced in API endpoints.
   */
  id: string;

  /**
   * The delta containing the fields that have changed.
   */
  delta: ReasoningDelta;
}

/**
 * The reasoning delta containing the fields that have changed on the Message.
 */
export interface ReasoningDelta {
  /**
   * The content of the message in array of text and/or images.
   */
  content?: MessageContentComplex[];
}

export type MessageDeltaUpdate = {
  type: ContentTypes.TEXT;
  text: string;
  tool_call_ids?: string[];
};
export type ReasoningDeltaUpdate = { type: ContentTypes.THINK; think: string };

export type ContentType = 'text' | 'image_url' | 'tool_call' | 'think' | string;

export type ReasoningContentText = {
  type: ContentTypes.THINK;
  think: string;
};

/** Vertex AI / Google Common - Reasoning Content Block Format */
export type GoogleReasoningContentText = {
  type: ContentTypes.REASONING;
  reasoning: string;
};

/** Anthropic's Reasoning Content Block Format */
export type ThinkingContentText = {
  type: ContentTypes.THINKING;
  index?: number;
  signature?: string;
  thinking?: string;
};

/** Bedrock's Reasoning Content Block Format */
export type BedrockReasoningContentText = {
  type: ContentTypes.REASONING_CONTENT;
  index?: number;
  reasoningText: { text?: string; signature?: string };
};

/**
 * A call to a tool.
 */
export type ToolCallPart = {
  /** Type ("tool_call") according to Assistants Tool Call Structure */
  type: ContentTypes.TOOL_CALL;
  /** The name of the tool to be called */
  name?: string;
  /** The arguments to the tool call */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args?: string | Record<string, any>;

  /** If provided, an identifier associated with the tool call */
  id?: string;
  /** If provided, the output of the tool call */
  output?: string;
  /** Auth URL */
  auth?: string;
  /** Expiration time */
  expires_at?: number;
};

export type ToolCallContent = {
  type: ContentTypes.TOOL_CALL;
  tool_call?: ToolCallPart;
};

export type ToolResultContent = {
  content:
    | string
    | Record<string, unknown>
    | Array<string | Record<string, unknown>>
    | AnthropicContentBlock[];
  type: 'tool_result' | 'web_search_result' | 'web_search_tool_result';
  tool_use_id?: string;
  input?: string | Record<string, unknown>;
  index?: number;
};

export type MessageContentComplex = (
  | ToolResultContent
  | ThinkingContentText
  | AgentUpdate
  | ToolCallContent
  | ReasoningContentText
  | MessageContentText
  | MessageContentImageUrl
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | (Record<string, any> & {
      type?: 'text' | 'image_url' | 'think' | 'thinking' | string;
    })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | (Record<string, any> & {
      type?: never;
    })
) & {
  tool_call_ids?: string[];
};

export interface TMessage {
  role?: string;
  content?: MessageContentComplex[] | string;
  [key: string]: unknown;
}

export type TPayload = Array<Partial<TMessage>>;

export type CustomChunkDelta =
  | null
  | undefined
  | (Partial<OpenAITypes.Chat.Completions.ChatCompletionChunk.Choice.Delta> & {
      reasoning?: string | null;
      reasoning_content?: string | null;
    });
export type CustomChunkChoice = Partial<
  Omit<OpenAITypes.Chat.Completions.ChatCompletionChunk.Choice, 'delta'> & {
    delta?: CustomChunkDelta;
  }
>;
export type CustomChunk = Partial<OpenAITypes.ChatCompletionChunk> & {
  choices?: Partial<Array<CustomChunkChoice>>;
};

export type SplitStreamHandlers = Partial<{
  [GraphEvents.ON_RUN_STEP]: ({
    event,
    data,
  }: {
    event: GraphEvents;
    data: RunStep;
  }) => void;
  [GraphEvents.ON_MESSAGE_DELTA]: ({
    event,
    data,
  }: {
    event: GraphEvents;
    data: MessageDeltaEvent;
  }) => void;
  [GraphEvents.ON_REASONING_DELTA]: ({
    event,
    data,
  }: {
    event: GraphEvents;
    data: ReasoningDeltaEvent;
  }) => void;
}>;

export type ContentAggregator = ({
  event,
  data,
}: {
  event: GraphEvents;
  data:
    | RunStep
    | MessageDeltaEvent
    | RunStepDeltaEvent
    | {
        result: ToolEndEvent;
      };
}) => void;
/**
 * Metadata for content parts in multi-agent runs.
 * - agentId: present for all MultiAgentGraph runs (enables agent labels in UI)
 * - groupId: present only for parallel execution (enables column rendering)
 */
export type ContentMetadata = {
  agentId?: string;
  groupId?: number;
};

export type ContentAggregatorResult = {
  stepMap: Map<string, RunStep | undefined>;
  contentParts: Array<MessageContentComplex | undefined>;
  /** Map of content index to metadata (agentId, groupId). Only populated for MultiAgentGraph runs. */
  contentMetadataMap: Map<number, ContentMetadata>;
  aggregateContent: ContentAggregator;
};
