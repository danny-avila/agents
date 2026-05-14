import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { BaseCheckpointSaver } from '@langchain/langgraph';
import type * as t from '@/types';

export type JsonPrimitive = string | number | boolean | null;

export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | {
      [key: string]: JsonValue;
    };

export type JsonObject = {
  [key: string]: JsonValue;
};

export type SessionEntryType =
  | 'message'
  | 'summary'
  | 'compaction'
  | 'checkpoint'
  | 'label'
  | 'run_event'
  | 'session_state';

export interface SessionHeader {
  type: 'session';
  version: 1;
  id: string;
  timestamp: string;
  cwd: string;
  name?: string;
  parentSession?: string;
}

export interface SessionEntryBase<TType extends SessionEntryType, TData> {
  type: TType;
  id: string;
  parentId: string | null;
  timestamp: string;
  data: TData;
}

export interface SerializedSessionMessage {
  messageType: string;
  content: JsonValue;
  additionalKwargs?: JsonObject;
  responseMetadata?: JsonObject;
  id?: string;
  name?: string;
  toolCallId?: string;
  toolCalls?: JsonValue;
  usageMetadata?: JsonObject;
}

export type SessionMessageEntry = SessionEntryBase<
  'message',
  {
    role: string;
    message: SerializedSessionMessage;
  }
>;

export type SessionSummaryEntry = SessionEntryBase<
  'summary',
  {
    text: string;
    tokenCount?: number;
    retainedEntryIds: JsonValue[];
    summarizedEntryIds: JsonValue[];
    instructions?: string;
  }
>;

export type SessionCompactionEntry = SessionEntryBase<
  'compaction',
  {
    summaryEntryId: string;
    retainedEntryIds: JsonValue[];
    summarizedEntryIds: JsonValue[];
  }
>;

export type SessionCheckpointEntry = SessionEntryBase<
  'checkpoint',
  {
    provider: 'langgraph';
    source: 'run' | 'resume' | 'reset';
    threadId: string;
    runId?: string;
    checkpointId?: string;
    checkpointNs?: string;
    parentCheckpointId?: string;
    reason?: string;
  }
>;

export type SessionLabelEntry = SessionEntryBase<
  'label',
  {
    targetEntryId: string;
    label: string;
  }
>;

export type SessionRunEventEntry = SessionEntryBase<
  'run_event',
  {
    event: string;
    runId?: string;
    threadId?: string;
    payload?: JsonValue;
  }
>;

export type SessionStateEntry = SessionEntryBase<
  'session_state',
  {
    leafId: string | null;
  }
>;

export type SessionEntry =
  | SessionMessageEntry
  | SessionSummaryEntry
  | SessionCompactionEntry
  | SessionCheckpointEntry
  | SessionLabelEntry
  | SessionRunEventEntry
  | SessionStateEntry;

export interface SessionTreeNode {
  entry: SessionEntry;
  children: SessionTreeNode[];
}

export interface SessionListItem {
  id: string;
  path: string;
  cwd: string;
  timestamp: string;
  name?: string;
  leafId?: string | null;
}

export type SessionPosition = 'before' | 'at';

export interface SessionForkOptions {
  position?: SessionPosition;
  cwd?: string;
  name?: string;
}

export interface SessionBranchOptions {
  position?: SessionPosition;
  summarizeAbandoned?: boolean | { instructions?: string };
}

export interface SessionCompactOptions {
  instructions?: string;
  retainRecentTurns?: number;
}

export interface AgentSessionRunResult {
  text: string;
  content: t.MessageContentComplex[];
  messages: BaseMessage[];
  usage: AgentSessionUsage;
  steps: t.RunStep[];
  interrupt: t.RunInterruptResult | undefined;
  haltedReason: string | undefined;
  runId: string;
  threadId: string;
}

export interface AgentSessionUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface AgentSessionCheckpointReference {
  provider: 'langgraph';
  threadId: string;
  checkpointId?: string;
  checkpointNs?: string;
  parentCheckpointId?: string;
}

export interface AgentSessionCheckpointLookupOptions {
  threadId?: string;
  checkpointNs?: string;
  config?: RunnableConfig;
}

export interface AgentSessionCheckpointingOptions {
  enabled?: boolean;
  checkpointer?: BaseCheckpointSaver;
}

export type AgentSessionCheckpointing =
  | boolean
  | AgentSessionCheckpointingOptions;

export type AgentSessionInput = string | BaseMessage | BaseMessage[] | t.IState;

export interface AgentSessionRunOptions {
  runId?: string;
  threadId?: string;
  config?: Partial<RunnableConfig> & { version?: 'v1' | 'v2' };
  streamOptions?: t.EventStreamOptions;
}

export interface AgentSessionStreamEvent {
  type:
    | 'run.started'
    | 'message.delta'
    | 'reasoning.delta'
    | 'tool.started'
    | 'tool.delta'
    | 'tool.completed'
    | 'usage.updated'
    | 'run.completed'
    | 'run.failed'
    | 'run.interrupted'
    | 'run.halted';
  sequence: number;
  runId: string;
  threadId: string;
  timestamp: string;
  data?: JsonValue;
}

export interface AgentSessionStream
  extends AsyncIterable<AgentSessionStreamEvent> {
  toTextStream(): AsyncIterable<string>;
  finalResult(): Promise<AgentSessionRunResult>;
}

export interface AgentSessionHandlersResult {
  contentParts: Array<t.MessageContentComplex | undefined>;
  steps: t.RunStep[];
  usage: AgentSessionUsage;
  events: AgentSessionStreamEvent[];
  handlers: Record<string, t.EventHandler>;
}

export type AgentSessionConfig =
  | {
      runConfig: t.RunConfig;
      cwd?: string;
      sessionPath?: string;
      sessionId?: string;
      name?: string;
      ephemeral?: boolean;
      checkpointing?: AgentSessionCheckpointing;
    }
  | ({
      runId?: string;
      cwd?: string;
      sessionPath?: string;
      sessionId?: string;
      name?: string;
      ephemeral?: boolean;
      checkpointing?: AgentSessionCheckpointing;
    } & Omit<t.RunConfig, 'runId'>);

export interface CreateSessionFileOptions {
  cwd: string;
  name?: string;
  parentSession?: string;
  sessionId?: string;
}
