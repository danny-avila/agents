// src/types/run.ts
import type * as z from 'zod';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  BaseCallbackHandler,
  CallbackHandlerMethods,
} from '@langchain/core/callbacks/base';
import type * as s from '@/types/stream';
import type * as e from '@/common/enum';
import type * as g from '@/types/graph';
import type * as l from '@/types/llm';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ZodObjectAny = z.ZodObject<any, any, any, any>;
export type BaseGraphConfig = {
  llmConfig: l.LLMConfig;
  provider?: e.Providers;
  clientOptions?: l.ClientOptions;
  /** Optional compile options for workflow.compile() */
  compileOptions?: g.CompileOptions;
};
export type LegacyGraphConfig = BaseGraphConfig & {
  type?: 'standard';
} & Omit<g.StandardGraphInput, 'provider' | 'clientOptions' | 'agents'> &
  Omit<g.AgentInputs, 'provider' | 'clientOptions' | 'agentId'>;

/* Supervised graph (opt-in) */
export type SupervisedGraphConfig = BaseGraphConfig & {
  type: 'supervised';
  /** Enable supervised router; when false, fall back to standard loop */
  routerEnabled?: boolean;
  /** Table-driven routing policy per stage */
  routingPolicies?: Array<{
    stage: string;
    agents?: string[];
    model?: e.Providers;
    parallel?: boolean;
    /** Optional simple condition on content/tools */
    when?:
      | 'always'
      | 'has_tools'
      | 'no_tools'
      | { includes?: string[]; excludes?: string[] };
  }>;
  /** Opt-in feature flags */
  featureFlags?: {
    multi_model_routing?: boolean;
    fan_out?: boolean;
    fan_out_retries?: number;
    fan_out_backoff_ms?: number;
    fan_out_concurrency?: number;
  };
  /** Optional per-stage model configs */
  models?: Record<string, l.LLMConfig>;
} & Omit<g.StandardGraphInput, 'provider' | 'clientOptions'>;

export type RunTitleOptions = {
  inputText: string;
  provider: e.Providers;
  contentParts: (s.MessageContentComplex | undefined)[];
  titlePrompt?: string;
  skipLanguage?: boolean;
  clientOptions?: l.ClientOptions;
  chainOptions?: Partial<RunnableConfig> | undefined;
  omitOptions?: Set<string>;
  titleMethod?: e.TitleMethod;
  titlePromptTemplate?: string;
};

export interface AgentStateChannels {
  messages: BaseMessage[];
  next: string;
  [key: string]: unknown;
  instructions?: string;
  additional_instructions?: string;
}

export interface Member {
  name: string;
  systemPrompt: string;
  tools: StructuredTool[];
  llmConfig: l.LLMConfig;
}

export type CollaborativeGraphConfig = {
  type: 'collaborative';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type TaskManagerGraphConfig = {
  type: 'taskmanager';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type MultiAgentGraphConfig = {
  type: 'multi-agent';
  compileOptions?: g.CompileOptions;
  agents: g.AgentInputs[];
  edges: g.GraphEdge[];
  modelSpecs?: g.ModelSpecsConfig;
};

export type StandardGraphConfig = Omit<
  MultiAgentGraphConfig,
  'edges' | 'type'
> & { type?: 'standard'; signal?: AbortSignal };

export type RunConfig = {
  runId: string;
  graphConfig: LegacyGraphConfig | StandardGraphConfig | MultiAgentGraphConfig;
  customHandlers?: Record<string, g.EventHandler>;
  returnContent?: boolean;
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
};

export type ProvidedCallbacks =
  | (BaseCallbackHandler | CallbackHandlerMethods)[]
  | undefined;

export type TokenCounter = (message: BaseMessage) => number;
export type EventStreamOptions = {
  callbacks?: g.ClientCallbacks;
  keepContent?: boolean;
};
