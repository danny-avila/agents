import type { RunnableConfig } from '@langchain/core/runnables';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ClientOptions } from '@/types/llm';
import type { Providers } from '@/common';

/** Lifecycle state of the visible reasoning snapshot being labeled. */
export type ReasoningLabelStatus = 'streaming' | 'complete';

/** Result of one reasoning-label revision. */
export type ReasoningLabelResult = {
  label?: string;
  /** Provider-reported usage for host-side billing after a durable commit. */
  usage?: UsageMetadata;
};

/** Options for `Run.generateReasoningLabel`. */
export type RunReasoningLabelOptions = {
  provider: Providers;
  clientOptions?: ClientOptions;
  /**
   * Complete user-visible reasoning accumulated for this step so far. Hidden
   * chain-of-thought must never be supplied through this API.
   */
  visibleReasoning: string;
  /** Stable run-step identity shared by every revision of this label. */
  reasoningStepId: string;
  /** Monotonically increasing host revision for this reasoning step. */
  revision: number;
  /** Whether the snapshot can still grow. Default `streaming`. */
  status?: ReasoningLabelStatus;
  /**
   * Last durably visible label for this step. The model repeats it exactly
   * when the reasoning direction has not materially changed, allowing hosts
   * to avoid redundant UI updates.
   */
  previousLabel?: string;
  /**
   * Agent that emitted the reasoning. Selects its Langfuse overlay and
   * redaction policy. Unknown agents and omitted multi-agent ownership fail
   * closed before tracing or generation.
   */
  agentId?: string;
  /** Override for the default reasoning-label system prompt. */
  prompt?: string;
  /** Maximum reasoning characters retained in the prompt. Default 6000. */
  charLimit?: number;
  /** LangChain runnable config carrier (signal, callbacks, thread/user ids). */
  chainOptions?: Partial<RunnableConfig> & {
    configurable?: Record<string, unknown>;
  };
  /** Deterministic seed for this reasoning-label revision trace. */
  traceSeed?: string;
  /** Stable source run identifier recorded on the observation. */
  sourceRunId?: string;
  /** Source Langfuse trace id retained as correlation metadata. */
  sourceTraceId?: string;
  /** Host response/message identifier recorded on the observation. */
  responseId?: string;
};
