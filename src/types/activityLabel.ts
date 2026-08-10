import type { RunnableConfig } from '@langchain/core/runnables';
import type { AssistantTextPhase } from '@/types/assistantPhase';
import type { ClientOptions } from '@/types/llm';
import type { Providers } from '@/common';

/** One tool call's contribution to the label payload (host-assembled). */
export type ActivityLabelToolEntry = {
  toolName: string;
  toolInput: unknown;
  toolOutput?: unknown;
  error?: string;
  status: 'success' | 'error';
};

/**
 * Options for `Run.generateActivityLabel`. The payload deliberately contains
 * NO human messages: intent context comes from the assistant's own last text
 * (Claude Code's pattern) and the block's reasoning excerpts (claude.ai's
 * pattern) — user text stays out of this low-scrutiny pathway entirely.
 *
 * This SDK defines NO activity-label graph event and never dispatches one.
 * Label lifecycle streaming is entirely host-owned: a host claims its own
 * content slots and emits on its own transport, with a payload shape only
 * it defines. The SDK surface here is exactly this method plus the
 * `activity_label` content type's formatter exclusions.
 */
export type RunActivityLabelOptions = {
  provider: Providers;
  clientOptions?: ClientOptions;
  /**
   * Agent that executed the labeled batch. Selects that agent's Langfuse
   * overlay (trace metadata AND tool-output redaction policy) instead of
   * the graph default — a stricter per-agent policy must not be bypassed
   * by labeling work the default agent never performed.
   */
  agentId?: string;
  entries: ActivityLabelToolEntry[];
  /** Truncated reasoning excerpts from the block being labeled. */
  thinkingExcerpts?: string[];
  /** Assistant's last text before the block (~200 chars), as intent context. */
  lastAssistantText?: string;
  /**
   * Provider-authored semantic phase for `lastAssistantText`. Hosts can use
   * this to pass commentary as intent while excluding final-answer text.
   */
  lastAssistantPhase?: AssistantTextPhase;
  /**
   * Headers already committed for earlier batches in this run (run order,
   * most recent last). Continuity context: the prompt shows them so the new
   * header extends the run's story instead of restating a line already on
   * screen. Hosts should pass only COMMITTED labels — a pending slot's text
   * is empty and a dropped fill never surfaced to the user.
   */
  previousLabels?: string[];
  /** Override for the default label system prompt. */
  prompt?: string;
  /** Per-entry serialization cap for the prompt. Default 600. */
  charLimit?: number;
  /** LangChain runnable config carrier (signal, callbacks, thread/user ids). */
  chainOptions?: Partial<RunnableConfig> & {
    configurable?: Record<string, unknown>;
  };
  /**
   * Seed for deterministic Langfuse trace ids (e.g. `${runId}-${slotIndex}`)
   * so each batch's label gets a distinct, reproducible trace. When omitted,
   * a per-run sequence keeps batches from collapsing into one trace.
   */
  traceSeed?: string;
};

/** One logical activity contributing to a run-wide phase summary. */
export type ActivityPhaseEntry = {
  /** Already committed child activity label, preferred over raw evidence. */
  label?: string;
  /** Raw fallback for a child label that was disabled, pending, or dropped. */
  entries?: ActivityLabelToolEntry[];
  /** Standalone or tool-attached reasoning excerpts for this activity. */
  thinkingExcerpts?: string[];
  /** Agent lane that produced the activity. */
  agentId?: string;
  /** Failed activities remain useful evidence and still count. */
  status?: 'success' | 'partial' | 'error';
};

/** Options for `Run.generateActivityPhaseLabel`. */
export type RunActivityPhaseLabelOptions = {
  provider: Providers;
  clientOptions?: ClientOptions;
  /**
   * Logical activities in run order. The SDK requires at least two so hosts
   * cannot accidentally spend a model call summarizing a single activity.
   */
  activities: ActivityPhaseEntry[];
  /** Full count when the host retained only bounded prompt evidence. */
  totalActivityCount?: number;
  /**
   * Bounded assistant commentary emitted inside the phase. Human messages
   * must not be supplied through this low-scrutiny summarization path.
   */
  assistantContext?: string[];
  /** Semantic phase of the text boundary that closed the activity phase. */
  closingTextPhase?: AssistantTextPhase;
  /** Override for the dedicated phase-label system prompt. */
  prompt?: string;
  /** Per-evidence serialization cap for the prompt. Default 600. */
  charLimit?: number;
  /** LangChain runnable config carrier (signal, callbacks, thread/user ids). */
  chainOptions?: Partial<RunnableConfig> & {
    configurable?: Record<string, unknown>;
  };
  /** Deterministic seed for this phase label trace. */
  traceSeed?: string;
  /** Stable source run identifier recorded on the phase observation. */
  sourceRunId?: string;
  /** Source Langfuse trace id for linking this detached summary trace. */
  sourceTraceId?: string;
  /** Host response/message identifier recorded on the phase observation. */
  responseId?: string;
  /** Zero-based index of the phase within the source run. */
  phaseIndex?: number;
  /** Completion state recorded on the phase observation. */
  status?: 'completed' | 'partial' | 'failed';
  /**
   * Contributing agents. Their Langfuse redaction policies are combined into
   * the strictest union before any phase evidence enters the model or trace.
   */
  agentIds?: string[];
};
