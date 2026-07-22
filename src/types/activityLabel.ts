import type { RunnableConfig } from '@langchain/core/runnables';
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
 * Event emission is intentionally host-owned: hosts claim content slots and
 * stream label lifecycle events on their own transport (e.g. LibreChat's
 * `on_activity_label` SSE event). The SDK's contract is limited to this
 * method and the `activity_label` content type's formatter exclusions.
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
