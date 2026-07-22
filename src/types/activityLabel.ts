import type { Providers } from '@/common';

/**
 * Deterministic classification of a completed tool batch, computed from tool
 * names with zero model calls. Rendered immediately as a fallback header
 * ("read 2 files · ran 1 command") until the fast-model label arrives.
 */
export type ActivityCounts = {
  searches: number;
  reads: number;
  writes: number;
  commands: number;
  other: number;
};

/**
 * Configuration for fast-model activity labeling. Fully configurable like
 * title generation: the label model never derives from the main loop model —
 * hosts resolve provider/model independently (e.g. titleModel-style
 * precedence) and pass the result here at run creation.
 */
export type ActivityLabelConfig = {
  enabled?: boolean;
  provider?: Providers;
  model?: string;
  parameters?: Record<string, unknown>;
  /** Override for the label system prompt. */
  prompt?: string;
  /** Per-tool input/output truncation for the label prompt. Default 300 chars. */
  charLimit?: number;
  /** Cost guard: maximum labels generated per run. Default 20. */
  maxPerRun?: number;
};

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
 */
export type RunActivityLabelOptions = {
  provider: Providers;
  clientOptions?: Record<string, unknown>;
  entries: ActivityLabelToolEntry[];
  /** Truncated reasoning excerpts from the block being labeled. */
  thinkingExcerpts?: string[];
  /** Assistant's last text before the block (~200 chars), as intent context. */
  lastAssistantText?: string;
  /** Override for the default label system prompt. */
  prompt?: string;
  /** Per-entry serialization cap for the prompt. Default 300. */
  charLimit?: number;
  /** LangChain runnable config carrier (signal, callbacks, thread/user ids). */
  chainOptions?: Record<string, unknown>;
  /**
   * Seed for deterministic Langfuse trace ids (e.g. `${runId}-${slotIndex}`)
   * so each batch's label gets a distinct, reproducible trace.
   */
  traceSeed?: string;
};

/**
 * Payload of `GraphEvents.ON_ACTIVITY_LABEL`. Additive event: hosts and SDK
 * consumers that do not handle it are unaffected. Emitted once per tool
 * batch — first with `label` undefined at batch completion (deterministic
 * counts only), then again with the fast-model `label` when it resolves.
 */
export type ActivityLabelEvent = {
  runId: string;
  /** Run step id of the tool batch this label describes. */
  stepId?: string;
  /** Tool call ids covered by this label — the trace/grouping anchor. */
  toolCallIds: string[];
  /** Fast-model generated label (~30 chars, git-commit-subject style). */
  label?: string;
  counts: ActivityCounts;
  /** ok = all tools succeeded, failed = all failed, partial = mixed. */
  status: 'ok' | 'partial' | 'failed';
  /** Owning agent in multi-agent graphs, for lane grouping. */
  agentId?: string;
};
