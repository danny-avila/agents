import type { SummaryContentBlock } from '@/types/stream';
import type { Providers } from '@/common';

export type SummarizationTrigger = {
  type:
    | 'token_ratio'
    | 'remaining_tokens'
    | 'messages_to_refine'
    | (string & {});
  value: number;
};

/**
 * Controls how many recent messages are preserved verbatim during
 * compaction.  The most recent user-led turn is always preserved
 * regardless of these caps, so a single oversized first message is
 * never destroyed by summarization.
 */
export type RetainRecentConfig = {
  /**
   * Maximum number of recent user-led turns to keep in the tail.  A turn
   * begins at a HumanMessage and includes every following AIMessage and
   * ToolMessage up to (but not including) the next HumanMessage.  Cutting
   * at turn boundaries guarantees tool_use / tool_result pairs are never
   * split.  Set to `0` to disable the recency window (legacy behavior:
   * summarize everything).  Defaults to `2`.
   */
  turns?: number;
  /**
   * Optional cap on retained-recent tokens beyond the most recent turn.
   * Older turns are added whole only while cumulative tokens stay below
   * the cap.  Defaults to undefined (no cap; bounded only by `turns`).
   */
  tokens?: number;
};

export type SummarizationConfig = {
  provider?: Providers;
  model?: string;
  parameters?: Record<string, unknown>;
  prompt?: string;
  updatePrompt?: string;
  trigger?: SummarizationTrigger;
  maxSummaryTokens?: number;
  /** Fraction of the token budget reserved as headroom (0–1). Defaults to 0.05. */
  reserveRatio?: number;
  /**
   * Recent-message preservation policy.  When unset, defaults to
   * `{ turns: 2 }` so the last two user-led turns are kept verbatim
   * while older content is summarized.  Setting `{ turns: 0 }` reverts
   * to the legacy behavior of summarizing every message.
   */
  retainRecent?: RetainRecentConfig;
};

export interface SummarizeResult {
  text: string;
  tokenCount: number;
  model?: string;
  provider?: string;
}

export interface SummarizationNodeInput {
  remainingContextTokens: number;
  agentId: string;
}

export interface SummarizeStartEvent {
  agentId: string;
  provider: string;
  model?: string;
  messagesToRefineCount: number;
  /** Which summarization cycle this is (1-based, increments each time summarization fires) */
  summaryVersion: number;
}

export interface SummarizeDeltaEvent {
  id: string;
  delta: {
    summary: SummaryContentBlock;
  };
}

export interface SummarizeCompleteEvent {
  id: string;
  agentId: string;
  summary?: SummaryContentBlock;
  error?: string;
}
