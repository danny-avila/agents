import type { BaseMessage } from '@langchain/core/messages';
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

export type SummarizationConfig = {
  enabled?: boolean;
  provider?: Providers;
  model?: string;
  parameters?: Record<string, unknown>;
  prompt?: string;
  /** Prompt used when updating an existing summary with new messages. Falls back to `prompt` if not set. */
  updatePrompt?: string;
  trigger?: SummarizationTrigger;
  /** When false, disables streaming for summarization LLM calls (uses invoke instead of stream). Defaults to true. */
  stream?: boolean;
  /** Maximum output tokens for the summarization model. Defaults to 2048. */
  maxSummaryTokens?: number;
  /**
   * Fraction of the effective token budget to reserve as headroom (0–1).
   * Pruning triggers when conversation tokens exceed `effectiveMax * (1 - reserveRatio)`,
   * preventing the context from filling to 100% before pruning kicks in.
   * This compensates for approximate token counting on non-OpenAI providers and
   * gives the model breathing room. Defaults to 0.05 (5%).
   */
  reserveRatio?: number;
};

export interface SummarizeResult {
  text: string;
  tokenCount: number;
  model?: string;
  provider?: string;
  targetMessageId?: string;
  targetContentIndex?: number;
}

export interface SummarizationNodeInput {
  messagesToRefine: BaseMessage[];
  context: BaseMessage[];
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
  agentId: string;
  summary: SummaryContentBlock;
  error?: string;
}
