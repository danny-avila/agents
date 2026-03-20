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
  provider?: Providers;
  model?: string;
  parameters?: Record<string, unknown>;
  prompt?: string;
  updatePrompt?: string;
  trigger?: SummarizationTrigger;
  maxSummaryTokens?: number;
  /** Fraction of the token budget reserved as headroom (0–1). Defaults to 0.05. */
  reserveRatio?: number;
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
