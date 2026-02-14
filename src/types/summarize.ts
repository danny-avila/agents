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
  trigger?: SummarizationTrigger;
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
