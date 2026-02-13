import { shouldTriggerSummarization } from '@/summarization';

describe('shouldTriggerSummarization', () => {
  it('uses pre-prune pressure for token_ratio triggers when messages were pruned', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'token_ratio', value: 0.8 },
      maxContextTokens: 2500,
      prePruneTotalTokens: 3200,
      remainingContextTokens: 1200,
      messagesToRefineCount: 4,
    });

    expect(result).toBe(true);
  });

  it('uses pre-prune remaining tokens for remaining_tokens triggers when available', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'remaining_tokens', value: 500 },
      maxContextTokens: 2500,
      prePruneTotalTokens: 2300,
      remainingContextTokens: 1400,
      messagesToRefineCount: 2,
    });

    expect(result).toBe(true);
  });

  it('falls back to post-prune remaining tokens when pre-prune totals are unavailable', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'token_ratio', value: 0.6 },
      maxContextTokens: 2500,
      remainingContextTokens: 1200,
      messagesToRefineCount: 2,
    });

    expect(result).toBe(false);
  });

  it('does not trigger when there is nothing to refine', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'token_ratio', value: 0.1 },
      maxContextTokens: 2500,
      prePruneTotalTokens: 2400,
      remainingContextTokens: 100,
      messagesToRefineCount: 0,
    });

    expect(result).toBe(false);
  });
});
