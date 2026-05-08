import {
  shouldTriggerSummarization,
  _resetUnrecognizedTriggerWarnings,
} from '@/summarization';

describe('shouldTriggerSummarization', () => {
  it('uses pre-prune pressure for token_ratio triggers when messages were pruned', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'token_ratio', value: 0.8 },
      maxContextTokens: 2500,
      prePruneContextTokens: 3200,
      remainingContextTokens: 1200,
      messagesToRefineCount: 4,
    });

    expect(result).toBe(true);
  });

  it('uses pre-prune remaining tokens for remaining_tokens triggers when available', () => {
    const result = shouldTriggerSummarization({
      trigger: { type: 'remaining_tokens', value: 500 },
      maxContextTokens: 2500,
      prePruneContextTokens: 2300,
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
      prePruneContextTokens: 2400,
      remainingContextTokens: 100,
      messagesToRefineCount: 0,
    });

    expect(result).toBe(false);
  });

  describe('unrecognized trigger type', () => {
    let warnSpy: jest.SpyInstance;

    beforeEach(() => {
      _resetUnrecognizedTriggerWarnings();
      warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    });

    afterEach(() => {
      warnSpy.mockRestore();
    });

    it('does not fire and warns once per unrecognized type', () => {
      const baseParams = {
        maxContextTokens: 2500,
        prePruneContextTokens: 2400,
        remainingContextTokens: 100,
        messagesToRefineCount: 4,
      };

      // Cast via `unknown` because the type union guards against this at compile
      // time; we are intentionally exercising the runtime fallback.
      const result1 = shouldTriggerSummarization({
        ...baseParams,
        trigger: { type: 'token_count', value: 8000 } as unknown as {
          type: 'token_ratio';
          value: number;
        },
      });

      expect(result1).toBe(false);
      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(warnSpy.mock.calls[0][0]).toContain('token_count');
      expect(warnSpy.mock.calls[0][0]).toContain('token_ratio');
      expect(warnSpy.mock.calls[0][0]).toContain('remaining_tokens');
      expect(warnSpy.mock.calls[0][0]).toContain('messages_to_refine');

      // Same unrecognized type a second time: no duplicate warning.
      shouldTriggerSummarization({
        ...baseParams,
        trigger: { type: 'token_count', value: 8000 } as unknown as {
          type: 'token_ratio';
          value: number;
        },
      });
      expect(warnSpy).toHaveBeenCalledTimes(1);

      // Different unrecognized type: warns again, once.
      shouldTriggerSummarization({
        ...baseParams,
        trigger: { type: 'nonsense', value: 1 } as unknown as {
          type: 'token_ratio';
          value: number;
        },
      });
      expect(warnSpy).toHaveBeenCalledTimes(2);
      expect(warnSpy.mock.calls[1][0]).toContain('nonsense');
    });

    it('does not grow memory unboundedly under a flood of unique types', () => {
      const baseParams = {
        maxContextTokens: 2500,
        prePruneContextTokens: 2400,
        remainingContextTokens: 100,
        messagesToRefineCount: 4,
      };

      for (let i = 0; i < 500; i++) {
        shouldTriggerSummarization({
          ...baseParams,
          trigger: { type: `bogus-${i}`, value: 1 } as unknown as {
            type: 'token_ratio';
            value: number;
          },
        });
      }

      // Still logged each new type (up to the cap) — we never silently dropped
      // warnings; we just evicted oldest entries from the dedup set.
      expect(warnSpy).toHaveBeenCalledTimes(500);

      // Re-warns for a recently-seen type that should still be in the cache
      // (last one just inserted). No duplicate warning means the dedup set
      // still functions; the size cap did not break the dedup contract.
      const beforeRecent = warnSpy.mock.calls.length;
      shouldTriggerSummarization({
        ...baseParams,
        trigger: { type: 'bogus-499', value: 1 } as unknown as {
          type: 'token_ratio';
          value: number;
        },
      });
      expect(warnSpy).toHaveBeenCalledTimes(beforeRecent);
    });
  });
});
