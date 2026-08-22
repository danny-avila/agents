import {
  SUMMARY_WRAPPER_OVERHEAD_TOKENS,
  DEFAULT_SUMMARIZATION_PROMPT,
  DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
  separateSummarizationParameters,
  buildSummarizationInstruction,
} from '@/summarization';

describe('summarization primitives', () => {
  it('exports the wrapper overhead the summary token count is measured against', () => {
    expect(SUMMARY_WRAPPER_OVERHEAD_TOKENS).toBe(33);
  });

  describe('separateSummarizationParameters', () => {
    it('lifts maxSummaryTokens out of the parameters passed to the model', () => {
      const result = separateSummarizationParameters({
        maxSummaryTokens: 512,
        temperature: 0.2,
      });
      expect(result).toEqual({
        llmParams: { temperature: 0.2 },
        maxSummaryTokens: 512,
      });
    });

    it('ignores a maxSummaryTokens that is not a positive number', () => {
      expect(separateSummarizationParameters({ maxSummaryTokens: 0 })).toEqual({
        llmParams: {},
        maxSummaryTokens: undefined,
      });
      expect(
        separateSummarizationParameters({ maxSummaryTokens: '512' })
      ).toEqual({ llmParams: {}, maxSummaryTokens: undefined });
    });
  });

  describe('buildSummarizationInstruction', () => {
    it('sends the fresh prompt when there is no prior summary', () => {
      expect(
        buildSummarizationInstruction(
          DEFAULT_SUMMARIZATION_PROMPT,
          DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
          undefined
        )
      ).toBe(DEFAULT_SUMMARIZATION_PROMPT);
    });

    it('sends the update prompt with the prior summary appended', () => {
      const result = buildSummarizationInstruction(
        'fresh',
        'update',
        'earlier checkpoint'
      );
      expect(result).toBe(
        'update\n\n<previous-summary>\nearlier checkpoint\n</previous-summary>'
      );
    });

    it('falls back to the fresh prompt when no update prompt is configured', () => {
      expect(buildSummarizationInstruction('fresh', undefined, 'prior')).toBe(
        'fresh\n\n<previous-summary>\nprior\n</previous-summary>'
      );
    });

    it('treats a blank prior summary as absent rather than consolidating nothing', () => {
      expect(buildSummarizationInstruction('fresh', 'update', '   \n ')).toBe(
        'fresh'
      );
    });
  });
});
