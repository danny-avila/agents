import { describe, it, expect } from '@jest/globals';
import {
  HARD_MAX_TOOL_RESULT_CHARS,
  HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE,
  calculateMaxToolResultChars,
  calculateMaxTotalToolOutputSize,
} from './truncation';

describe('truncation helpers', () => {
  describe('calculateMaxToolResultChars', () => {
    it('returns the hard cap when context tokens are missing', () => {
      expect(calculateMaxToolResultChars()).toBe(HARD_MAX_TOOL_RESULT_CHARS);
      expect(calculateMaxToolResultChars(undefined)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
      expect(calculateMaxToolResultChars(0)).toBe(HARD_MAX_TOOL_RESULT_CHARS);
      expect(calculateMaxToolResultChars(-100)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
    });

    it('computes 30% of context-window characters for normal inputs', () => {
      // 100k tokens * 0.3 = 30k tokens * 4 chars/token = 120k chars
      expect(calculateMaxToolResultChars(100_000)).toBe(120_000);
    });

    it('clamps to the hard cap for large context windows', () => {
      // 1M tokens * 0.3 * 4 = 1.2M chars, exceeds 400k cap
      expect(calculateMaxToolResultChars(1_000_000)).toBe(
        HARD_MAX_TOOL_RESULT_CHARS
      );
    });
  });

  describe('calculateMaxTotalToolOutputSize', () => {
    it('returns the absolute hard cap when no per-output is provided', () => {
      expect(calculateMaxTotalToolOutputSize()).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      expect(calculateMaxTotalToolOutputSize(0)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      expect(calculateMaxTotalToolOutputSize(-1)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
    });

    it('doubles the per-output cap by default', () => {
      expect(calculateMaxTotalToolOutputSize(100_000)).toBe(200_000);
      expect(calculateMaxTotalToolOutputSize(1)).toBe(2);
    });

    it('clamps the doubled value to HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE', () => {
      // 4M * 2 = 8M, exceeds 5M
      expect(calculateMaxTotalToolOutputSize(4_000_000)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
      // Right at the boundary: 2.5M * 2 = 5M (no clamp).
      expect(calculateMaxTotalToolOutputSize(2_500_000)).toBe(5_000_000);
      // Just past it: 2_500_001 * 2 = 5_000_002 -> clamped.
      expect(calculateMaxTotalToolOutputSize(2_500_001)).toBe(
        HARD_MAX_TOTAL_TOOL_OUTPUT_SIZE
      );
    });
  });
});
