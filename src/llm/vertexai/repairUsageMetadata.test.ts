import { expect, test, describe } from '@jest/globals';
import type { UsageMetadata } from '@langchain/core/messages';
import { repairStreamUsageMetadata } from './index';

const goodUsage: UsageMetadata = {
  input_tokens: 80657,
  output_tokens: 2608,
  total_tokens: 83265,
  output_token_details: { reasoning: 1842 },
};

const buggyFallbackUsage: UsageMetadata = {
  input_tokens: 80657,
  output_tokens: 766,
  total_tokens: 83265,
};

describe('repairStreamUsageMetadata', () => {
  test('replaces buggy fallback usage with tracked good usage from generationInfo', () => {
    const result = repairStreamUsageMetadata(buggyFallbackUsage, goodUsage);
    expect(result).toBe(goodUsage);
  });

  test('returns current unchanged when no generationInfo usage was tracked', () => {
    const result = repairStreamUsageMetadata(buggyFallbackUsage, undefined);
    expect(result).toBe(buggyFallbackUsage);
  });

  test('returns undefined unchanged', () => {
    const result = repairStreamUsageMetadata(undefined, goodUsage);
    expect(result).toBeUndefined();
  });

  test('does not replace when total_tokens differ (different request)', () => {
    const stale: UsageMetadata = { ...goodUsage, total_tokens: 100 };
    const result = repairStreamUsageMetadata(buggyFallbackUsage, stale);
    expect(result).toBe(buggyFallbackUsage);
  });

  test('does not replace when generationInfo output_tokens is not larger (already correct)', () => {
    const equivalent: UsageMetadata = {
      ...buggyFallbackUsage,
      output_tokens: buggyFallbackUsage.output_tokens,
    };
    const result = repairStreamUsageMetadata(buggyFallbackUsage, equivalent);
    expect(result).toBe(buggyFallbackUsage);
  });

  test('does not replace when generationInfo output_tokens is smaller', () => {
    const smaller: UsageMetadata = { ...goodUsage, output_tokens: 100 };
    const result = repairStreamUsageMetadata(buggyFallbackUsage, smaller);
    expect(result).toBe(buggyFallbackUsage);
  });
});
