import { expect, test, describe } from '@jest/globals';
import type { UsageMetadata } from '@langchain/core/messages';
import { repairStreamUsageMetadata } from './index';

describe('repairStreamUsageMetadata', () => {
  test('adds reasoning to output_tokens when google-common stream omits it', () => {
    const usage: UsageMetadata = {
      input_tokens: 80657,
      output_tokens: 766,
      total_tokens: 83265,
      output_token_details: { reasoning: 1842 },
    };
    repairStreamUsageMetadata(usage);
    expect(usage.output_tokens).toBe(2608);
    expect(usage.output_token_details).toEqual({ reasoning: 1842 });
    expect(usage.total_tokens).toBe(83265);
  });

  test('leaves output_tokens alone when reasoning is already included', () => {
    const usage: UsageMetadata = {
      input_tokens: 80657,
      output_tokens: 2608,
      total_tokens: 83265,
      output_token_details: { reasoning: 1842 },
    };
    repairStreamUsageMetadata(usage);
    expect(usage.output_tokens).toBe(2608);
  });

  test('no-op when output_token_details.reasoning is missing', () => {
    const usage: UsageMetadata = {
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    };
    repairStreamUsageMetadata(usage);
    expect(usage.output_tokens).toBe(50);
  });

  test('no-op when reasoning is zero', () => {
    const usage: UsageMetadata = {
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      output_token_details: { reasoning: 0 },
    };
    repairStreamUsageMetadata(usage);
    expect(usage.output_tokens).toBe(50);
  });

  test('no-op when undefined', () => {
    expect(() => repairStreamUsageMetadata(undefined)).not.toThrow();
  });

  test('does not double-count: total - input <= output means reasoning already in output', () => {
    const usage: UsageMetadata = {
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      output_token_details: { reasoning: 30 },
    };
    repairStreamUsageMetadata(usage);
    expect(usage.output_tokens).toBe(50);
  });
});
