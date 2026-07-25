import { describe, expect, it } from '@jest/globals';
import {
  planContextOverflowRecovery,
  DEFAULT_MAX_OVERFLOW_RECOVERIES,
} from '@/llm/contextOverflowRecovery';
import { OVERFLOW_SIGNATURES } from '@/utils/__tests__/fixtures/contextOverflowSignatures';
import { Providers } from '@/common';

function signatureFor(model: string) {
  const signature = OVERFLOW_SIGNATURES.find((s) => s.model === model);
  if (signature == null) {
    throw new Error(`missing fixture for ${model}`);
  }
  return signature;
}

describe('planContextOverflowRecovery', () => {
  it('targets the ceiling the provider reported, with headroom', () => {
    const anthropic = signatureFor('claude-haiku-4-5-20251001');
    const plan = planContextOverflowRecovery({
      error: anthropic.error,
      provider: Providers.ANTHROPIC,
      /** Misconfigured: caller believed the window was far larger. */
      maxContextTokens: 1_000_000,
      estimatedPromptTokens: 274_468,
      attemptsSoFar: 0,
    });

    expect(plan).not.toBeNull();
    expect(plan?.budgetTokens).toBeLessThan(200_000);
    expect(plan?.budgetTokens).toBeGreaterThan(180_000);
    expect(plan?.info.limitTokens).toBe(200_000);
  });

  it('converts the reported ceiling into our units when we under-count', () => {
    const openrouter = signatureFor('qwen/qwen-2.5-7b-instruct');
    /**
     * OpenRouter counted 56,811 input tokens for a prompt we estimated at
     * 42,599 — it bills roughly 1.33 of our tokens per token, so a budget set
     * to its raw 32,768 ceiling would still overflow.
     */
    const plan = planContextOverflowRecovery({
      error: openrouter.error,
      provider: Providers.OPENROUTER,
      maxContextTokens: 32_768,
      estimatedPromptTokens: 42_599,
      attemptsSoFar: 0,
    });

    expect(plan?.observedCalibrationRatio).toBeCloseTo(1.333, 2);
    expect(plan?.budgetTokens).toBeLessThan(32_768 / 1.3);
  });

  it('calibrates on the prompt, never on a completion-inclusive total', () => {
    const openai = signatureFor('gpt-5-nano');
    /**
     * The 429 quotes "Requested 480002", which includes the output
     * allowance. Reading that as the prompt size would invent a ratio and
     * shrink the budget far past what the overflow calls for.
     */
    const plan = planContextOverflowRecovery({
      error: openai.error,
      provider: Providers.OPENAI,
      maxContextTokens: 400_000,
      estimatedPromptTokens: 240_000,
      attemptsSoFar: 0,
    });

    expect(plan?.observedCalibrationRatio).toBeUndefined();
    expect(plan?.budgetTokens).toBe(Math.floor(200_000 * 0.95));
  });

  it('reserves the completion allowance the provider counted against the ceiling', () => {
    /**
     * A 64k completion allowance inside a 128k ceiling leaves 64k for the
     * prompt; targeting the full ceiling would overflow again no matter how
     * far the prompt is compacted.
     */
    const error = {
      name: 'ContextOverflowError',
      message:
        '400 This model\'s maximum context length is 128000 tokens. However, you requested 190000 tokens (126000 in the messages, 64000 in the completion). Please reduce the length of the messages or completion.',
    };
    const plan = planContextOverflowRecovery({
      error,
      provider: Providers.DEEPSEEK,
      maxContextTokens: 128_000,
      estimatedPromptTokens: 126_000,
      attemptsSoFar: 0,
    });

    expect(plan?.budgetTokens).toBe(Math.floor((128_000 - 64_000) * 0.95));
  });

  it('shrinks blindly when the provider named no numbers', () => {
    const bedrock = signatureFor(
      'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    );
    const plan = planContextOverflowRecovery({
      error: bedrock.error,
      provider: Providers.BEDROCK,
      maxContextTokens: 200_000,
      estimatedPromptTokens: 190_000,
      attemptsSoFar: 0,
    });

    expect(plan?.info.limitTokens).toBeUndefined();
    expect(plan?.budgetTokens).toBe(Math.floor(190_000 * 0.7));
  });

  it('recovers the mid-stream Bedrock overflow that arrives as HTTP 200', () => {
    const nova = signatureFor('us.amazon.nova-lite-v1:0');
    const plan = planContextOverflowRecovery({
      error: nova.error,
      provider: Providers.BEDROCK,
      maxContextTokens: 300_000,
      estimatedPromptTokens: 290_000,
      attemptsSoFar: 0,
    });

    expect(plan).not.toBeNull();
    expect(plan?.budgetTokens).toBeLessThan(300_000);
  });

  it('recovers an OpenAI token-bucket rejection', () => {
    const openai = signatureFor('gpt-5-nano');
    const plan = planContextOverflowRecovery({
      error: openai.error,
      provider: Providers.OPENAI,
      maxContextTokens: 400_000,
      estimatedPromptTokens: 480_000,
      attemptsSoFar: 0,
    });

    expect(plan?.info.kind).toBe('request_too_large');
    /** The bucket, not the context window, is the binding constraint. */
    expect(plan?.budgetTokens).toBeLessThan(200_000);
  });

  it('stops once the per-run recovery budget is spent', () => {
    const anthropic = signatureFor('claude-haiku-4-5-20251001');
    const params = {
      error: anthropic.error,
      provider: Providers.ANTHROPIC,
      maxContextTokens: 1_000_000,
      estimatedPromptTokens: 274_468,
    };

    expect(
      planContextOverflowRecovery({
        ...params,
        attemptsSoFar: DEFAULT_MAX_OVERFLOW_RECOVERIES - 1,
      })
    ).not.toBeNull();
    expect(
      planContextOverflowRecovery({
        ...params,
        attemptsSoFar: DEFAULT_MAX_OVERFLOW_RECOVERIES,
      })
    ).toBeNull();
  });

  it('declines when the budget cannot meaningfully shrink further', () => {
    const anthropic = signatureFor('claude-haiku-4-5-20251001');
    const plan = planContextOverflowRecovery({
      error: anthropic.error,
      provider: Providers.ANTHROPIC,
      maxContextTokens: 3_000,
      estimatedPromptTokens: 2_900,
      attemptsSoFar: 0,
    });

    expect(plan).toBeNull();
  });

  it('declines for failures compaction cannot fix', () => {
    const plan = planContextOverflowRecovery({
      error: {
        status: 429,
        message: '429 Rate limit reached. Try again in 4ms.',
      },
      provider: Providers.OPENAI,
      maxContextTokens: 128_000,
      estimatedPromptTokens: 100_000,
      attemptsSoFar: 0,
    });

    expect(plan).toBeNull();
  });

  it('successive recoveries keep shrinking', () => {
    const bedrock = signatureFor(
      'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    );
    const first = planContextOverflowRecovery({
      error: bedrock.error,
      provider: Providers.BEDROCK,
      maxContextTokens: 200_000,
      estimatedPromptTokens: 190_000,
      attemptsSoFar: 0,
    });
    const second = planContextOverflowRecovery({
      error: bedrock.error,
      provider: Providers.BEDROCK,
      maxContextTokens: first?.budgetTokens,
      estimatedPromptTokens: first?.budgetTokens,
      attemptsSoFar: 1,
    });

    expect(second?.budgetTokens).toBeLessThan(first?.budgetTokens ?? 0);
  });
});
