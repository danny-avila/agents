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

  it('scales our own estimate by the measured overage, needing no unit conversion', () => {
    const openrouter = signatureFor('qwen/qwen-2.5-7b-instruct');
    /**
     * OpenRouter counted 56,811 input tokens for a prompt we estimated at
     * 42,599 — a ~1.33 divergence. The ratio `promptCeiling / promptTokens`
     * is unit-free, so applying it to our estimate lands on a correct target
     * without converting between tokenizers. Converting instead would
     * double-count against the pruner's own calibration, which already
     * divides the budget by its learned ratio.
     */
    const plan = planContextOverflowRecovery({
      error: openrouter.error,
      provider: Providers.OPENROUTER,
      maxContextTokens: 40_000,
      estimatedPromptTokens: 42_599,
      attemptsSoFar: 0,
    });

    expect(plan?.observedCalibrationRatio).toBeCloseTo(1.333, 2);
    const overageScale = (32_768 - 16) / 56_811;
    expect(plan?.budgetTokens).toBe(Math.floor(42_599 * overageScale * 0.95));
  });

  it('recovers on a small-window model despite the absolute floor', () => {
    /** 4,096-token window: 95% of the ceiling lands under MIN_RECOVERY_BUDGET. */
    const error = {
      name: 'ContextOverflowError',
      message:
        '400 This model\'s maximum context length is 4096 tokens. However, your messages resulted in 6000 tokens. Please reduce the length of the messages.',
    };
    const plan = planContextOverflowRecovery({
      error,
      provider: Providers.OPENAI,
      maxContextTokens: 8_192,
      estimatedPromptTokens: 6_000,
      instructionTokens: 200,
      attemptsSoFar: 0,
    });

    expect(plan?.budgetTokens).toBe(Math.floor(4_096 * 0.95));
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

  it('reserves the configured completion allowance on a total-only error', () => {
    const openai = signatureFor('gpt-5-nano');
    /**
     * The 429 quotes a 200k bucket and no breakdown. With a 32k completion
     * allowance configured, a retry aimed at the whole bucket would still be
     * `prompt + 32k` and over the limit.
     */
    const plan = planContextOverflowRecovery({
      error: openai.error,
      provider: Providers.OPENAI,
      maxContextTokens: 400_000,
      estimatedPromptTokens: 240_000,
      configuredCompletionTokens: 32_000,
      attemptsSoFar: 0,
    });

    expect(plan?.budgetTokens).toBe(Math.floor((200_000 - 32_000) * 0.95));
  });

  it('prefers the provider breakdown over the configured allowance', () => {
    const openrouter = signatureFor('qwen/qwen-2.5-7b-instruct');
    /** The error itemized 16 output tokens; that beats a stale config value. */
    const withConfig = planContextOverflowRecovery({
      error: openrouter.error,
      provider: Providers.OPENROUTER,
      maxContextTokens: 32_768,
      estimatedPromptTokens: 42_599,
      configuredCompletionTokens: 8_000,
      attemptsSoFar: 0,
    });
    const withoutConfig = planContextOverflowRecovery({
      error: openrouter.error,
      provider: Providers.OPENROUTER,
      maxContextTokens: 32_768,
      estimatedPromptTokens: 42_599,
      attemptsSoFar: 0,
    });

    expect(withConfig?.budgetTokens).toBe(withoutConfig?.budgetTokens);
  });

  it('declines when the completion allowance alone exceeds the ceiling', () => {
    const openai = signatureFor('gpt-5-nano');
    /**
     * A 240k output allowance against a 200k bucket: no prompt, however
     * small, makes this request fit. Spending recovery attempts on it would
     * bury the real problem.
     */
    const plan = planContextOverflowRecovery({
      error: openai.error,
      provider: Providers.OPENAI,
      maxContextTokens: 400_000,
      estimatedPromptTokens: 240_000,
      configuredCompletionTokens: 240_000,
      attemptsSoFar: 0,
    });

    expect(plan).toBeNull();
  });

  it('declines when the corrected budget would not clear the instructions', () => {
    const anthropic = signatureFor('claude-haiku-4-5-20251001');
    /**
     * Tool schemas alone fill the provider's real window, so compaction has
     * nowhere to put the messages and the summarize node would refuse to run —
     * the detour would bounce between nodes without shrinking anything.
     */
    const plan = planContextOverflowRecovery({
      error: anthropic.error,
      provider: Providers.ANTHROPIC,
      maxContextTokens: 1_000_000,
      estimatedPromptTokens: 274_468,
      instructionTokens: 199_000,
      attemptsSoFar: 0,
    });

    expect(plan).toBeNull();
  });

  it('still recovers when the instructions leave room for messages', () => {
    const anthropic = signatureFor('claude-haiku-4-5-20251001');
    const plan = planContextOverflowRecovery({
      error: anthropic.error,
      provider: Providers.ANTHROPIC,
      maxContextTokens: 1_000_000,
      estimatedPromptTokens: 274_468,
      instructionTokens: 20_000,
      attemptsSoFar: 0,
    });

    expect(plan?.budgetTokens).toBeGreaterThan(180_000);
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
