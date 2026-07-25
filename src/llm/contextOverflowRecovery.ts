/**
 * Recovery policy for provider context-overflow rejections.
 *
 * Detection (`@/utils/errors`) answers "was this an overflow, and what did
 * the provider disclose?". This module answers the follow-up: "what budget
 * should the retry target?" — deliberately kept as pure functions so the
 * policy can be reasoned about and tested without a graph.
 *
 * The interesting case is unit mismatch. A provider reports its ceiling in
 * *its* tokenizer's units, while our budget is expressed in ours. When the
 * error also names the size it attributed to the prompt we just sent, the
 * two numbers give us the conversion factor for free, so the retry targets a
 * budget that is genuinely under the limit instead of one that merely looks
 * smaller.
 */
import type { ContextOverflowInfo } from '@/utils/errors';
import type { Providers } from '@/common';
import { getContextOverflowInfo } from '@/utils/errors';

/** Fraction of the previous budget used when the provider named no ceiling. */
const BLIND_SHRINK_RATIO = 0.7;

/** Slack left below a known ceiling so the retry is not sized to the edge. */
const CEILING_HEADROOM_RATIO = 0.95;

/**
 * Floor for a recovered budget. Below this the prompt could not hold a system
 * prompt plus one turn, so shrinking further trades an overflow error for an
 * empty-context error.
 */
const MIN_RECOVERY_BUDGET_TOKENS = 4_000;

/**
 * Room a corrected budget must leave above the instructions for the messages
 * themselves. Without it, a budget that merely clears the system prompt and
 * tool schemas is not a budget anything can be compacted into.
 */
const MIN_MESSAGE_HEADROOM_TOKENS = 2_000;

/** Bound on forced-compaction retries per agent, per run. */
export const DEFAULT_MAX_OVERFLOW_RECOVERIES = 2;

export interface OverflowRecoveryPlan {
  /** Budget the retry should target, in the SDK's own token units. */
  budgetTokens: number;
  /** What the provider disclosed. Carried through for logging. */
  info: ContextOverflowInfo;
  /**
   * Provider-reported prompt size divided by our own estimate, when both are
   * known. Greater than 1 means we systematically under-count for this
   * provider.
   *
   * Derived from `info.promptTokens`, never from `info.requestedTokens`:
   * several providers fold the requested completion allowance into the
   * latter, which would inflate the ratio and shrink the prompt far more than
   * the overflow actually calls for.
   */
  observedCalibrationRatio?: number;
}

export interface OverflowRecoveryParams {
  error: unknown;
  provider: Providers;
  /** Budget in force when the rejected prompt was built. */
  maxContextTokens?: number;
  /** Our own estimate of the prompt we actually sent. */
  estimatedPromptTokens?: number;
  /**
   * System prompt plus tool schemas — the part of the budget compaction
   * cannot touch. A corrected budget at or below this leaves no room for
   * messages, and the summarize node refuses to run, so recovery is declined
   * rather than entered.
   */
  instructionTokens?: number;
  /**
   * Completion allowance the caller configured. Providers count it against
   * the same ceiling, so it has to come off the top when the error itself did
   * not break the total down.
   */
  configuredCompletionTokens?: number;
  /** Recoveries already attempted for this agent in this run. */
  attemptsSoFar: number;
  maxAttempts?: number;
}

function isUsable(value: number | undefined): value is number {
  return value != null && Number.isFinite(value) && value > 0;
}

/**
 * Converts the provider's reported ceiling into our token units using the
 * ratio between what it said the prompt cost and what we estimated.
 * Only ratios above 1 are applied: if we over-count relative to the provider,
 * our budget is already conservative and scaling it up would undo that.
 */
function toLocalUnits(limitTokens: number, ratio: number | undefined): number {
  if (ratio == null || ratio <= 1) {
    return limitTokens;
  }
  return limitTokens / ratio;
}

/**
 * The completion allowance the provider counted against the same ceiling,
 * when it reported both the total and the prompt portion. The retry budget
 * governs the prompt only, so this has to come off the ceiling first —
 * otherwise a large `maxTokens` keeps the request over the limit no matter
 * how far the prompt is compacted.
 */
function reservedForCompletion(
  info: ContextOverflowInfo,
  configuredCompletionTokens: number | undefined
): number {
  if (isUsable(info.requestedTokens) && isUsable(info.promptTokens)) {
    const difference = info.requestedTokens - info.promptTokens;
    if (difference > 0) {
      return difference;
    }
  }
  /**
   * No breakdown on offer. Fall back to what the caller configured, because
   * the provider still counts it: targeting the whole ceiling would leave the
   * retry at `prompt + maxTokens` and over the limit however far the prompt
   * is compacted.
   */
  return isUsable(configuredCompletionTokens) ? configuredCompletionTokens : 0;
}

function resolveTargetBudget(
  info: ContextOverflowInfo,
  ratio: number | undefined,
  maxContextTokens: number | undefined,
  estimatedPromptTokens: number | undefined,
  configuredCompletionTokens: number | undefined
): number | null {
  if (isUsable(info.limitTokens)) {
    /** Subtract in provider units, then convert the remainder to ours. */
    const promptCeiling =
      info.limitTokens -
      reservedForCompletion(info, configuredCompletionTokens);
    /**
     * A completion allowance at or above the ceiling leaves nothing for the
     * prompt: even an empty one plus the requested output overruns the limit.
     * Compaction cannot fix that, so declining surfaces the real problem
     * instead of burning the recovery budget on retries that must fail.
     */
    return promptCeiling > 0
      ? toLocalUnits(promptCeiling, ratio) * CEILING_HEADROOM_RATIO
      : null;
  }
  if (isUsable(estimatedPromptTokens)) {
    return estimatedPromptTokens * BLIND_SHRINK_RATIO;
  }
  if (isUsable(maxContextTokens)) {
    return maxContextTokens * BLIND_SHRINK_RATIO;
  }
  return null;
}

/**
 * Decides whether a failed model call is a recoverable context overflow and,
 * if so, what budget the retry should be re-pruned against.
 *
 * Returns `null` when the error is something compaction cannot fix, or when
 * the per-run recovery budget is spent — in both cases the caller should let
 * its normal failure handling proceed.
 */
export function planContextOverflowRecovery({
  error,
  provider,
  maxContextTokens,
  estimatedPromptTokens,
  instructionTokens,
  configuredCompletionTokens,
  attemptsSoFar,
  maxAttempts = DEFAULT_MAX_OVERFLOW_RECOVERIES,
}: OverflowRecoveryParams): OverflowRecoveryPlan | null {
  if (attemptsSoFar >= maxAttempts) {
    return null;
  }

  const info = getContextOverflowInfo(error, {
    provider,
    estimatedPromptTokens,
    maxContextTokens,
  });
  if (info == null) {
    return null;
  }

  const observedCalibrationRatio =
    isUsable(info.promptTokens) && isUsable(estimatedPromptTokens)
      ? info.promptTokens / estimatedPromptTokens
      : undefined;

  const target = resolveTargetBudget(
    info,
    observedCalibrationRatio,
    maxContextTokens,
    estimatedPromptTokens,
    configuredCompletionTokens
  );
  if (target == null) {
    return null;
  }

  /**
   * A retry that does not actually shrink the prompt would just reproduce the
   * same rejection, so a ceiling that lands at or above the budget we already
   * had is replaced by a blind shrink.
   */
  const bounded =
    isUsable(maxContextTokens) && target >= maxContextTokens
      ? maxContextTokens * BLIND_SHRINK_RATIO
      : target;

  const budgetTokens = Math.floor(bounded);

  /**
   * Below the floor there is no usable budget left: either nothing survives
   * pruning, or the instructions alone fill the window, in which case the
   * summarize node refuses to run and the detour would bounce between the
   * agent and summarize nodes without ever shrinking the prompt. Declining
   * lets the existing "instructions exceed context budget" guidance surface
   * instead.
   */
  const floorTokens = Math.max(
    MIN_RECOVERY_BUDGET_TOKENS,
    isUsable(instructionTokens)
      ? instructionTokens + MIN_MESSAGE_HEADROOM_TOKENS
      : 0
  );
  if (budgetTokens < floorTokens) {
    return null;
  }

  /**
   * Refuse to report a "recovery" that changes nothing: when the budget in
   * force is already at or below the target, re-pruning cannot free space.
   */
  if (isUsable(maxContextTokens) && budgetTokens >= maxContextTokens) {
    return null;
  }

  return { budgetTokens, info, observedCalibrationRatio };
}
