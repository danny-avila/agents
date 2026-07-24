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

function resolveTargetBudget(
  info: ContextOverflowInfo,
  ratio: number | undefined,
  maxContextTokens: number | undefined,
  estimatedPromptTokens: number | undefined
): number | null {
  if (isUsable(info.limitTokens)) {
    return toLocalUnits(info.limitTokens, ratio) * CEILING_HEADROOM_RATIO;
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
    isUsable(info.requestedTokens) && isUsable(estimatedPromptTokens)
      ? info.requestedTokens / estimatedPromptTokens
      : undefined;

  const target = resolveTargetBudget(
    info,
    observedCalibrationRatio,
    maxContextTokens,
    estimatedPromptTokens
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

  const budgetTokens = Math.max(
    MIN_RECOVERY_BUDGET_TOKENS,
    Math.floor(bounded)
  );

  /**
   * Refuse to report a "recovery" that changes nothing: when the floor is
   * already at or above the budget in force, re-pruning cannot free space.
   */
  if (isUsable(maxContextTokens) && budgetTokens >= maxContextTokens) {
    return null;
  }

  return { budgetTokens, info, observedCalibrationRatio };
}
