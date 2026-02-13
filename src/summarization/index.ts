import type { SummarizationTrigger } from '@/types';

/**
 * Determines whether summarization should be triggered based on the configured trigger
 * and current context state.
 *
 * Default behavior (no trigger configured): returns `true` whenever messages were pruned.
 * This is intentional — when an admin enables summarization without specifying a trigger,
 * summarization fires on any context overflow that causes pruning.
 *
 * When a trigger IS configured but required runtime data is missing (e.g., maxContextTokens
 * unavailable for a token_ratio trigger), returns `false` — we cannot evaluate the condition,
 * so we do not fire.
 */
export function shouldTriggerSummarization(params: {
  trigger?: SummarizationTrigger;
  maxContextTokens?: number;
  prePruneTotalTokens?: number;
  remainingContextTokens?: number;
  messagesToRefineCount: number;
}): boolean {
  const {
    trigger,
    maxContextTokens,
    prePruneTotalTokens,
    remainingContextTokens,
    messagesToRefineCount,
  } = params;
  if (messagesToRefineCount <= 0) {
    return false;
  }

  // No trigger configured: default to always summarize when pruning occurs.
  if (!trigger || typeof trigger.type !== 'string') {
    return true;
  }

  const triggerValue =
    typeof trigger.value === 'number' && Number.isFinite(trigger.value)
      ? trigger.value
      : undefined;

  // Trigger configured but value is invalid: cannot evaluate, do not fire.
  if (triggerValue == null) {
    return false;
  }

  if (trigger.type === 'token_ratio') {
    const prePruneRemainingContextTokens =
      maxContextTokens != null &&
      Number.isFinite(maxContextTokens) &&
      maxContextTokens > 0 &&
      prePruneTotalTokens != null &&
      Number.isFinite(prePruneTotalTokens)
        ? maxContextTokens - prePruneTotalTokens
        : undefined;
    const effectiveRemainingContextTokens =
      prePruneRemainingContextTokens ?? remainingContextTokens;

    // Required runtime data missing: cannot evaluate token_ratio, do not fire.
    if (
      maxContextTokens == null ||
      !Number.isFinite(maxContextTokens) ||
      maxContextTokens <= 0 ||
      effectiveRemainingContextTokens == null ||
      !Number.isFinite(effectiveRemainingContextTokens)
    ) {
      return false;
    }
    const usedRatio = 1 - effectiveRemainingContextTokens / maxContextTokens;
    return usedRatio >= triggerValue;
  }

  if (trigger.type === 'remaining_tokens') {
    const prePruneRemainingContextTokens =
      maxContextTokens != null &&
      Number.isFinite(maxContextTokens) &&
      maxContextTokens > 0 &&
      prePruneTotalTokens != null &&
      Number.isFinite(prePruneTotalTokens)
        ? maxContextTokens - prePruneTotalTokens
        : undefined;
    const effectiveRemainingContextTokens =
      prePruneRemainingContextTokens ?? remainingContextTokens;

    // Required runtime data missing: cannot evaluate remaining_tokens, do not fire.
    if (
      effectiveRemainingContextTokens == null ||
      !Number.isFinite(effectiveRemainingContextTokens)
    ) {
      return false;
    }
    return effectiveRemainingContextTokens <= triggerValue;
  }

  if (trigger.type === 'messages_to_refine') {
    return messagesToRefineCount >= triggerValue;
  }

  // Unrecognized trigger type: cannot evaluate, do not fire.
  return false;
}
