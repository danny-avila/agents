import type { SummarizationTrigger } from '@/types';

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
  if (!trigger || typeof trigger.type !== 'string') {
    return true;
  }
  const triggerValue =
    typeof trigger.value === 'number' && Number.isFinite(trigger.value)
      ? trigger.value
      : undefined;
  if (triggerValue == null) {
    return true;
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
    if (
      maxContextTokens == null ||
      !Number.isFinite(maxContextTokens) ||
      maxContextTokens <= 0 ||
      effectiveRemainingContextTokens == null ||
      !Number.isFinite(effectiveRemainingContextTokens)
    ) {
      return true;
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
    if (
      effectiveRemainingContextTokens == null ||
      !Number.isFinite(effectiveRemainingContextTokens)
    ) {
      return true;
    }
    return effectiveRemainingContextTokens <= triggerValue;
  }

  if (trigger.type === 'messages_to_refine') {
    return messagesToRefineCount >= triggerValue;
  }

  return true;
}
