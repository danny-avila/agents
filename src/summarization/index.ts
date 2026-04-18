import type { SummarizationTrigger } from '@/types';

const VALID_TRIGGER_TYPES = [
  'token_ratio',
  'remaining_tokens',
  'messages_to_refine',
] as const;

const warnedUnrecognizedTriggerTypes = new Set<string>();

/**
 * Warn (once per process, per unrecognized type) when the configured trigger
 * type is something the runtime does not evaluate. Without this, a misconfigured
 * `trigger.type` silently disables summarization with no visible signal.
 */
function warnUnrecognizedTriggerType(type: string): void {
  if (warnedUnrecognizedTriggerTypes.has(type)) {
    return;
  }
  warnedUnrecognizedTriggerTypes.add(type);
  console.warn(
    `[shouldTriggerSummarization] Unrecognized trigger.type: "${type}". ` +
      `Summarization will not fire. Valid values: ${VALID_TRIGGER_TYPES.join(', ')}.`
  );
}

/** For tests only. Resets the dedup set so warnings can be observed again. */
export function _resetUnrecognizedTriggerWarnings(): void {
  warnedUnrecognizedTriggerTypes.clear();
}

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
  prePruneContextTokens?: number;
  remainingContextTokens?: number;
  messagesToRefineCount: number;
}): boolean {
  const {
    trigger,
    maxContextTokens,
    prePruneContextTokens,
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
      prePruneContextTokens != null &&
      Number.isFinite(prePruneContextTokens)
        ? maxContextTokens - prePruneContextTokens
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
      prePruneContextTokens != null &&
      Number.isFinite(prePruneContextTokens)
        ? maxContextTokens - prePruneContextTokens
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
  warnUnrecognizedTriggerType(trigger.type);
  return false;
}
