import type { FadingTier } from '@/types/graph';
import {
  calculateMaxToolCallInputChars,
  calculateMaxToolResultChars,
} from '@/utils/truncation';

export const FADING_TIER_VERSION = 1;

/** Context pressure at which observation masking activates. */
export const PRESSURE_THRESHOLD_MASKING = 0.8;

/** Smallest token budget a rung can shrink to; keeps the emergency floor near 200 chars. */
export const FADING_MIN_BUDGET_TOKENS = 170;

/** Floor for masked (consumed) tool results. */
export const MASKED_RESULT_MIN_CHARS = 300;

/** Fraction of a fresh result's cap that a masked (consumed) result keeps. */
const MASKED_RESULT_CAP_RATIO = 0.1;

/** Largest share of the effective budget a single fresh tool result may occupy. */
const FIT_SHARE = 0.5;

/**
 * Pressure bands expressed as extra rungs on top of the fit rung, so the
 * budget factors land at ×0.5 (85 %), ×0.25 (90 %) and ×0.0625 (99 %).
 */
const PRESSURE_BAND_RUNGS: readonly (readonly [number, number])[] = [
  [0.99, 4],
  [0.9, 2],
  [0.85, 1],
];

export type FadingSignals = {
  /** calibratedTotal / pruningBudget, measured before any truncation. */
  contextPressure: number;
  /** (pruningBudget − instruction tokens) ÷ calibrationRatio, in raw token space. */
  effectiveRawTokens: number;
  summarizationEnabled: boolean;
  /** Recovery paths force at least this rung. */
  minRung?: number;
};

export type FadingCaps = {
  budgetTokens: number;
  /** Cap for tool results the model has not answered yet. */
  resultChars: number;
  /** Cap for consumed tool results; equals `resultChars` until masking activates. */
  consumedChars: number;
  /** Cap for historical tool-call inputs. */
  inputChars: number;
};

function floorBudgetTokens(window: number): number {
  return Math.min(FADING_MIN_BUDGET_TOKENS, window);
}

export function createFadingTier(window: number): FadingTier {
  return { v: FADING_TIER_VERSION, window, rung: 0, masked: false };
}

export function isFadingTier(value: unknown): value is FadingTier {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const { v, window, rung, masked } = value as Partial<
    Record<keyof FadingTier, unknown>
  >;
  return (
    v === FADING_TIER_VERSION &&
    typeof window === 'number' &&
    Number.isFinite(window) &&
    window > 0 &&
    typeof rung === 'number' &&
    Number.isInteger(rung) &&
    rung >= 0 &&
    typeof masked === 'boolean'
  );
}

/** Deepest rung for a window: the point where the budget reaches its floor. */
export function maxFadingRung(window: number): number {
  const floor = floorBudgetTokens(window);
  if (!(floor > 0)) {
    return 0;
  }
  return Math.max(0, Math.ceil(Math.log2(window / floor)));
}

/** Token budget at a rung: the window halved per rung, never below the floor. */
export function fadingBudgetTokens(window: number, rung: number): number {
  return Math.max(floorBudgetTokens(window), Math.floor(window / 2 ** rung));
}

/**
 * Restores a tier persisted by the host. Anything invalid, or derived for a
 * different context window (model switch), starts fresh: the prefix cache is
 * invalid in that case anyway.
 */
export function seedFadingTier(window: number, seed?: unknown): FadingTier {
  if (!isFadingTier(seed) || seed.window !== window) {
    return createFadingTier(window);
  }
  return {
    v: FADING_TIER_VERSION,
    window,
    rung: Math.min(seed.rung, maxFadingRung(window)),
    masked: seed.masked,
  };
}

/** Shallowest rung whose fresh-result cap is at most `targetChars`. */
export function fadingRungForResultChars(
  window: number,
  targetChars: number
): number {
  const deepest = maxFadingRung(window);
  for (let rung = 0; rung < deepest; rung++) {
    if (
      calculateMaxToolResultChars(fadingBudgetTokens(window, rung)) <=
      targetChars
    ) {
      return rung;
    }
  }
  return deepest;
}

/**
 * Shallowest rung at which a single fresh tool result fits within
 * `FIT_SHARE` of `rawTokens`, the effective budget in raw token space.
 * This is the fit guarantee: summarization never sees an empty context
 * because one result alone overflowed the budget.
 */
export function fadingRungForBudget(window: number, rawTokens: number): number {
  if (!(rawTokens > 0)) {
    return 0;
  }
  return fadingRungForResultChars(
    window,
    Math.floor(rawTokens * FIT_SHARE) * 4
  );
}

/** Character caps for a tier; a pure function of `(window, rung, masked)`. */
export function resolveFadingCaps(
  tier: FadingTier,
  maxToolResultChars?: number
): FadingCaps {
  const budgetTokens = fadingBudgetTokens(tier.window, tier.rung);
  const windowResultChars = calculateMaxToolResultChars(budgetTokens);
  const resultChars =
    maxToolResultChars != null && maxToolResultChars > 0
      ? Math.min(windowResultChars, maxToolResultChars)
      : windowResultChars;
  const consumedChars = tier.masked
    ? Math.min(
      resultChars,
      Math.max(
        MASKED_RESULT_MIN_CHARS,
        Math.floor(resultChars * MASKED_RESULT_CAP_RATIO)
      )
    )
    : resultChars;
  return {
    budgetTokens,
    resultChars,
    consumedChars,
    inputChars: calculateMaxToolCallInputChars(budgetTokens),
  };
}

/**
 * Advances a tier from this call's signals. The rung only ever deepens and
 * masking only ever activates, which is the hysteresis: drift in calibration,
 * instruction overhead or message count can escalate once at a boundary but
 * never oscillate. Returns the same object when nothing changes.
 *
 * - The fit rung guarantees a single tool result (30 % of the rung budget)
 *   fits the effective budget, so summarization never sees an empty context.
 * - Pressure bands add rungs on top of the fit rung when summarization is
 *   off, mirroring the staged budget factors of progressive context fading.
 */
export function resolveFadingTier(
  tier: FadingTier,
  signals: FadingSignals
): FadingTier {
  const { window } = tier;
  const fitRung = fadingRungForBudget(window, signals.effectiveRawTokens);
  const bandRung = signals.summarizationEnabled
    ? 0
    : (PRESSURE_BAND_RUNGS.find(
      ([threshold]) => signals.contextPressure >= threshold
    )?.[1] ?? 0);
  const rung = Math.min(
    maxFadingRung(window),
    Math.max(tier.rung, fitRung + bandRung, signals.minRung ?? 0)
  );
  const masked =
    tier.masked || signals.contextPressure >= PRESSURE_THRESHOLD_MASKING;
  if (rung === tier.rung && masked === tier.masked) {
    return tier;
  }
  return { v: FADING_TIER_VERSION, window, rung, masked };
}
