import type { BaseMessage } from '@langchain/core/messages';

const stableMessageTokenCounters = new WeakSet<
  (message: BaseMessage) => number
>();

/**
 * Opts a deterministic counter into stable-message count reuse. The counter
 * must depend only on the message surface consumed by `getTokenCountForMessage`.
 */
export function markTokenCounterCacheCompatible<
  TCounter extends (message: BaseMessage) => number,
>(tokenCounter: TCounter): TCounter {
  stableMessageTokenCounters.add(tokenCounter);
  return tokenCounter;
}

export function isTokenCounterCacheCompatible(
  tokenCounter: (message: BaseMessage) => number
): boolean {
  return stableMessageTokenCounters.has(tokenCounter);
}
