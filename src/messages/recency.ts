import type { BaseMessage } from '@langchain/core/messages';

/**
 * Configuration for splitting a message list into a head (to be summarized)
 * and a tail (to be preserved verbatim).
 */
export interface RecencyWindowOptions {
  /**
   * Maximum number of recent user-led turns to keep in the tail.  A "turn"
   * begins at a HumanMessage and includes every following AIMessage and
   * ToolMessage up to (but not including) the next HumanMessage.  Cutting
   * at turn boundaries guarantees that tool_use / tool_result pairs are
   * never split across the head/tail divide.
   *
   * The most recent turn is always preserved regardless of this value or
   * the token cap, so that a single oversized first message is never
   * destroyed by summarization.
   *
   * Defaults to `2`.  A value of `0` disables the recency window (head =
   * everything, tail = empty), restoring the pre-recency-window behavior.
   */
  turns?: number;
  /**
   * Optional cap on tail size in tokens.  When set, additional turns
   * beyond the most recent one are added to the tail only while the
   * cumulative token count stays at or below this cap.  Turns are added
   * whole — never partially — so a turn that would exceed the cap is
   * left in the head.
   *
   * The most recent turn is always preserved even if it exceeds the cap.
   */
  tokens?: number;
  /** Token-counter used to evaluate the optional `tokens` cap. */
  tokenCounter?: (m: BaseMessage) => number;
}

export interface RecencySplit {
  /** Older messages eligible for summarization.  Empty when nothing to summarize. */
  head: BaseMessage[];
  /** Recent messages preserved verbatim.  Always contains the most recent turn when any HumanMessage exists. */
  tail: BaseMessage[];
  /** Number of user-led turns retained in the tail (0 if no HumanMessage exists). */
  tailTurnCount: number;
  /** Index in the original `messages` array where the tail begins. */
  tailStartIndex: number;
}

/**
 * Splits `messages` into a head (older, to summarize) and a tail (recent,
 * to preserve verbatim) at user-message boundaries.  The most recent
 * user-led turn is always included in the tail; additional older turns
 * are added subject to `turns` and `tokens` caps.
 *
 * Cutting strictly at HumanMessage boundaries ensures that:
 * - tool_use ↔ tool_result pairs are never split (they always live within
 *   the same turn);
 * - the first user message is never replaced by a summary, addressing
 *   the "first turn destruction" failure mode where a single large
 *   user-pasted payload would otherwise be replaced by a generic summary.
 *
 * When `messages` contains no HumanMessage (degenerate state — e.g. system
 * + assistant messages from a programmatic preamble), everything is
 * placed in the head and the tail is empty.  The summarize node treats
 * an empty tail as "nothing recent to preserve" and falls through to its
 * existing logic.
 */
export function splitAtRecencyBoundary(
  messages: BaseMessage[],
  options: RecencyWindowOptions = {}
): RecencySplit {
  const turnsCap = options.turns ?? 2;

  if (messages.length === 0 || turnsCap <= 0) {
    return {
      head: messages,
      tail: [],
      tailTurnCount: 0,
      tailStartIndex: messages.length,
    };
  }

  const turnStarts: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].getType() === 'human') {
      turnStarts.push(i);
    }
  }

  if (turnStarts.length === 0) {
    return {
      head: messages,
      tail: [],
      tailTurnCount: 0,
      tailStartIndex: messages.length,
    };
  }

  const lastTurnStart = turnStarts[turnStarts.length - 1] as number;
  let tailStartIndex = lastTurnStart;
  let tailTurnCount = 1;

  const tokensCap = options.tokens;
  const tokenCounter = options.tokenCounter;
  const trackTokens =
    tokensCap != null && Number.isFinite(tokensCap) && tokenCounter != null;

  /**
   * Token-counting strategy: each candidate turn `t` spans the half-open
   * range `[turnStarts[t], turnStarts[t + 1])` (or `[turnStarts[t], messages.length)`
   * for the most recent turn).  Successive iterations of the outer loop
   * walk older turns one at a time and never revisit messages from a
   * later turn — so each message contributes to `tokenCounter` at most
   * once across the entire selection, making the boundary search
   * `O(messages_in_visited_turns)` and bounded by `O(messages.length)`
   * even before the `turnsCap` short-circuit applies.  The inner upper
   * bound uses `turnStarts[t + 1]` (a value derived from immutable
   * `turnStarts`) rather than the mutated `tailStartIndex` to make the
   * disjoint-range invariant self-evident.
   */
  let tailTokens = 0;
  if (trackTokens) {
    for (let i = lastTurnStart; i < messages.length; i++) {
      tailTokens += tokenCounter(messages[i] as BaseMessage);
    }
  }

  for (let t = turnStarts.length - 2; t >= 0; t--) {
    if (tailTurnCount >= turnsCap) {
      break;
    }
    const turnStart = turnStarts[t] as number;
    const turnEnd = turnStarts[t + 1] as number;

    if (trackTokens) {
      let turnTokens = 0;
      for (let i = turnStart; i < turnEnd; i++) {
        turnTokens += tokenCounter(messages[i] as BaseMessage);
      }
      if (tailTokens + turnTokens > (tokensCap as number)) {
        break;
      }
      tailTokens += turnTokens;
    }

    tailStartIndex = turnStart;
    tailTurnCount += 1;
  }

  return {
    head: messages.slice(0, tailStartIndex),
    tail: messages.slice(tailStartIndex),
    tailTurnCount,
    tailStartIndex,
  };
}
