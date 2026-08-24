import type {
  AIMessage,
  BaseMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { getProviderSourceMessageIds } from './provenance';
import {
  getBoundedProviderPairingArray,
  getBoundedProviderPairingArrayProperty,
  getProviderToolCallPartDescriptor,
  getProviderToolResultPartDescriptor,
  PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS,
} from './toolResultTypes';

export const DEFAULT_RETAIN_RECENT_TURNS = 2;
export const DEFAULT_INTRA_TURN_RETAIN_RATIO = 0.16;

/**
 * Configuration for splitting a message list into a head (to be summarized)
 * and a tail (to be preserved verbatim).
 */
export interface RecencyWindowOptions {
  /**
   * Maximum number of recent user-led turns to keep in the tail. A "turn"
   * begins at a user-authored HumanMessage and includes every following
   * AIMessage and tool result up to the next user-authored HumanMessage.
   * Provider-native HumanMessages containing only tool results remain in the
   * current turn, so the boundary cannot split them from their calls.
   *
   * The most recent turn is preserved unless `intraTurnTokens` enables the
   * pairing-balanced fallback for a tool-heavy history. A lone oversized user
   * message is never eligible for that fallback.
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
  /**
   * Minimum token budget to retain when the turn window would otherwise make
   * the whole history indivisible. When set with `tokenCounter`, older closed
   * tool-call/result units may be summarized from within the earliest retained
   * turn. A lone user payload and open tool units remain indivisible.
   */
  intraTurnTokens?: number;
}

export interface RecencySplit {
  /** Older messages eligible for summarization.  Empty when nothing to summarize. */
  head: BaseMessage[];
  /** Recent messages preserved verbatim, beginning at a pairing-balanced boundary. */
  tail: BaseMessage[];
  /** Number of user-led turns retained in the tail (0 if no HumanMessage exists). */
  tailTurnCount: number;
  /** Index in the original `messages` array where the tail begins. */
  tailStartIndex: number;
}

export function resolveIntraTurnRetainTokens({
  tokens,
  maxContextTokens,
}: {
  tokens?: number;
  maxContextTokens?: number;
}): number | undefined {
  if (tokens != null) {
    return Number.isFinite(tokens) && tokens > 0
      ? Math.floor(tokens)
      : undefined;
  }
  if (
    maxContextTokens == null ||
    !Number.isFinite(maxContextTokens) ||
    maxContextTokens <= 0
  ) {
    return undefined;
  }
  return Math.max(
    1,
    Math.floor(maxContextTokens * DEFAULT_INTRA_TURN_RETAIN_RATIO)
  );
}

function readOwnString(value: unknown, key: string): string | undefined {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor != null &&
      'value' in descriptor &&
      typeof descriptor.value === 'string' &&
      descriptor.value !== '' &&
      descriptor.value.length <= PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS
      ? descriptor.value
      : undefined;
  } catch {
    return undefined;
  }
}

function addToolPairingId(ids: Set<string>, candidate: unknown): void {
  if (
    typeof candidate === 'string' &&
    candidate !== '' &&
    candidate.length <= PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS
  ) {
    ids.add(candidate);
  }
}

function getToolCallIds(message: BaseMessage): Set<string> {
  const ids = new Set<string>();
  const messageRole = (message as BaseMessage & { role?: unknown }).role;
  if (message.getType() !== 'ai' && messageRole !== 'assistant') {
    return ids;
  }

  for (const toolCall of (message as AIMessage).tool_calls ?? []) {
    addToolPairingId(ids, toolCall.id);
  }

  const rawToolCalls = getBoundedProviderPairingArrayProperty(
    message.additional_kwargs,
    'tool_calls'
  );
  if (rawToolCalls != null) {
    for (const toolCall of rawToolCalls) {
      const id = readOwnString(toolCall, 'id');
      addToolPairingId(ids, id);
    }
  }

  const content = getBoundedProviderPairingArray(message.content);
  if (content != null) {
    for (const part of content) {
      const descriptor = getProviderToolCallPartDescriptor(part);
      if (descriptor != null) {
        addToolPairingId(ids, descriptor.callId);
      }
    }
  }

  return ids;
}

function getToolResultIds(message: BaseMessage): Set<string> {
  const ids = new Set<string>();
  if (message.getType() === 'tool') {
    const toolMessage = message as ToolMessage & { toolCallId?: unknown };
    addToolPairingId(ids, toolMessage.tool_call_id);
    addToolPairingId(ids, toolMessage.toolCallId);
  }

  const content = getBoundedProviderPairingArray(message.content);
  if (content != null) {
    for (const part of content) {
      const descriptor = getProviderToolResultPartDescriptor(part);
      if (descriptor?.toolCallId != null) {
        addToolPairingId(ids, descriptor.toolCallId);
      }
    }
  }

  return ids;
}

function isToolResultOnlyMessage(message: BaseMessage): boolean {
  if (message.getType() === 'tool') {
    return true;
  }
  const content = getBoundedProviderPairingArray(message.content);
  if (content == null || content.length === 0) {
    return false;
  }
  for (const part of content) {
    if (getProviderToolResultPartDescriptor(part)?.toolCallId == null) {
      return false;
    }
  }
  return true;
}

function getCoverageSourceIds(message: BaseMessage): string[] {
  const sourceIds = getProviderSourceMessageIds(message);
  if (sourceIds.length > 0) {
    return sourceIds;
  }
  const messageId = message.id?.trim();
  return messageId != null && messageId !== '' ? [messageId] : [];
}

function findIntraTurnBoundary(
  messages: BaseMessage[],
  countMessageAt: (index: number) => number,
  retainTokens: number
): number | undefined {
  let remainingTokens = 0;
  for (let i = 0; i < messages.length; i++) {
    remainingTokens += countMessageAt(i);
  }
  if (remainingTokens <= retainTokens) {
    return undefined;
  }

  const sourceIdsByIndex = new Array<string[]>(messages.length);
  const lastIndexBySourceId = new Map<string, number>();
  for (let i = 0; i < messages.length; i++) {
    const sourceIds = getCoverageSourceIds(messages[i] as BaseMessage);
    sourceIdsByIndex[i] = sourceIds;
    for (const sourceId of sourceIds) {
      lastIndexBySourceId.set(sourceId, i);
    }
  }

  const pendingToolCallIds = new Set<string>();
  const straddlingSourceIds = new Set<string>();
  let completedToolCalls = 0;
  let boundary: number | undefined;

  for (let i = 0; i < messages.length - 1; i++) {
    const message = messages[i] as BaseMessage;
    remainingTokens -= countMessageAt(i);

    for (const callId of getToolCallIds(message)) {
      pendingToolCallIds.add(callId);
    }

    const toolResultIds = getToolResultIds(message);
    for (const toolCallId of toolResultIds) {
      if (pendingToolCallIds.delete(toolCallId)) {
        completedToolCalls += 1;
      }
    }

    for (const sourceId of sourceIdsByIndex[i] as string[]) {
      if (lastIndexBySourceId.get(sourceId) === i) {
        straddlingSourceIds.delete(sourceId);
      } else {
        straddlingSourceIds.add(sourceId);
      }
    }

    if (remainingTokens < retainTokens) {
      break;
    }
    if (
      completedToolCalls > 0 &&
      pendingToolCallIds.size === 0 &&
      straddlingSourceIds.size === 0 &&
      (message.getType() !== 'human' || isToolResultOnlyMessage(message))
    ) {
      boundary = i + 1;
    }
  }

  return boundary;
}

/**
 * Splits `messages` into a head (older, to summarize) and a tail (recent,
 * to preserve verbatim), preferring user-message boundaries. The most recent
 * user-led turn is normally included in the tail; additional older turns are
 * added subject to `turns` and `tokens` caps.
 *
 * When that policy exposes no compactable head, `intraTurnTokens` may select
 * a boundary after older closed tool-call/result units. This keeps runaway
 * first-turn tool loops compactable without splitting parallel calls from
 * their results. A user payload without a completed tool unit stays intact.
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
  const turnsCap = options.turns ?? DEFAULT_RETAIN_RECENT_TURNS;

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
    if (
      messages[i].getType() === 'human' &&
      !isToolResultOnlyMessage(messages[i] as BaseMessage)
    ) {
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
  const messageTokens: Array<number | undefined> = new Array(messages.length);
  const countMessageAt = (index: number): number => {
    const cached = messageTokens[index];
    if (cached != null) {
      return cached;
    }
    const count = tokenCounter?.(messages[index] as BaseMessage) ?? 0;
    messageTokens[index] = count;
    return count;
  };

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
      tailTokens += countMessageAt(i);
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
        turnTokens += countMessageAt(i);
      }
      if (tailTokens + turnTokens > (tokensCap as number)) {
        break;
      }
      tailTokens += turnTokens;
    }

    tailStartIndex = turnStart;
    tailTurnCount += 1;
  }

  if (tailStartIndex === turnStarts[0]) {
    const intraTurnTokens = options.intraTurnTokens;
    if (
      intraTurnTokens != null &&
      Number.isFinite(intraTurnTokens) &&
      intraTurnTokens > 0 &&
      tokenCounter != null
    ) {
      const intraTurnBoundary = findIntraTurnBoundary(
        messages,
        countMessageAt,
        intraTurnTokens
      );
      if (intraTurnBoundary != null) {
        return {
          head: messages.slice(0, intraTurnBoundary),
          tail: messages.slice(intraTurnBoundary),
          tailTurnCount,
          tailStartIndex: intraTurnBoundary,
        };
      }
    }
  }

  return {
    head: messages.slice(0, tailStartIndex),
    tail: messages.slice(tailStartIndex),
    tailTurnCount,
    tailStartIndex,
  };
}
