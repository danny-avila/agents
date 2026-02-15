/**
 * Position-based context pruning for tool results.
 *
 * Adapted from openclaw's context-pruning/pruner.ts. Since agentus doesn't have
 * per-message timestamps, we use position-based age: the distance of a message
 * from the conversation end as a fraction of total messages.
 *
 * Two degradation levels:
 * - Soft-trim: Keep head + tail of tool result content, drop middle.
 * - Hard-clear: Replace entire content with a placeholder.
 *
 * Messages in the "protected zone" (recent assistant turns, system/pre-first-human
 * messages, and messages with image content) are never pruned.
 */

import { ToolMessage, type BaseMessage } from '@langchain/core/messages';
import type { TokenCounter } from '@/types/run';
import type { ContextPruningConfig } from '@/types/graph';
import type { ContextPruningSettings } from './contextPruningSettings';
import { resolveContextPruningSettings } from './contextPruningSettings';

/**
 * Checks if a message contains image content blocks (should not be pruned).
 */
function hasImageContent(message: BaseMessage): boolean {
  if (!Array.isArray(message.content)) {
    return false;
  }
  return message.content.some(
    (block) =>
      typeof block === 'object' &&
      'type' in block &&
      (block.type === 'image_url' || block.type === 'image')
  );
}

/**
 * Applies head+tail soft-trim to tool result content.
 */
function softTrimContent(
  content: string,
  settings: ContextPruningSettings['softTrim']
): string {
  const { headChars, tailChars } = settings;
  const indicator = `\n\n… [soft-trimmed: ${content.length} chars → ${headChars + tailChars} chars, middle removed] …\n\n`;
  return content.slice(0, headChars) + indicator + content.slice(-tailChars);
}

export interface ContextPruningResult {
  /** Number of messages that were soft-trimmed. */
  softTrimmed: number;
  /** Number of messages that were hard-cleared. */
  hardCleared: number;
}

/**
 * Applies position-based context pruning to tool result messages.
 *
 * Modifies messages in-place and updates indexTokenCountMap with recounted
 * token values for modified messages.
 *
 * @param params.messages - The full message array (modified in-place).
 * @param params.indexTokenCountMap - Token count map (updated in-place).
 * @param params.tokenCounter - Function to recount tokens after modification.
 * @param params.config - Partial context pruning config (merged with defaults).
 * @returns Counts of soft-trimmed and hard-cleared messages.
 */
export function applyContextPruning(params: {
  messages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
  config?: ContextPruningConfig;
}): ContextPruningResult {
  const { messages, indexTokenCountMap, tokenCounter, config } = params;
  const settings = resolveContextPruningSettings(config);

  if (!settings.enabled || messages.length === 0) {
    return { softTrimmed: 0, hardCleared: 0 };
  }

  const totalMessages = messages.length;
  let softTrimmed = 0;
  let hardCleared = 0;

  // Find the protected zone: last N assistant turns from the end.
  // An "assistant turn" is a contiguous sequence of AI + Tool messages.
  const protectedIndices = new Set<number>();

  // Always protect the system message (index 0 if present)
  if (messages[0]?.getType() === 'system') {
    protectedIndices.add(0);
  }

  // Protect messages before the first human message
  for (let i = 0; i < totalMessages; i++) {
    if (messages[i].getType() === 'human') {
      break;
    }
    protectedIndices.add(i);
  }

  // Protect the last N assistant turns (walking backwards)
  let assistantTurnsFound = 0;
  let inAssistantSequence = false;
  for (let i = totalMessages - 1; i >= 0; i--) {
    const type = messages[i].getType();
    if (type === 'ai' || type === 'tool') {
      protectedIndices.add(i);
      if (!inAssistantSequence) {
        inAssistantSequence = true;
      }
    } else {
      if (inAssistantSequence) {
        assistantTurnsFound++;
        inAssistantSequence = false;
        if (assistantTurnsFound >= settings.keepLastAssistants) {
          break;
        }
      }
      // Protect the human message between assistant turns in the protected zone
      if (assistantTurnsFound < settings.keepLastAssistants) {
        protectedIndices.add(i);
      }
    }
  }

  // Process each tool message outside the protected zone
  for (let i = 0; i < totalMessages; i++) {
    const message = messages[i];
    if (message.getType() !== 'tool') {
      continue;
    }
    if (protectedIndices.has(i)) {
      continue;
    }
    if (hasImageContent(message)) {
      continue;
    }

    const content = message.content;
    if (typeof content !== 'string') {
      continue;
    }
    if (content.length < settings.minPrunableToolChars) {
      continue;
    }

    // Compute age ratio: how far back from the end (0 = latest, 1 = oldest)
    const ageRatio = (totalMessages - i) / totalMessages;

    if (ageRatio >= settings.hardClearRatio && settings.hardClear.enabled) {
      // Hard-clear: replace with placeholder
      (message as ToolMessage).content = settings.hardClear.placeholder;
      indexTokenCountMap[i] = tokenCounter(message);
      hardCleared++;
    } else if (ageRatio >= settings.softTrimRatio) {
      // Soft-trim: keep head + tail
      if (content.length > settings.softTrim.maxChars) {
        (message as ToolMessage).content = softTrimContent(
          content,
          settings.softTrim
        );
        indexTokenCountMap[i] = tokenCounter(message);
        softTrimmed++;
      }
    }
  }

  return { softTrimmed, hardCleared };
}
