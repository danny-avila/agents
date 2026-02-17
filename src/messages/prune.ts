import {
  AIMessage,
  BaseMessage,
  ToolMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type {
  ThinkingContentText,
  MessageContentComplex,
  ReasoningContentText,
} from '@/types/stream';
import type { TokenCounter } from '@/types/run';
import type { ContextPruningConfig } from '@/types/graph';
import {
  calculateMaxToolResultChars,
  truncateToolResultContent,
  truncateToolInput,
} from '@/utils/truncation';
import { applyContextPruning } from './contextPruning';
import { ContentTypes, Providers } from '@/common';

export type PruneMessagesFactoryParams = {
  provider?: Providers;
  maxTokens: number;
  startIndex: number;
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
  thinkingEnabled?: boolean;
  /** Context pruning configuration for position-based tool result degradation. */
  contextPruningConfig?: ContextPruningConfig;
  /**
   * Returns the current instruction-token overhead (system message + tool schemas + summary).
   * Called on each prune invocation so the budget reflects dynamic changes
   * (e.g. summary added between turns).  When messages don't include a leading
   * SystemMessage, these tokens are subtracted from the available budget so
   * the pruner correctly reserves space for the system prompt that will be
   * prepended later by `buildSystemRunnable`.
   */
  getInstructionTokens?: () => number;
  /**
   * Fraction of the effective token budget to reserve as headroom (0–1).
   * When set, pruning triggers at `effectiveMax * (1 - reserveRatio)` instead of
   * filling the context window to 100%.  Defaults to 0 (no reserve) when omitted.
   */
  reserveRatio?: number;
  /** Optional diagnostic log callback wired by the graph for observability. */
  log?: (
    level: 'debug' | 'info' | 'warn' | 'error',
    message: string,
    data?: Record<string, unknown>
  ) => void;
};
export type PruneMessagesParams = {
  messages: BaseMessage[];
  usageMetadata?: Partial<UsageMetadata>;
  startType?: ReturnType<BaseMessage['getType']>;
  /**
   * Usage from the most recent LLM call only (not accumulated).
   * When provided, calibration uses this instead of usageMetadata
   * to avoid inflated ratios from N×cacheRead accumulation.
   */
  lastCallUsage?: {
    totalTokens: number;
  };
  /**
   * Whether the token data is fresh (from a just-completed LLM call).
   * When false, provider calibration is skipped to avoid applying
   * stale ratios.
   */
  totalTokensFresh?: boolean;
};

function getToolCallIds(message: BaseMessage): Set<string> {
  if (message.getType() !== 'ai') {
    return new Set<string>();
  }

  const ids = new Set<string>();
  const aiMessage = message as AIMessage;
  for (const toolCall of aiMessage.tool_calls ?? []) {
    if (typeof toolCall.id === 'string' && toolCall.id.length > 0) {
      ids.add(toolCall.id);
    }
  }

  if (Array.isArray(aiMessage.content)) {
    for (const part of aiMessage.content) {
      if (typeof part !== 'object') {
        continue;
      }
      const record = part as { type?: unknown; id?: unknown };
      if (
        (record.type === 'tool_use' || record.type === 'tool_call') &&
        typeof record.id === 'string' &&
        record.id.length > 0
      ) {
        ids.add(record.id);
      }
    }
  }

  return ids;
}

function getToolResultId(message: BaseMessage): string | null {
  if (message.getType() !== 'tool') {
    return null;
  }
  const toolMessage = message as ToolMessage & {
    tool_call_id?: unknown;
    toolCallId?: unknown;
  };
  if (
    typeof toolMessage.tool_call_id === 'string' &&
    toolMessage.tool_call_id.length > 0
  ) {
    return toolMessage.tool_call_id;
  }
  if (
    typeof toolMessage.toolCallId === 'string' &&
    toolMessage.toolCallId.length > 0
  ) {
    return toolMessage.toolCallId;
  }
  return null;
}

function resolveTokenCountForMessage({
  message,
  allMessages,
  tokenCounter,
  indexTokenCountMap,
}: {
  message: BaseMessage;
  allMessages: BaseMessage[];
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
}): number {
  const originalIndex = allMessages.findIndex(
    (candidateMessage) => candidateMessage === message
  );
  if (originalIndex > -1 && indexTokenCountMap[originalIndex] != null) {
    return indexTokenCountMap[originalIndex] as number;
  }
  return tokenCounter(message);
}

export function repairOrphanedToolMessages({
  context,
  allMessages,
  tokenCounter,
  indexTokenCountMap,
}: {
  context: BaseMessage[];
  allMessages: BaseMessage[];
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
}): {
  context: BaseMessage[];
  reclaimedTokens: number;
  droppedOrphanCount: number;
  /** Messages removed from context during orphan repair.  These should be
   *  appended to `messagesToRefine` so that summarization can still see them
   *  (e.g. a ToolMessage whose parent AI was pruned). */
  droppedMessages: BaseMessage[];
} {
  // Collect all tool_call IDs from AI messages in context
  const validToolCallIds = new Set<string>();
  for (const message of context) {
    for (const id of getToolCallIds(message)) {
      validToolCallIds.add(id);
    }
  }

  // Collect all tool_result IDs from ToolMessages in context
  const presentToolResultIds = new Set<string>();
  for (const message of context) {
    const resultId = getToolResultId(message);
    if (resultId != null) {
      presentToolResultIds.add(resultId);
    }
  }

  let reclaimedTokens = 0;
  let droppedOrphanCount = 0;
  const repairedContext: BaseMessage[] = [];
  const droppedMessages: BaseMessage[] = [];

  for (const message of context) {
    // Drop orphan ToolMessages whose AI tool_call is not in context.
    // This covers two cases: (a) AI messages exist but don't claim this tool_call_id,
    // (b) no AI messages with tool_calls survived pruning at all.
    if (message.getType() === 'tool') {
      const toolResultId = getToolResultId(message);
      if (toolResultId == null || !validToolCallIds.has(toolResultId)) {
        droppedOrphanCount += 1;
        reclaimedTokens += resolveTokenCountForMessage({
          message,
          allMessages,
          tokenCounter,
          indexTokenCountMap,
        });
        droppedMessages.push(message);
        continue;
      }
      repairedContext.push(message);
      continue;
    }

    // Strip orphan tool_use blocks from AI messages whose ToolMessages are not in context.
    // Providers like Anthropic/Bedrock require every tool_use to have a matching tool_result.
    if (message.getType() === 'ai' && message instanceof AIMessage) {
      const toolCallIds = getToolCallIds(message);
      if (toolCallIds.size > 0) {
        const hasOrphanToolCalls = Array.from(toolCallIds).some(
          (id) => !presentToolResultIds.has(id)
        );
        if (hasOrphanToolCalls) {
          const originalTokens = resolveTokenCountForMessage({
            message,
            allMessages,
            tokenCounter,
            indexTokenCountMap,
          });
          const stripped = stripOrphanToolUseBlocks(
            message,
            presentToolResultIds
          );
          if (stripped != null) {
            const strippedTokens = tokenCounter(stripped);
            reclaimedTokens += originalTokens - strippedTokens;
            repairedContext.push(stripped);
          } else {
            // AI message had only tool_use blocks, nothing left after stripping
            droppedOrphanCount += 1;
            reclaimedTokens += originalTokens;
            droppedMessages.push(message);
          }
          continue;
        }
      }
    }

    repairedContext.push(message);
  }

  return {
    context: repairedContext,
    reclaimedTokens,
    droppedOrphanCount,
    droppedMessages,
  };
}

/**
 * Strips tool_use content blocks and tool_calls entries from an AI message
 * when their corresponding ToolMessages are not in the context.
 * Returns null if the message has no content left after stripping.
 */
function stripOrphanToolUseBlocks(
  message: AIMessage,
  presentToolResultIds: Set<string>
): AIMessage | null {
  // Strip tool_calls array entries
  const keptToolCalls = (message.tool_calls ?? []).filter(
    (tc) => typeof tc.id === 'string' && presentToolResultIds.has(tc.id)
  );

  // Strip content blocks
  let keptContent: MessageContentComplex[] | string;
  if (Array.isArray(message.content)) {
    const filtered = (message.content as MessageContentComplex[]).filter(
      (block) => {
        if (typeof block !== 'object') {
          return true;
        }
        const record = block as { type?: unknown; id?: unknown };
        if (
          (record.type === 'tool_use' || record.type === 'tool_call') &&
          typeof record.id === 'string'
        ) {
          return presentToolResultIds.has(record.id);
        }
        return true;
      }
    );

    // If nothing left, return null
    if (filtered.length === 0) {
      return null;
    }
    keptContent = filtered;
  } else {
    keptContent = message.content;
  }

  return new AIMessage({
    ...message,
    content: keptContent,
    tool_calls: keptToolCalls.length > 0 ? keptToolCalls : undefined,
  });
}

/**
 * Lightweight structural cleanup: strips orphan tool_use blocks from AI messages
 * and drops orphan ToolMessages whose AI counterpart is missing.
 *
 * Unlike `repairOrphanedToolMessages`, this does NOT track tokens — it is
 * intended as a final safety net in Graph.ts right before model invocation
 * to prevent Anthropic/Bedrock structural validation errors.
 *
 * Uses duck-typing instead of `getType()` because messages at this stage
 * may be plain objects (from LangGraph state serialization) rather than
 * proper BaseMessage class instances.
 *
 * Includes a fast-path: if every tool_call has a matching tool_result and
 * vice-versa, the original array is returned immediately with zero allocation.
 */
export function sanitizeOrphanToolBlocks(
  messages: BaseMessage[]
): BaseMessage[] {
  const allToolCallIds = new Set<string>();
  const allToolResultIds = new Set<string>();

  for (const msg of messages) {
    const msgAny = msg as unknown as Record<string, unknown>;
    const toolCalls = msgAny.tool_calls as Array<{ id?: string }> | undefined;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        if (typeof tc.id === 'string' && tc.id.length > 0) {
          allToolCallIds.add(tc.id);
        }
      }
    }
    if (Array.isArray(msgAny.content)) {
      for (const block of msgAny.content as Array<Record<string, unknown>>) {
        if (
          typeof block === 'object' &&
          (block.type === 'tool_use' || block.type === 'tool_call') &&
          typeof block.id === 'string'
        ) {
          allToolCallIds.add(block.id);
        }
      }
    }
    const toolCallId = msgAny.tool_call_id as string | undefined;
    if (typeof toolCallId === 'string' && toolCallId.length > 0) {
      allToolResultIds.add(toolCallId);
    }
  }

  // Fast-path: if every tool_call has a result and every result has a call,
  // there are no orphans — return the original array immediately.
  let hasOrphans = false;
  for (const id of allToolCallIds) {
    if (!allToolResultIds.has(id)) {
      hasOrphans = true;
      break;
    }
  }
  if (!hasOrphans) {
    for (const id of allToolResultIds) {
      if (!allToolCallIds.has(id)) {
        hasOrphans = true;
        break;
      }
    }
  }
  if (!hasOrphans) {
    return messages;
  }

  const result: BaseMessage[] = [];
  // Track indices of AI messages that were modified (tool_use stripped).
  const strippedAiIndices = new Set<number>();

  for (const msg of messages) {
    const msgAny = msg as unknown as Record<string, unknown>;
    const msgType =
      typeof (msg as { getType?: unknown }).getType === 'function'
        ? msg.getType()
        : ((msgAny.role as string | undefined) ??
          (msgAny._type as string | undefined));

    // Drop orphan ToolMessages whose AI tool_call is missing.
    const toolCallId = msgAny.tool_call_id as string | undefined;
    if (
      (msgType === 'tool' || msg instanceof ToolMessage) &&
      typeof toolCallId === 'string' &&
      !allToolCallIds.has(toolCallId)
    ) {
      continue;
    }

    // Strip orphan tool_use blocks from AI messages.
    const toolCalls = msgAny.tool_calls as Array<{ id?: string }> | undefined;
    if (
      (msgType === 'ai' ||
        msgType === 'assistant' ||
        msg instanceof AIMessage) &&
      Array.isArray(toolCalls) &&
      toolCalls.length > 0
    ) {
      const hasOrphanCalls = toolCalls.some(
        (tc) => typeof tc.id === 'string' && !allToolResultIds.has(tc.id)
      );
      if (hasOrphanCalls) {
        if (msg instanceof AIMessage) {
          const stripped = stripOrphanToolUseBlocks(msg, allToolResultIds);
          if (stripped != null) {
            strippedAiIndices.add(result.length);
            result.push(stripped);
          }
          continue;
        }
        // For plain objects, filter tool_calls and content in-place.
        const keptToolCalls = toolCalls.filter(
          (tc) => typeof tc.id === 'string' && allToolResultIds.has(tc.id)
        );
        const keptContent = Array.isArray(msgAny.content)
          ? (msgAny.content as Array<Record<string, unknown>>).filter(
            (block) => {
              if (typeof block !== 'object') return true;
              if (
                (block.type === 'tool_use' || block.type === 'tool_call') &&
                  typeof block.id === 'string'
              ) {
                return allToolResultIds.has(block.id);
              }
              return true;
            }
          )
          : msgAny.content;
        if (
          keptToolCalls.length === 0 &&
          Array.isArray(keptContent) &&
          keptContent.length === 0
        ) {
          continue;
        }
        strippedAiIndices.add(result.length);
        msgAny.tool_calls = keptToolCalls.length > 0 ? keptToolCalls : [];
        msgAny.content = keptContent;
        result.push(msg);
        continue;
      }
    }

    result.push(msg);
  }

  // If the conversation now ends with a stripped AI message (one whose
  // tool_use was just removed — an incomplete tool call), drop it.
  // Bedrock/Anthropic require the conversation to end with a user message,
  // and a stripped AI message represents a dead-end exchange.
  while (result.length > 0 && strippedAiIndices.has(result.length - 1)) {
    result.pop();
  }

  return result;
}

/**
 * Truncates an oversized tool_use `input` field using head+tail, preserving
 * it as a valid JSON object. Head gets ~70%, tail gets ~30% so the model
 * sees both the beginning (what was called) and end (closing structure/values).
 * Falls back to head-only when the budget is too small for a meaningful tail.
 */
function isIndexInContext(
  arrayA: unknown[],
  arrayB: unknown[],
  targetIndex: number
): boolean {
  const startingIndexInA = arrayA.length - arrayB.length;
  return targetIndex >= startingIndexInA;
}

function addThinkingBlock(
  message: AIMessage,
  thinkingBlock: ThinkingContentText | ReasoningContentText
): AIMessage {
  const content: MessageContentComplex[] = Array.isArray(message.content)
    ? (message.content as MessageContentComplex[])
    : [
      {
        type: ContentTypes.TEXT,
        text: message.content,
      },
    ];
  /** Edge case, the message already has the thinking block */
  if (content[0].type === thinkingBlock.type) {
    return message;
  }
  content.unshift(thinkingBlock);
  return new AIMessage({
    ...message,
    content,
  });
}

/**
 * Calculates the total tokens from a single usage object
 *
 * @param usage The usage metadata object containing token information
 * @returns An object containing the total input and output tokens
 */
export function calculateTotalTokens(
  usage: Partial<UsageMetadata>
): UsageMetadata {
  const baseInputTokens = Number(usage.input_tokens) || 0;
  const cacheCreation = Number(usage.input_token_details?.cache_creation) || 0;
  const cacheRead = Number(usage.input_token_details?.cache_read) || 0;

  const totalInputTokens = baseInputTokens + cacheCreation + cacheRead;
  const totalOutputTokens = Number(usage.output_tokens) || 0;

  return {
    input_tokens: totalInputTokens,
    output_tokens: totalOutputTokens,
    total_tokens: totalInputTokens + totalOutputTokens,
  };
}

export type PruningResult = {
  context: BaseMessage[];
  remainingContextTokens: number;
  messagesToRefine: BaseMessage[];
  thinkingStartIndex?: number;
};

/**
 * Processes an array of messages and returns a context of messages that fit within a specified token limit.
 * It iterates over the messages from newest to oldest, adding them to the context until the token limit is reached.
 *
 * @param options Configuration options for processing messages
 * @returns Object containing the message context, remaining tokens, messages not included, and summary index
 */
export function getMessagesWithinTokenLimit({
  messages: _messages,
  maxContextTokens,
  indexTokenCountMap,
  startType: _startType,
  thinkingEnabled,
  tokenCounter,
  thinkingStartIndex: _thinkingStartIndex = -1,
  reasoningType = ContentTypes.THINKING,
  instructionTokens: _instructionTokens = 0,
}: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  startType?: string | string[];
  thinkingEnabled?: boolean;
  tokenCounter: TokenCounter;
  thinkingStartIndex?: number;
  reasoningType?: ContentTypes.THINKING | ContentTypes.REASONING_CONTENT;
  /**
   * Token overhead for instructions (system message + tool schemas + summary)
   * that are NOT included in `messages`.  When messages[0] is already a
   * SystemMessage the budget is deducted from its indexTokenCountMap entry
   * as before; otherwise this value is subtracted from the available budget.
   */
  instructionTokens?: number;
}): PruningResult {
  // Every reply is primed with <|start|>assistant<|message|>, so we
  // start with 3 tokens for the label after all messages have been counted.
  let currentTokenCount = 3;
  const instructions =
    _messages[0]?.getType() === 'system' ? _messages[0] : undefined;
  // When messages include a system message at index 0, use its token count
  // from the map.  Otherwise fall back to the explicit instructionTokens
  // parameter (system prompt prepended later by buildSystemRunnable).
  const instructionsTokenCount =
    instructions != null ? (indexTokenCountMap[0] ?? 0) : _instructionTokens;
  const initialContextTokens = maxContextTokens - instructionsTokenCount;
  let remainingContextTokens = initialContextTokens;
  let startType = _startType;
  const originalLength = _messages.length;
  const messages = [..._messages];
  /**
   * IMPORTANT: this context array gets reversed at the end, since the latest messages get pushed first.
   *
   * This may be confusing to read, but it is done to ensure the context is in the correct order for the model.
   * */
  let context: Array<BaseMessage | undefined> = [];

  let thinkingStartIndex = _thinkingStartIndex;
  let thinkingEndIndex = -1;
  let thinkingBlock: ThinkingContentText | ReasoningContentText | undefined;
  const endIndex = instructions != null ? 1 : 0;
  const prunedMemory: BaseMessage[] = [];

  if (_thinkingStartIndex > -1) {
    const thinkingMessageContent = messages[_thinkingStartIndex]?.content;
    if (Array.isArray(thinkingMessageContent)) {
      thinkingBlock = thinkingMessageContent.find(
        (content) => content.type === reasoningType
      ) as ThinkingContentText | undefined;
    }
  }

  if (currentTokenCount < remainingContextTokens) {
    let currentIndex = messages.length;
    while (
      messages.length > 0 &&
      currentTokenCount < remainingContextTokens &&
      currentIndex > endIndex
    ) {
      currentIndex--;
      if (messages.length === 1 && instructions) {
        break;
      }
      const poppedMessage = messages.pop();
      if (!poppedMessage) continue;
      const messageType = poppedMessage.getType();
      if (
        thinkingEnabled === true &&
        thinkingEndIndex === -1 &&
        currentIndex === originalLength - 1 &&
        (messageType === 'ai' || messageType === 'tool')
      ) {
        thinkingEndIndex = currentIndex;
      }
      if (
        thinkingEndIndex > -1 &&
        !thinkingBlock &&
        thinkingStartIndex < 0 &&
        messageType === 'ai' &&
        Array.isArray(poppedMessage.content)
      ) {
        thinkingBlock = poppedMessage.content.find(
          (content) => content.type === reasoningType
        ) as ThinkingContentText | undefined;
        thinkingStartIndex = thinkingBlock != null ? currentIndex : -1;
      }
      /** False start, the latest message was not part of a multi-assistant/tool sequence of messages */
      if (
        thinkingEndIndex > -1 &&
        currentIndex === thinkingEndIndex - 1 &&
        messageType !== 'ai' &&
        messageType !== 'tool'
      ) {
        thinkingEndIndex = -1;
      }

      const tokenCount = indexTokenCountMap[currentIndex] ?? 0;

      if (
        prunedMemory.length === 0 &&
        currentTokenCount + tokenCount <= remainingContextTokens
      ) {
        context.push(poppedMessage);
        currentTokenCount += tokenCount;
      } else {
        prunedMemory.push(poppedMessage);
        if (thinkingEndIndex > -1 && thinkingStartIndex < 0) {
          continue;
        }
        break;
      }
    }

    if (context[context.length - 1]?.getType() === 'tool') {
      startType = ['ai', 'human'];
    }

    if (startType != null && startType.length > 0 && context.length > 0) {
      let requiredTypeIndex = -1;

      let totalTokens = 0;
      for (let i = context.length - 1; i >= 0; i--) {
        const currentType = context[i]?.getType() ?? '';
        if (
          Array.isArray(startType)
            ? startType.includes(currentType)
            : currentType === startType
        ) {
          requiredTypeIndex = i + 1;
          break;
        }
        const originalIndex = originalLength - 1 - i;
        totalTokens += indexTokenCountMap[originalIndex] ?? 0;
      }

      if (requiredTypeIndex > 0) {
        currentTokenCount -= totalTokens;
        context = context.slice(0, requiredTypeIndex);
      }
    }
  }

  if (instructions && originalLength > 0) {
    context.push(_messages[0] as BaseMessage);
    messages.shift();
  }

  // The backward iteration pushed messages in reverse chronological order
  // (newest first).  Restore correct chronological order before prepending
  // the remaining (older) messages so that messagesToRefine is always
  // ordered oldest → newest.  Without this, callers that rely on
  // messagesToRefine order (e.g. the summarization node extracting the
  // latest turn) would see tool_use/tool_result pairs in the wrong order.
  prunedMemory.reverse();

  if (messages.length > 0) {
    prunedMemory.unshift(...messages);
  }

  remainingContextTokens -= currentTokenCount;
  const result: PruningResult = {
    remainingContextTokens,
    context: [] as BaseMessage[],
    messagesToRefine: prunedMemory,
  };

  if (thinkingStartIndex > -1) {
    result.thinkingStartIndex = thinkingStartIndex;
  }

  if (
    prunedMemory.length === 0 ||
    thinkingEndIndex < 0 ||
    (thinkingStartIndex > -1 &&
      isIndexInContext(_messages, context, thinkingStartIndex))
  ) {
    // we reverse at this step to ensure the context is in the correct order for the model, and we need to work backwards
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  if (thinkingEndIndex > -1 && thinkingStartIndex < 0) {
    throw new Error(
      'The payload is malformed. There is a thinking sequence but no "AI" messages with thinking blocks.'
    );
  }

  if (!thinkingBlock) {
    throw new Error(
      'The payload is malformed. There is a thinking sequence but no thinking block found.'
    );
  }

  // Since we have a thinking sequence, we need to find the last assistant message
  // in the latest AI/tool sequence to add the thinking block that falls outside of the current context
  // Latest messages are ordered first.
  let assistantIndex = -1;
  for (let i = 0; i < context.length; i++) {
    const currentMessage = context[i];
    const type = currentMessage?.getType();
    if (type === 'ai') {
      assistantIndex = i;
    }
    if (assistantIndex > -1 && (type === 'human' || type === 'system')) {
      break;
    }
  }

  if (assistantIndex === -1) {
    // No AI messages survived pruning — skip thinking block reattachment.
    // The caller handles empty/insufficient context via overflow recovery.
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  thinkingStartIndex = originalLength - 1 - assistantIndex;
  const thinkingTokenCount = tokenCounter(
    new AIMessage({ content: [thinkingBlock] })
  );
  const newRemainingCount = remainingContextTokens - thinkingTokenCount;
  const newMessage = addThinkingBlock(
    context[assistantIndex] as AIMessage,
    thinkingBlock
  );
  context[assistantIndex] = newMessage;
  if (newRemainingCount > 0) {
    result.context = context.reverse() as BaseMessage[];
    return result;
  }

  const thinkingMessage: AIMessage = context[assistantIndex] as AIMessage;
  // now we need to an additional round of pruning but making the thinking block fit
  const newThinkingMessageTokenCount =
    (indexTokenCountMap[thinkingStartIndex] ?? 0) + thinkingTokenCount;
  remainingContextTokens = initialContextTokens - newThinkingMessageTokenCount;
  currentTokenCount = 3;
  let newContext: BaseMessage[] = [];
  const secondRoundMessages = [..._messages];
  let currentIndex = secondRoundMessages.length;
  while (
    secondRoundMessages.length > 0 &&
    currentTokenCount < remainingContextTokens &&
    currentIndex > thinkingStartIndex
  ) {
    currentIndex--;
    const poppedMessage = secondRoundMessages.pop();
    if (!poppedMessage) continue;
    const tokenCount = indexTokenCountMap[currentIndex] ?? 0;
    if (currentTokenCount + tokenCount <= remainingContextTokens) {
      newContext.push(poppedMessage);
      currentTokenCount += tokenCount;
    } else {
      messages.push(poppedMessage);
      break;
    }
  }

  const firstMessage: AIMessage = newContext[newContext.length - 1];
  const firstMessageType = newContext[newContext.length - 1].getType();
  if (firstMessageType === 'tool') {
    startType = ['ai', 'human'];
  }

  if (startType != null && startType.length > 0 && newContext.length > 0) {
    let requiredTypeIndex = -1;

    let totalTokens = 0;
    for (let i = newContext.length - 1; i >= 0; i--) {
      const currentType = newContext[i]?.getType() ?? '';
      if (
        Array.isArray(startType)
          ? startType.includes(currentType)
          : currentType === startType
      ) {
        requiredTypeIndex = i + 1;
        break;
      }
      const originalIndex = originalLength - 1 - i;
      totalTokens += indexTokenCountMap[originalIndex] ?? 0;
    }

    if (requiredTypeIndex > 0) {
      currentTokenCount -= totalTokens;
      newContext = newContext.slice(0, requiredTypeIndex);
    }
  }

  if (firstMessageType === 'ai') {
    const newMessage = addThinkingBlock(firstMessage, thinkingBlock);
    newContext[newContext.length - 1] = newMessage;
  } else {
    newContext.push(thinkingMessage);
  }

  if (instructions && originalLength > 0) {
    newContext.push(_messages[0] as BaseMessage);
    secondRoundMessages.shift();
  }

  result.context = newContext.reverse();
  return result;
}

export function checkValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && value > 0;
}

/**
 * Pre-flight truncation: truncates oversized ToolMessage content before the
 * main backward-iteration pruning runs. Unlike the ingestion guard (which caps
 * at tool-execution time), pre-flight truncation applies per-turn based on the
 * current context window budget (which may have shrunk due to growing conversation).
 *
 * After truncation, recounts tokens via tokenCounter and updates indexTokenCountMap
 * so subsequent pruning works with accurate counts.
 *
 * @returns The number of tool messages that were truncated.
 */
export function preFlightTruncateToolResults(params: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
}): number {
  const { messages, maxContextTokens, indexTokenCountMap, tokenCounter } =
    params;
  const maxChars = calculateMaxToolResultChars(maxContextTokens);
  let truncatedCount = 0;

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.getType() !== 'tool') {
      continue;
    }
    const content = message.content;
    if (typeof content !== 'string' || content.length <= maxChars) {
      continue;
    }

    const truncated = truncateToolResultContent(content, maxChars);
    // Mutate the message content in place (ToolMessage.content is writable)
    (message as ToolMessage).content = truncated;
    // Recount tokens for the modified message
    indexTokenCountMap[i] = tokenCounter(message);
    truncatedCount++;
  }

  return truncatedCount;
}

/**
 * Pre-flight truncation: truncates oversized `tool_use` input fields in AI messages.
 *
 * Tool call inputs (arguments) can be very large — e.g., code evaluation payloads from
 * MCP tools like chrome-devtools. Since these tool calls have already been executed,
 * the model only needs a summary of what was called, not the full arguments. Truncating
 * them before pruning can prevent entire messages from being dropped.
 *
 * Uses 15% of the context window (in estimated characters, ~4 chars/token) as the
 * per-input cap, capped at 200K chars.
 *
 * @returns The number of AI messages that had tool_use inputs truncated.
 */
export function preFlightTruncateToolCallInputs(params: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
}): number {
  const { messages, maxContextTokens, indexTokenCountMap, tokenCounter } =
    params;
  const maxInputChars = Math.min(
    Math.floor(maxContextTokens * 0.15) * 4,
    200_000
  );
  let truncatedCount = 0;

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.getType() !== 'ai') {
      continue;
    }
    if (!Array.isArray(message.content)) {
      continue;
    }

    const originalContent = message.content as MessageContentComplex[];
    const newContent = originalContent.map((block) => {
      if (typeof block !== 'object') {
        return block;
      }
      const record = block as Record<string, unknown>;
      if (record.type !== 'tool_use' && record.type !== 'tool_call') {
        return block;
      }

      const input = record.input;
      if (input == null) {
        return block;
      }
      const serialized =
        typeof input === 'string' ? input : JSON.stringify(input);
      if (serialized.length <= maxInputChars) {
        return block;
      }

      return {
        ...record,
        input: truncateToolInput(input, maxInputChars),
      };
    });

    // Check if any blocks were actually replaced
    if (newContent.every((block, idx) => block === originalContent[idx])) {
      continue;
    }

    // Also truncate the matching tool_calls[].args entries
    const aiMsg = message as AIMessage;
    const newToolCalls = (aiMsg.tool_calls ?? []).map((tc) => {
      const serializedArgs = JSON.stringify(tc.args);
      if (serializedArgs.length <= maxInputChars) {
        return tc;
      }
      return {
        ...tc,
        args: truncateToolInput(tc.args, maxInputChars),
      };
    });

    messages[i] = new AIMessage({
      ...aiMsg,
      content: newContent,
      tool_calls: newToolCalls.length > 0 ? newToolCalls : undefined,
    });
    indexTokenCountMap[i] = tokenCounter(messages[i]);
    truncatedCount++;
  }

  return truncatedCount;
}

type ThinkingBlocks = {
  thinking_blocks?: Array<{
    type: 'thinking';
    thinking: string;
    signature: string;
  }>;
};

export function createPruneMessages(factoryParams: PruneMessagesFactoryParams) {
  const indexTokenCountMap = { ...factoryParams.indexTokenCountMap };
  let lastTurnStartIndex = factoryParams.startIndex;
  let lastCutOffIndex = 0;
  let totalTokens = Object.values(indexTokenCountMap).reduce(
    (a = 0, b = 0) => a + b,
    0
  ) as number;
  let runThinkingStartIndex = -1;
  return function pruneMessages(params: PruneMessagesParams): {
    context: BaseMessage[];
    indexTokenCountMap: Record<string, number | undefined>;
    messagesToRefine?: BaseMessage[];
    prePruneTotalTokens?: number;
    remainingContextTokens?: number;
  } {
    // Guard: empty messages array (e.g. after REMOVE_ALL with empty context).
    // Nothing to prune — return immediately to avoid out-of-bounds access.
    if (params.messages.length === 0) {
      return {
        context: [],
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneTotalTokens: 0,
        remainingContextTokens: factoryParams.maxTokens,
      };
    }

    if (
      factoryParams.provider === Providers.OPENAI &&
      factoryParams.thinkingEnabled === true
    ) {
      for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
        const m = params.messages[i];
        if (
          m.getType() === 'ai' &&
          typeof m.additional_kwargs.reasoning_content === 'string' &&
          Array.isArray(
            (
              m.additional_kwargs.provider_specific_fields as
                | ThinkingBlocks
                | undefined
            )?.thinking_blocks
          ) &&
          (m as AIMessage).tool_calls &&
          ((m as AIMessage).tool_calls?.length ?? 0) > 0
        ) {
          const message = m as AIMessage;
          const thinkingBlocks = (
            message.additional_kwargs.provider_specific_fields as ThinkingBlocks
          ).thinking_blocks;
          const signature =
            thinkingBlocks?.[thinkingBlocks.length - 1].signature;
          const thinkingBlock: ThinkingContentText = {
            signature,
            type: ContentTypes.THINKING,
            thinking: message.additional_kwargs.reasoning_content as string,
          };

          params.messages[i] = new AIMessage({
            ...message,
            content: [thinkingBlock],
            additional_kwargs: {
              ...message.additional_kwargs,
              reasoning_content: undefined,
            },
          });
        }
      }
    }

    let currentUsage: UsageMetadata | undefined;
    if (
      params.usageMetadata &&
      (checkValidNumber(params.usageMetadata.input_tokens) ||
        (checkValidNumber(params.usageMetadata.input_token_details) &&
          (checkValidNumber(
            params.usageMetadata.input_token_details.cache_creation
          ) ||
            checkValidNumber(
              params.usageMetadata.input_token_details.cache_read
            )))) &&
      checkValidNumber(params.usageMetadata.output_tokens)
    ) {
      currentUsage = calculateTotalTokens(params.usageMetadata);
      totalTokens = currentUsage.total_tokens;
    }

    const newOutputs = new Set<number>();
    for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
      const message = params.messages[i];
      if (
        i === lastTurnStartIndex &&
        indexTokenCountMap[i] === undefined &&
        currentUsage
      ) {
        indexTokenCountMap[i] = currentUsage.output_tokens;
      } else if (indexTokenCountMap[i] === undefined) {
        indexTokenCountMap[i] = factoryParams.tokenCounter(message);
        if (currentUsage) {
          newOutputs.add(i);
        }
        totalTokens += indexTokenCountMap[i] ?? 0;
      }
    }

    // Distribute the current total tokens to our `indexTokenCountMap` in a weighted manner,
    // so that the total token count aligns with provider-reported usage.
    // Uses lastCallUsage (single-call) when available to avoid inflated ratios from
    // accumulated cacheRead across N tool-call round-trips.
    // Gated on totalTokensFresh to avoid applying stale ratios.
    if (currentUsage && params.totalTokensFresh !== false) {
      // Prefer lastCallUsage (single-call, not accumulated) for calibration.
      const calibrationTokens =
        params.lastCallUsage?.totalTokens ?? currentUsage.total_tokens;

      let totalIndexTokens = 0;
      const firstIsSystem =
        params.messages.length > 0 && params.messages[0].getType() === 'system';
      if (firstIsSystem) {
        totalIndexTokens += indexTokenCountMap[0] ?? 0;
      }
      for (let i = lastCutOffIndex; i < params.messages.length; i++) {
        if (i === 0 && firstIsSystem) {
          continue;
        }
        if (newOutputs.has(i)) {
          continue;
        }
        totalIndexTokens += indexTokenCountMap[i] ?? 0;
      }

      // Calculate ratio based only on messages that remain in the context
      const ratio = calibrationTokens / totalIndexTokens;
      const isRatioSafe = ratio >= 1 / 3 && ratio <= 2.5;

      // Apply the ratio adjustment only to messages at or after lastCutOffIndex, and only if the ratio is safe
      if (isRatioSafe) {
        // Snapshot the map before calibration for sanity revert
        const preCalibrationSnapshot: Record<string, number | undefined> = {};
        for (let i = lastCutOffIndex; i < params.messages.length; i++) {
          preCalibrationSnapshot[i] = indexTokenCountMap[i];
        }
        if (firstIsSystem) {
          preCalibrationSnapshot[0] = indexTokenCountMap[0];
        }

        if (firstIsSystem && lastCutOffIndex !== 0) {
          indexTokenCountMap[0] = Math.round(
            (indexTokenCountMap[0] ?? 0) * ratio
          );
        }

        for (let i = lastCutOffIndex; i < params.messages.length; i++) {
          if (newOutputs.has(i)) {
            continue;
          }
          indexTokenCountMap[i] = Math.round(
            (indexTokenCountMap[i] ?? 0) * ratio
          );
        }

        // Post-calibration sanity check: verify calibrated sum is within 0.25×–3× of raw sum.
        // If not, revert to pre-calibration values to prevent wildly incorrect pruning.
        let calibratedSum = 0;
        for (let i = lastCutOffIndex; i < params.messages.length; i++) {
          if (!newOutputs.has(i)) {
            calibratedSum += indexTokenCountMap[i] ?? 0;
          }
        }
        const sanityRatio =
          totalIndexTokens > 0 ? calibratedSum / totalIndexTokens : 1;
        if (sanityRatio < 0.25 || sanityRatio > 3) {
          // Revert — calibration produced unreasonable values
          for (const [key, value] of Object.entries(preCalibrationSnapshot)) {
            indexTokenCountMap[key] = value;
          }
        }
      }
    }

    // Get instruction token overhead (system message + tool schemas + summary).
    // This budget is reserved for content prepended after pruning by buildSystemRunnable.
    // Computed BEFORE pre-flight truncation so the effective available budget can
    // drive truncation thresholds — without this, truncation thresholds based on
    // maxTokens are too generous and leave individual messages larger than the
    // actual available budget.
    const currentInstructionTokens =
      factoryParams.getInstructionTokens?.() ?? 0;

    // Apply reserve ratio: reduce the budget ceiling so pruning triggers before
    // the context fills to 100%.  This compensates for approximate token counting
    // and gives the model headroom.  Defaults to 5% when not configured.
    const DEFAULT_RESERVE_RATIO = 0.05;
    const reserveRatio = factoryParams.reserveRatio ?? DEFAULT_RESERVE_RATIO;
    const reserveTokens =
      reserveRatio > 0 && reserveRatio < 1
        ? Math.round(factoryParams.maxTokens * reserveRatio)
        : 0;
    const pruningBudget = factoryParams.maxTokens - reserveTokens;

    const effectiveMaxTokens = Math.max(
      0,
      pruningBudget - currentInstructionTokens
    );

    // P1: Log budget computation
    factoryParams.log?.('debug', 'Budget computed', {
      maxTokens: factoryParams.maxTokens,
      reserveTokens,
      pruningBudget,
      instructionTokens: currentInstructionTokens,
      effectiveMax: effectiveMaxTokens,
      messageCount: params.messages.length,
      totalTokens,
    });

    // Pre-flight truncation: truncate oversized tool results before pruning.
    // Uses the raw maxContextTokens (total context window) so the 30% threshold
    // in calculateMaxToolResultChars reflects the model's capacity, not the
    // post-deduction budget. Using effectiveMaxTokens here was overly aggressive
    // for tight contexts — it could truncate small tool results and destroy
    // information needed by post-LLM enrichment (e.g., error messages).
    const preFlightResultCount = preFlightTruncateToolResults({
      messages: params.messages,
      maxContextTokens: factoryParams.maxTokens,
      indexTokenCountMap,
      tokenCounter: factoryParams.tokenCounter,
    });

    // Pre-flight truncation: truncate oversized tool_use inputs (args) in AI messages.
    const preFlightInputCount = preFlightTruncateToolCallInputs({
      messages: params.messages,
      maxContextTokens: factoryParams.maxTokens,
      indexTokenCountMap,
      tokenCounter: factoryParams.tokenCounter,
    });

    if (preFlightResultCount > 0 || preFlightInputCount > 0) {
      factoryParams.log?.('debug', 'Pre-flight truncation applied', {
        toolResultsTruncated: preFlightResultCount,
        toolInputsTruncated: preFlightInputCount,
      });
    }

    // Position-based context pruning: degrade old tool results before the main prune.
    if (factoryParams.contextPruningConfig?.enabled === true) {
      applyContextPruning({
        messages: params.messages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        config: factoryParams.contextPruningConfig,
      });
    }

    // Recalculate totalTokens from indexTokenCountMap after pre-flight truncation
    // and context pruning may have reduced token counts for modified messages.
    // Without this, prePruneTotalTokens and early-return remainingContextTokens
    // would be stale (inflated), causing unnecessary summarization triggers.
    const preTruncationTotalTokens = totalTokens;
    totalTokens = 0;
    for (let i = 0; i < params.messages.length; i++) {
      totalTokens += indexTokenCountMap[i] ?? 0;
    }

    if (totalTokens !== preTruncationTotalTokens) {
      factoryParams.log?.('debug', 'Post-truncation token recount', {
        before: preTruncationTotalTokens,
        after: totalTokens,
        saved: preTruncationTotalTokens - totalTokens,
      });
    }

    lastTurnStartIndex = params.messages.length;
    if (
      lastCutOffIndex === 0 &&
      totalTokens + currentInstructionTokens <= pruningBudget
    ) {
      return {
        context: params.messages,
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneTotalTokens: totalTokens,
        remainingContextTokens:
          pruningBudget - totalTokens - currentInstructionTokens,
      };
    }

    const {
      context: initialContext,
      thinkingStartIndex,
      messagesToRefine,
      remainingContextTokens: initialRemainingContextTokens,
    } = getMessagesWithinTokenLimit({
      maxContextTokens: pruningBudget,
      messages: params.messages,
      indexTokenCountMap,
      startType: params.startType,
      thinkingEnabled: factoryParams.thinkingEnabled,
      tokenCounter: factoryParams.tokenCounter,
      instructionTokens: currentInstructionTokens,
      reasoningType:
        factoryParams.provider === Providers.BEDROCK
          ? ContentTypes.REASONING_CONTENT
          : ContentTypes.THINKING,
      thinkingStartIndex:
        factoryParams.thinkingEnabled === true
          ? runThinkingStartIndex
          : undefined,
    });

    const {
      context: repairedContext,
      reclaimedTokens: initialReclaimedTokens,
      droppedMessages,
    } = repairOrphanedToolMessages({
      context: initialContext,
      allMessages: params.messages,
      tokenCounter: factoryParams.tokenCounter,
      indexTokenCountMap,
    });

    // P2: Log pruning result with per-message type breakdown
    const contextBreakdown = repairedContext.map((msg) => {
      const type = msg.getType();
      const name = type === 'tool' ? (msg.name ?? 'unknown') : '';
      return name !== '' ? `${type}(${name})` : type;
    });
    factoryParams.log?.('debug', 'Pruning complete', {
      contextLength: repairedContext.length,
      contextTypes: contextBreakdown.join(', '),
      messagesToRefineCount: messagesToRefine.length,
      droppedOrphans: droppedMessages.length,
      remainingTokens: initialRemainingContextTokens,
    });

    let context = repairedContext;
    let reclaimedTokens = initialReclaimedTokens;

    // Orphan repair may drop ToolMessages whose parent AI was pruned.
    // Append them to messagesToRefine so summarization can still see the
    // tool results (otherwise the summary says "in progress" for a tool
    // call that already completed, causing the model to repeat it).
    if (droppedMessages.length > 0) {
      messagesToRefine.push(...droppedMessages);
    }

    // ---------------------------------------------------------------
    // Emergency truncation: if pruning produced an empty context but
    // messages exist, aggressively truncate all tool_call inputs and
    // tool results, then retry.  Budget is proportional to the
    // effective token limit (~4 chars/token, spread across messages)
    // with a floor of 200 chars so content is never completely blank.
    // Uses head+tail so the model sees both what was called and the
    // final outcome (e.g., return value at the end of a script eval).
    // ---------------------------------------------------------------
    if (
      context.length === 0 &&
      params.messages.length > 0 &&
      effectiveMaxTokens > 0
    ) {
      // Proportional budget: divide effective budget across messages,
      // convert to chars (~4 chars/token), floor at 200 chars.
      const perMessageTokenBudget = Math.floor(
        effectiveMaxTokens / Math.max(1, params.messages.length)
      );
      const emergencyMaxChars = Math.max(200, perMessageTokenBudget * 4);

      // P3: Log emergency path entry
      factoryParams.log?.(
        'warn',
        'Empty context, entering emergency truncation',
        {
          messageCount: params.messages.length,
          effectiveMax: effectiveMaxTokens,
          emergencyMaxChars,
        }
      );

      // Clone the messages array so emergency truncation doesn't permanently
      // mutate graph state.  The originals remain intact for future turns
      // where more budget may be available.  Also snapshot indexTokenCountMap
      // entries so the closure doesn't retain stale (too-small) counts for
      // the original un-truncated messages on the next turn.
      const emergencyMessages = [...params.messages];
      const preEmergencyTokenCounts: Record<string, number | undefined> = {};
      for (let i = 0; i < params.messages.length; i++) {
        preEmergencyTokenCounts[i] = indexTokenCountMap[i];
      }

      let emergencyTruncatedCount = 0;
      for (let i = 0; i < emergencyMessages.length; i++) {
        const message = emergencyMessages[i];
        // Truncate ToolMessage content (uses head+tail via truncateToolResultContent)
        if (message.getType() === 'tool') {
          const content = message.content;
          if (
            typeof content === 'string' &&
            content.length > emergencyMaxChars
          ) {
            const beforeLen = content.length;
            // Clone the ToolMessage to avoid mutating the original in graph state
            const cloned = new ToolMessage({
              content: truncateToolResultContent(content, emergencyMaxChars),
              tool_call_id: (message as ToolMessage).tool_call_id,
              name: message.name,
              id: message.id,
              additional_kwargs: message.additional_kwargs,
              response_metadata: message.response_metadata,
            });
            emergencyMessages[i] = cloned;
            indexTokenCountMap[i] = factoryParams.tokenCounter(cloned);
            emergencyTruncatedCount++;
            factoryParams.log?.('debug', 'Emergency truncated tool result', {
              index: i,
              toolName: message.name ?? 'unknown',
              beforeChars: beforeLen,
              afterChars: emergencyMaxChars,
            });
          }
        }
        // Truncate AI message tool_call inputs
        if (message.getType() === 'ai' && Array.isArray(message.content)) {
          const aiMsg = message as AIMessage;
          const contentBlocks = aiMsg.content as MessageContentComplex[];
          const needsTruncation = contentBlocks.some((block) => {
            if (typeof block !== 'object') return false;
            const record = block as Record<string, unknown>;
            if (
              (record.type === 'tool_use' || record.type === 'tool_call') &&
              record.input != null
            ) {
              const serialized =
                typeof record.input === 'string'
                  ? record.input
                  : JSON.stringify(record.input);
              return serialized.length > emergencyMaxChars;
            }
            return false;
          });
          if (needsTruncation) {
            const newContent = contentBlocks.map((block) => {
              if (typeof block !== 'object') return block;
              const record = block as Record<string, unknown>;
              if (
                (record.type === 'tool_use' || record.type === 'tool_call') &&
                record.input != null
              ) {
                const serialized =
                  typeof record.input === 'string'
                    ? record.input
                    : JSON.stringify(record.input);
                if (serialized.length > emergencyMaxChars) {
                  return {
                    ...record,
                    input: truncateToolInput(record.input, emergencyMaxChars),
                  };
                }
              }
              return block;
            });
            const newToolCalls = (aiMsg.tool_calls ?? []).map((tc) => {
              const serializedArgs = JSON.stringify(tc.args);
              if (serializedArgs.length > emergencyMaxChars) {
                return {
                  ...tc,
                  args: truncateToolInput(tc.args, emergencyMaxChars),
                };
              }
              return tc;
            });
            emergencyMessages[i] = new AIMessage({
              ...aiMsg,
              content: newContent,
              tool_calls: newToolCalls.length > 0 ? newToolCalls : undefined,
            });
            indexTokenCountMap[i] = factoryParams.tokenCounter(
              emergencyMessages[i]
            );
            emergencyTruncatedCount++;
            factoryParams.log?.('debug', 'Emergency truncated tool input', {
              index: i,
            });
          }
        }
      }

      factoryParams.log?.('info', 'Emergency truncation complete', {
        truncatedCount: emergencyTruncatedCount,
        emergencyMaxChars,
      });

      // Retry pruning with the emergency-truncated (cloned) messages
      const retryResult = getMessagesWithinTokenLimit({
        maxContextTokens: pruningBudget,
        messages: emergencyMessages,
        indexTokenCountMap,
        startType: params.startType,
        thinkingEnabled: factoryParams.thinkingEnabled,
        tokenCounter: factoryParams.tokenCounter,
        instructionTokens: currentInstructionTokens,
        reasoningType:
          factoryParams.provider === Providers.BEDROCK
            ? ContentTypes.REASONING_CONTENT
            : ContentTypes.THINKING,
        thinkingStartIndex:
          factoryParams.thinkingEnabled === true
            ? runThinkingStartIndex
            : undefined,
      });

      const repaired = repairOrphanedToolMessages({
        context: retryResult.context,
        allMessages: emergencyMessages,
        tokenCounter: factoryParams.tokenCounter,
        indexTokenCountMap,
      });

      context = repaired.context;
      reclaimedTokens = repaired.reclaimedTokens;
      messagesToRefine.push(...retryResult.messagesToRefine);
      if (repaired.droppedMessages.length > 0) {
        messagesToRefine.push(...repaired.droppedMessages);
      }

      // Restore the closure's indexTokenCountMap to pre-emergency values so the
      // next turn counts old messages at their original (un-truncated) size.
      // The emergency-truncated counts were only needed for this turn's
      // getMessagesWithinTokenLimit retry.
      for (const [key, value] of Object.entries(preEmergencyTokenCounts)) {
        indexTokenCountMap[key] = value;
      }
    }

    const remainingContextTokens = Math.max(
      0,
      Math.min(pruningBudget, initialRemainingContextTokens + reclaimedTokens)
    );

    runThinkingStartIndex = thinkingStartIndex ?? -1;
    /** The index is the first value of `context`, index relative to `params.messages` */
    lastCutOffIndex = Math.max(
      params.messages.length -
        (context.length - (context[0]?.getType() === 'system' ? 1 : 0)),
      0
    );

    return {
      context,
      indexTokenCountMap,
      messagesToRefine,
      prePruneTotalTokens: totalTokens,
      remainingContextTokens,
    };
  };
}
