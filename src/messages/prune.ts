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
} {
  const validToolCallIds = new Set<string>();
  for (const message of context) {
    for (const id of getToolCallIds(message)) {
      validToolCallIds.add(id);
    }
  }

  if (validToolCallIds.size === 0) {
    return {
      context,
      reclaimedTokens: 0,
      droppedOrphanCount: 0,
    };
  }

  let reclaimedTokens = 0;
  let droppedOrphanCount = 0;
  const repairedContext: BaseMessage[] = [];
  for (const message of context) {
    if (message.getType() !== 'tool') {
      repairedContext.push(message);
      continue;
    }

    const toolResultId = getToolResultId(message);
    if (toolResultId == null || !validToolCallIds.has(toolResultId)) {
      droppedOrphanCount += 1;
      reclaimedTokens += resolveTokenCountForMessage({
        message,
        allMessages,
        tokenCounter,
        indexTokenCountMap,
      });
      continue;
    }

    repairedContext.push(message);
  }

  return {
    context: repairedContext,
    reclaimedTokens,
    droppedOrphanCount,
  };
}

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
}: {
  messages: BaseMessage[];
  maxContextTokens: number;
  indexTokenCountMap: Record<string, number | undefined>;
  startType?: string | string[];
  thinkingEnabled?: boolean;
  tokenCounter: TokenCounter;
  thinkingStartIndex?: number;
  reasoningType?: ContentTypes.THINKING | ContentTypes.REASONING_CONTENT;
}): PruningResult {
  // Every reply is primed with <|start|>assistant<|message|>, so we
  // start with 3 tokens for the label after all messages have been counted.
  let currentTokenCount = 3;
  const instructions =
    _messages[0]?.getType() === 'system' ? _messages[0] : undefined;
  const instructionsTokenCount =
    instructions != null ? (indexTokenCountMap[0] ?? 0) : 0;
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
    throw new Error(
      'Context window exceeded: aggressive pruning removed all AI messages (likely due to an oversized tool response). Increase max context tokens or reduce tool output size.'
    );
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
      if (params.messages[0].getType() === 'system') {
        totalIndexTokens += indexTokenCountMap[0] ?? 0;
      }
      for (let i = lastCutOffIndex; i < params.messages.length; i++) {
        if (i === 0 && params.messages[0].getType() === 'system') {
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
        if (params.messages[0].getType() === 'system') {
          preCalibrationSnapshot[0] = indexTokenCountMap[0];
        }

        if (
          params.messages[0].getType() === 'system' &&
          lastCutOffIndex !== 0
        ) {
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

    // Pre-flight truncation: truncate oversized tool results before pruning.
    // This runs after calibration so token counts are reasonably accurate,
    // but before getMessagesWithinTokenLimit so pruning works with correct counts.
    preFlightTruncateToolResults({
      messages: params.messages,
      maxContextTokens: factoryParams.maxTokens,
      indexTokenCountMap,
      tokenCounter: factoryParams.tokenCounter,
    });

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
    totalTokens = 0;
    for (let i = 0; i < params.messages.length; i++) {
      totalTokens += indexTokenCountMap[i] ?? 0;
    }

    lastTurnStartIndex = params.messages.length;
    if (lastCutOffIndex === 0 && totalTokens <= factoryParams.maxTokens) {
      return {
        context: params.messages,
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneTotalTokens: totalTokens,
        remainingContextTokens: factoryParams.maxTokens - totalTokens,
      };
    }

    const {
      context: initialContext,
      thinkingStartIndex,
      messagesToRefine,
      remainingContextTokens: initialRemainingContextTokens,
    } = getMessagesWithinTokenLimit({
      maxContextTokens: factoryParams.maxTokens,
      messages: params.messages,
      indexTokenCountMap,
      startType: params.startType,
      thinkingEnabled: factoryParams.thinkingEnabled,
      tokenCounter: factoryParams.tokenCounter,
      reasoningType:
        factoryParams.provider === Providers.BEDROCK
          ? ContentTypes.REASONING_CONTENT
          : ContentTypes.THINKING,
      thinkingStartIndex:
        factoryParams.thinkingEnabled === true
          ? runThinkingStartIndex
          : undefined,
    });

    const { context, reclaimedTokens } = repairOrphanedToolMessages({
      context: initialContext,
      allMessages: params.messages,
      tokenCounter: factoryParams.tokenCounter,
      indexTokenCountMap,
    });

    const remainingContextTokens = Math.max(
      0,
      Math.min(
        factoryParams.maxTokens,
        initialRemainingContextTokens + reclaimedTokens
      )
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
