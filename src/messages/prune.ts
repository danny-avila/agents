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
import { resolveContextPruningSettings } from './contextPruningSettings';
import { ContentTypes, Providers, Constants } from '@/common';
import { applyContextPruning } from './contextPruning';

function sumTokenCounts(
  tokenMap: Record<string, number | undefined>,
  count: number
): number {
  let total = 0;
  for (let i = 0; i < count; i++) {
    total += tokenMap[i] ?? 0;
  }
  return total;
}

/** Default fraction of the token budget reserved as headroom (5 %). */
export const DEFAULT_RESERVE_RATIO = 0.05;

/** Context pressure at which observation masking and context fading activate. */
const PRESSURE_THRESHOLD_MASKING = 0.8;

/** Pressure band thresholds paired with budget factors for progressive context fading. */
const PRESSURE_BANDS: [number, number][] = [
  [0.99, 0.05],
  [0.9, 0.2],
  [0.85, 0.5],
  [0.8, 1.0],
];

/** Maximum character length for masked (consumed) tool results. */
const MASKED_RESULT_MAX_CHARS = 300;

/** Hard cap for the originalToolContent store (~2 MB estimated from char length). */
const ORIGINAL_CONTENT_MAX_CHARS = 2_000_000;

/** Minimum cumulative calibration ratio — provider can't count fewer tokens
 *  than our raw estimate (within reason). Prevents divide-by-zero edge cases. */
const CALIBRATION_RATIO_MIN = 0.5;

/** Maximum cumulative calibration ratio — sanity cap for the running ratio. */
const CALIBRATION_RATIO_MAX = 5;

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
   * When true, context pressure fading (pre-flight tool result truncation)
   * is skipped.  Summarization replaces pruning as the primary context
   * management strategy — the summarizer needs full un-truncated tool results
   * to produce an accurate summary.  Hard pruning still runs as a fallback
   * when summarization is skipped or capped.
   */
  summarizationEnabled?: boolean;
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
   * filling the context window to 100%.  Defaults to 5 % (0.05) when omitted.
   */
  reserveRatio?: number;
  /**
   * Initial calibration ratio from a previous run's persisted contextMeta.
   * Seeds the running EMA so new messages are scaled immediately instead
   * of waiting for the first provider response.  Ignored when <= 0.
   */
  calibrationRatio?: number;
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
    inputTokens?: number;
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
  messageIndexMap,
  tokenCounter,
  indexTokenCountMap,
}: {
  message: BaseMessage;
  messageIndexMap: Map<BaseMessage, number>;
  tokenCounter: TokenCounter;
  indexTokenCountMap: Record<string, number | undefined>;
}): number {
  const originalIndex = messageIndexMap.get(message) ?? -1;
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
  const messageIndexMap = new Map<BaseMessage, number>();
  for (let i = 0; i < allMessages.length; i++) {
    messageIndexMap.set(allMessages[i], i);
  }

  const validToolCallIds = new Set<string>();
  const presentToolResultIds = new Set<string>();
  for (const message of context) {
    for (const id of getToolCallIds(message)) {
      validToolCallIds.add(id);
    }
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
    if (message.getType() === 'tool') {
      const toolResultId = getToolResultId(message);
      if (toolResultId == null || !validToolCallIds.has(toolResultId)) {
        droppedOrphanCount += 1;
        reclaimedTokens += resolveTokenCountForMessage({
          message,
          tokenCounter,
          messageIndexMap,
          indexTokenCountMap,
        });
        droppedMessages.push(message);
        continue;
      }
      repairedContext.push(message);
      continue;
    }

    if (message.getType() === 'ai' && message instanceof AIMessage) {
      const toolCallIds = getToolCallIds(message);
      if (toolCallIds.size > 0) {
        let hasOrphanToolCalls = false;
        for (const id of toolCallIds) {
          if (!presentToolResultIds.has(id)) {
            hasOrphanToolCalls = true;
            break;
          }
        }
        if (hasOrphanToolCalls) {
          const originalTokens = resolveTokenCountForMessage({
            message,
            messageIndexMap,
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
  const keptToolCalls = (message.tool_calls ?? []).filter(
    (tc) => typeof tc.id === 'string' && presentToolResultIds.has(tc.id)
  );

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
        if (
          typeof tc.id === 'string' &&
          tc.id.length > 0 &&
          !tc.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
        ) {
          allToolCallIds.add(tc.id);
        }
      }
    }
    if (Array.isArray(msgAny.content)) {
      for (const block of msgAny.content as Array<Record<string, unknown>>) {
        if (
          typeof block === 'object' &&
          (block.type === 'tool_use' || block.type === 'tool_call') &&
          typeof block.id === 'string' &&
          !block.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
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
  const strippedAiIndices = new Set<number>();

  for (const msg of messages) {
    const msgAny = msg as unknown as Record<string, unknown>;
    const msgType =
      typeof (msg as { getType?: unknown }).getType === 'function'
        ? msg.getType()
        : ((msgAny.role as string | undefined) ??
          (msgAny._type as string | undefined));

    const toolCallId = msgAny.tool_call_id as string | undefined;
    if (
      (msgType === 'tool' || msg instanceof ToolMessage) &&
      typeof toolCallId === 'string' &&
      !allToolCallIds.has(toolCallId)
    ) {
      continue;
    }

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
        const patched = Object.create(
          Object.getPrototypeOf(msg),
          Object.getOwnPropertyDescriptors(msg)
        );
        patched.tool_calls = keptToolCalls.length > 0 ? keptToolCalls : [];
        patched.content = keptContent;
        result.push(patched as BaseMessage);
        continue;
      }
    }

    result.push(msg);
  }

  // Bedrock/Anthropic require the conversation to end with a user message;
  // a stripped AI message (tool_use removed) represents a dead-end exchange.
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
  const totalOutputTokens = Number(usage.output_tokens) || 0;
  const cacheSum = cacheCreation + cacheRead;
  // Anthropic: input_tokens excludes cache, cache_read can be much larger than input_tokens.
  // OpenAI: input_tokens includes cache, cache_read is always <= input_tokens.
  const cacheIsAdditive = cacheSum > 0 && cacheSum > baseInputTokens;
  const totalInputTokens = cacheIsAdditive
    ? baseInputTokens + cacheSum
    : baseInputTokens;

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
 * Observation masking: replaces consumed ToolMessage content with tight
 * head+tail truncations that serve as informative placeholders.
 *
 * A ToolMessage is "consumed" when a subsequent AI message exists that is NOT
 * purely tool calls — meaning the model has already read and acted on the
 * result. Unconsumed results (the latest tool outputs the model hasn't
 * responded to yet) are left intact so the model can still use them.
 *
 * AI messages are never masked — they contain the model's own reasoning and
 * conclusions, which is what prevents the model from repeating work after
 * its tool results are masked.
 *
 * @returns The number of tool messages that were masked.
 */
export function maskConsumedToolResults(params: {
  messages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  tokenCounter: TokenCounter;
  /** Raw-space token budget available for all consumed tool results combined.
   *  When provided, the budget is distributed across consumed results weighted
   *  by recency (newest get the most, oldest get MASKED_RESULT_MAX_CHARS min).
   *  When omitted, falls back to a flat MASKED_RESULT_MAX_CHARS per result. */
  availableRawBudget?: number;
  /** When provided, original (pre-masking) content is stored here keyed by
   *  message index — only for entries that actually get truncated. */
  originalContentStore?: Map<number, string>;
  /** Called after storing content with the char length of the stored entry. */
  onContentStored?: (charLength: number) => void;
}): number {
  const { messages, indexTokenCountMap, tokenCounter } = params;
  let maskedCount = 0;

  // Pass 1 (backward): identify consumed tool message indices.
  // A ToolMessage is "consumed" once we've seen a subsequent AI message with
  // substantive text content (not just tool calls).
  // Collected in forward order (oldest first) for recency weighting.
  let seenNonToolCallAI = false;
  const consumedIndices: number[] = [];

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const type = msg.getType();

    if (type === 'ai') {
      const hasText =
        typeof msg.content === 'string'
          ? msg.content.trim().length > 0
          : Array.isArray(msg.content) &&
            msg.content.some(
              (b) =>
                typeof b === 'object' &&
                (b as Record<string, unknown>).type === 'text' &&
                typeof (b as Record<string, unknown>).text === 'string' &&
                ((b as Record<string, unknown>).text as string).trim().length >
                  0
            );
      if (hasText) {
        seenNonToolCallAI = true;
      }
    } else if (type === 'tool' && seenNonToolCallAI) {
      consumedIndices.push(i);
    }
  }

  if (consumedIndices.length === 0) {
    return 0;
  }

  consumedIndices.reverse();

  const totalBudgetChars =
    params.availableRawBudget != null && params.availableRawBudget > 0
      ? params.availableRawBudget * 4
      : 0;

  const count = consumedIndices.length;

  for (let c = 0; c < count; c++) {
    const i = consumedIndices[c];
    const message = messages[i];
    const content = message.content;
    if (typeof content !== 'string') {
      continue;
    }

    let maxChars: number;
    if (totalBudgetChars > 0) {
      const position = count > 1 ? c / (count - 1) : 1;
      const weight = 0.2 + 0.8 * position;
      const totalWeight = count > 1 ? 0.6 * count : 1;
      const share = (weight / totalWeight) * totalBudgetChars;
      maxChars = Math.max(MASKED_RESULT_MAX_CHARS, Math.floor(share));
    } else {
      maxChars = MASKED_RESULT_MAX_CHARS;
    }

    if (content.length <= maxChars) {
      continue;
    }

    if (params.originalContentStore && !params.originalContentStore.has(i)) {
      params.originalContentStore.set(i, content);
      if (params.onContentStored) {
        params.onContentStored(content.length);
      }
    }

    const cloned = new ToolMessage({
      content: truncateToolResultContent(content, maxChars),
      tool_call_id: (message as ToolMessage).tool_call_id,
      name: message.name,
      id: message.id,
      additional_kwargs: message.additional_kwargs,
      response_metadata: message.response_metadata,
    });
    messages[i] = cloned;
    indexTokenCountMap[i] = tokenCounter(cloned);
    maskedCount++;
  }

  return maskedCount;
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
  const baseMaxChars = calculateMaxToolResultChars(maxContextTokens);
  let truncatedCount = 0;

  const toolIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].getType() === 'tool') {
      toolIndices.push(i);
    }
  }

  for (let t = 0; t < toolIndices.length; t++) {
    const i = toolIndices[t];
    const message = messages[i];
    const content = message.content;
    if (typeof content !== 'string') {
      continue;
    }

    const position = toolIndices.length > 1 ? t / (toolIndices.length - 1) : 1;
    const recencyFactor = 0.2 + 0.8 * position;
    const maxChars = Math.max(200, Math.floor(baseMaxChars * recencyFactor));

    if (content.length <= maxChars) {
      continue;
    }

    const truncated = truncateToolResultContent(content, maxChars);
    const cloned = new ToolMessage({
      content: truncated,
      tool_call_id: (message as ToolMessage).tool_call_id,
      name: message.name,
      id: message.id,
      additional_kwargs: message.additional_kwargs,
      response_metadata: message.response_metadata,
    });
    messages[i] = cloned;
    indexTokenCountMap[i] = tokenCounter(cloned);
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
    const state = { changed: false };
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

      state.changed = true;
      // Replaces original input with { _truncated, _originalChars } —
      // safe because the tool call already executed in a prior turn.
      return {
        ...record,
        input: truncateToolInput(serialized, maxInputChars),
      };
    });

    if (!state.changed) {
      continue;
    }

    const aiMsg = message as AIMessage;
    const newToolCalls = (aiMsg.tool_calls ?? []).map((tc) => {
      const serializedArgs = JSON.stringify(tc.args);
      if (serializedArgs.length <= maxInputChars) {
        return tc;
      }
      // Replaces original args with { _truncated, _originalChars } —
      // safe because the tool call already executed in a prior turn.
      return {
        ...tc,
        args: truncateToolInput(serializedArgs, maxInputChars),
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
  let totalTokens = 0;
  for (const key in indexTokenCountMap) {
    totalTokens += indexTokenCountMap[key] ?? 0;
  }
  let runThinkingStartIndex = -1;
  /** Cumulative raw tiktoken tokens we've sent to the provider (messages only,
   *  excludes instruction overhead and new outputs not yet seen by provider). */
  let cumulativeRawSent = 0;
  /** Cumulative provider-reported message tokens (providerInput - instructionOverhead). */
  let cumulativeProviderReported = 0;
  /** Stable calibration ratio = cumulativeProviderReported / cumulativeRawSent.
   *  Converges monotonically as data accumulates. Falls back to seeded value. */
  let calibrationRatio =
    factoryParams.calibrationRatio != null && factoryParams.calibrationRatio > 0
      ? factoryParams.calibrationRatio
      : 1;
  /** Best observed instruction overhead from a near-zero variance turn.
   *  Self-seeds from provider observations within the run. */
  let bestInstructionOverhead: number | undefined;
  let bestVarianceAbs = Infinity;
  /** Local estimate at the time bestInstructionOverhead was observed.
   *  Used to invalidate the cached overhead when instructions change
   *  mid-run (e.g. tool discovery adds tools to the bound set). */
  let bestInstructionEstimate: number | undefined;
  /** Original (pre-masking) tool result content keyed by message index.
   *  Allows the summarizer to see full tool outputs even after masking
   *  has truncated them in the live message array. Cleared when the
   *  pruner is recreated after summarization. */
  const originalToolContent = new Map<number, string>();
  let originalToolContentSize = 0;
  const contextPruningSettings = resolveContextPruningSettings(
    factoryParams.contextPruningConfig
  );

  return function pruneMessages(params: PruneMessagesParams): {
    context: BaseMessage[];
    indexTokenCountMap: Record<string, number | undefined>;
    messagesToRefine?: BaseMessage[];
    prePruneContextTokens?: number;
    remainingContextTokens?: number;
    contextPressure?: number;
    originalToolContent?: Map<number, string>;
    calibrationRatio?: number;
    resolvedInstructionOverhead?: number;
  } {
    if (params.messages.length === 0) {
      return {
        context: [],
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneContextTokens: 0,
        remainingContextTokens: factoryParams.maxTokens,
        calibrationRatio,
        resolvedInstructionOverhead: bestInstructionOverhead,
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
    }

    const newOutputs = new Set<number>();
    let outputTokensAssigned = false;
    for (let i = lastTurnStartIndex; i < params.messages.length; i++) {
      const message = params.messages[i];
      if (indexTokenCountMap[i] !== undefined) {
        continue;
      }

      // Assign output_tokens to the first uncounted AI message — this is the
      // model's response.  Previous code blindly targeted lastTurnStartIndex
      // which could hit a pre-counted HumanMessage or miss the AI entirely.
      if (!outputTokensAssigned && currentUsage && message.getType() === 'ai') {
        indexTokenCountMap[i] = currentUsage.output_tokens;
        newOutputs.add(i);
        outputTokensAssigned = true;
      } else {
        // Always store raw tiktoken count — the map stays in raw space.
        // Budget decisions multiply by calibrationRatio on the fly.
        indexTokenCountMap[i] = factoryParams.tokenCounter(message);
        if (currentUsage) {
          newOutputs.add(i);
        }
      }
      totalTokens += indexTokenCountMap[i] ?? 0;
    }

    // Cumulative calibration: accumulate raw tiktoken tokens and provider-
    // reported tokens across turns.  The ratio of the two running totals
    // converges monotonically to the true provider multiplier — no EMA,
    // no per-turn oscillation, no map mutation.
    if (currentUsage && params.totalTokensFresh !== false) {
      const instructionOverhead = factoryParams.getInstructionTokens?.() ?? 0;
      const providerInputTokens =
        params.lastCallUsage?.inputTokens ?? currentUsage.input_tokens;

      // Sum raw tiktoken counts for messages the provider saw (excludes
      // new outputs from this turn — the provider hasn't seen them yet).
      let rawSentThisTurn = 0;
      const firstIsSystem =
        params.messages.length > 0 && params.messages[0].getType() === 'system';
      if (firstIsSystem) {
        rawSentThisTurn += indexTokenCountMap[0] ?? 0;
      }
      for (let i = lastCutOffIndex; i < params.messages.length; i++) {
        if ((i === 0 && firstIsSystem) || newOutputs.has(i)) {
          continue;
        }
        rawSentThisTurn += indexTokenCountMap[i] ?? 0;
      }

      const providerMessageTokens = Math.max(
        0,
        providerInputTokens - instructionOverhead
      );

      if (rawSentThisTurn > 0 && providerMessageTokens > 0) {
        cumulativeRawSent += rawSentThisTurn;
        cumulativeProviderReported += providerMessageTokens;
        const newRatio = cumulativeProviderReported / cumulativeRawSent;
        calibrationRatio = Math.max(
          CALIBRATION_RATIO_MIN,
          Math.min(CALIBRATION_RATIO_MAX, newRatio)
        );
      }

      const calibratedOurTotal =
        instructionOverhead + rawSentThisTurn * calibrationRatio;
      const overallRatio =
        calibratedOurTotal > 0 ? providerInputTokens / calibratedOurTotal : 0;
      const variancePct = Math.round((overallRatio - 1) * 100);

      const absVariance = Math.abs(overallRatio - 1);
      if (absVariance < bestVarianceAbs && rawSentThisTurn > 0) {
        bestVarianceAbs = absVariance;
        bestInstructionOverhead = Math.max(
          0,
          Math.round(providerInputTokens - rawSentThisTurn * calibrationRatio)
        );
        bestInstructionEstimate = factoryParams.getInstructionTokens?.() ?? 0;
      }

      factoryParams.log?.('debug', 'Calibration observed', {
        providerInputTokens,
        calibratedEstimate: Math.round(calibratedOurTotal),
        variance: `${variancePct > 0 ? '+' : ''}${variancePct}%`,
        calibrationRatio: Math.round(calibrationRatio * 100) / 100,
        instructionOverhead,
        cumulativeRawSent,
        cumulativeProviderReported,
      });
    }

    // Computed BEFORE pre-flight truncation so the effective budget can drive
    // truncation thresholds — without this, thresholds based on maxTokens are
    // too generous and leave individual messages larger than the actual budget.
    const estimatedInstructionTokens =
      factoryParams.getInstructionTokens?.() ?? 0;
    const estimateStable =
      bestInstructionEstimate != null &&
      bestInstructionEstimate > 0 &&
      Math.abs(estimatedInstructionTokens - bestInstructionEstimate) /
        bestInstructionEstimate <
        0.1;
    const currentInstructionTokens =
      bestInstructionOverhead != null &&
      bestInstructionOverhead <= estimatedInstructionTokens &&
      estimateStable
        ? bestInstructionOverhead
        : estimatedInstructionTokens;

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

    let calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);

    factoryParams.log?.('debug', 'Budget', {
      maxTokens: factoryParams.maxTokens,
      pruningBudget,
      effectiveMax: effectiveMaxTokens,
      instructionTokens: currentInstructionTokens,
      messageCount: params.messages.length,
      calibratedTotalTokens,
      calibrationRatio: Math.round(calibrationRatio * 100) / 100,
    });

    // When instructions alone consume the entire budget, no message can
    // fit regardless of truncation.  Short-circuit: yield all messages for
    // summarization and return an empty context so the Graph can route to
    // the summarize node immediately instead of falling through to the
    // emergency path that would reach the same outcome more expensively.
    if (
      effectiveMaxTokens === 0 &&
      factoryParams.summarizationEnabled === true &&
      params.messages.length > 0
    ) {
      factoryParams.log?.(
        'warn',
        'Instructions consume entire budget — yielding all messages for summarization',
        {
          instructionTokens: currentInstructionTokens,
          pruningBudget,
          messageCount: params.messages.length,
        }
      );

      lastTurnStartIndex = params.messages.length;
      return {
        context: [],
        indexTokenCountMap,
        messagesToRefine: [...params.messages],
        prePruneContextTokens: calibratedTotalTokens,
        remainingContextTokens: 0,
        contextPressure:
          pruningBudget > 0 ? calibratedTotalTokens / pruningBudget : 0,
        calibrationRatio,
        resolvedInstructionOverhead: bestInstructionOverhead,
      };
    }

    // ---------------------------------------------------------------------------
    // Progressive context fading — inspired by Claude Code's staged compaction.
    // Below 80%: no modifications, tool results retain full size.
    // Above 80%: graduated truncation with increasing aggression per pressure band.
    // Recency weighting ensures older results fade first, newer results last.
    //
    // At the gentlest level, truncation preserves most content (head+tail).
    // At the most aggressive level, the result is effectively a one-line placeholder.
    //
    //   80%: gentle — budget factor 1.0, oldest get light truncation
    //   85%: moderate — budget factor 0.50, older results shrink significantly
    //   90%: aggressive — budget factor 0.20, most results heavily truncated
    //   99%: emergency — budget factor 0.05, effectively placeholders for old results
    // ---------------------------------------------------------------------------
    totalTokens = sumTokenCounts(indexTokenCountMap, params.messages.length);
    calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);
    const contextPressure =
      pruningBudget > 0 ? calibratedTotalTokens / pruningBudget : 0;
    let preFlightResultCount = 0;
    let preFlightInputCount = 0;

    // -----------------------------------------------------------------------
    // Observation masking (80%+ pressure, both paths):
    // Replace consumed ToolMessage content with tight head+tail placeholders.
    // AI messages stay intact so the model can read its own prior reasoning
    // and won't repeat work.  Unconsumed results (latest tool outputs the
    // model hasn't acted on yet) stay full.
    //
    // When summarization is enabled, snapshot messages first so the
    // summarizer can see the full originals when compaction fires.
    // -----------------------------------------------------------------------
    let observationsMasked = 0;

    if (contextPressure >= PRESSURE_THRESHOLD_MASKING) {
      const rawMessageBudget =
        calibrationRatio > 0
          ? Math.floor(effectiveMaxTokens / calibrationRatio)
          : effectiveMaxTokens;
      // When summarization is enabled, use half the reserve ratio as extra
      // masking headroom — the LLM keeps more context while the summarizer
      // gets full content from originalToolContent regardless. The remaining
      // half of the reserve covers estimation errors.
      const reserveHeadroom =
        factoryParams.summarizationEnabled === true
          ? Math.floor(
            rawMessageBudget *
                (factoryParams.reserveRatio ?? DEFAULT_RESERVE_RATIO) *
                0.5
          )
          : 0;
      observationsMasked = maskConsumedToolResults({
        messages: params.messages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        availableRawBudget: rawMessageBudget + reserveHeadroom,
        originalContentStore:
          factoryParams.summarizationEnabled === true
            ? originalToolContent
            : undefined,
        onContentStored:
          factoryParams.summarizationEnabled === true
            ? (charLen: number): void => {
              originalToolContentSize += charLen;
              while (
                originalToolContentSize > ORIGINAL_CONTENT_MAX_CHARS &&
                  originalToolContent.size > 0
              ) {
                const oldest = originalToolContent.keys().next();
                if (oldest.done === true) {
                  break;
                }
                const removed = originalToolContent.get(oldest.value);
                if (removed != null) {
                  originalToolContentSize -= removed.length;
                }
                originalToolContent.delete(oldest.value);
              }
            }
            : undefined,
      });
      if (observationsMasked > 0) {
        cumulativeRawSent = 0;
        cumulativeProviderReported = 0;
      }
    }

    if (
      contextPressure >= PRESSURE_THRESHOLD_MASKING &&
      factoryParams.summarizationEnabled !== true
    ) {
      const budgetFactor =
        PRESSURE_BANDS.find(
          ([threshold]) => contextPressure >= threshold
        )?.[1] ?? 1.0;

      const baseBudget = Math.max(
        1024,
        Math.floor(effectiveMaxTokens * budgetFactor)
      );

      preFlightResultCount = preFlightTruncateToolResults({
        messages: params.messages,
        maxContextTokens: baseBudget,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });

      preFlightInputCount = preFlightTruncateToolCallInputs({
        messages: params.messages,
        maxContextTokens: baseBudget,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });
    }
    if (
      factoryParams.contextPruningConfig?.enabled === true &&
      factoryParams.summarizationEnabled !== true
    ) {
      applyContextPruning({
        messages: params.messages,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
        resolvedSettings: contextPruningSettings,
      });
    }

    // Fit-to-budget: when summarization is enabled and individual messages
    // exceed the effective budget, truncate them so every message can fit in
    // a single context slot.  Without this, oversized tool results (e.g.
    // take_snapshot at 9K chars) cause empty context → emergency truncation
    // → immediate re-summarization after just one tool call.
    //
    // This is NOT the lossy position-based fading above — it only targets
    // messages that individually exceed the budget, using the full effective
    // budget as the cap (not a pressure-scaled fraction).
    // Fit-to-budget caps are in raw space (divide by ratio) so that after
    // calibration the truncated results actually fit within the budget.
    const rawSpaceEffectiveMax =
      calibrationRatio > 0
        ? Math.round(effectiveMaxTokens / calibrationRatio)
        : effectiveMaxTokens;

    if (
      factoryParams.summarizationEnabled === true &&
      rawSpaceEffectiveMax > 0
    ) {
      preFlightResultCount = preFlightTruncateToolResults({
        messages: params.messages,
        maxContextTokens: rawSpaceEffectiveMax,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });

      preFlightInputCount = preFlightTruncateToolCallInputs({
        messages: params.messages,
        maxContextTokens: rawSpaceEffectiveMax,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });
    }

    const preTruncationTotalTokens = totalTokens;
    totalTokens = sumTokenCounts(indexTokenCountMap, params.messages.length);
    calibratedTotalTokens = Math.round(totalTokens * calibrationRatio);

    const anyAdjustment =
      observationsMasked > 0 ||
      preFlightResultCount > 0 ||
      preFlightInputCount > 0 ||
      totalTokens !== preTruncationTotalTokens;

    if (anyAdjustment) {
      factoryParams.log?.('debug', 'Context adjusted', {
        contextPressure: Math.round(contextPressure * 100),
        observationsMasked,
        toolOutputsTruncated: preFlightResultCount,
        toolInputsTruncated: preFlightInputCount,
        tokensBefore: preTruncationTotalTokens,
        tokensAfter: totalTokens,
        tokensSaved: preTruncationTotalTokens - totalTokens,
      });
    }

    lastTurnStartIndex = params.messages.length;
    if (
      lastCutOffIndex === 0 &&
      calibratedTotalTokens + currentInstructionTokens <= pruningBudget
    ) {
      return {
        context: params.messages,
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneContextTokens: calibratedTotalTokens,
        remainingContextTokens:
          pruningBudget - calibratedTotalTokens - currentInstructionTokens,
        contextPressure,
        originalToolContent:
          originalToolContent.size > 0 ? originalToolContent : undefined,
        calibrationRatio,
        resolvedInstructionOverhead: bestInstructionOverhead,
      };
    }

    const rawSpaceBudget =
      calibrationRatio > 0
        ? Math.round(pruningBudget / calibrationRatio)
        : pruningBudget;

    const rawSpaceInstructionTokens =
      calibrationRatio > 0
        ? Math.round(currentInstructionTokens / calibrationRatio)
        : currentInstructionTokens;

    const {
      context: initialContext,
      thinkingStartIndex,
      messagesToRefine,
      remainingContextTokens: initialRemainingContextTokens,
    } = getMessagesWithinTokenLimit({
      maxContextTokens: rawSpaceBudget,
      messages: params.messages,
      indexTokenCountMap,
      startType: params.startType,
      thinkingEnabled: factoryParams.thinkingEnabled,
      tokenCounter: factoryParams.tokenCounter,
      instructionTokens: rawSpaceInstructionTokens,
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
    // Fallback fading: when summarization skipped fading earlier and
    // pruning still produced an empty context, apply lossy pressure-band
    // fading and retry.  This is a last resort before emergency truncation
    // — the summarizer already saw the full messages, so fading the
    // surviving context for the LLM is acceptable.
    // ---------------------------------------------------------------
    if (
      context.length === 0 &&
      params.messages.length > 0 &&
      effectiveMaxTokens > 0 &&
      factoryParams.summarizationEnabled === true
    ) {
      const fadingBudget = Math.max(1024, effectiveMaxTokens);

      factoryParams.log?.(
        'debug',
        'Fallback fading — empty context with summarization',
        {
          messageCount: params.messages.length,
          effectiveMaxTokens,
          fadingBudget,
        }
      );

      const fadedMessages = [...params.messages];
      const preFadingTokenCounts: Record<string, number | undefined> = {};
      for (let i = 0; i < params.messages.length; i++) {
        preFadingTokenCounts[i] = indexTokenCountMap[i];
      }

      preFlightTruncateToolResults({
        messages: fadedMessages,
        maxContextTokens: fadingBudget,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });
      preFlightTruncateToolCallInputs({
        messages: fadedMessages,
        maxContextTokens: fadingBudget,
        indexTokenCountMap,
        tokenCounter: factoryParams.tokenCounter,
      });

      const fadingRetry = getMessagesWithinTokenLimit({
        maxContextTokens: pruningBudget,
        messages: fadedMessages,
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

      const fadingRepaired = repairOrphanedToolMessages({
        context: fadingRetry.context,
        allMessages: fadedMessages,
        tokenCounter: factoryParams.tokenCounter,
        indexTokenCountMap,
      });

      if (fadingRepaired.context.length > 0) {
        context = fadingRepaired.context;
        reclaimedTokens = fadingRepaired.reclaimedTokens;
        messagesToRefine.push(...fadingRetry.messagesToRefine);
        if (fadingRepaired.droppedMessages.length > 0) {
          messagesToRefine.push(...fadingRepaired.droppedMessages);
        }

        factoryParams.log?.('debug', 'Fallback fading recovered context', {
          contextLength: context.length,
          messagesToRefineCount: messagesToRefine.length,
          remainingTokens: fadingRetry.remainingContextTokens,
        });

        for (const [key, value] of Object.entries(preFadingTokenCounts)) {
          indexTokenCountMap[key] = value;
        }
      } else {
        for (const [key, value] of Object.entries(preFadingTokenCounts)) {
          indexTokenCountMap[key] = value;
        }
      }
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
      const perMessageTokenBudget = Math.floor(
        effectiveMaxTokens / Math.max(1, params.messages.length)
      );
      const emergencyMaxChars = Math.max(200, perMessageTokenBudget * 4);

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

      try {
        let emergencyTruncatedCount = 0;
        for (let i = 0; i < emergencyMessages.length; i++) {
          const message = emergencyMessages[i];
          if (message.getType() === 'tool') {
            const content = message.content;
            if (
              typeof content === 'string' &&
              content.length > emergencyMaxChars
            ) {
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
            }
          }
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
                    // Replaces original input with { _truncated, _originalChars } —
                    // safe because the tool call already executed in a prior turn.
                    return {
                      ...record,
                      input: truncateToolInput(serialized, emergencyMaxChars),
                    };
                  }
                }
                return block;
              });
              const newToolCalls = (aiMsg.tool_calls ?? []).map((tc) => {
                const serializedArgs = JSON.stringify(tc.args);
                if (serializedArgs.length > emergencyMaxChars) {
                  // Replaces original args with { _truncated, _originalChars } —
                  // safe because the tool call already executed in a prior turn.
                  return {
                    ...tc,
                    args: truncateToolInput(serializedArgs, emergencyMaxChars),
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
            }
          }
        }

        factoryParams.log?.('info', 'Emergency truncation complete');
        factoryParams.log?.('debug', 'Emergency truncation details', {
          truncatedCount: emergencyTruncatedCount,
          emergencyMaxChars,
        });

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

        factoryParams.log?.('debug', 'Emergency truncation retry result', {
          contextLength: context.length,
          messagesToRefineCount: messagesToRefine.length,
          remainingTokens: retryResult.remainingContextTokens,
        });
      } finally {
        // Restore the closure's indexTokenCountMap to pre-emergency values so the
        // next turn counts old messages at their original (un-truncated) size.
        // The emergency-truncated counts were only needed for this turn's
        // getMessagesWithinTokenLimit retry.
        for (const [key, value] of Object.entries(preEmergencyTokenCounts)) {
          indexTokenCountMap[key] = value;
        }
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
      prePruneContextTokens: calibratedTotalTokens,
      remainingContextTokens,
      contextPressure,
      originalToolContent:
        originalToolContent.size > 0 ? originalToolContent : undefined,
      calibrationRatio,
      resolvedInstructionOverhead: bestInstructionOverhead,
    };
  };
}
