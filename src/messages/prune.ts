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
  /**
   * Returns the current instruction-token overhead (system message + tool schemas + summary).
   * Called on each prune invocation so the budget reflects dynamic changes
   * (e.g. summary added between turns).  When messages don't include a leading
   * SystemMessage, these tokens are subtracted from the available budget so
   * the pruner correctly reserves space for the system prompt that will be
   * prepended later by `buildSystemRunnable`.
   */
  getInstructionTokens?: () => number;
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
 * Truncates an oversized tool_use `input` field, preserving it as a valid JSON object.
 * The truncated result contains a `_truncated` key with a head-prefix of the
 * serialized original, plus a `_originalChars` field for visibility.
 */
function truncateToolInput(
  input: unknown,
  maxChars: number
): { _truncated: string; _originalChars: number } {
  const serialized = typeof input === 'string' ? input : JSON.stringify(input);
  const head = serialized.slice(0, maxChars);
  return {
    _truncated:
      head + `\n… [truncated: ${serialized.length} → ${maxChars} chars]`,
    _originalChars: serialized.length,
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
    const effectiveMaxTokens = Math.max(
      0,
      factoryParams.maxTokens - currentInstructionTokens
    );

    // Pre-flight truncation: truncate oversized tool results before pruning.
    // Uses effectiveMaxTokens (after instruction overhead) so thresholds reflect
    // the real budget available for messages.
    preFlightTruncateToolResults({
      messages: params.messages,
      maxContextTokens: effectiveMaxTokens,
      indexTokenCountMap,
      tokenCounter: factoryParams.tokenCounter,
    });

    // Pre-flight truncation: truncate oversized tool_use inputs (args) in AI messages.
    preFlightTruncateToolCallInputs({
      messages: params.messages,
      maxContextTokens: effectiveMaxTokens,
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
    if (
      lastCutOffIndex === 0 &&
      totalTokens + currentInstructionTokens <= factoryParams.maxTokens
    ) {
      return {
        context: params.messages,
        indexTokenCountMap,
        messagesToRefine: [],
        prePruneTotalTokens: totalTokens,
        remainingContextTokens:
          factoryParams.maxTokens - totalTokens - currentInstructionTokens,
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

    let { context, reclaimedTokens } = repairOrphanedToolMessages({
      context: initialContext,
      allMessages: params.messages,
      tokenCounter: factoryParams.tokenCounter,
      indexTokenCountMap,
    });

    // ---------------------------------------------------------------
    // Emergency truncation: if pruning produced an empty context but
    // messages exist, aggressively truncate all tool_call inputs and
    // tool results to a minimal stub, then retry.  This handles the
    // scenario where a single oversized tool_call (e.g. evaluate_script
    // with thousands of chars of JavaScript) exceeds the available
    // budget on its own, making the pruner give up entirely.
    // ---------------------------------------------------------------
    if (
      context.length === 0 &&
      params.messages.length > 0 &&
      effectiveMaxTokens > 0
    ) {
      // Aggressively truncate: tool result content and tool_call inputs
      // are reduced to a minimal stub so the model knows what was called
      // but the payload is removed.
      const EMERGENCY_MAX_CHARS = 150;
      for (let i = 0; i < params.messages.length; i++) {
        const message = params.messages[i];
        // Truncate ToolMessage content
        if (message.getType() === 'tool') {
          const content = message.content;
          if (
            typeof content === 'string' &&
            content.length > EMERGENCY_MAX_CHARS
          ) {
            (message as ToolMessage).content =
              content.slice(0, EMERGENCY_MAX_CHARS) +
              `\n… [emergency truncated: ${content.length} → ${EMERGENCY_MAX_CHARS} chars]`;
            indexTokenCountMap[i] = factoryParams.tokenCounter(message);
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
              return serialized.length > EMERGENCY_MAX_CHARS;
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
                if (serialized.length > EMERGENCY_MAX_CHARS) {
                  return {
                    ...record,
                    input: truncateToolInput(record.input, EMERGENCY_MAX_CHARS),
                  };
                }
              }
              return block;
            });
            const newToolCalls = (aiMsg.tool_calls ?? []).map((tc) => {
              const serializedArgs = JSON.stringify(tc.args);
              if (serializedArgs.length > EMERGENCY_MAX_CHARS) {
                return {
                  ...tc,
                  args: truncateToolInput(tc.args, EMERGENCY_MAX_CHARS),
                };
              }
              return tc;
            });
            params.messages[i] = new AIMessage({
              ...aiMsg,
              content: newContent,
              tool_calls: newToolCalls.length > 0 ? newToolCalls : undefined,
            });
            indexTokenCountMap[i] = factoryParams.tokenCounter(
              params.messages[i]
            );
          }
        }
      }

      // Retry pruning with the emergency-truncated messages
      const retryResult = getMessagesWithinTokenLimit({
        maxContextTokens: factoryParams.maxTokens,
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

      const repaired = repairOrphanedToolMessages({
        context: retryResult.context,
        allMessages: params.messages,
        tokenCounter: factoryParams.tokenCounter,
        indexTokenCountMap,
      });

      context = repaired.context;
      reclaimedTokens = repaired.reclaimedTokens;
      messagesToRefine.push(...retryResult.messagesToRefine);
    }

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
