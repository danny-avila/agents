import { BaseMessage, MessageContentComplex } from '@langchain/core/messages';
import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

/**
 * Deep clones a message's content to prevent mutation of the original.
 */
function deepCloneContent<T extends string | MessageContentComplex[]>(
  content: T
): T {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((block) => ({ ...block })) as T;
  }
  return content;
}

/**
 * Clones a message with deep-cloned content, explicitly excluding LangChain
 * serialization metadata to prevent coercion issues.
 */
function cloneMessage<T extends MessageWithContent>(
  message: T,
  content: string | MessageContentComplex[]
): T {
  const {
    lc_kwargs: _lc_kwargs,
    lc_serializable: _lc_serializable,
    lc_namespace: _lc_namespace,
    ...rest
  } = message as T & {
    lc_kwargs?: unknown;
    lc_serializable?: unknown;
    lc_namespace?: unknown;
  };

  const cloned = { ...rest, content } as T;

  // LangChain messages don't have a direct 'role' property - derive it from getType()
  if (
    'getType' in message &&
    typeof message.getType === 'function' &&
    !('role' in cloned)
  ) {
    const msgType = (message as unknown as BaseMessage).getType();
    const roleMap: Record<string, string> = {
      human: 'user',
      ai: 'assistant',
      system: 'system',
      tool: 'tool',
    };
    (cloned as Record<string, unknown>).role = roleMap[msgType] || msgType;
  }

  return cloned;
}

/**
 * Checks if a message's content needs cache control stripping.
 * Returns true if content has cachePoint blocks or cache_control fields.
 */
function needsCacheStripping(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (isCachePoint(block)) return true;
    if ('cache_control' in block) return true;
  }
  return false;
}

/**
 * Anthropic API: Adds cache control to the appropriate user messages in the payload.
 * Strips ALL existing cache control (both Anthropic and Bedrock formats) from all messages,
 * then adds fresh cache control to the last 2 user messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache control added.
 */
export function addCacheControl<T extends AnthropicMessage | BaseMessage>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let userMessagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;
    const isUserMessage =
      ('getType' in originalMessage && originalMessage.getType() === 'human') ||
      ('role' in originalMessage && originalMessage.role === 'user');

    const hasArrayContent = Array.isArray(content);
    const needsStripping =
      hasArrayContent &&
      needsCacheStripping(content as MessageContentComplex[]);
    const needsCacheAdd =
      userMessagesModified < 2 &&
      isUserMessage &&
      (typeof content === 'string' || hasArrayContent);

    if (!needsStripping && !needsCacheAdd) {
      continue;
    }

    let workingContent: MessageContentComplex[];

    if (hasArrayContent) {
      workingContent = deepCloneContent(
        content as MessageContentComplex[]
      ).filter((block) => !isCachePoint(block as MessageContentComplex));

      for (let j = 0; j < workingContent.length; j++) {
        const block = workingContent[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    } else if (typeof content === 'string') {
      workingContent = [
        { type: 'text', text: content },
      ] as MessageContentComplex[];
    } else {
      workingContent = [];
    }

    if (userMessagesModified >= 2 || !isUserMessage) {
      updatedMessages[i] = cloneMessage(
        originalMessage as MessageWithContent,
        workingContent
      ) as T;
      continue;
    }

    for (let j = workingContent.length - 1; j >= 0; j--) {
      const contentPart = workingContent[j];
      if ('type' in contentPart && contentPart.type === 'text') {
        (contentPart as Anthropic.TextBlockParam).cache_control = {
          type: 'ephemeral',
        };
        userMessagesModified++;
        break;
      }
    }

    updatedMessages[i] = cloneMessage(
      originalMessage as MessageWithContent,
      workingContent
    ) as T;
  }

  return updatedMessages;
}

/**
 * Checks if a content block is a cache point
 */
function isCachePoint(block: MessageContentComplex): boolean {
  return 'cachePoint' in block && !('type' in block);
}

/**
 * Checks if a message's content has Anthropic cache_control fields.
 */
function hasAnthropicCacheControl(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    if ('cache_control' in content[i]) return true;
  }
  return false;
}

/**
 * Removes all Anthropic cache_control fields from messages
 * Used when switching from Anthropic to Bedrock provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripAnthropicCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasAnthropicCacheControl(content)) {
      continue;
    }

    const clonedContent = deepCloneContent(content);
    for (let j = 0; j < clonedContent.length; j++) {
      const block = clonedContent[j] as Record<string, unknown>;
      if ('cache_control' in block) {
        delete block.cache_control;
      }
    }
    updatedMessages[i] = cloneMessage(originalMessage, clonedContent);
  }

  return updatedMessages;
}

/**
 * Checks if a message's content has Bedrock cachePoint blocks.
 */
function hasBedrockCachePoint(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    if (isCachePoint(content[i])) return true;
  }
  return false;
}

/**
 * Removes all Bedrock cachePoint blocks from messages
 * Used when switching from Bedrock to Anthropic provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripBedrockCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasBedrockCachePoint(content)) {
      continue;
    }

    const clonedContent = deepCloneContent(content).filter(
      (block) => !isCachePoint(block as MessageContentComplex)
    );
    updatedMessages[i] = cloneMessage(originalMessage, clonedContent);
  }

  return updatedMessages;
}

/**
 * Adds Bedrock Converse API cache points to the last two messages.
 * Inserts `{ cachePoint: { type: 'default' } }` as a separate content block
 * immediately after the last text block in each targeted message.
 * Strips ALL existing cache control (both Bedrock and Anthropic formats) from all messages,
 * then adds fresh cache points to the last 2 messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache points added.
 */
export function addBedrockCacheControl<
  T extends Partial<BaseMessage> & MessageWithContent,
>(messages: T[]): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let messagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const isToolMessage =
      'getType' in originalMessage &&
      typeof originalMessage.getType === 'function' &&
      originalMessage.getType() === 'tool';

    const content = originalMessage.content;
    const hasArrayContent = Array.isArray(content);
    const needsStripping =
      hasArrayContent &&
      needsCacheStripping(content as MessageContentComplex[]);
    const isEmptyString = typeof content === 'string' && content === '';
    const needsCacheAdd =
      messagesModified < 2 &&
      !isToolMessage &&
      !isEmptyString &&
      (typeof content === 'string' || hasArrayContent);

    if (!needsStripping && !needsCacheAdd) {
      continue;
    }

    let workingContent: MessageContentComplex[];

    if (hasArrayContent) {
      workingContent = deepCloneContent(
        content as MessageContentComplex[]
      ).filter((block) => !isCachePoint(block));

      for (let j = 0; j < workingContent.length; j++) {
        const block = workingContent[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    } else if (typeof content === 'string') {
      workingContent = [{ type: ContentTypes.TEXT, text: content }];
    } else {
      workingContent = [];
    }

    if (messagesModified >= 2 || isToolMessage || isEmptyString) {
      updatedMessages[i] = cloneMessage(originalMessage, workingContent);
      continue;
    }

    if (workingContent.length === 0) {
      continue;
    }

    let hasCacheableContent = false;
    for (const block of workingContent) {
      if (block.type === ContentTypes.TEXT) {
        if (typeof block.text === 'string' && block.text !== '') {
          hasCacheableContent = true;
          break;
        }
      }
    }

    if (!hasCacheableContent) {
      updatedMessages[i] = cloneMessage(originalMessage, workingContent);
      continue;
    }

    let inserted = false;
    for (let j = workingContent.length - 1; j >= 0; j--) {
      const block = workingContent[j] as MessageContentComplex;
      const type = (block as { type?: string }).type;
      if (type === ContentTypes.TEXT || type === 'text') {
        const text = (block as { text?: string }).text;
        if (text === '' || text === undefined) {
          continue;
        }
        workingContent.splice(j + 1, 0, {
          cachePoint: { type: 'default' },
        } as MessageContentComplex);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      workingContent.push({
        cachePoint: { type: 'default' },
      } as MessageContentComplex);
    }

    updatedMessages[i] = cloneMessage(originalMessage, workingContent);
    messagesModified++;
  }

  return updatedMessages;
}
