import { BaseMessage, MessageContentComplex } from '@langchain/core/messages';
import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

/**
 * Deep clones a message's content to prevent mutation of the original.
 * Handles both string and array content types.
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
 * Creates a shallow clone of a message with deep-cloned content.
 * This ensures modifications to content don't affect the original message.
 */
function cloneMessageWithContent<T extends MessageWithContent>(message: T): T {
  if (message.content === undefined) {
    return { ...message };
  }

  const clonedContent = deepCloneContent(message.content);
  const cloned = {
    ...message,
    content: clonedContent,
  };

  /**
   * LangChain messages store internal state in lc_kwargs.
   * Clone it but don't sync content yet - that happens after all modifications.
   */
  const lcKwargs = (message as Record<string, unknown>).lc_kwargs;
  if (lcKwargs != null && typeof lcKwargs === 'object') {
    (cloned as Record<string, unknown>).lc_kwargs = { ...lcKwargs };
  }

  return cloned;
}

/**
 * Syncs lc_kwargs.content with the message's content property.
 * Call this after all modifications to ensure LangChain serialization works correctly.
 */
function syncLcKwargsContent<T extends MessageWithContent>(message: T): void {
  const lcKwargs = (message as Record<string, unknown>).lc_kwargs;
  if (lcKwargs != null && typeof lcKwargs === 'object') {
    (lcKwargs as Record<string, unknown>).content = message.content;
  }
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
    const isUserMessage =
      ('getType' in originalMessage && originalMessage.getType() === 'human') ||
      ('role' in originalMessage && originalMessage.role === 'user');

    const hasArrayContent = Array.isArray(originalMessage.content);
    const needsStripping =
      hasArrayContent &&
      needsCacheStripping(originalMessage.content as MessageContentComplex[]);
    const needsCacheAdd =
      userMessagesModified < 2 &&
      isUserMessage &&
      (typeof originalMessage.content === 'string' || hasArrayContent);

    if (!needsStripping && !needsCacheAdd) {
      continue;
    }

    const message = cloneMessageWithContent(
      originalMessage as MessageWithContent
    ) as T;
    updatedMessages[i] = message;

    if (hasArrayContent) {
      message.content = (message.content as MessageContentComplex[]).filter(
        (block) => !isCachePoint(block as MessageContentComplex)
      ) as typeof message.content;

      for (
        let j = 0;
        j < (message.content as MessageContentComplex[]).length;
        j++
      ) {
        const block = (message.content as MessageContentComplex[])[j] as Record<
          string,
          unknown
        >;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    }

    if (userMessagesModified >= 2 || !isUserMessage) {
      syncLcKwargsContent(message);
      continue;
    }

    if (typeof message.content === 'string') {
      message.content = [
        {
          type: 'text',
          text: message.content,
          cache_control: { type: 'ephemeral' },
        },
      ];
      userMessagesModified++;
    } else if (Array.isArray(message.content)) {
      for (let j = message.content.length - 1; j >= 0; j--) {
        const contentPart = message.content[j];
        if ('type' in contentPart && contentPart.type === 'text') {
          (contentPart as Anthropic.TextBlockParam).cache_control = {
            type: 'ephemeral',
          };
          userMessagesModified++;
          break;
        }
      }
    }

    syncLcKwargsContent(message);
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

    const message = cloneMessageWithContent(originalMessage);
    updatedMessages[i] = message;

    for (
      let j = 0;
      j < (message.content as MessageContentComplex[]).length;
      j++
    ) {
      const block = (message.content as MessageContentComplex[])[j] as Record<
        string,
        unknown
      >;
      if ('cache_control' in block) {
        delete block.cache_control;
      }
    }
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

    const message = cloneMessageWithContent(originalMessage);
    updatedMessages[i] = message;

    message.content = (message.content as MessageContentComplex[]).filter(
      (block) => !isCachePoint(block as MessageContentComplex)
    ) as typeof content;
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

    const message = cloneMessageWithContent(originalMessage);
    updatedMessages[i] = message;

    if (hasArrayContent) {
      message.content = (message.content as MessageContentComplex[]).filter(
        (block) => !isCachePoint(block)
      ) as typeof content;

      for (
        let j = 0;
        j < (message.content as MessageContentComplex[]).length;
        j++
      ) {
        const block = (message.content as MessageContentComplex[])[j] as Record<
          string,
          unknown
        >;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    }

    if (messagesModified >= 2 || isToolMessage || isEmptyString) {
      syncLcKwargsContent(message);
      continue;
    }

    if (typeof message.content === 'string') {
      message.content = [
        { type: ContentTypes.TEXT, text: message.content },
        { cachePoint: { type: 'default' } },
      ] as MessageContentComplex[];
      messagesModified++;
      syncLcKwargsContent(message);
      continue;
    }

    if (Array.isArray(message.content)) {
      let hasCacheableContent = false;
      for (const block of message.content) {
        if (block.type === ContentTypes.TEXT) {
          if (typeof block.text === 'string' && block.text !== '') {
            hasCacheableContent = true;
            break;
          }
        }
      }

      if (!hasCacheableContent) {
        syncLcKwargsContent(message);
        continue;
      }

      let inserted = false;
      for (let j = message.content.length - 1; j >= 0; j--) {
        const block = message.content[j] as MessageContentComplex;
        const type = (block as { type?: string }).type;
        if (type === ContentTypes.TEXT || type === 'text') {
          const text = (block as { text?: string }).text;
          if (text === '' || text === undefined) {
            continue;
          }
          message.content.splice(j + 1, 0, {
            cachePoint: { type: 'default' },
          } as MessageContentComplex);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        message.content.push({
          cachePoint: { type: 'default' },
        } as MessageContentComplex);
      }
      messagesModified++;
    }

    syncLcKwargsContent(message);
  }

  return updatedMessages;
}
