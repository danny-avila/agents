import { BaseMessage, MessageContentComplex } from '@langchain/core/messages';
import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

/**
 * Anthropic API: Adds cache control to the appropriate user messages in the payload.
 * Strips ALL existing cache control (both Anthropic and Bedrock formats) from all messages,
 * then adds fresh cache control to the last 2 user messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * @param messages - The array of message objects.
 * @returns - The updated array of message objects with cache control added.
 */
export function addCacheControl<T extends AnthropicMessage | BaseMessage>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages = [...messages];
  let userMessagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const message = updatedMessages[i];
    const isUserMessage =
      ('getType' in message && message.getType() === 'human') ||
      ('role' in message && message.role === 'user');

    if (Array.isArray(message.content)) {
      message.content = message.content.filter(
        (block) => !isCachePoint(block as MessageContentComplex)
      ) as typeof message.content;

      for (let j = 0; j < message.content.length; j++) {
        const block = message.content[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    }

    if (userMessagesModified >= 2 || !isUserMessage) {
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
 * Removes all Anthropic cache_control fields from messages
 * Used when switching from Anthropic to Bedrock provider
 */
export function stripAnthropicCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const message = updatedMessages[i];
    const content = message.content;

    if (Array.isArray(content)) {
      for (let j = 0; j < content.length; j++) {
        const block = content[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    }
  }

  return updatedMessages;
}

/**
 * Removes all Bedrock cachePoint blocks from messages
 * Used when switching from Bedrock to Anthropic provider
 */
export function stripBedrockCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const message = updatedMessages[i];
    const content = message.content;

    if (Array.isArray(content)) {
      message.content = content.filter(
        (block) => !isCachePoint(block as MessageContentComplex)
      ) as typeof content;
    }
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
 * @param messages - The array of message objects.
 * @returns - The updated array of message objects with cache points added.
 */
export function addBedrockCacheControl<
  T extends Partial<BaseMessage> & MessageWithContent,
>(messages: T[]): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = messages.slice();
  let messagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const message = updatedMessages[i];
    const isToolMessage =
      'getType' in message &&
      typeof message.getType === 'function' &&
      message.getType() === 'tool';

    const content = message.content;

    if (Array.isArray(content)) {
      message.content = content.filter(
        (block) => !isCachePoint(block)
      ) as typeof content;

      for (let j = 0; j < message.content.length; j++) {
        const block = message.content[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    }

    if (messagesModified >= 2 || isToolMessage) {
      continue;
    }

    if (typeof content === 'string' && content === '') {
      continue;
    }

    if (typeof content === 'string') {
      message.content = [
        { type: ContentTypes.TEXT, text: content },
        { cachePoint: { type: 'default' } },
      ] as MessageContentComplex[];
      messagesModified++;
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
  }

  return updatedMessages;
}
