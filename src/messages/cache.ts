import { BaseMessage, MessageContentComplex } from '@langchain/core/messages';
import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

/**
 * Anthropic API: Adds cache control to the appropriate user messages in the payload.
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

  for (
    let i = updatedMessages.length - 1;
    i >= 0 && userMessagesModified < 2;
    i--
  ) {
    const message = updatedMessages[i];
    if ('getType' in message && message.getType() !== 'human') {
      continue;
    } else if ('role' in message && message.role !== 'user') {
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
 * Adds Bedrock Converse API cache points to the last two messages.
 * Inserts `{ cachePoint: { type: 'default' } }` as a separate content block
 * immediately after the last text block in each targeted message.
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

  for (
    let i = updatedMessages.length - 1;
    i >= 0 && messagesModified < 2;
    i--
  ) {
    const message = updatedMessages[i];

    if (
      'getType' in message &&
      typeof message.getType === 'function' &&
      message.getType() === 'tool'
    ) {
      continue;
    }

    const content = message.content;

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

    if (Array.isArray(content)) {
      let hasCacheableContent = false;
      for (const block of content) {
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
      for (let j = content.length - 1; j >= 0; j--) {
        const block = content[j] as MessageContentComplex;
        const type = (block as { type?: string }).type;
        if (type === ContentTypes.TEXT || type === 'text') {
          const text = (block as { text?: string }).text;
          if (text === '' || text === undefined) {
            continue;
          }
          content.splice(j + 1, 0, {
            cachePoint: { type: 'default' },
          } as MessageContentComplex);
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        content.push({
          cachePoint: { type: 'default' },
        } as MessageContentComplex);
      }
      messagesModified++;
    }
  }

  return updatedMessages;
}
