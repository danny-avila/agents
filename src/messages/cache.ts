import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { BaseMessage } from '@langchain/core/messages';

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
