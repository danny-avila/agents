import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import { ContentTypes } from '@/common';

/**
 * Whether {@link formatContentStrings} will flatten this message's content:
 * a human/ai/system message whose content is an array of text-only blocks.
 */
export const isLegacyConvertible = (message: BaseMessage): boolean => {
  const messageType = message.getType();
  const isValidMessage =
    messageType === 'human' || messageType === 'ai' || messageType === 'system';
  if (!isValidMessage) {
    return false;
  }
  if (!Array.isArray(message.content)) {
    return false;
  }
  return message.content.every((block) => block.type === ContentTypes.TEXT);
};

/**
 * Formats an array of messages for LangChain, making sure all content fields are strings
 * @param {Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>} payload - The array of messages to format.
 * @returns {Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>} - The array of formatted LangChain messages, including ToolMessages for tool calls.
 */
export const formatContentStrings = (
  payload: Array<BaseMessage>
): Array<BaseMessage> => {
  // Create a new array to store the processed messages
  const result: Array<BaseMessage> = [];

  for (const message of payload) {
    if (!isLegacyConvertible(message)) {
      result.push(message);
      continue;
    }

    // Reduce text types to a single string
    const blocks = message.content as MessageContentComplex[];
    const content = blocks.reduce((acc, curr) => {
      if (curr.type === ContentTypes.TEXT) {
        return `${acc}${curr[ContentTypes.TEXT] || ''}\n`;
      }
      return acc;
    }, '');

    message.content = content.trim();
    result.push(message);
  }

  return result;
};
