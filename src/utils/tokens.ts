import { Tiktoken } from 'js-tiktoken/lite';
import type { BaseMessage } from '@langchain/core/messages';
import { ContentTypes } from '@/common/enum';

export function getTokenCountForMessage(
  message: BaseMessage,
  getTokenCount: (text: string) => number
): number {
  const tokensPerMessage = 3;

  const processValue = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const item of value) {
        if (
          !item ||
          !item.type ||
          item.type === ContentTypes.ERROR ||
          item.type === ContentTypes.IMAGE_URL
        ) {
          continue;
        }

        if (item.type === ContentTypes.TOOL_CALL && item.tool_call != null) {
          const toolName = item.tool_call?.name || '';
          if (toolName != null && toolName && typeof toolName === 'string') {
            numTokens += getTokenCount(toolName);
          }

          const args = item.tool_call?.args || '';
          if (args != null && args && typeof args === 'string') {
            numTokens += getTokenCount(args);
          }

          const output = item.tool_call?.output || '';
          if (output != null && output && typeof output === 'string') {
            numTokens += getTokenCount(output);
          }
          continue;
        }

        const nestedValue = item[item.type];

        if (!nestedValue) {
          continue;
        }

        processValue(nestedValue);
      }
    } else if (typeof value === 'string') {
      numTokens += getTokenCount(value);
    } else if (typeof value === 'number') {
      numTokens += getTokenCount(value.toString());
    } else if (typeof value === 'boolean') {
      numTokens += getTokenCount(value.toString());
    }
  };

  let numTokens = tokensPerMessage;
  processValue(message.content);
  return numTokens;
}

let encoderPromise: Promise<Tiktoken> | undefined;

async function getSharedEncoder(): Promise<Tiktoken> {
  if (encoderPromise) {
    return encoderPromise;
  }
  encoderPromise = (async (): Promise<Tiktoken> => {
    const res = await fetch('https://tiktoken.pages.dev/js/o200k_base.json');
    const o200k_base = await res.json();
    return new Tiktoken(o200k_base);
  })();
  return encoderPromise;
}

export const createTokenCounter = async () => {
  const enc = await getSharedEncoder();
  const countTokens = (text: string): number => enc.encode(text).length;
  return (message: BaseMessage): number =>
    getTokenCountForMessage(message, countTokens);
};
