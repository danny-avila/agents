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
let tokenCounterPromise: Promise<(message: BaseMessage) => number> | undefined;

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

/**
 * Creates a singleton token counter function that reuses the same encoder instance.
 * This avoids creating multiple function closures and prevents potential memory issues.
 */
export const createTokenCounter = async (): Promise<
  (message: BaseMessage) => number
> => {
  if (tokenCounterPromise) {
    return tokenCounterPromise;
  }

  tokenCounterPromise = (async (): Promise<
    (message: BaseMessage) => number
  > => {
    const enc = await getSharedEncoder();
    const countTokens = (text: string): number => enc.encode(text).length;
    return (message: BaseMessage): number =>
      getTokenCountForMessage(message, countTokens);
  })();

  return tokenCounterPromise;
};

/**
 * Utility to manage the token encoder lifecycle explicitly.
 * Useful for applications that need fine-grained control over resource management.
 */
export const TokenEncoderManager = {
  /**
   * Pre-initializes the encoder. This can be called during app startup
   * to avoid lazy loading delays later.
   */
  async initialize(): Promise<void> {
    await getSharedEncoder();
  },

  /**
   * Clears the cached encoder and token counter.
   * Useful for testing or when you need to force a fresh reload.
   */
  reset(): void {
    encoderPromise = undefined;
    tokenCounterPromise = undefined;
  },

  /**
   * Checks if the encoder has been initialized.
   */
  isInitialized(): boolean {
    return encoderPromise !== undefined;
  },
};
