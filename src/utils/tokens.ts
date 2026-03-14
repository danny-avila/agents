import { Tokenizer } from 'ai-tokenizer';
import type { BaseMessage } from '@langchain/core/messages';
import { ContentTypes } from '@/common/enum';

export type EncodingName = 'o200k_base' | 'claude';

const tokenizers: Partial<Record<EncodingName, Tokenizer>> = {};

async function getTokenizer(
  encoding: EncodingName = 'o200k_base'
): Promise<Tokenizer> {
  const cached = tokenizers[encoding];
  if (cached) {
    return cached;
  }
  const data =
    encoding === 'claude'
      ? await import('ai-tokenizer/encoding/claude')
      : await import('ai-tokenizer/encoding/o200k_base');
  const instance = new Tokenizer(data);
  tokenizers[encoding] = instance;
  return instance;
}

export function encodingForModel(model: string): EncodingName {
  if (model.toLowerCase().includes('claude')) {
    return 'claude';
  }
  return 'o200k_base';
}

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

/**
 * Anthropic's API consistently reports ~10% more tokens than the local
 * claude tokenizer due to internal message framing and content encoding.
 * Verified empirically across content types via the count_tokens endpoint.
 */
const CLAUDE_TOKEN_CORRECTION = 1.1;

/**
 * Creates a token counter function using the specified encoding.
 * Lazily loads the encoding data on first use via dynamic import.
 */
export const createTokenCounter = async (
  encoding: EncodingName = 'o200k_base'
): Promise<(message: BaseMessage) => number> => {
  const tok = await getTokenizer(encoding);
  const countTokens = (text: string): number => tok.count(text);
  const isClaude = encoding === 'claude';
  return (message: BaseMessage): number => {
    const count = getTokenCountForMessage(message, countTokens);
    return isClaude ? Math.ceil(count * CLAUDE_TOKEN_CORRECTION) : count;
  };
};

/** Utility to manage the token encoder lifecycle explicitly. */
export const TokenEncoderManager = {
  async initialize(): Promise<void> {
    // No-op: ai-tokenizer is synchronously initialized from bundled data.
  },

  reset(): void {
    for (const key of Object.keys(tokenizers)) {
      delete tokenizers[key as EncodingName];
    }
  },

  isInitialized(): boolean {
    return Object.keys(tokenizers).length > 0;
  },
};
