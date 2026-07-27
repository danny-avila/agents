// src/messages/alternation.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, MessageContent } from '@langchain/core/messages';
import { Providers } from '@/common';

/**
 * Providers whose APIs require strict user/assistant alternation and reject
 * consecutive user turns. Anthropic's Messages API, OpenAI and Gemini all
 * accept them, so they are deliberately absent.
 */
export const strictAlternationProviders: ReadonlySet<Providers> = new Set([
  Providers.BEDROCK,
  Providers.MISTRAL,
  Providers.MISTRALAI,
]);

const TOOL_RESULT_TYPES = new Set(['tool_result', 'toolResult']);

/**
 * True when every block is a tool result. Both vendored converters already
 * merge adjacent runs of these, and folding one into a text turn would break
 * the tool pairing they depend on — so they are left alone here.
 */
function isToolResultMessage(message: BaseMessage): boolean {
  const { content } = message;
  if (typeof content === 'string' || content.length === 0) {
    return false;
  }
  return content.every(
    (block) =>
      typeof block.type === 'string' && TOOL_RESULT_TYPES.has(block.type)
  );
}

function toBlocks(content: MessageContent): Exclude<MessageContent, string> {
  if (typeof content === 'string') {
    return content === '' ? [] : [{ type: 'text', text: content }];
  }
  return content;
}

function joinContent(
  left: MessageContent,
  right: MessageContent
): MessageContent {
  if (typeof left === 'string' && typeof right === 'string') {
    return left === '' ? right : `${left}\n\n${right}`;
  }
  return [...toBlocks(left), ...toBlocks(right)];
}

/**
 * Merges runs of consecutive human turns into one, for providers that reject
 * them. Purely a wire-shaping pass: it returns a new array of new messages,
 * so graph state and the host's persisted messages keep the per-message
 * identity that steer rendering and the trailing-steer anchor rely on.
 *
 * Tool-result turns are excluded — the converters merge those themselves, and
 * combining one with a text turn would orphan the pairing.
 */
export function coalesceAdjacentUserTurns(
  messages: BaseMessage[]
): BaseMessage[] {
  const result: BaseMessage[] = [];
  for (const message of messages) {
    const previous = result[result.length - 1];
    const mergeable =
      result.length > 0 &&
      previous.getType() === 'human' &&
      message.getType() === 'human' &&
      !isToolResultMessage(previous) &&
      !isToolResultMessage(message);

    if (!mergeable) {
      result.push(message);
      continue;
    }

    result[result.length - 1] = new HumanMessage({
      content: joinContent(previous.content, message.content),
      additional_kwargs: previous.additional_kwargs,
      ...(previous.id != null && { id: previous.id }),
    });
  }
  return result;
}
