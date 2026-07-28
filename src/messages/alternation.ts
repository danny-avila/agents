// src/messages/alternation.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, MessageContent } from '@langchain/core/messages';
import { Providers } from '@/common';

/**
 * Providers whose APIs specify strict user/assistant alternation. Mistral
 * rejects consecutive user turns outright. Bedrock's Converse API documents
 * the alternation requirement across many model families; enforcement varies
 * by family — Claude on Converse currently tolerates adjacent user turns
 * (verified live, 2026-07-28) — so the payload is normalized for all of them
 * rather than betting on per-family leniency. Anthropic's own Messages API,
 * OpenAI and Gemini all accept consecutive user turns, so they are
 * deliberately absent.
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
  let mergedAny = false;
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
    mergedAny = true;

    result[result.length - 1] = new HumanMessage({
      content: joinContent(previous.content, message.content),
      /**
       * The LATER turn's kwargs, deliberately. The one provider-path consumer
       * of these flags is the prompt-cache tail anchor, and it reasons
       * positionally: `isSyntheticMetaMessage` decides whether a breakpoint
       * may be inserted after the message's LAST text block. Merging keeps
       * the last block's provenance only if the last part's kwargs survive —
       * a skill body absorbed into a real user turn must stay anchorable
       * (the real turn ends it), while a real steer absorbed into a trailing
       * skill body must not pin the cache to the volatile body. The first
       * turn's id is kept so origin tracking can re-attach by key.
       */
      additional_kwargs: message.additional_kwargs,
      ...(previous.id != null && { id: previous.id }),
    });
  }
  /**
   * Identity on the no-merge path. The pass runs twice for a primary
   * Bedrock/Mistral call — once in `createCallModel` (must precede the cache
   * breakpoint) and once in the `attemptInvoke` funnel (must cover fallback
   * and summarization sends) — so the already-normalized second pass returns
   * the SAME array rather than reallocating a context-sized copy, and
   * callers can cheaply detect "nothing changed" by identity.
   */
  return mergedAny ? result : messages;
}
