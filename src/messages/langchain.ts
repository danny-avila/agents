import type { MessageContent } from '@langchain/core/messages';
import type * as t from '@/types';

type LibreChatMessageContent =
  | MessageContent
  | string
  | t.MessageContentComplex[]
  | t.ExtendedMessageContent[];

type WithLangChainContent<T extends { content: LibreChatMessageContent }> =
  Omit<T, 'content'> & {
    content: MessageContent;
  };

/**
 * Bridges LibreChat's extended content blocks to LangChain 1.x MessageContent.
 *
 * LangChain 1.x narrowed message constructor types around ContentBlock, while
 * LibreChat still carries provider-specific blocks through the same content
 * field. This helper keeps the runtime shape unchanged during the dependency
 * upgrade; tracking issue: https://github.com/danny-avila/agents/issues/130.
 */
export function toLangChainContent(
  content: LibreChatMessageContent
): MessageContent {
  return content as MessageContent;
}

/**
 * Applies the same LangChain 1.x content bridge to message constructor fields.
 *
 * Keep this cast-only helper local to constructor boundaries so follow-up work
 * can replace it with aligned content types or explicit conversion logic.
 */
export function toLangChainMessageFields<
  T extends { content: LibreChatMessageContent },
>(message: T): WithLangChainContent<T> {
  return message as WithLangChainContent<T>;
}
