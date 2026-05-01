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

export function toLangChainContent(
  content: LibreChatMessageContent
): MessageContent {
  return content as MessageContent;
}

export function toLangChainMessageFields<
  T extends { content: LibreChatMessageContent },
>(message: T): WithLangChainContent<T> {
  return message as WithLangChainContent<T>;
}
