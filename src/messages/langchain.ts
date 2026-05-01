import type { MessageContent } from '@langchain/core/messages';
import type * as t from '@/types';

type LibreChatMessageContent =
  | MessageContent
  | string
  | t.MessageContentComplex[]
  | t.ExtendedMessageContent[];

export function toLangChainContent(
  content: LibreChatMessageContent
): MessageContent {
  return content as MessageContent;
}
