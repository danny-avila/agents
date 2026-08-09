import type { AssistantTextPhase } from '@/types/assistantPhase';
import type { MessageContentComplex } from '@/types/stream';
import { ContentTypes } from '@/common';

export type MessageCreationContentMetadata = {
  content_type?: ContentTypes.TEXT | ContentTypes.THINK;
  phase?: AssistantTextPhase;
};

function toAssistantTextPhase(value: unknown): AssistantTextPhase | undefined {
  return value === 'commentary' || value === 'final_answer' ? value : undefined;
}

/** Reads both provider-native and LangChain standard-content phase fields. */
export function getAssistantTextPhase(
  contentPart: MessageContentComplex
): AssistantTextPhase | undefined {
  return (
    toAssistantTextPhase(contentPart.phase) ??
    toAssistantTextPhase(contentPart.extras?.phase)
  );
}

function isTextPart(contentPart: MessageContentComplex): boolean {
  return contentPart.type?.startsWith(ContentTypes.TEXT) ?? false;
}

function isReasoningPart(contentPart: MessageContentComplex): boolean {
  return (
    contentPart.type === ContentTypes.THINK ||
    (contentPart.type?.startsWith(ContentTypes.THINKING) ?? false) ||
    (contentPart.type?.startsWith(ContentTypes.REASONING) ?? false) ||
    (contentPart.type?.startsWith(ContentTypes.REASONING_CONTENT) ?? false) ||
    contentPart.type === 'redacted_thinking'
  );
}

/**
 * Derives additive message-creation metadata before a content delta is
 * dispatched. The fallback covers string-only providers whose semantic lane
 * is tracked by the stream handler rather than represented on a block.
 */
export function getMessageCreationContentMetadata(
  content: string | MessageContentComplex[] | undefined,
  fallbackContentType?: ContentTypes.TEXT | ContentTypes.THINK
): MessageCreationContentMetadata {
  if (!Array.isArray(content)) {
    return fallbackContentType == null
      ? {}
      : { content_type: fallbackContentType };
  }
  const textPart = content.find(isTextPart);
  if (textPart != null) {
    const phase = getAssistantTextPhase(textPart);
    return {
      content_type: ContentTypes.TEXT,
      ...(phase == null ? {} : { phase }),
    };
  }
  if (content.some(isReasoningPart)) {
    return { content_type: ContentTypes.THINK };
  }
  return fallbackContentType == null
    ? {}
    : { content_type: fallbackContentType };
}
