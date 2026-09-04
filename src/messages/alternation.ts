// src/messages/alternation.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, MessageContent } from '@langchain/core/messages';
import type { ProviderMessageProvenancePart } from './provenance';
import type { ProviderToolCallIndex } from './toolResultTypes';
import {
  appendProviderToolCallDescriptor,
  consumeProviderToolResultPair,
  getBoundedProviderPairingArrayProperty,
  getProviderAIMessageToolCallDescriptor,
  getProviderToolCallPartDescriptor,
  getProviderToolResultPartDescriptor,
} from './toolResultTypes';
import {
  inspectProviderMessageProvenance,
  inspectProviderSourceMessageIds,
  setInvalidProviderMessageProvenance,
  setProviderMessageProvenance,
} from './provenance';
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

/**
 * True when every block is a tool result. Both vendored converters already
 * merge adjacent runs of these, and folding one into a text turn would break
 * the tool pairing they depend on — so they are left alone here.
 */
function collectProviderToolCalls(message: BaseMessage): ProviderToolCallIndex {
  const calls: ProviderToolCallIndex = new Map();
  const content = getBoundedProviderPairingArrayProperty(message, 'content');
  if (content != null) {
    for (let index = 0; index < content.length; index++) {
      const descriptor = getProviderToolCallPartDescriptor(content[index]);
      if (descriptor != null) {
        appendProviderToolCallDescriptor(calls, descriptor);
      }
    }
  }
  const toolCalls = getBoundedProviderPairingArrayProperty(
    message,
    'tool_calls'
  );
  if (toolCalls != null) {
    for (let index = 0; index < toolCalls.length; index++) {
      const descriptor = getProviderAIMessageToolCallDescriptor(
        toolCalls[index]
      );
      if (descriptor != null) {
        appendProviderToolCallDescriptor(calls, descriptor);
      }
    }
  }
  return calls;
}

function isToolResultMessage(
  message: BaseMessage,
  pairedToolCalls: ProviderToolCallIndex
): boolean {
  const content = getBoundedProviderPairingArrayProperty(message, 'content');
  if (content == null || content.length === 0) {
    return false;
  }
  for (let index = 0; index < content.length; index++) {
    const descriptor = getProviderToolResultPartDescriptor(content[index]);
    if (
      descriptor?.allowHumanMessagePairing !== true ||
      !consumeProviderToolResultPair(descriptor, pairedToolCalls)
    ) {
      return false;
    }
  }
  return true;
}

function toBlocks(content: MessageContent): Exclude<MessageContent, string> {
  if (typeof content === 'string') {
    return content === '' ? [] : [{ type: 'text', text: content }];
  }
  return content;
}

function joinStringContents(
  messages: readonly BaseMessage[],
  endIndex = messages.length
): string {
  const contents: string[] = [];
  for (let index = 0; index < endIndex; index++) {
    const content = messages[index].content as string;
    if (contents.length === 0 && content === '') {
      continue;
    }
    contents.push(content);
  }
  return contents.join('\n\n');
}

function joinContents(messages: readonly BaseMessage[]): MessageContent {
  const firstArrayIndex = messages.findIndex((message) =>
    Array.isArray(message.content)
  );
  if (firstArrayIndex === -1) {
    return joinStringContents(messages);
  }

  const prefix = joinStringContents(messages, firstArrayIndex);
  const blocks: Exclude<MessageContent, string> = [];
  const appendBlocks = (content: MessageContent): void => {
    const contentBlocks = toBlocks(content);
    for (let index = 0; index < contentBlocks.length; index++) {
      blocks.push(contentBlocks[index]);
    }
  };
  appendBlocks(prefix);
  for (let index = firstArrayIndex; index < messages.length; index++) {
    appendBlocks(messages[index].content);
  }
  return blocks;
}

function getHumanMessageProvenanceParts(
  message: BaseMessage
): ProviderMessageProvenancePart[] | null {
  const provenanceState = inspectProviderMessageProvenance(message);
  const sourceMessageIdsState = inspectProviderSourceMessageIds(message);
  if (
    provenanceState.status === 'invalid' ||
    sourceMessageIdsState.status === 'invalid'
  ) {
    return null;
  }
  const sourceMessageIds =
    sourceMessageIdsState.status === 'valid'
      ? sourceMessageIdsState.sourceMessageIds
      : [];
  if (provenanceState.status === 'valid') {
    const parts = [...provenanceState.provenance.parts];
    const representedSourceIds = new Set<string>();
    for (const part of parts) {
      if (part.sourceMessageId != null) {
        representedSourceIds.add(part.sourceMessageId);
      }
    }
    const missingSourceIds = sourceMessageIds.filter(
      (sourceMessageId) => !representedSourceIds.has(sourceMessageId)
    );
    if (
      parts.length === 1 &&
      parts[0].sourceMessageId == null &&
      missingSourceIds.length === 1
    ) {
      return missingSourceIds.map((sourceMessageId) => ({
        ...parts[0],
        sourceMessageId,
      }));
    }
    for (const sourceMessageId of missingSourceIds) {
      parts.push({ attribution: 'user', sourceMessageId });
    }
    return parts;
  }
  return sourceMessageIds.length > 0
    ? sourceMessageIds.map((sourceMessageId) => ({
      attribution: 'user' as const,
      sourceMessageId,
    }))
    : [{ attribution: 'user' }];
}

function hasProviderVisibleContent(message: BaseMessage): boolean {
  return typeof message.content === 'string'
    ? message.content.length > 0
    : message.content.length > 0;
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
  let result: BaseMessage[] | null = null;
  let pairedToolCalls: ProviderToolCallIndex = new Map();
  let index = 0;
  while (index < messages.length) {
    const first = messages[index];
    if (first.getType() === 'ai') {
      pairedToolCalls = collectProviderToolCalls(first);
      result?.push(first);
      index++;
      continue;
    }
    if (first.getType() !== 'human') {
      result?.push(first);
      index++;
      continue;
    }
    if (isToolResultMessage(first, pairedToolCalls)) {
      result?.push(first);
      index++;
      continue;
    }

    let endIndex = index + 1;
    while (
      endIndex < messages.length &&
      messages[endIndex].getType() === 'human' &&
      !isToolResultMessage(messages[endIndex], pairedToolCalls)
    ) {
      endIndex++;
    }
    if (endIndex === index + 1) {
      result?.push(first);
      pairedToolCalls = new Map();
      index = endIndex;
      continue;
    }
    result ??= messages.slice(0, index);
    const run = messages.slice(index, endIndex);
    const last = run[run.length - 1];
    const provenanceParts: ProviderMessageProvenancePart[] = [];
    let hasInvalidProvenance = false;
    for (const message of run) {
      if (!hasProviderVisibleContent(message)) {
        continue;
      }
      const messageProvenanceParts = getHumanMessageProvenanceParts(message);
      if (messageProvenanceParts == null) {
        hasInvalidProvenance = true;
        continue;
      }
      for (const part of messageProvenanceParts) {
        provenanceParts.push(part);
      }
    }
    if (provenanceParts.length === 0) {
      provenanceParts.push({ attribution: 'user' });
    }

    const mergedMessage = new HumanMessage({
      content: joinContents(run),
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
      additional_kwargs: { ...last.additional_kwargs },
      ...(first.id != null && { id: first.id }),
    });
    if (hasInvalidProvenance) {
      setInvalidProviderMessageProvenance(mergedMessage);
    } else {
      setProviderMessageProvenance(mergedMessage, provenanceParts);
    }
    result.push(mergedMessage);
    pairedToolCalls = new Map();
    index = endIndex;
  }
  /**
   * Identity on the no-merge path. The pass runs twice for a primary
   * Bedrock/Mistral call — once in `createCallModel` (must precede the cache
   * breakpoint) and once in the `attemptInvoke` funnel (must cover fallback
   * and summarization sends) — so the already-normalized second pass returns
   * the SAME array rather than reallocating a context-sized copy, and
   * callers can cheaply detect "nothing changed" by identity.
   */
  return result ?? messages;
}
