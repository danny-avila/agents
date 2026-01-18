import {
  BaseMessage,
  MessageContentComplex,
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { AnthropicMessage } from '@/types/messages';
import type Anthropic from '@anthropic-ai/sdk';
import { ContentTypes } from '@/common/enum';

type MessageWithContent = {
  content?: string | MessageContentComplex[];
};

/**
 * Deep clones a message's content to prevent mutation of the original.
 */
function deepCloneContent<T extends string | MessageContentComplex[]>(
  content: T
): T {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map((block) => ({ ...block })) as T;
  }
  return content;
}

/**
 * Simple shallow clone with deep-cloned content.
 * Used for stripping cache control where we don't need proper LangChain instances.
 */
function shallowCloneMessage<T extends MessageWithContent>(message: T): T {
  return {
    ...message,
    content: deepCloneContent(message.content ?? ''),
  } as T;
}

/**
 * Creates a new LangChain message instance with the given content.
 * Required when adding cache points to ensure proper serialization.
 */
function createNewMessage<T extends MessageWithContent>(
  message: T,
  content: MessageContentComplex[]
): T {
  if ('getType' in message && typeof message.getType === 'function') {
    const baseMsg = message as unknown as BaseMessage;
    const msgType = baseMsg.getType();

    const baseFields = {
      content,
      name: baseMsg.name,
      additional_kwargs: { ...baseMsg.additional_kwargs },
      response_metadata: { ...baseMsg.response_metadata },
      id: baseMsg.id,
    };

    switch (msgType) {
    case 'human':
      return new HumanMessage(baseFields) as unknown as T;
    case 'ai': {
      const aiMsg = baseMsg as AIMessage;
      return new AIMessage({
        ...baseFields,
        tool_calls: aiMsg.tool_calls ? [...aiMsg.tool_calls] : [],
        invalid_tool_calls: aiMsg.invalid_tool_calls
          ? [...aiMsg.invalid_tool_calls]
          : [],
        usage_metadata: aiMsg.usage_metadata,
      }) as unknown as T;
    }
    case 'system':
      return new SystemMessage(baseFields) as unknown as T;
    case 'tool': {
      const toolMsg = baseMsg as ToolMessage;
      return new ToolMessage({
        ...baseFields,
        tool_call_id: toolMsg.tool_call_id,
        status: toolMsg.status,
        artifact: toolMsg.artifact,
      }) as unknown as T;
    }
    default:
      break;
    }
  }

  const cloned = { ...message, content } as T;
  const lcKwargs = (cloned as Record<string, unknown>).lc_kwargs as
    | Record<string, unknown>
    | undefined;
  if (lcKwargs != null) {
    (cloned as Record<string, unknown>).lc_kwargs = { ...lcKwargs, content };
  }
  return cloned;
}

/**
 * Checks if a message's content needs cache control stripping.
 * Returns true if content has cachePoint blocks or cache_control fields.
 */
function needsCacheStripping(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (isCachePoint(block)) return true;
    if ('cache_control' in block) return true;
  }
  return false;
}

/**
 * Anthropic API: Adds cache control to the appropriate user messages in the payload.
 * Strips ALL existing cache control (both Anthropic and Bedrock formats) from all messages,
 * then adds fresh cache control to the last 2 user messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache control added.
 */
export function addCacheControl<T extends AnthropicMessage | BaseMessage>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let userMessagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;
    const isUserMessage =
      ('getType' in originalMessage && originalMessage.getType() === 'human') ||
      ('role' in originalMessage && originalMessage.role === 'user');

    const hasArrayContent = Array.isArray(content);
    const needsStripping =
      hasArrayContent &&
      needsCacheStripping(content as MessageContentComplex[]);
    const needsCacheAdd =
      userMessagesModified < 2 &&
      isUserMessage &&
      (typeof content === 'string' || hasArrayContent);

    if (!needsStripping && !needsCacheAdd) {
      continue;
    }

    let workingContent: MessageContentComplex[];

    if (hasArrayContent) {
      workingContent = deepCloneContent(
        content as MessageContentComplex[]
      ).filter((block) => !isCachePoint(block as MessageContentComplex));

      for (let j = 0; j < workingContent.length; j++) {
        const block = workingContent[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    } else if (typeof content === 'string') {
      workingContent = [
        { type: 'text', text: content },
      ] as MessageContentComplex[];
    } else {
      workingContent = [];
    }

    if (userMessagesModified >= 2 || !isUserMessage) {
      updatedMessages[i] = shallowCloneMessage(
        originalMessage as MessageWithContent
      ) as T;
      (updatedMessages[i] as MessageWithContent).content = workingContent;
      continue;
    }

    let cacheAdded = false;
    for (let j = workingContent.length - 1; j >= 0; j--) {
      const contentPart = workingContent[j];
      if ('type' in contentPart && contentPart.type === 'text') {
        (contentPart as Anthropic.TextBlockParam).cache_control = {
          type: 'ephemeral',
        };
        cacheAdded = true;
        userMessagesModified++;
        break;
      }
    }

    if (cacheAdded) {
      updatedMessages[i] = createNewMessage(
        originalMessage as MessageWithContent,
        workingContent
      ) as T;
    } else {
      updatedMessages[i] = shallowCloneMessage(
        originalMessage as MessageWithContent
      ) as T;
      (updatedMessages[i] as MessageWithContent).content = workingContent;
    }
  }

  return updatedMessages;
}

/**
 * Checks if a content block is a cache point
 */
function isCachePoint(block: MessageContentComplex): boolean {
  return 'cachePoint' in block && !('type' in block);
}

/**
 * Checks if a message's content has Anthropic cache_control fields.
 */
function hasAnthropicCacheControl(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    if ('cache_control' in content[i]) return true;
  }
  return false;
}

/**
 * Removes all Anthropic cache_control fields from messages
 * Used when switching from Anthropic to Bedrock provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripAnthropicCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasAnthropicCacheControl(content)) {
      continue;
    }

    const message = shallowCloneMessage(originalMessage);
    updatedMessages[i] = message;

    for (
      let j = 0;
      j < (message.content as MessageContentComplex[]).length;
      j++
    ) {
      const block = (message.content as MessageContentComplex[])[j] as Record<
        string,
        unknown
      >;
      if ('cache_control' in block) {
        delete block.cache_control;
      }
    }
  }

  return updatedMessages;
}

/**
 * Checks if a message's content has Bedrock cachePoint blocks.
 */
function hasBedrockCachePoint(content: MessageContentComplex[]): boolean {
  for (let i = 0; i < content.length; i++) {
    if (isCachePoint(content[i])) return true;
  }
  return false;
}

/**
 * Removes all Bedrock cachePoint blocks from messages
 * Used when switching from Bedrock to Anthropic provider
 * Returns a new array - only clones messages that require modification.
 */
export function stripBedrockCacheControl<T extends MessageWithContent>(
  messages: T[]
): T[] {
  if (!Array.isArray(messages)) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];

  for (let i = 0; i < updatedMessages.length; i++) {
    const originalMessage = updatedMessages[i];
    const content = originalMessage.content;

    if (!Array.isArray(content) || !hasBedrockCachePoint(content)) {
      continue;
    }

    const message = shallowCloneMessage(originalMessage);
    updatedMessages[i] = message;

    message.content = (message.content as MessageContentComplex[]).filter(
      (block) => !isCachePoint(block as MessageContentComplex)
    ) as typeof content;
  }

  return updatedMessages;
}

/**
 * Adds Bedrock Converse API cache points to the last two messages.
 * Inserts `{ cachePoint: { type: 'default' } }` as a separate content block
 * immediately after the last text block in each targeted message.
 * Strips ALL existing cache control (both Bedrock and Anthropic formats) from all messages,
 * then adds fresh cache points to the last 2 messages in a single backward pass.
 * This ensures we don't accumulate stale cache points across multiple turns.
 * Returns a new array - only clones messages that require modification.
 * @param messages - The array of message objects.
 * @returns - A new array of message objects with cache points added.
 */
export function addBedrockCacheControl<
  T extends Partial<BaseMessage> & MessageWithContent,
>(messages: T[]): T[] {
  if (!Array.isArray(messages) || messages.length < 2) {
    return messages;
  }

  const updatedMessages: T[] = [...messages];
  let messagesModified = 0;

  for (let i = updatedMessages.length - 1; i >= 0; i--) {
    const originalMessage = updatedMessages[i];
    const isToolMessage =
      'getType' in originalMessage &&
      typeof originalMessage.getType === 'function' &&
      originalMessage.getType() === 'tool';

    const content = originalMessage.content;
    const hasArrayContent = Array.isArray(content);
    const needsStripping =
      hasArrayContent &&
      needsCacheStripping(content as MessageContentComplex[]);
    const isEmptyString = typeof content === 'string' && content === '';
    const needsCacheAdd =
      messagesModified < 2 &&
      !isToolMessage &&
      !isEmptyString &&
      (typeof content === 'string' || hasArrayContent);

    if (!needsStripping && !needsCacheAdd) {
      continue;
    }

    let workingContent: MessageContentComplex[];

    if (hasArrayContent) {
      workingContent = deepCloneContent(
        content as MessageContentComplex[]
      ).filter((block) => !isCachePoint(block));

      for (let j = 0; j < workingContent.length; j++) {
        const block = workingContent[j] as Record<string, unknown>;
        if ('cache_control' in block) {
          delete block.cache_control;
        }
      }
    } else if (typeof content === 'string') {
      workingContent = [{ type: ContentTypes.TEXT, text: content }];
    } else {
      workingContent = [];
    }

    if (messagesModified >= 2 || isToolMessage || isEmptyString) {
      updatedMessages[i] = shallowCloneMessage(originalMessage);
      (updatedMessages[i] as MessageWithContent).content = workingContent;
      continue;
    }

    if (workingContent.length === 0) {
      continue;
    }

    let hasCacheableContent = false;
    for (const block of workingContent) {
      if (block.type === ContentTypes.TEXT) {
        if (typeof block.text === 'string' && block.text !== '') {
          hasCacheableContent = true;
          break;
        }
      }
    }

    if (!hasCacheableContent) {
      updatedMessages[i] = shallowCloneMessage(originalMessage);
      (updatedMessages[i] as MessageWithContent).content = workingContent;
      continue;
    }

    let inserted = false;
    for (let j = workingContent.length - 1; j >= 0; j--) {
      const block = workingContent[j] as MessageContentComplex;
      const type = (block as { type?: string }).type;
      if (type === ContentTypes.TEXT || type === 'text') {
        const text = (block as { text?: string }).text;
        if (text === '' || text === undefined) {
          continue;
        }
        workingContent.splice(j + 1, 0, {
          cachePoint: { type: 'default' },
        } as MessageContentComplex);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      workingContent.push({
        cachePoint: { type: 'default' },
      } as MessageContentComplex);
    }

    updatedMessages[i] = createNewMessage(originalMessage, workingContent);
    messagesModified++;
  }

  return updatedMessages;
}
