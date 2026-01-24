// src/messages.ts
import {
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
  AIMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { Providers } from '@/common';

export function getConverseOverrideMessage({
  userMessage,
  lastMessageX,
  lastMessageY,
}: {
  userMessage: string[];
  lastMessageX: AIMessageChunk | null;
  lastMessageY: ToolMessage;
}): HumanMessage {
  const content = `
User: ${userMessage[1]}

---
# YOU HAVE ALREADY RESPONDED TO THE LATEST USER MESSAGE:

# Observations:
- ${lastMessageX?.content}

# Tool Calls:
- ${lastMessageX?.tool_calls?.join('\n- ')}

# Tool Responses:
- ${lastMessageY.content}
`;

  return new HumanMessage(content);
}

const _allowedTypes = ['image_url', 'text', 'tool_use', 'tool_result'];
const allowedTypesByProvider: Record<string, string[]> = {
  default: _allowedTypes,
  [Providers.ANTHROPIC]: [..._allowedTypes, 'thinking', 'redacted_thinking'],
  [Providers.BEDROCK]: [..._allowedTypes, 'reasoning_content'],
  [Providers.OPENAI]: _allowedTypes,
};

const modifyContent = ({
  provider,
  messageType,
  content,
}: {
  provider: Providers;
  messageType: string;
  content: t.ExtendedMessageContent[];
}): t.ExtendedMessageContent[] => {
  const allowedTypes =
    allowedTypesByProvider[provider] ?? allowedTypesByProvider.default;
  return content.map((item) => {
    if (
      item &&
      typeof item === 'object' &&
      'type' in item &&
      item.type != null &&
      item.type
    ) {
      let newType = item.type;
      if (newType.endsWith('_delta')) {
        newType = newType.replace('_delta', '');
      }
      if (!allowedTypes.includes(newType)) {
        newType = 'text';
      }

      /* Handle the edge case for empty object 'tool_use' input in AI messages */
      if (
        messageType === 'ai' &&
        newType === 'tool_use' &&
        'input' in item &&
        item.input === ''
      ) {
        return { ...item, type: newType, input: '{}' };
      }

      return { ...item, type: newType };
    }
    return item;
  });
};

type ContentBlock =
  | Partial<t.BedrockReasoningContentText>
  | t.MessageDeltaUpdate;

function reduceBlocks(blocks: ContentBlock[]): ContentBlock[] {
  const reduced: ContentBlock[] = [];

  for (const block of blocks) {
    const lastBlock = reduced[reduced.length - 1] as ContentBlock | undefined;

    // Merge consecutive 'reasoning_content'
    if (
      block.type === 'reasoning_content' &&
      lastBlock?.type === 'reasoning_content'
    ) {
      // append text if exists
      if (block.reasoningText?.text != null && block.reasoningText.text) {
        (
          lastBlock.reasoningText as t.BedrockReasoningContentText['reasoningText']
        ).text =
          (lastBlock.reasoningText?.text ?? '') + block.reasoningText.text;
      }
      // preserve the signature if exists
      if (
        block.reasoningText?.signature != null &&
        block.reasoningText.signature
      ) {
        (
          lastBlock.reasoningText as t.BedrockReasoningContentText['reasoningText']
        ).signature = block.reasoningText.signature;
      }
    }
    // Merge consecutive 'text'
    else if (block.type === 'text' && lastBlock?.type === 'text') {
      lastBlock.text += block.text;
    }
    // add a new block as it's a different type or first element
    else {
      // deep copy to avoid mutation of original
      reduced.push(JSON.parse(JSON.stringify(block)));
    }
  }

  return reduced;
}

export function modifyDeltaProperties(
  provider: Providers,
  obj?: AIMessageChunk
): AIMessageChunk | undefined {
  if (!obj || typeof obj !== 'object') return obj;

  const messageType = (obj as Partial<AIMessageChunk>)._getType
    ? obj._getType()
    : '';

  if (provider === Providers.BEDROCK && Array.isArray(obj.content)) {
    obj.content = reduceBlocks(obj.content as ContentBlock[]);
  }
  if (Array.isArray(obj.content)) {
    obj.content = modifyContent({
      provider,
      messageType,
      content: obj.content,
    });
  }
  if (
    (obj as Partial<AIMessageChunk>).lc_kwargs &&
    Array.isArray(obj.lc_kwargs.content)
  ) {
    obj.lc_kwargs.content = modifyContent({
      provider,
      messageType,
      content: obj.lc_kwargs.content,
    });
  }
  return obj;
}

export function formatAnthropicMessage(message: AIMessageChunk): AIMessage {
  if (!message.tool_calls || message.tool_calls.length === 0) {
    return new AIMessage({ content: message.content });
  }

  const toolCallMap = new Map(message.tool_calls.map((tc) => [tc.id, tc]));
  let formattedContent: string | t.ExtendedMessageContent[];

  if (Array.isArray(message.content)) {
    formattedContent = message.content.reduce<t.ExtendedMessageContent[]>(
      (acc, item) => {
        if (typeof item === 'object') {
          const extendedItem = item as t.ExtendedMessageContent;
          if (
            extendedItem.type === 'text' &&
            extendedItem.text != null &&
            extendedItem.text
          ) {
            acc.push({ type: 'text', text: extendedItem.text });
          } else if (
            extendedItem.type === 'tool_use' &&
            extendedItem.id != null &&
            extendedItem.id
          ) {
            const toolCall = toolCallMap.get(extendedItem.id);
            if (toolCall) {
              acc.push({
                type: 'tool_use',
                id: extendedItem.id,
                name: toolCall.name,
                input: toolCall.args as unknown as string,
              });
            }
          } else if (
            'input' in extendedItem &&
            extendedItem.input != null &&
            extendedItem.input
          ) {
            try {
              const parsedInput = JSON.parse(extendedItem.input);
              const toolCall = message.tool_calls?.find(
                (tc) => tc.args.input === parsedInput.input
              );
              if (toolCall) {
                acc.push({
                  type: 'tool_use',
                  id: toolCall.id,
                  name: toolCall.name,
                  input: toolCall.args as unknown as string,
                });
              }
            } catch {
              if (extendedItem.input) {
                acc.push({ type: 'text', text: extendedItem.input });
              }
            }
          }
        } else if (typeof item === 'string') {
          acc.push({ type: 'text', text: item });
        }
        return acc;
      },
      []
    );
  } else if (typeof message.content === 'string') {
    formattedContent = message.content;
  } else {
    formattedContent = [];
  }

  // const formattedToolCalls: ToolCall[] = message.tool_calls.map(toolCall => ({
  //   id: toolCall.id ?? '',
  //   name: toolCall.name,
  //   args: toolCall.args,
  //   type: 'tool_call',
  // }));

  const formattedToolCalls: t.AgentToolCall[] = message.tool_calls.map(
    (toolCall) => ({
      id: toolCall.id ?? '',
      type: 'function',
      function: {
        name: toolCall.name,
        arguments: toolCall.args,
      },
    })
  );

  return new AIMessage({
    content: formattedContent,
    tool_calls: formattedToolCalls as ToolCall[],
    additional_kwargs: {
      ...message.additional_kwargs,
    },
  });
}

export function convertMessagesToContent(
  messages: BaseMessage[]
): t.MessageContentComplex[] {
  const processedContent: t.MessageContentComplex[] = [];

  const addContentPart = (message: BaseMessage | null): void => {
    const content =
      message?.lc_kwargs.content != null
        ? message.lc_kwargs.content
        : message?.content;
    if (content === undefined) {
      return;
    }
    if (typeof content === 'string') {
      processedContent.push({
        type: 'text',
        text: content,
      });
    } else if (Array.isArray(content)) {
      const filteredContent = content.filter(
        (item) => item != null && item.type !== 'tool_use'
      );
      processedContent.push(...filteredContent);
    }
  };

  let currentAIMessageIndex = -1;
  const toolCallMap = new Map<string, t.CustomToolCall>();

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i] as BaseMessage | null;
    const messageType = message?._getType();

    if (
      messageType === 'ai' &&
      ((message as AIMessage).tool_calls?.length ?? 0) > 0
    ) {
      const tool_calls = (message as AIMessage).tool_calls || [];
      for (const tool_call of tool_calls) {
        if (tool_call.id == null || !tool_call.id) {
          continue;
        }

        toolCallMap.set(tool_call.id, tool_call);
      }

      addContentPart(message);
      currentAIMessageIndex = processedContent.length - 1;
      continue;
    } else if (
      messageType === 'tool' &&
      (message as ToolMessage).tool_call_id
    ) {
      const id = (message as ToolMessage).tool_call_id;
      const output = (message as ToolMessage).content;
      const tool_call = toolCallMap.get(id);
      if (currentAIMessageIndex === -1) {
        processedContent.push({ type: 'text', text: '' });
        currentAIMessageIndex = processedContent.length - 1;
      }
      const contentPart = processedContent[currentAIMessageIndex];
      processedContent.push({
        type: 'tool_call',
        tool_call: Object.assign({}, tool_call, { output }),
      });
      const tool_call_ids = contentPart.tool_call_ids || [];
      tool_call_ids.push(id);
      contentPart.tool_call_ids = tool_call_ids;
      continue;
    } else if (messageType !== 'ai') {
      continue;
    }

    addContentPart(message);
  }

  return processedContent;
}

export function formatAnthropicArtifactContent(messages: BaseMessage[]): void {
  const lastMessage = messages[messages.length - 1];
  if (!(lastMessage instanceof ToolMessage)) return;

  // Find the latest AIMessage with tool_calls that this tool message belongs to
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      (msg instanceof AIMessageChunk &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessage.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return;

  // Check if any tool message after the AI message has array artifact content
  const hasArtifactContent = messages.some(
    (msg, i) =>
      i > latestAIParentIndex &&
      msg instanceof ToolMessage &&
      msg.artifact != null &&
      msg.artifact?.content != null &&
      Array.isArray(msg.artifact.content)
  );

  if (!hasArtifactContent) return;

  const message = messages[latestAIParentIndex] as AIMessageChunk;
  const toolCallIds = message.tool_calls?.map((tc) => tc.id) ?? [];

  for (let j = latestAIParentIndex + 1; j < messages.length; j++) {
    const msg = messages[j];
    if (
      msg instanceof ToolMessage &&
      toolCallIds.includes(msg.tool_call_id) &&
      msg.artifact != null &&
      Array.isArray(msg.artifact?.content) &&
      Array.isArray(msg.content)
    ) {
      msg.content = msg.content.concat(msg.artifact.content);
    }
  }
}

export function formatArtifactPayload(
  messages: BaseMessage[],
  isVisionModel: boolean = true
): void {
  const lastMessageY = messages[messages.length - 1];
  if (!(lastMessageY instanceof ToolMessage)) return;

  // Find the latest AIMessage with tool_calls that this tool message belongs to
  const latestAIParentIndex = findLastIndex(
    messages,
    (msg) =>
      (msg instanceof AIMessageChunk &&
        (msg.tool_calls?.length ?? 0) > 0 &&
        msg.tool_calls?.some((tc) => tc.id === lastMessageY.tool_call_id)) ??
      false
  );

  if (latestAIParentIndex === -1) return;

  // Check if any tool message after the AI message has array artifact content
  const hasArtifactContent = messages.some(
    (msg, i) =>
      i > latestAIParentIndex &&
      msg instanceof ToolMessage &&
      msg.artifact != null &&
      msg.artifact?.content != null &&
      Array.isArray(msg.artifact.content)
  );

  if (!hasArtifactContent) return;

  // Collect all relevant tool messages and their artifacts
  const relevantMessages = messages
    .slice(latestAIParentIndex + 1)
    .filter((msg) => msg instanceof ToolMessage) as ToolMessage[];

  // Aggregate all content and artifacts
  const aggregatedContent: t.MessageContentComplex[] = [];

  relevantMessages.forEach((msg) => {
    if (!Array.isArray(msg.artifact?.content)) {
      return;
    }
    let currentContent = msg.content;
    if (!Array.isArray(currentContent)) {
      currentContent = [
        {
          type: 'text',
          text: msg.content,
        },
      ];
    }
    // Filter out image content from currentContent if model doesn't support vision
    const filteredCurrentContent = isVisionModel
      ? currentContent
      : currentContent.filter(
        (item: t.MessageContentComplex) => item.type !== 'image_url'
      );
    aggregatedContent.push(...filteredCurrentContent);
    msg.content =
      'Tool response is included in the next message as a Human message';

    // Filter out image artifacts if model doesn't support vision
    const artifactContent = isVisionModel
      ? msg.artifact.content
      : msg.artifact.content.filter(
        (item: t.MessageContentComplex) => item.type !== 'image_url'
      );

    aggregatedContent.push(...artifactContent);
  });

  // Add single HumanMessage with all aggregated content
  if (aggregatedContent.length > 0) {
    messages.push(new HumanMessage({ content: aggregatedContent }));
  }
}

export function findLastIndex<T>(
  array: T[],
  predicate: (value: T) => boolean
): number {
  for (let i = array.length - 1; i >= 0; i--) {
    if (predicate(array[i])) {
      return i;
    }
  }
  return -1;
}
