/* eslint-disable @typescript-eslint/no-explicit-any */
import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  getBufferString,
} from '@langchain/core/messages';
import type { MessageContentImageUrl } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  ExtendedMessageContent,
  MessageContentComplex,
  ToolCallPart,
  TPayload,
  TMessage,
} from '@/types';
import { Providers, ContentTypes } from '@/common';

interface MediaMessageParams {
  message: {
    role: string;
    content: string;
    name?: string;
    [key: string]: any;
  };
  mediaParts: MessageContentComplex[];
  endpoint?: Providers;
}

/**
 * Formats a message with media content (images, documents, videos, audios) to API payload format.
 *
 * @param params - The parameters for formatting.
 * @returns - The formatted message.
 */
export const formatMediaMessage = ({
  message,
  endpoint,
  mediaParts,
}: MediaMessageParams): {
  role: string;
  content: MessageContentComplex[];
  name?: string;
  [key: string]: any;
} => {
  // Create a new object to avoid mutating the input
  const result: {
    role: string;
    content: MessageContentComplex[];
    name?: string;
    [key: string]: any;
  } = {
    ...message,
    content: [] as MessageContentComplex[],
  };

  if (endpoint === Providers.ANTHROPIC) {
    result.content = [
      ...mediaParts,
      { type: ContentTypes.TEXT, text: message.content },
    ] as MessageContentComplex[];
    return result;
  }

  result.content = [
    { type: ContentTypes.TEXT, text: message.content },
    ...mediaParts,
  ] as MessageContentComplex[];

  return result;
};

interface MessageInput {
  role?: string;
  _name?: string;
  sender?: string;
  text?: string;
  content?: string | MessageContentComplex[];
  image_urls?: MessageContentImageUrl[];
  documents?: MessageContentComplex[];
  videos?: MessageContentComplex[];
  audios?: MessageContentComplex[];
  lc_id?: string[];
  [key: string]: any;
}

interface FormatMessageParams {
  message: MessageInput;
  userName?: string;
  assistantName?: string;
  endpoint?: Providers;
  langChain?: boolean;
}

interface FormattedMessage {
  role: string;
  content: string | MessageContentComplex[];
  name?: string;
  [key: string]: any;
}

/**
 * Formats a message to OpenAI payload format based on the provided options.
 *
 * @param params - The parameters for formatting.
 * @returns - The formatted message.
 */
export const formatMessage = ({
  message,
  userName,
  endpoint,
  assistantName,
  langChain = false,
}: FormatMessageParams):
  | FormattedMessage
  | HumanMessage
  | AIMessage
  | SystemMessage => {
  // eslint-disable-next-line prefer-const
  let { role: _role, _name, sender, text, content: _content, lc_id } = message;
  if (lc_id && lc_id[2] && !langChain) {
    const roleMapping: Record<string, string> = {
      SystemMessage: 'system',
      HumanMessage: 'user',
      AIMessage: 'assistant',
    };
    _role = roleMapping[lc_id[2]] || _role;
  }
  const role =
    _role ??
    (sender != null && sender && sender.toLowerCase() === 'user'
      ? 'user'
      : 'assistant');
  const content = _content ?? text ?? '';
  const formattedMessage: FormattedMessage = {
    role,
    content,
  };

  // Set name fields first
  if (_name != null && _name) {
    formattedMessage.name = _name;
  }

  if (userName != null && userName && formattedMessage.role === 'user') {
    formattedMessage.name = userName;
  }

  if (
    assistantName != null &&
    assistantName &&
    formattedMessage.role === 'assistant'
  ) {
    formattedMessage.name = assistantName;
  }

  if (formattedMessage.name != null && formattedMessage.name) {
    // Conform to API regex: ^[a-zA-Z0-9_-]{1,64}$
    // https://community.openai.com/t/the-format-of-the-name-field-in-the-documentation-is-incorrect/175684/2
    formattedMessage.name = formattedMessage.name.replace(
      /[^a-zA-Z0-9_-]/g,
      '_'
    );

    if (formattedMessage.name.length > 64) {
      formattedMessage.name = formattedMessage.name.substring(0, 64);
    }
  }

  const { image_urls, documents, videos, audios } = message;
  const mediaParts: MessageContentComplex[] = [];

  if (Array.isArray(documents) && documents.length > 0) {
    mediaParts.push(...documents);
  }

  if (Array.isArray(videos) && videos.length > 0) {
    mediaParts.push(...videos);
  }

  if (Array.isArray(audios) && audios.length > 0) {
    mediaParts.push(...audios);
  }

  if (Array.isArray(image_urls) && image_urls.length > 0) {
    mediaParts.push(...image_urls);
  }

  if (mediaParts.length > 0 && role === 'user') {
    const mediaMessage = formatMediaMessage({
      message: {
        ...formattedMessage,
        content:
          typeof formattedMessage.content === 'string'
            ? formattedMessage.content
            : '',
      },
      mediaParts,
      endpoint,
    });

    if (!langChain) {
      return mediaMessage;
    }

    return new HumanMessage(mediaMessage);
  }

  if (!langChain) {
    return formattedMessage;
  }

  if (role === 'user') {
    return new HumanMessage(formattedMessage);
  } else if (role === 'assistant') {
    return new AIMessage(formattedMessage);
  } else {
    return new SystemMessage(formattedMessage);
  }
};

/**
 * Formats an array of messages for LangChain.
 *
 * @param messages - The array of messages to format.
 * @param formatOptions - The options for formatting each message.
 * @returns - The array of formatted LangChain messages.
 */
export const formatLangChainMessages = (
  messages: Array<MessageInput>,
  formatOptions: Omit<FormatMessageParams, 'message' | 'langChain'>
): Array<HumanMessage | AIMessage | SystemMessage> => {
  return messages.map((msg) => {
    const formatted = formatMessage({
      ...formatOptions,
      message: msg,
      langChain: true,
    });
    return formatted as HumanMessage | AIMessage | SystemMessage;
  });
};

interface LangChainMessage {
  lc_kwargs?: {
    additional_kwargs?: Record<string, any>;
    [key: string]: any;
  };
  kwargs?: {
    additional_kwargs?: Record<string, any>;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * Formats a LangChain message object by merging properties from `lc_kwargs` or `kwargs` and `additional_kwargs`.
 *
 * @param message - The message object to format.
 * @returns - The formatted LangChain message.
 */
export const formatFromLangChain = (
  message: LangChainMessage
): Record<string, any> => {
  const kwargs = message.lc_kwargs ?? message.kwargs ?? {};
  const { additional_kwargs = {}, ...message_kwargs } = kwargs;
  return {
    ...message_kwargs,
    ...additional_kwargs,
  };
};

/**
 * Helper function to format an assistant message
 * @param message The message to format
 * @returns Array of formatted messages
 */
function formatAssistantMessage(
  message: Partial<TMessage>
): Array<AIMessage | ToolMessage> {
  const formattedMessages: Array<AIMessage | ToolMessage> = [];
  let currentContent: MessageContentComplex[] = [];
  let lastAIMessage: AIMessage | null = null;
  let hasReasoning = false;

  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (part.type === ContentTypes.TEXT && part.tool_call_ids) {
        /*
        If there's pending content, it needs to be aggregated as a single string to prepare for tool calls.
        For Anthropic models, the "tool_calls" field on a message is only respected if content is a string.
        */
        if (currentContent.length > 0) {
          let content = currentContent.reduce((acc, curr) => {
            if (curr.type === ContentTypes.TEXT) {
              return `${acc}${curr[ContentTypes.TEXT] || ''}\n`;
            }
            return acc;
          }, '');
          content =
            `${content}\n${part[ContentTypes.TEXT] ?? part.text ?? ''}`.trim();
          lastAIMessage = new AIMessage({ content });
          formattedMessages.push(lastAIMessage);
          currentContent = [];
          continue;
        }
        // Create a new AIMessage with this text and prepare for tool calls
        lastAIMessage = new AIMessage({
          content: part.text || '',
        });
        formattedMessages.push(lastAIMessage);
      } else if (part.type === ContentTypes.TOOL_CALL) {
        // Skip malformed tool call entries without tool_call property
        if (!part.tool_call) {
          continue;
        }

        // Note: `tool_calls` list is defined when constructed by `AIMessage` class, and outputs should be excluded from it
        const {
          output,
          args: _args,
          ..._tool_call
        } = part.tool_call as ToolCallPart;

        // Skip invalid tool calls that have no name AND no output
        if (!_tool_call.name && (output == null || output === '')) {
          continue;
        }

        if (!lastAIMessage) {
          // "Heal" the payload by creating an AIMessage to precede the tool call
          lastAIMessage = new AIMessage({ content: '' });
          formattedMessages.push(lastAIMessage);
        }

        const tool_call: ToolCallPart = _tool_call;
        // TODO: investigate; args as dictionary may need to be providers-or-tool-specific
        let args: any = _args;
        try {
          if (typeof _args === 'string') {
            args = JSON.parse(_args);
          }
        } catch {
          if (typeof _args === 'string') {
            args = { input: _args };
          }
        }

        tool_call.args = args;
        if (!lastAIMessage.tool_calls) {
          lastAIMessage.tool_calls = [];
        }
        lastAIMessage.tool_calls.push(tool_call as ToolCall);

        formattedMessages.push(
          new ToolMessage({
            tool_call_id: tool_call.id ?? '',
            name: tool_call.name,
            content: output || '',
          })
        );
      } else if (part.type === ContentTypes.THINK) {
        hasReasoning = true;
        continue;
      } else if (
        part.type === ContentTypes.ERROR ||
        part.type === ContentTypes.AGENT_UPDATE
      ) {
        continue;
      } else {
        currentContent.push(part);
      }
    }
  }

  if (hasReasoning && currentContent.length > 0) {
    const content = currentContent
      .reduce((acc, curr) => {
        if (curr.type === ContentTypes.TEXT) {
          return `${acc}${curr[ContentTypes.TEXT] || ''}\n`;
        }
        return acc;
      }, '')
      .trim();

    if (content) {
      formattedMessages.push(new AIMessage({ content }));
    }
  } else if (currentContent.length > 0) {
    formattedMessages.push(new AIMessage({ content: currentContent }));
  }

  return formattedMessages;
}

/**
 * Formats an array of messages for LangChain, handling tool calls and creating ToolMessage instances.
 *
 * @param payload - The array of messages to format.
 * @param indexTokenCountMap - Optional map of message indices to token counts.
 * @param tools - Optional set of tool names that are allowed in the request.
 * @returns - Object containing formatted messages and updated indexTokenCountMap if provided.
 */
export const formatAgentMessages = (
  payload: TPayload,
  indexTokenCountMap?: Record<number, number>,
  tools?: Set<string>
): {
  messages: Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>;
  indexTokenCountMap?: Record<number, number>;
} => {
  const messages: Array<
    HumanMessage | AIMessage | SystemMessage | ToolMessage
  > = [];
  // If indexTokenCountMap is provided, create a new map to track the updated indices
  const updatedIndexTokenCountMap: Record<number, number> = {};
  // Keep track of the mapping from original payload indices to result indices
  const indexMapping: Record<number, number[]> = {};

  // Process messages with tool conversion if tools set is provided
  for (let i = 0; i < payload.length; i++) {
    const message = payload[i];
    // Q: Store the current length of messages to track where this payload message starts in the result?
    // const startIndex = messages.length;
    if (typeof message.content === 'string') {
      message.content = [
        { type: ContentTypes.TEXT, [ContentTypes.TEXT]: message.content },
      ];
    }
    if (message.role !== 'assistant') {
      messages.push(
        formatMessage({
          message: message as MessageInput,
          langChain: true,
        }) as HumanMessage | AIMessage | SystemMessage
      );

      // Update the index mapping for this message
      indexMapping[i] = [messages.length - 1];
      continue;
    }

    // For assistant messages, track the starting index before processing
    const startMessageIndex = messages.length;

    // If tools set is provided, we need to check if we need to convert tool messages to a string
    if (tools) {
      // First, check if this message contains tool calls
      let hasToolCalls = false;
      let hasInvalidTool = false;
      const toolNames: string[] = [];

      const content = message.content;
      if (content && Array.isArray(content)) {
        for (const part of content) {
          if (part.type === ContentTypes.TOOL_CALL) {
            hasToolCalls = true;
            if (tools.size === 0) {
              hasInvalidTool = true;
              break;
            }
            // Protect against malformed tool call entries
            if (
              !part.tool_call ||
              part.tool_call.name == null ||
              part.tool_call.name === ''
            ) {
              hasInvalidTool = true;
              continue;
            }
            const toolName = part.tool_call.name;
            toolNames.push(toolName);
            if (!tools.has(toolName)) {
              hasInvalidTool = true;
            }
          }
        }
      }

      // If this message has tool calls and at least one is invalid, we need to convert it
      if (hasToolCalls && hasInvalidTool) {
        // We need to collect all related messages (this message and any subsequent tool messages)
        const toolSequence: BaseMessage[] = [];
        let sequenceEndIndex = i;

        // Process the current assistant message to get the AIMessage with tool calls
        const formattedMessages = formatAssistantMessage(message);
        toolSequence.push(...formattedMessages);

        // Look ahead for any subsequent assistant messages that might be part of this tool sequence
        let j = i + 1;
        while (j < payload.length && payload[j].role === 'assistant') {
          // Check if this is a continuation of the tool sequence
          let isToolResponse = false;
          const content = payload[j].content;
          if (content && Array.isArray(content)) {
            for (const part of content) {
              if (part.type === ContentTypes.TOOL_CALL) {
                isToolResponse = true;
                break;
              }
            }
          }

          if (isToolResponse) {
            // This is part of the tool sequence, add it
            const nextMessages = formatAssistantMessage(payload[j]);
            toolSequence.push(...nextMessages);
            sequenceEndIndex = j;
            j++;
          } else {
            // This is not part of the tool sequence, stop looking
            break;
          }
        }

        // Convert the sequence to a string
        const bufferString = getBufferString(toolSequence);
        messages.push(new AIMessage({ content: bufferString }));

        // Skip the messages we've already processed
        i = sequenceEndIndex;

        // Update the index mapping for this sequence
        const resultIndices = [messages.length - 1];
        for (let k = i; k >= i && k <= sequenceEndIndex; k++) {
          indexMapping[k] = resultIndices;
        }

        continue;
      }
    }

    // Process the assistant message using the helper function
    const formattedMessages = formatAssistantMessage(message);
    messages.push(...formattedMessages);

    // Update the index mapping for this assistant message
    // Store all indices that were created from this original message
    const endMessageIndex = messages.length;
    const resultIndices = [];
    for (let j = startMessageIndex; j < endMessageIndex; j++) {
      resultIndices.push(j);
    }
    indexMapping[i] = resultIndices;
  }

  // Update the token count map if it was provided
  if (indexTokenCountMap) {
    for (
      let originalIndex = 0;
      originalIndex < payload.length;
      originalIndex++
    ) {
      const resultIndices = indexMapping[originalIndex] || [];
      const tokenCount = indexTokenCountMap[originalIndex];

      if (tokenCount !== undefined) {
        if (resultIndices.length === 1) {
          // Simple 1:1 mapping
          updatedIndexTokenCountMap[resultIndices[0]] = tokenCount;
        } else if (resultIndices.length > 1) {
          // If one message was split into multiple, distribute the token count
          // This is a simplification - in reality, you might want a more sophisticated distribution
          const countPerMessage = Math.floor(tokenCount / resultIndices.length);
          resultIndices.forEach((resultIndex, idx) => {
            if (idx === resultIndices.length - 1) {
              // Give any remainder to the last message
              updatedIndexTokenCountMap[resultIndex] =
                tokenCount - countPerMessage * (resultIndices.length - 1);
            } else {
              updatedIndexTokenCountMap[resultIndex] = countPerMessage;
            }
          });
        }
      }
    }
  }

  return {
    messages,
    indexTokenCountMap: indexTokenCountMap
      ? updatedIndexTokenCountMap
      : undefined,
  };
};

/**
 * Adds a value at key 0 for system messages and shifts all key indices by one in an indexTokenCountMap.
 * This is useful when adding a system message at the beginning of a conversation.
 *
 * @param indexTokenCountMap - The original map of message indices to token counts
 * @param instructionsTokenCount - The token count for the system message to add at index 0
 * @returns A new map with the system message at index 0 and all other indices shifted by 1
 */
export function shiftIndexTokenCountMap(
  indexTokenCountMap: Record<number, number>,
  instructionsTokenCount: number
): Record<number, number> {
  // Create a new map to avoid modifying the original
  const shiftedMap: Record<number, number> = {};
  shiftedMap[0] = instructionsTokenCount;

  // Shift all existing indices by 1
  for (const [indexStr, tokenCount] of Object.entries(indexTokenCountMap)) {
    const index = Number(indexStr);
    shiftedMap[index + 1] = tokenCount;
  }

  return shiftedMap;
}

/**
 * Ensures compatibility when switching from a non-thinking agent to a thinking-enabled agent.
 * Converts AI messages with tool calls (that lack thinking blocks) into buffer strings,
 * avoiding the thinking block signature requirement.
 *
 * @param messages - Array of messages to process
 * @param provider - The provider being used (unused but kept for future compatibility)
 * @returns The messages array with tool sequences converted to buffer strings if necessary
 */
export function ensureThinkingBlockInMessages(
  messages: BaseMessage[],
  _provider: Providers
): BaseMessage[] {
  const result: BaseMessage[] = [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];
    const isAI = msg instanceof AIMessage || msg instanceof AIMessageChunk;

    if (!isAI) {
      result.push(msg);
      i++;
      continue;
    }

    const aiMsg = msg as AIMessage | AIMessageChunk;
    const hasToolCalls = aiMsg.tool_calls && aiMsg.tool_calls.length > 0;
    const contentIsArray = Array.isArray(aiMsg.content);

    // Check if the message has tool calls or tool_use content
    let hasToolUse = hasToolCalls ?? false;
    let firstContentType: string | undefined;

    if (contentIsArray && aiMsg.content.length > 0) {
      const content = aiMsg.content as ExtendedMessageContent[];
      firstContentType = content[0]?.type;
      hasToolUse =
        hasToolUse ||
        content.some((c) => typeof c === 'object' && c.type === 'tool_use');
    }

    // If message has tool use but no thinking block, convert to buffer string
    if (
      hasToolUse &&
      firstContentType !== ContentTypes.THINKING &&
      firstContentType !== 'redacted_thinking'
    ) {
      // Collect the AI message and any following tool messages
      const toolSequence: BaseMessage[] = [msg];
      let j = i + 1;

      // Look ahead for tool messages that belong to this AI message
      while (j < messages.length && messages[j] instanceof ToolMessage) {
        toolSequence.push(messages[j]);
        j++;
      }

      // Convert the sequence to a buffer string and wrap in a HumanMessage
      // This avoids the thinking block requirement which only applies to AI messages
      const bufferString = getBufferString(toolSequence);
      result.push(
        new HumanMessage({
          content: `[Previous agent context]\n${bufferString}`,
        })
      );

      // Skip the messages we've processed
      i = j;
    } else {
      // Keep the message as is
      result.push(msg);
      i++;
    }
  }

  return result;
}
