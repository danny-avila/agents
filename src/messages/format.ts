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
  ReasoningContentText,
  SummaryContentBlock,
  ToolCallContent,
  ToolCallPart,
  TPayload,
  TMessage,
} from '@/types';
import type { RunnableConfig } from '@langchain/core/runnables';
import { emitAgentLog } from '@/utils/events';
import { Providers, ContentTypes, Constants } from '@/common';

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
    for (const part of message.content as Array<
      MessageContentComplex | undefined | null
    >) {
      if (part == null) {
        continue;
      }
      if (part.type === ContentTypes.TEXT && part.tool_call_ids) {
        /*
        If there's pending content, it needs to be aggregated as a single string to prepare for tool calls.
        For Anthropic models, the "tool_calls" field on a message is only respected if content is a string.
        */
        if (currentContent.length > 0) {
          let content = currentContent.reduce((acc, curr) => {
            if (curr.type === ContentTypes.TEXT) {
              return `${acc}${String(curr[ContentTypes.TEXT] ?? '')}\n`;
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
          content: part.text != null ? part.text : '',
        });
        formattedMessages.push(lastAIMessage);
      } else if (part.type === ContentTypes.TOOL_CALL) {
        // Skip malformed tool call entries without tool_call property
        if (part.tool_call == null) {
          continue;
        }

        // Note: `tool_calls` list is defined when constructed by `AIMessage` class, and outputs should be excluded from it
        const {
          output,
          args: _args,
          ..._tool_call
        } = part.tool_call as ToolCallPart;

        // Skip invalid tool calls that have no name AND no output
        if (
          _tool_call.name == null ||
          (_tool_call.name === '' && (output == null || output === ''))
        ) {
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
            content: output != null ? output : '',
          })
        );
      } else if (part.type === ContentTypes.THINK) {
        hasReasoning = true;
        continue;
      } else if (
        part.type === ContentTypes.ERROR ||
        part.type === ContentTypes.AGENT_UPDATE ||
        part.type === ContentTypes.SUMMARY
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
          return `${acc}${String(curr[ContentTypes.TEXT] ?? '')}\n`;
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

function getSourceMessageId(message: Partial<TMessage>): string | undefined {
  const candidate =
    (message as { messageId?: string }).messageId ??
    (message as { id?: string }).id;
  if (typeof candidate !== 'string') {
    return undefined;
  }
  const normalized = candidate.trim();
  return normalized.length > 0 ? normalized : undefined;
}

/**
 * Labels all agent content for parallel patterns (fan-out/fan-in)
 * Groups consecutive content by agent and wraps with clear labels
 */
function labelAllAgentContent(
  contentParts: MessageContentComplex[],
  agentIdMap: Record<number, string>,
  agentNames?: Record<string, string>
): MessageContentComplex[] {
  const result: MessageContentComplex[] = [];
  let currentAgentId: string | undefined;
  let agentContentBuffer: MessageContentComplex[] = [];

  const flushAgentBuffer = (): void => {
    if (agentContentBuffer.length === 0) {
      return;
    }

    if (currentAgentId != null && currentAgentId !== '') {
      const agentName = (agentNames?.[currentAgentId] ?? '') || currentAgentId;
      const formattedParts: string[] = [];

      formattedParts.push(`--- ${agentName} ---`);

      for (const part of agentContentBuffer) {
        if (part.type === ContentTypes.THINK) {
          const thinkContent = (part as ReasoningContentText).think || '';
          if (thinkContent) {
            formattedParts.push(
              `${agentName}: ${JSON.stringify({
                type: 'think',
                think: thinkContent,
              })}`
            );
          }
        } else if (part.type === ContentTypes.TEXT) {
          const textContent: string = part.text ?? '';
          if (textContent) {
            formattedParts.push(`${agentName}: ${textContent}`);
          }
        } else if (part.type === ContentTypes.TOOL_CALL) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'tool_call',
              tool_call: (part as ToolCallContent).tool_call,
            })}`
          );
        }
      }

      formattedParts.push(`--- End of ${agentName} ---`);

      // Create a single text content part with all agent content
      result.push({
        type: ContentTypes.TEXT,
        text: formattedParts.join('\n\n'),
      } as MessageContentComplex);
    } else {
      // No agent ID, pass through as-is
      result.push(...agentContentBuffer);
    }

    agentContentBuffer = [];
  };

  for (let i = 0; i < contentParts.length; i++) {
    const part = contentParts[i];
    const agentId = agentIdMap[i];

    // If agent changed, flush previous buffer
    if (agentId !== currentAgentId && currentAgentId !== undefined) {
      flushAgentBuffer();
    }

    currentAgentId = agentId;
    agentContentBuffer.push(part);
  }

  // Flush any remaining content
  flushAgentBuffer();

  return result;
}

/**
 * Groups content parts by agent and formats them with agent labels
 * This preprocesses multi-agent content to prevent identity confusion
 *
 * @param contentParts - The content parts from a run
 * @param agentIdMap - Map of content part index to agent ID
 * @param agentNames - Optional map of agent ID to display name
 * @param options - Configuration options
 * @param options.labelNonTransferContent - If true, labels all agent transitions (for parallel patterns)
 * @returns Modified content parts with agent labels where appropriate
 */
export const labelContentByAgent = (
  contentParts: MessageContentComplex[],
  agentIdMap?: Record<number, string>,
  agentNames?: Record<string, string>,
  options?: { labelNonTransferContent?: boolean }
): MessageContentComplex[] => {
  if (!agentIdMap || Object.keys(agentIdMap).length === 0) {
    return contentParts;
  }

  // If labelNonTransferContent is true, use a different strategy for parallel patterns
  if (options?.labelNonTransferContent === true) {
    return labelAllAgentContent(contentParts, agentIdMap, agentNames);
  }

  const result: MessageContentComplex[] = [];
  let currentAgentId: string | undefined;
  let agentContentBuffer: MessageContentComplex[] = [];
  let transferToolCallIndex: number | undefined;
  let transferToolCallId: string | undefined;

  const flushAgentBuffer = (): void => {
    if (agentContentBuffer.length === 0) {
      return;
    }

    // If this is content from a transferred agent, format it specially
    if (
      currentAgentId != null &&
      currentAgentId !== '' &&
      transferToolCallIndex !== undefined
    ) {
      const agentName = (agentNames?.[currentAgentId] ?? '') || currentAgentId;
      const formattedParts: string[] = [];

      formattedParts.push(`--- Transfer to ${agentName} ---`);

      for (const part of agentContentBuffer) {
        if (part.type === ContentTypes.THINK) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'think',
              think: (part as ReasoningContentText).think,
            })}`
          );
        } else if ('text' in part && part.type === ContentTypes.TEXT) {
          const textContent: string = part.text ?? '';
          if (textContent) {
            formattedParts.push(
              `${agentName}: ${JSON.stringify({
                type: 'text',
                text: textContent,
              })}`
            );
          }
        } else if (part.type === ContentTypes.TOOL_CALL) {
          formattedParts.push(
            `${agentName}: ${JSON.stringify({
              type: 'tool_call',
              tool_call: (part as ToolCallContent).tool_call,
            })}`
          );
        }
      }

      formattedParts.push(`--- End of ${agentName} response ---`);

      // Find the tool call that triggered this transfer and update its output
      if (transferToolCallIndex < result.length) {
        const transferToolCall = result[transferToolCallIndex];
        if (
          transferToolCall.type === ContentTypes.TOOL_CALL &&
          transferToolCall.tool_call?.id === transferToolCallId
        ) {
          transferToolCall.tool_call.output = formattedParts.join('\n\n');
        }
      }
    } else {
      // Not from a transfer, add as-is
      result.push(...agentContentBuffer);
    }

    agentContentBuffer = [];
    transferToolCallIndex = undefined;
    transferToolCallId = undefined;
  };

  for (let i = 0; i < contentParts.length; i++) {
    const part = contentParts[i];
    const agentId = agentIdMap[i];

    // Check if this is a transfer tool call
    const isTransferTool =
      (part.type === ContentTypes.TOOL_CALL &&
        (part as ToolCallContent).tool_call?.name?.startsWith(
          'lc_transfer_to_'
        )) ??
      false;

    // If agent changed, flush previous buffer
    if (agentId !== currentAgentId && currentAgentId !== undefined) {
      flushAgentBuffer();
    }

    currentAgentId = agentId;

    if (isTransferTool) {
      // Flush any existing buffer first
      flushAgentBuffer();
      // Add the transfer tool call to result
      result.push(part);
      // Mark that the next agent's content should be captured
      transferToolCallIndex = result.length - 1;
      transferToolCallId = (part as ToolCallContent).tool_call?.id;
      currentAgentId = undefined; // Reset to capture the next agent
    } else {
      agentContentBuffer.push(part);
    }
  }

  flushAgentBuffer();

  return result;
};

/** Extracts tool names from a tool_search output JSON string. */
function extractToolNamesFromSearchOutput(output: string): string[] {
  try {
    const parsed: unknown = JSON.parse(output);
    if (
      typeof parsed === 'object' &&
      parsed !== null &&
      Array.isArray((parsed as Record<string, unknown>).tools)
    ) {
      return (
        (parsed as Record<string, unknown>).tools as Array<{ name?: string }>
      )
        .map((t) => t.name)
        .filter((name): name is string => typeof name === 'string');
    }
  } catch {
    /** Output may have warnings prepended, try to find JSON within it */
    const jsonMatch = output.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        const parsed: unknown = JSON.parse(jsonMatch[0]);
        if (
          typeof parsed === 'object' &&
          parsed !== null &&
          Array.isArray((parsed as Record<string, unknown>).tools)
        ) {
          return (
            (parsed as Record<string, unknown>).tools as Array<{
              name?: string;
            }>
          )
            .map((t) => t.name)
            .filter((name): name is string => typeof name === 'string');
        }
      } catch {
        /* ignore */
      }
    }
  }
  return [];
}

type SummaryBoundary = {
  messageIndex: number;
  contentIndex: number;
  text: string;
  tokenCount: number;
};

function getLatestSummaryBoundary(
  payload: TPayload
): SummaryBoundary | undefined {
  let summaryBoundary: SummaryBoundary | undefined;

  for (let i = 0; i < payload.length; i++) {
    const message = payload[i];
    if (!Array.isArray(message.content)) {
      continue;
    }

    for (let j = 0; j < message.content.length; j++) {
      const part = message.content[j];
      if (part == null || part.type !== ContentTypes.SUMMARY) {
        continue;
      }

      const summaryPart = part as Partial<SummaryContentBlock>;
      const summaryText =
        typeof summaryPart.text === 'string' ? summaryPart.text.trim() : '';
      if (summaryText.length === 0) {
        continue;
      }

      summaryBoundary = {
        messageIndex: i,
        contentIndex: j,
        text: summaryText,
        tokenCount:
          typeof summaryPart.tokenCount === 'number' &&
          Number.isFinite(summaryPart.tokenCount)
            ? summaryPart.tokenCount
            : 0,
      };
    }
  }

  return summaryBoundary;
}

function applySummaryBoundary(
  message: Partial<TMessage>,
  messageIndex: number,
  summaryBoundary?: SummaryBoundary
): Partial<TMessage> | null {
  if (!summaryBoundary) {
    return message;
  }

  if (messageIndex < summaryBoundary.messageIndex) {
    return null;
  }

  if (
    messageIndex !== summaryBoundary.messageIndex ||
    !Array.isArray(message.content)
  ) {
    return message;
  }

  return {
    ...message,
    content: message.content.slice(summaryBoundary.contentIndex + 1),
  };
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
  indexTokenCountMap?: Record<number, number | undefined>,
  tools?: Set<string>
): {
  messages: Array<HumanMessage | AIMessage | SystemMessage | ToolMessage>;
  indexTokenCountMap?: Record<number, number>;
  /** Cross-run summary extracted from the payload. Should be forwarded to the
   *  agent run so it can be included in the system message via AgentContext. */
  summary?: { text: string; tokenCount: number };
} => {
  const messages: Array<
    HumanMessage | AIMessage | SystemMessage | ToolMessage
  > = [];
  // If indexTokenCountMap is provided, create a new map to track the updated indices
  const updatedIndexTokenCountMap: Record<number, number> = {};
  // Keep track of the mapping from original payload indices to result indices
  const indexMapping: Record<number, number[] | undefined> = {};
  const summaryBoundary = getLatestSummaryBoundary(payload);

  // Summary metadata is returned to the caller so it can be forwarded to the
  // agent run and included in the single system message via AgentContext.
  // We intentionally do NOT create a SystemMessage here — that would conflict
  // with the agent's own system message (instructions + summary combined).

  /**
   * Create a mutable copy of the tools set that can be expanded dynamically.
   * When we encounter tool_search results, we add discovered tools to this set,
   * making their subsequent tool calls valid.
   */
  const discoveredTools = tools ? new Set(tools) : undefined;

  // Process messages with tool conversion if tools set is provided
  for (let i = 0; i < payload.length; i++) {
    const rawMessage = payload[i];
    const sourceMessageId = getSourceMessageId(rawMessage);
    let message = applySummaryBoundary(rawMessage, i, summaryBoundary);
    if (!message) {
      indexMapping[i] = [];
      continue;
    }

    // Q: Store the current length of messages to track where this payload message starts in the result?
    // const startIndex = messages.length;
    if (typeof message.content === 'string') {
      message = {
        ...message,
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: message.content },
        ],
      };
    } else if (Array.isArray(message.content) && message.content.length === 0) {
      indexMapping[i] = [];
      continue;
    }

    if (message.role !== 'assistant') {
      const formattedMessage = formatMessage({
        message: message as MessageInput,
        langChain: true,
      }) as HumanMessage | AIMessage | SystemMessage;
      if (sourceMessageId) {
        formattedMessage.id = sourceMessageId;
      }
      messages.push(formattedMessage);

      // Update the index mapping for this message
      indexMapping[i] = [messages.length - 1];
      continue;
    }

    // For assistant messages, track the starting index before processing
    const startMessageIndex = messages.length;

    /**
     * If tools set is provided, process tool_calls:
     * - Keep valid tool_calls (tools in the set or dynamically discovered)
     * - Convert invalid tool_calls to string representation for context preservation
     * - Dynamically expand the set when tool_search results are encountered
     */
    let processedMessage = message;
    if (discoveredTools) {
      const content = message.content;
      if (content && Array.isArray(content)) {
        const filteredContent: typeof content = [];
        const invalidToolCallIds = new Set<string>();
        const invalidToolStrings: string[] = [];

        for (const part of content) {
          if (part.type !== ContentTypes.TOOL_CALL) {
            filteredContent.push(part);
            continue;
          }

          /** Skip malformed tool_call entries */
          if (
            part.tool_call == null ||
            part.tool_call.name == null ||
            part.tool_call.name === ''
          ) {
            if (
              typeof part.tool_call?.id === 'string' &&
              part.tool_call.id !== ''
            ) {
              invalidToolCallIds.add(part.tool_call.id);
            }
            continue;
          }

          const toolName = part.tool_call.name;

          /**
           * If this is a tool_search result with output, extract discovered tool names
           * and add them to the discoveredTools set for subsequent validation.
           */
          if (
            toolName === Constants.TOOL_SEARCH &&
            typeof part.tool_call.output === 'string' &&
            part.tool_call.output !== ''
          ) {
            const extracted = extractToolNamesFromSearchOutput(
              part.tool_call.output
            );
            for (const name of extracted) {
              discoveredTools.add(name);
            }
          }

          if (discoveredTools.has(toolName)) {
            /** Valid tool - keep it */
            filteredContent.push(part);
          } else {
            /** Invalid tool - convert to string for context preservation */
            if (
              typeof part.tool_call.id === 'string' &&
              part.tool_call.id !== ''
            ) {
              invalidToolCallIds.add(part.tool_call.id);
            }
            const output = part.tool_call.output ?? '';
            invalidToolStrings.push(`Tool: ${toolName}, ${output}`);
          }
        }

        /** Remove tool_call_ids references to invalid tools from text parts */
        if (invalidToolCallIds.size > 0) {
          for (const part of filteredContent) {
            if (
              part.type === ContentTypes.TEXT &&
              Array.isArray(part.tool_call_ids)
            ) {
              part.tool_call_ids = part.tool_call_ids.filter(
                (id: string) => !invalidToolCallIds.has(id)
              );
              if (part.tool_call_ids.length === 0) {
                delete part.tool_call_ids;
              }
            }
          }
        }

        /** Append invalid tool strings to the content for context preservation */
        if (invalidToolStrings.length > 0) {
          /** Find the last text part or create one */
          let lastTextPartIndex = -1;
          for (let j = filteredContent.length - 1; j >= 0; j--) {
            if (filteredContent[j].type === ContentTypes.TEXT) {
              lastTextPartIndex = j;
              break;
            }
          }

          const invalidToolText = invalidToolStrings.join('\n');
          if (lastTextPartIndex >= 0) {
            const lastTextPart = filteredContent[lastTextPartIndex] as {
              type: string;
              [ContentTypes.TEXT]?: string;
              text?: string;
            };
            const existingText =
              lastTextPart[ContentTypes.TEXT] ?? lastTextPart.text ?? '';
            filteredContent[lastTextPartIndex] = {
              ...lastTextPart,
              [ContentTypes.TEXT]: existingText
                ? `${existingText}\n${invalidToolText}`
                : invalidToolText,
            };
          } else {
            /** No text part exists, create one */
            filteredContent.push({
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: invalidToolText,
            });
          }
        }

        /** Use filtered content if we made any changes */
        if (
          filteredContent.length !== content.length ||
          invalidToolStrings.length > 0
        ) {
          processedMessage = { ...message, content: filteredContent };
        }
      }
    }

    // Process the assistant message using the helper function
    const formattedMessages = formatAssistantMessage(processedMessage);
    if (sourceMessageId) {
      for (const formattedMessage of formattedMessages) {
        formattedMessage.id = sourceMessageId;
      }
    }
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

  if (indexTokenCountMap) {
    for (
      let originalIndex = 0;
      originalIndex < payload.length;
      originalIndex++
    ) {
      const resultIndices = indexMapping[originalIndex] || [];
      const tokenCount = indexTokenCountMap[originalIndex];

      if (tokenCount === undefined) {
        continue;
      }

      const msgCount = resultIndices.length;
      if (msgCount === 1) {
        updatedIndexTokenCountMap[resultIndices[0]] = tokenCount;
        continue;
      }

      if (msgCount < 2) {
        continue;
      }

      let totalLength = 0;
      const lastIdx = msgCount - 1;
      const lengths = new Array<number>(msgCount);
      for (let k = 0; k < msgCount; k++) {
        const msg = messages[resultIndices[k]];
        const { content } = msg;
        let len = 0;
        if (typeof content === 'string') {
          len = content.length;
        } else if (Array.isArray(content)) {
          for (const part of content as Array<
            Record<string, unknown> | string | undefined
          >) {
            if (typeof part === 'string') {
              len += part.length;
            } else if (part != null && typeof part === 'object') {
              const val = part.text ?? part.content;
              if (typeof val === 'string') {
                len += val.length;
              }
            }
          }
        }
        const toolCalls = (msg as AIMessage).tool_calls;
        if (Array.isArray(toolCalls)) {
          for (const tc of toolCalls as Array<Record<string, unknown>>) {
            if (typeof tc.name === 'string') {
              len += tc.name.length;
            }
            const { args } = tc;
            if (typeof args === 'string') {
              len += args.length;
            } else if (args != null) {
              len += JSON.stringify(args).length;
            }
          }
        }
        lengths[k] = len;
        totalLength += len;
      }

      if (totalLength === 0) {
        const countPerMessage = Math.floor(tokenCount / msgCount);
        for (let k = 0; k < lastIdx; k++) {
          updatedIndexTokenCountMap[resultIndices[k]] = countPerMessage;
        }
        updatedIndexTokenCountMap[resultIndices[lastIdx]] =
          tokenCount - countPerMessage * lastIdx;
      } else {
        let distributed = 0;
        for (let k = 0; k < lastIdx; k++) {
          const share = Math.floor((lengths[k] / totalLength) * tokenCount);
          updatedIndexTokenCountMap[resultIndices[k]] = share;
          distributed += share;
        }
        updatedIndexTokenCountMap[resultIndices[lastIdx]] =
          tokenCount - distributed;
      }
    }
  }

  return {
    messages,
    indexTokenCountMap: indexTokenCountMap
      ? updatedIndexTokenCountMap
      : undefined,
    summary: summaryBoundary
      ? { text: summaryBoundary.text, tokenCount: summaryBoundary.tokenCount }
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
 * Converts AI messages with tool calls (that lack thinking/reasoning blocks) into buffer strings,
 * avoiding the thinking block signature requirement.
 *
 * Recognizes the following as valid thinking/reasoning blocks:
 * - ContentTypes.THINKING (Anthropic)
 * - ContentTypes.REASONING_CONTENT (Bedrock)
 * - ContentTypes.REASONING (VertexAI / Google)
 * - 'redacted_thinking'
 *
 * @param messages - Array of messages to process
 * @param provider - The provider being used (unused but kept for future compatibility)
 * @param config - Optional RunnableConfig for structured agent logging
 * @returns The messages array with tool sequences converted to buffer strings if necessary
 */
export function ensureThinkingBlockInMessages(
  messages: BaseMessage[],
  _provider: Providers,
  config?: RunnableConfig
): BaseMessage[] {
  if (messages.length === 0) {
    return messages;
  }

  // If the last message is already a HumanMessage, there is no trailing tool
  // sequence to convert — return early to preserve prompt caching and avoid
  // redundant token overhead from re-processing the entire history.
  const lastMsg = messages[messages.length - 1];
  const lastIsHuman =
    lastMsg instanceof HumanMessage ||
    ('role' in lastMsg && (lastMsg as any).role === 'user');
  if (lastIsHuman) {
    return messages;
  }

  const result: BaseMessage[] = [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];
    /** Detect AI messages by instanceof OR by role, in case cache-control cloning
     produced a plain object that lost the LangChain prototype. */
    const isAI =
      msg instanceof AIMessage ||
      msg instanceof AIMessageChunk ||
      ('role' in msg && (msg as any).role === 'assistant');

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
    let hasThinkingBlock = false;

    if (contentIsArray && aiMsg.content.length > 0) {
      const content = aiMsg.content as ExtendedMessageContent[];
      hasToolUse =
        hasToolUse ||
        content.some((c) => typeof c === 'object' && c.type === 'tool_use');
      // Check ALL content blocks for thinking/reasoning, not just [0].
      // Bedrock may emit a whitespace text chunk before the thinking block,
      // pushing the reasoning_content to index 1+.
      hasThinkingBlock = content.some(
        (c) =>
          typeof c === 'object' &&
          (c.type === ContentTypes.THINKING ||
            c.type === ContentTypes.REASONING_CONTENT ||
            c.type === ContentTypes.REASONING ||
            c.type === 'redacted_thinking')
      );
    }

    // Bedrock also stores reasoning in additional_kwargs (may not be in content array)
    if (
      !hasThinkingBlock &&
      aiMsg.additional_kwargs.reasoning_content != null
    ) {
      hasThinkingBlock = true;
    }

    // If message has tool use but no thinking block, check whether this is a
    // continuation of a thinking-enabled agent's chain before converting.
    // Bedrock reasoning models can produce multiple AI→Tool rounds after an
    // initial reasoning response: the first AI message has reasoning_content,
    // but follow-ups have content: "" with only tool_calls. These are the
    // same agent's turn and must NOT be converted to HumanMessages.
    if (hasToolUse && !hasThinkingBlock) {
      // Walk backwards — if an earlier AI message in the same chain (before
      // the nearest HumanMessage) has a thinking/reasoning block, this is a
      // continuation of a thinking-enabled turn, not a non-thinking handoff.
      if (chainHasThinkingBlock(messages, i)) {
        result.push(msg);
        i++;
        continue;
      }

      // Collect the AI message and any following tool messages
      const toolSequence: BaseMessage[] = [msg];
      let j = i + 1;

      // Look ahead for tool messages that belong to this AI message
      const isToolMsg = (m: BaseMessage): boolean =>
        m instanceof ToolMessage || ('role' in m && (m as any).role === 'tool');
      while (j < messages.length && isToolMsg(messages[j])) {
        toolSequence.push(messages[j]);
        j++;
      }

      // Convert the sequence to a buffer string and wrap in a HumanMessage
      // This avoids the thinking block requirement which only applies to AI messages
      const bufferString = getBufferString(toolSequence);
      emitAgentLog(
        config,
        'warn',
        'format',
        'ensureThinkingBlockInMessages: injecting [Previous agent context] HumanMessage' +
          ` (${toolSequence.length} msgs at index ${i}, no thinking block in chain)`
      );
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

/**
 * Walks backwards from `currentIndex` through the message array to check
 * whether an earlier AI message in the same "chain" (no HumanMessage boundary)
 * contains a thinking/reasoning block.
 *
 * A "chain" is a contiguous sequence of AI + Tool messages with no intervening
 * HumanMessage. Bedrock reasoning models produce reasoning on the first AI
 * response, then issue follow-up tool calls with `content: ""` and no
 * reasoning block. These follow-ups are part of the same thinking-enabled
 * turn and should not be converted.
 */
function chainHasThinkingBlock(
  messages: BaseMessage[],
  currentIndex: number
): boolean {
  for (let k = currentIndex - 1; k >= 0; k--) {
    const prev = messages[k];

    // HumanMessage = turn boundary — stop searching
    if (
      prev instanceof HumanMessage ||
      ('role' in prev && (prev as any).role === 'user')
    ) {
      return false;
    }

    // Check AI messages for thinking/reasoning blocks
    const isPrevAI =
      prev instanceof AIMessage ||
      prev instanceof AIMessageChunk ||
      ('role' in prev && (prev as any).role === 'assistant');

    if (isPrevAI) {
      const prevAiMsg = prev as AIMessage | AIMessageChunk;

      if (Array.isArray(prevAiMsg.content) && prevAiMsg.content.length > 0) {
        const content = prevAiMsg.content as ExtendedMessageContent[];
        if (
          content.some(
            (c) =>
              typeof c === 'object' &&
              (c.type === ContentTypes.THINKING ||
                c.type === ContentTypes.REASONING_CONTENT ||
                c.type === ContentTypes.REASONING ||
                c.type === 'redacted_thinking')
          )
        ) {
          return true;
        }
      }

      // Bedrock also stores reasoning in additional_kwargs
      if (prevAiMsg.additional_kwargs.reasoning_content != null) {
        return true;
      }
    }

    // ToolMessages are part of the chain — keep walking back
  }

  return false;
}
