/**
 * Utility functions for converting Bedrock Converse responses to LangChain messages.
 * Ported from @langchain/aws common.js
 */
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type {
  BedrockMessage,
  ConverseResponse,
  ContentBlockDeltaEvent,
  ConverseStreamMetadataEvent,
  ContentBlockStartEvent,
  ReasoningContentBlock,
  ReasoningContentBlockDelta,
  MessageContentReasoningBlock,
  MessageContentReasoningBlockReasoningTextPartial,
  MessageContentReasoningBlockRedacted,
} from '../types';

/**
 * Convert a Bedrock reasoning block delta to a LangChain partial reasoning block.
 */
export function bedrockReasoningDeltaToLangchainPartialReasoningBlock(
  reasoningContent: ReasoningContentBlockDelta
):
  | MessageContentReasoningBlockReasoningTextPartial
  | MessageContentReasoningBlockRedacted {
  const { text, redactedContent, signature } =
    reasoningContent as ReasoningContentBlockDelta & {
      text?: string;
      redactedContent?: Uint8Array;
      signature?: string;
    };

  if (typeof text === 'string') {
    return {
      type: 'reasoning_content',
      reasoningText: { text },
    };
  }
  if (signature != null) {
    return {
      type: 'reasoning_content',
      reasoningText: { signature },
    };
  }
  if (redactedContent != null) {
    return {
      type: 'reasoning_content',
      redactedContent: Buffer.from(redactedContent).toString('base64'),
    };
  }
  throw new Error('Invalid reasoning content');
}

/**
 * Convert a Bedrock reasoning block to a LangChain reasoning block.
 */
export function bedrockReasoningBlockToLangchainReasoningBlock(
  reasoningContent: ReasoningContentBlock
): MessageContentReasoningBlock {
  const { reasoningText, redactedContent } =
    reasoningContent as ReasoningContentBlock & {
      reasoningText?: { text?: string; signature?: string };
      redactedContent?: Uint8Array;
    };

  if (reasoningText != null) {
    return {
      type: 'reasoning_content',
      reasoningText: reasoningText,
    };
  }
  if (redactedContent != null) {
    return {
      type: 'reasoning_content',
      redactedContent: Buffer.from(redactedContent).toString('base64'),
    };
  }
  throw new Error('Invalid reasoning content');
}

/**
 * Convert a Bedrock Converse message to a LangChain message.
 */
export function convertConverseMessageToLangChainMessage(
  message: BedrockMessage,
  responseMetadata: Omit<ConverseResponse, 'output'>
): AIMessage {
  if (message.content == null) {
    throw new Error('No message content found in response.');
  }
  if (message.role !== 'assistant') {
    throw new Error(
      `Unsupported message role received in ChatBedrockConverse response: ${message.role}`
    );
  }

  let requestId: string | undefined;
  if (
    '$metadata' in responseMetadata &&
    responseMetadata.$metadata != null &&
    typeof responseMetadata.$metadata === 'object' &&
    'requestId' in responseMetadata.$metadata
  ) {
    requestId = responseMetadata.$metadata.requestId as string;
  }

  let tokenUsage:
    | { input_tokens: number; output_tokens: number; total_tokens: number }
    | undefined;
  if (responseMetadata.usage != null) {
    const input_tokens = responseMetadata.usage.inputTokens ?? 0;
    const output_tokens = responseMetadata.usage.outputTokens ?? 0;
    tokenUsage = {
      input_tokens,
      output_tokens,
      total_tokens:
        responseMetadata.usage.totalTokens ?? input_tokens + output_tokens,
    };
  }

  if (
    message.content.length === 1 &&
    'text' in message.content[0] &&
    typeof message.content[0].text === 'string'
  ) {
    return new AIMessage({
      content: message.content[0].text,
      response_metadata: responseMetadata,
      usage_metadata: tokenUsage,
      id: requestId,
    });
  } else {
    const toolCalls: Array<{
      id?: string;
      name: string;
      args: Record<string, unknown>;
      type: 'tool_call';
    }> = [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const content: any[] = [];

    message.content.forEach((c) => {
      if (
        'toolUse' in c &&
        c.toolUse != null &&
        c.toolUse.name != null &&
        c.toolUse.name !== '' &&
        c.toolUse.input != null &&
        typeof c.toolUse.input === 'object'
      ) {
        toolCalls.push({
          id: c.toolUse.toolUseId,
          name: c.toolUse.name,
          args: c.toolUse.input as Record<string, unknown>,
          type: 'tool_call',
        });
      } else if ('text' in c && typeof c.text === 'string') {
        content.push({ type: 'text', text: c.text });
      } else if ('reasoningContent' in c && c.reasoningContent != null) {
        content.push(
          bedrockReasoningBlockToLangchainReasoningBlock(c.reasoningContent)
        );
      } else {
        content.push(c);
      }
    });

    return new AIMessage({
      content: content.length ? content : '',
      tool_calls: toolCalls.length ? toolCalls : undefined,
      response_metadata: responseMetadata,
      usage_metadata: tokenUsage,
      id: requestId,
    });
  }
}

/**
 * Handle a content block delta event from Bedrock Converse stream.
 */
export function handleConverseStreamContentBlockDelta(
  contentBlockDelta: ContentBlockDeltaEvent
): ChatGenerationChunk {
  if (contentBlockDelta.delta == null) {
    throw new Error('No delta found in content block.');
  }

  if (typeof contentBlockDelta.delta.text === 'string') {
    return new ChatGenerationChunk({
      text: contentBlockDelta.delta.text,
      message: new AIMessageChunk({
        content: contentBlockDelta.delta.text,
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
        },
      }),
    });
  } else if (contentBlockDelta.delta.toolUse != null) {
    const index = contentBlockDelta.contentBlockIndex;
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            args: contentBlockDelta.delta.toolUse.input as string,
            index,
            type: 'tool_call_chunk',
          },
        ],
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
        },
      }),
    });
  } else if (contentBlockDelta.delta.reasoningContent != null) {
    const reasoningBlock =
      bedrockReasoningDeltaToLangchainPartialReasoningBlock(
        contentBlockDelta.delta.reasoningContent
      );
    let reasoningText = '';
    if ('reasoningText' in reasoningBlock) {
      reasoningText = reasoningBlock.reasoningText.text ?? '';
    } else if ('redactedContent' in reasoningBlock) {
      reasoningText = reasoningBlock.redactedContent;
    }
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: [reasoningBlock],
        additional_kwargs: {
          // Set reasoning_content for stream handler to detect reasoning mode
          reasoning_content: reasoningText,
        },
        response_metadata: {
          contentBlockIndex: contentBlockDelta.contentBlockIndex,
        },
      }),
    });
  } else {
    throw new Error(
      `Unsupported content block type(s): ${JSON.stringify(contentBlockDelta.delta, null, 2)}`
    );
  }
}

/**
 * Handle a content block start event from Bedrock Converse stream.
 */
export function handleConverseStreamContentBlockStart(
  contentBlockStart: ContentBlockStartEvent
): ChatGenerationChunk | null {
  const index = contentBlockStart.contentBlockIndex;

  if (contentBlockStart.start?.toolUse != null) {
    return new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          {
            name: contentBlockStart.start.toolUse.name,
            id: contentBlockStart.start.toolUse.toolUseId,
            index,
            type: 'tool_call_chunk',
          },
        ],
        response_metadata: {
          contentBlockIndex: index,
        },
      }),
    });
  }

  // Return null for non-tool content block starts (text blocks don't need special handling)
  return null;
}

/**
 * Handle a metadata event from Bedrock Converse stream.
 */
export function handleConverseStreamMetadata(
  metadata: ConverseStreamMetadataEvent,
  extra: { streamUsage: boolean }
): ChatGenerationChunk {
  const inputTokens = metadata.usage?.inputTokens ?? 0;
  const outputTokens = metadata.usage?.outputTokens ?? 0;
  const usage_metadata = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: metadata.usage?.totalTokens ?? inputTokens + outputTokens,
  };

  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: '',
      usage_metadata: extra.streamUsage ? usage_metadata : undefined,
      response_metadata: {
        // Use the same key as returned from the Converse API
        metadata,
      },
    }),
  });
}
