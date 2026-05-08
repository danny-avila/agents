/**
 * Utility functions for converting LangChain messages to Bedrock Converse messages.
 * Ported from @langchain/aws common.js
 */
import {
  type BaseMessage,
  isAIMessage,
  type Data,
  parseBase64DataUrl,
  parseMimeType,
  MessageContentComplex,
  type StandardContentBlockConverter,
  convertToProviderContentBlock,
  isDataContentBlock,
} from '@langchain/core/messages';
import type {
  AudioFormat,
  AudioSource,
  DocumentFormat,
  DocumentSource,
  VideoFormat,
  VideoSource,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  BedrockMessage,
  BedrockSystemContentBlock,
  BedrockContentBlock,
  MessageContentReasoningBlock,
} from '../types';

/**
 * Convert a LangChain reasoning block to a Bedrock reasoning block.
 */
export function langchainReasoningBlockToBedrockReasoningBlock(
  content: MessageContentReasoningBlock
): {
  reasoningText?: { text?: string; signature?: string };
  redactedContent?: Uint8Array;
} {
  if (content.reasoningText != null) {
    return {
      reasoningText: content.reasoningText,
    };
  }
  if (content.redactedContent != null && content.redactedContent !== '') {
    return {
      redactedContent: new Uint8Array(
        Buffer.from(content.redactedContent, 'base64')
      ),
    };
  }
  throw new Error('Invalid reasoning content');
}

/**
 * Concatenate consecutive reasoning blocks in content array.
 */
export function concatenateLangchainReasoningBlocks(
  content: Array<MessageContentComplex | MessageContentReasoningBlock>
): Array<MessageContentComplex | MessageContentReasoningBlock> {
  const result: Array<MessageContentComplex | MessageContentReasoningBlock> =
    [];

  for (const block of content) {
    if (block.type === 'reasoning_content') {
      const currentReasoning = block as MessageContentReasoningBlock;
      const lastIndex = result.length - 1;

      // Check if we can merge with the previous block
      if (lastIndex >= 0) {
        const lastBlock = result[lastIndex];
        if (
          lastBlock.type === 'reasoning_content' &&
          (lastBlock as MessageContentReasoningBlock).reasoningText != null &&
          currentReasoning.reasoningText != null
        ) {
          const lastReasoning = lastBlock as MessageContentReasoningBlock;
          // Merge consecutive reasoning text blocks
          const lastText = lastReasoning.reasoningText?.text;
          const currentText = currentReasoning.reasoningText.text;
          if (
            lastText != null &&
            lastText !== '' &&
            currentText != null &&
            currentText !== ''
          ) {
            lastReasoning.reasoningText!.text = lastText + currentText;
          } else if (
            currentReasoning.reasoningText.signature != null &&
            currentReasoning.reasoningText.signature !== ''
          ) {
            lastReasoning.reasoningText!.signature =
              currentReasoning.reasoningText.signature;
          }
          continue;
        }
      }

      result.push({ ...block } as MessageContentReasoningBlock);
    } else {
      result.push(block);
    }
  }

  return result;
}

/**
 * Extract image info from a base64 string or URL.
 */
export function extractImageInfo(base64: string): BedrockContentBlock {
  // Extract the format from the base64 string
  const formatMatch = base64.match(/^data:image\/(\w+);base64,/);
  let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
  if (formatMatch) {
    const extractedFormat = formatMatch[1].toLowerCase();
    if (['gif', 'jpeg', 'png', 'webp'].includes(extractedFormat)) {
      format = extractedFormat as typeof format;
    }
  }

  // Remove the data URL prefix if present
  const base64Data = base64.replace(/^data:image\/\w+;base64,/, '');

  // Convert base64 to Uint8Array
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return {
    image: {
      format,
      source: {
        bytes,
      },
    },
  };
}

type MediaContentBlock = MessageContentComplex & {
  data?: string | Uint8Array;
  url?: string;
  fileId?: string;
  mimeType?: string;
};

const mimeTypeToVideoFormat: Record<string, VideoFormat> = {
  'video/flv': 'flv',
  'video/mkv': 'mkv',
  'video/mov': 'mov',
  'video/mp4': 'mp4',
  'video/mpeg': 'mpeg',
  'video/mpg': 'mpg',
  'video/three_gp': 'three_gp',
  'video/webm': 'webm',
  'video/wmv': 'wmv',
};

const mimeTypeToAudioFormat: Record<string, AudioFormat> = {
  'audio/aac': 'aac',
  'audio/flac': 'flac',
  'audio/m4a': 'm4a',
  'audio/mka': 'mka',
  'audio/mkv': 'mkv',
  'audio/mp3': 'mp3',
  'audio/mp4': 'mp4',
  'audio/mpeg': 'mpeg',
  'audio/mpga': 'mpga',
  'audio/ogg': 'ogg',
  'audio/opus': 'opus',
  'audio/pcm': 'pcm',
  'audio/wav': 'wav',
  'audio/webm': 'webm',
  'audio/x-aac': 'x-aac',
};

const mimeTypeToDocumentFormat: Partial<Record<string, DocumentFormat>> = {
  'text/csv': 'csv',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
    'docx',
  'text/html': 'html',
  'text/markdown': 'md',
  'application/pdf': 'pdf',
  'text/plain': 'txt',
  'application/vnd.ms-excel': 'xls',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
};

function base64ToBytes(data: string): Uint8Array {
  return Uint8Array.from(atob(data), (char) => char.charCodeAt(0));
}

function getMediaFormat<T extends string>(
  mimeType: string | undefined,
  formatMap: Record<string, T>
): T | undefined {
  if (mimeType == null || mimeType === '') {
    return undefined;
  }
  return formatMap[mimeType] ?? (parseMimeType(mimeType).subtype as T);
}

function resolveMediaSource(
  block: MediaContentBlock
): AudioSource | VideoSource {
  if (typeof block.data === 'string') {
    return { bytes: base64ToBytes(block.data) };
  }
  if (block.data instanceof Uint8Array) {
    return { bytes: block.data };
  }
  if (typeof block.url === 'string') {
    const parsedData = parseBase64DataUrl({
      dataUrl: block.url,
      asTypedArray: true,
    });
    if (parsedData != null) {
      return { bytes: parsedData.data as Uint8Array };
    }
    throw new Error(
      `Only base64 data URLs are supported for ${block.type} blocks with 'url' field with ChatBedrockConverse.`
    );
  }
  if (typeof block.fileId === 'string') {
    return { s3Location: { uri: block.fileId } };
  }
  throw new Error(
    `${block.type} block must include one of: 'data' (base64 string or Uint8Array), 'url' (base64 data URL), or 'fileId' (S3 URI).`
  );
}

function convertMultimodalVideoBlock(
  block: MediaContentBlock
): BedrockContentBlock {
  return {
    video: {
      format: getMediaFormat(block.mimeType, mimeTypeToVideoFormat),
      source: resolveMediaSource(block) as VideoSource,
    },
  } as BedrockContentBlock;
}

function convertMultimodalAudioBlock(
  block: MediaContentBlock
): BedrockContentBlock {
  return {
    audio: {
      format: getMediaFormat(block.mimeType, mimeTypeToAudioFormat),
      source: resolveMediaSource(block) as AudioSource,
    },
  } as BedrockContentBlock;
}

function getDocumentName(block: Data.StandardFileBlock): string {
  return (
    (block.metadata?.name as string | undefined) ??
    (block.metadata?.filename as string | undefined) ??
    (block.metadata?.title as string | undefined) ??
    globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 12)
  );
}

function getDocumentFormat(
  mimeType: string | undefined
): DocumentFormat | undefined {
  if (mimeType == null || mimeType === '') {
    return undefined;
  }
  const parsedMimeType = parseMimeType(mimeType);
  const format =
    mimeTypeToDocumentFormat[
      `${parsedMimeType.type}/${parsedMimeType.subtype}`
    ];
  if (format === undefined) {
    throw new Error(
      `Unsupported file mime type: "${mimeType}" ChatBedrockConverse only supports ${Object.keys(
        mimeTypeToDocumentFormat
      ).join(', ')} formats.`
    );
  }
  return format;
}

const standardContentBlockConverter: StandardContentBlockConverter<{
  text: BedrockContentBlock;
  image: BedrockContentBlock;
  file: BedrockContentBlock;
}> = {
  providerName: 'ChatBedrockConverse',

  fromStandardTextBlock(block: Data.StandardTextBlock): BedrockContentBlock {
    return { text: block.text };
  },

  fromStandardImageBlock(block: Data.StandardImageBlock): BedrockContentBlock {
    if (block.source_type === 'url') {
      const parsedData = parseBase64DataUrl({
        dataUrl: block.url,
        asTypedArray: true,
      });
      if (parsedData == null) {
        throw new Error(
          [
            'Only base64 data URLs are supported for image blocks with source type ',
            'url',
            ' with ChatBedrockConverse.',
          ].join(String.fromCharCode(39))
        );
      }
      return {
        image: {
          format: parseMimeType(parsedData.mime_type).subtype as
            | 'gif'
            | 'jpeg'
            | 'png'
            | 'webp',
          source: { bytes: parsedData.data as Uint8Array },
        },
      };
    }
    if (block.source_type === 'base64') {
      let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
      if (block.mime_type != null && block.mime_type !== '') {
        format = parseMimeType(block.mime_type).subtype as typeof format;
      }
      if (format != null && !['gif', 'jpeg', 'png', 'webp'].includes(format)) {
        throw new Error(
          `Unsupported image mime type: "${block.mime_type}" ChatBedrockConverse only supports "image/gif", "image/jpeg", "image/png", and "image/webp" formats.`
        );
      }
      return {
        image: {
          format,
          source: { bytes: base64ToBytes(block.data) },
        },
      };
    }
    throw new Error(
      `Image source type '${block.source_type}' not supported with ChatBedrockConverse.`
    );
  },

  fromStandardFileBlock(block: Data.StandardFileBlock): BedrockContentBlock {
    const name = getDocumentName(block);
    if (block.source_type === 'text') {
      return {
        document: {
          name,
          format: 'txt',
          source: { bytes: new TextEncoder().encode(block.text) },
        },
      } as BedrockContentBlock;
    }
    if (block.source_type === 'url') {
      const parsedData = parseBase64DataUrl({
        dataUrl: block.url,
        asTypedArray: true,
      });
      if (parsedData == null) {
        throw new Error(
          [
            'Only base64 data URLs are supported for file blocks with source type ',
            'url',
            ' with ChatBedrockConverse.',
          ].join(String.fromCharCode(39))
        );
      }
      return {
        document: {
          name,
          format: getDocumentFormat(parsedData.mime_type),
          source: { bytes: parsedData.data as Uint8Array } as DocumentSource,
        },
      } as BedrockContentBlock;
    }
    if (block.source_type === 'base64') {
      return {
        document: {
          name,
          format: getDocumentFormat(block.mime_type),
          source: { bytes: base64ToBytes(block.data) } as DocumentSource,
        },
      } as BedrockContentBlock;
    }
    throw new Error(
      `File source type '${block.source_type}' not supported with ChatBedrockConverse.`
    );
  },
};

/**
 * Check if a block has a cache point.
 */
function isDefaultCachePoint(block: unknown): boolean {
  if (typeof block !== 'object' || block === null) {
    return false;
  }
  if (!('cachePoint' in block)) {
    return false;
  }
  const cachePoint = (block as { cachePoint?: unknown }).cachePoint;
  if (typeof cachePoint !== 'object' || cachePoint === null) {
    return false;
  }
  if (!('type' in cachePoint)) {
    return false;
  }
  return (cachePoint as { type?: string }).type === 'default';
}

/**
 * Convert a LangChain content block to a Bedrock Converse content block.
 */
function convertLangChainContentBlockToConverseContentBlock({
  block,
  onUnknown = 'throw',
}: {
  block: string | MessageContentComplex;
  onUnknown?: 'throw' | 'passthrough';
}): BedrockContentBlock {
  if (typeof block === 'string') {
    return { text: block };
  }

  if (isDataContentBlock(block)) {
    return convertToProviderContentBlock(block, standardContentBlockConverter);
  }

  if (block.type === 'text') {
    return { text: (block as { text: string }).text };
  }

  if (block.type === 'image_url') {
    const imageUrl =
      typeof (block as { image_url: string | { url: string } }).image_url ===
      'string'
        ? (block as { image_url: string }).image_url
        : (block as { image_url: { url: string } }).image_url.url;
    return extractImageInfo(imageUrl);
  }

  if (block.type === 'image') {
    // Handle standard image block format
    const imageBlock = block as {
      source_type?: string;
      url?: string;
      data?: string;
      mime_type?: string;
    };
    if (
      imageBlock.source_type === 'url' &&
      imageBlock.url != null &&
      imageBlock.url !== ''
    ) {
      const parsedData = parseBase64DataUrl({
        dataUrl: imageBlock.url,
        asTypedArray: true,
      });
      if (parsedData != null) {
        const parsedMimeType = parseMimeType(parsedData.mime_type);
        return {
          image: {
            format: parsedMimeType.subtype as 'gif' | 'jpeg' | 'png' | 'webp',
            source: {
              bytes: parsedData.data as Uint8Array,
            },
          },
        };
      }
    } else if (
      imageBlock.source_type === 'base64' &&
      imageBlock.data != null &&
      imageBlock.data !== ''
    ) {
      let format: 'gif' | 'jpeg' | 'png' | 'webp' | undefined;
      if (imageBlock.mime_type != null && imageBlock.mime_type !== '') {
        const parsedMimeType = parseMimeType(imageBlock.mime_type);
        format = parsedMimeType.subtype as typeof format;
      }
      return {
        image: {
          format,
          source: {
            bytes: Uint8Array.from(atob(imageBlock.data), (c) =>
              c.charCodeAt(0)
            ),
          },
        },
      };
    }
    // If it already has the Bedrock image structure, pass through
    if ((block as { image?: unknown }).image !== undefined) {
      return {
        image: (block as { image: unknown }).image,
      } as BedrockContentBlock;
    }
  }

  if (
    block.type === 'video' &&
    (block as { video?: unknown }).video !== undefined
  ) {
    return {
      video: (block as { video: unknown }).video,
    } as BedrockContentBlock;
  }

  if (block.type === 'video') {
    return convertMultimodalVideoBlock(block as MediaContentBlock);
  }

  if (
    block.type === 'audio' &&
    (block as { audio?: unknown }).audio !== undefined
  ) {
    return {
      audio: (block as { audio: unknown }).audio,
    } as BedrockContentBlock;
  }

  if (block.type === 'audio') {
    return convertMultimodalAudioBlock(block as MediaContentBlock);
  }

  if (
    block.type === 'document' &&
    (block as { document?: unknown }).document !== undefined
  ) {
    return {
      document: (block as { document: unknown }).document,
    } as BedrockContentBlock;
  }

  if (isDefaultCachePoint(block)) {
    return {
      cachePoint: {
        type: 'default',
      },
    } as BedrockContentBlock;
  }

  if (onUnknown === 'throw') {
    throw new Error(`Unsupported content block type: ${block.type}`);
  } else {
    return block as unknown as BedrockContentBlock;
  }
}

/**
 * Convert a system message to Bedrock system content blocks.
 */
function convertSystemMessageToConverseMessage(
  msg: BaseMessage
): BedrockSystemContentBlock[] {
  if (typeof msg.content === 'string') {
    return [{ text: msg.content }];
  } else if (Array.isArray(msg.content) && msg.content.length > 0) {
    const contentBlocks: BedrockSystemContentBlock[] = [];
    for (const block of msg.content) {
      if (
        typeof block === 'object' &&
        block.type === 'text' &&
        typeof (block as { text?: string }).text === 'string'
      ) {
        contentBlocks.push({
          text: (block as { text: string }).text,
        });
      } else if (isDefaultCachePoint(block)) {
        contentBlocks.push({
          cachePoint: {
            type: 'default',
          },
        } as BedrockSystemContentBlock);
      } else {
        break;
      }
    }
    if (msg.content.length === contentBlocks.length) {
      return contentBlocks;
    }
  }
  throw new Error(
    'System message content must be either a string, or an array of text blocks, optionally including a cache point.'
  );
}

/**
 * Convert an AI message to a Bedrock message.
 */
function convertAIMessageToConverseMessage(msg: BaseMessage): BedrockMessage {
  // Check for v1 format from other providers (PR #9766 fix)
  const responseMetadata = msg.response_metadata as
    | { output_version?: string }
    | undefined;
  if (responseMetadata?.output_version === 'v1') {
    return convertFromV1ToChatBedrockConverseMessage(msg);
  }

  const assistantMsg: BedrockMessage = {
    role: 'assistant',
    content: [],
  };

  if (typeof msg.content === 'string' && msg.content !== '') {
    assistantMsg.content?.push({ text: msg.content });
  } else if (Array.isArray(msg.content)) {
    const concatenatedBlocks = concatenateLangchainReasoningBlocks(
      msg.content as Array<MessageContentComplex | MessageContentReasoningBlock>
    );
    const contentBlocks: BedrockContentBlock[] = [];

    concatenatedBlocks.forEach((block) => {
      if (block.type === 'text') {
        const text = (block as { text?: string }).text ?? '';
        // Skip completely empty text blocks (common in AI messages with tool_use blocks)
        if (text === '') {
          return;
        }
        // Merge whitespace/newlines with previous text blocks to avoid validation errors.
        const cleanedText = text.replace(/\n/g, '').trim();
        if (cleanedText === '') {
          if (contentBlocks.length > 0) {
            const lastBlock = contentBlocks[contentBlocks.length - 1];
            if ('text' in lastBlock) {
              const mergedTextContent = `${lastBlock.text}${text}`;
              (lastBlock as { text: string }).text = mergedTextContent;
            }
          }
        } else {
          contentBlocks.push({ text });
        }
      } else if (block.type === 'reasoning_content') {
        contentBlocks.push({
          reasoningContent: langchainReasoningBlockToBedrockReasoningBlock(
            block as MessageContentReasoningBlock
          ),
        } as BedrockContentBlock);
      } else if (isDefaultCachePoint(block)) {
        contentBlocks.push({
          cachePoint: {
            type: 'default',
          },
        } as BedrockContentBlock);
      } else {
        const blockValues = Object.fromEntries(
          Object.entries(block).filter(([key]) => key !== 'type')
        );
        throw new Error(
          `Unsupported content block type: ${block.type} with content of ${JSON.stringify(blockValues, null, 2)}`
        );
      }
    });

    assistantMsg.content = [...(assistantMsg.content ?? []), ...contentBlocks];
  }

  // Important: this must be placed after any reasoning content blocks
  if (isAIMessage(msg) && msg.tool_calls != null && msg.tool_calls.length > 0) {
    const toolUseBlocks = msg.tool_calls.map((tc) => ({
      toolUse: {
        toolUseId: tc.id,
        name: tc.name,
        input: tc.args as Record<string, unknown>,
      },
    }));
    assistantMsg.content = [
      ...(assistantMsg.content ?? []),
      ...toolUseBlocks,
    ] as BedrockContentBlock[];
  }

  return assistantMsg;
}

/**
 * Convert a v1 format message from other providers to Bedrock format.
 * This handles messages with standard content blocks like tool_call and reasoning.
 * (Implements PR #9766 fix for output_version v1 detection)
 */
function convertFromV1ToChatBedrockConverseMessage(
  msg: BaseMessage
): BedrockMessage {
  const assistantMsg: BedrockMessage = {
    role: 'assistant',
    content: [],
  };

  if (Array.isArray(msg.content)) {
    for (const block of msg.content as Array<
      MessageContentComplex | MessageContentReasoningBlock
    >) {
      if (typeof block === 'string') {
        assistantMsg.content?.push({ text: block });
      } else if (block.type === 'text') {
        assistantMsg.content?.push({ text: (block as { text: string }).text });
      } else if (block.type === 'tool_call') {
        const toolCall = block as {
          id: string;
          name: string;
          args: Record<string, unknown>;
        };
        assistantMsg.content?.push({
          toolUse: {
            toolUseId: toolCall.id,
            name: toolCall.name,
            input: toolCall.args as Record<string, unknown>,
          },
        } as BedrockContentBlock);
      } else if (block.type === 'reasoning') {
        const reasoning = block as { reasoning: string };
        assistantMsg.content?.push({
          reasoningContent: {
            reasoningText: { text: reasoning.reasoning },
          },
        } as BedrockContentBlock);
      } else if (block.type === 'reasoning_content') {
        assistantMsg.content?.push({
          reasoningContent: langchainReasoningBlockToBedrockReasoningBlock(
            block as MessageContentReasoningBlock
          ),
        } as BedrockContentBlock);
      }
    }
  } else if (typeof msg.content === 'string' && msg.content !== '') {
    assistantMsg.content?.push({ text: msg.content });
  }

  // Also handle tool_calls from the message
  if (isAIMessage(msg) && msg.tool_calls != null && msg.tool_calls.length > 0) {
    // Check if tool calls are already in content
    const existingToolUseIds = new Set(
      assistantMsg.content
        ?.filter((c) => 'toolUse' in c)
        .map(
          (c) => (c as { toolUse: { toolUseId: string } }).toolUse.toolUseId
        ) ?? []
    );

    for (const tc of msg.tool_calls) {
      if (!existingToolUseIds.has(tc.id ?? '')) {
        assistantMsg.content?.push({
          toolUse: {
            toolUseId: tc.id,
            name: tc.name,
            input: tc.args as Record<string, unknown>,
          },
        } as BedrockContentBlock);
      }
    }
  }

  return assistantMsg;
}

/**
 * Convert a human message to a Bedrock message.
 */
function convertHumanMessageToConverseMessage(
  msg: BaseMessage
): BedrockMessage {
  const userMessage: BedrockMessage = {
    role: 'user',
    content: [],
  };

  if (typeof msg.content === 'string') {
    userMessage.content = [{ text: msg.content }];
  } else if (Array.isArray(msg.content)) {
    userMessage.content = msg.content.map((block) =>
      convertLangChainContentBlockToConverseContentBlock({ block })
    );
  }

  return userMessage;
}

/**
 * Convert a tool message to a Bedrock message.
 */
function convertToolMessageToConverseMessage(msg: BaseMessage): BedrockMessage {
  const toolCallId = (msg as { tool_call_id?: string }).tool_call_id;

  let content: BedrockContentBlock[];
  if (typeof msg.content === 'string') {
    content = [{ text: msg.content }];
  } else if (Array.isArray(msg.content)) {
    content = msg.content.map((block) =>
      convertLangChainContentBlockToConverseContentBlock({
        block,
        onUnknown: 'passthrough',
      })
    );
  } else {
    content = [{ text: String(msg.content) }];
  }

  return {
    role: 'user',
    content: [
      {
        toolResult: {
          toolUseId: toolCallId,
          content: content as { text: string }[],
        },
      },
    ],
  };
}

/**
 * Convert LangChain messages to Bedrock Converse messages.
 */
export function convertToConverseMessages(messages: BaseMessage[]): {
  converseMessages: BedrockMessage[];
  converseSystem: BedrockSystemContentBlock[];
} {
  const converseSystem = messages
    .filter((msg) => msg._getType() === 'system')
    .flatMap((msg) => convertSystemMessageToConverseMessage(msg));

  const converseMessages = messages
    .filter((msg) => msg._getType() !== 'system')
    .map((msg) => {
      if (msg._getType() === 'ai') {
        return convertAIMessageToConverseMessage(msg);
      } else if (msg._getType() === 'human' || msg._getType() === 'generic') {
        return convertHumanMessageToConverseMessage(msg);
      } else if (msg._getType() === 'tool') {
        return convertToolMessageToConverseMessage(msg);
      } else {
        throw new Error(`Unsupported message type: ${msg._getType()}`);
      }
    });

  // Combine consecutive user tool result messages into a single message
  const combinedConverseMessages = converseMessages.reduce<BedrockMessage[]>(
    (acc, curr) => {
      if (acc.length === 0) {
        acc.push(curr);
        return acc;
      }
      const lastMessage = acc[acc.length - 1];
      const lastHasToolResult =
        lastMessage.content?.some((c) => 'toolResult' in c) === true;
      const currHasToolResult =
        curr.content?.some((c) => 'toolResult' in c) === true;
      if (
        lastMessage.role === 'user' &&
        lastHasToolResult &&
        curr.role === 'user' &&
        currHasToolResult
      ) {
        lastMessage.content = lastMessage.content?.concat(curr.content ?? []);
      } else {
        acc.push(curr);
      }
      return acc;
    },
    []
  );

  return { converseMessages: combinedConverseMessages, converseSystem };
}
