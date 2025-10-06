import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type { MessageContentComplex } from '@/types';
import {
  formatMessage,
  formatLangChainMessages,
  formatFromLangChain,
  formatMediaMessage,
} from './format';
import { Providers } from '@/common';

const NO_PARENT = '00000000-0000-0000-0000-000000000000';

/**
 * Type for formatted message results with media content
 */
interface FormattedMediaMessage {
  role: string;
  content: MessageContentComplex[];
  name?: string;
}

/**
 * Type guard to check if result is a FormattedMediaMessage
 */
function isFormattedMediaMessage(
  result: unknown
): result is FormattedMediaMessage {
  return (
    typeof result === 'object' &&
    result !== null &&
    'role' in result &&
    'content' in result &&
    Array.isArray((result as FormattedMediaMessage).content)
  );
}

describe('formatMessage', () => {
  it('formats user message', () => {
    const input = {
      message: {
        sender: 'user',
        text: 'Hello',
      },
      userName: 'John',
    };
    const result = formatMessage(input);
    expect(result).toEqual({
      role: 'user',
      content: 'Hello',
      name: 'John',
    });
  });

  it('sanitizes the name by replacing invalid characters (per OpenAI)', () => {
    const input = {
      message: {
        sender: 'user',
        text: 'Hello',
      },
      userName: ' John$Doe@Example! ',
    };
    const result = formatMessage(input);
    expect(result).toEqual({
      role: 'user',
      content: 'Hello',
      name: '_John_Doe_Example__',
    });
  });

  it('trims the name to a maximum length of 64 characters', () => {
    const longName = 'a'.repeat(100);
    const input = {
      message: {
        sender: 'user',
        text: 'Hello',
      },
      userName: longName,
    };
    const result = formatMessage(input);
    expect(result.name?.length).toBe(64);
    expect(result.name).toBe('a'.repeat(64));
  });

  it('formats a realistic user message', () => {
    const input = {
      message: {
        _id: '6512cdfb92cbf69fea615331',
        messageId: 'b620bf73-c5c3-4a38-b724-76886aac24c4',
        __v: 0,
        conversationId: '5c23d24f-941f-4aab-85df-127b596c8aa5',
        createdAt: Date.now(),
        error: false,
        finish_reason: null,
        isCreatedByUser: true,
        model: null,
        parentMessageId: NO_PARENT,
        sender: 'User',
        text: 'hi',
        tokenCount: 5,
        unfinished: false,
        updatedAt: Date.now(),
        user: '6512cdf475f05c86d44c31d2',
      },
      userName: 'John',
    };
    const result = formatMessage(input);
    expect(result).toEqual({
      role: 'user',
      content: 'hi',
      name: 'John',
    });
  });

  it('formats assistant message', () => {
    const input = {
      message: {
        sender: 'assistant',
        text: 'Hi there',
      },
      assistantName: 'Assistant',
    };
    const result = formatMessage(input);
    expect(result).toEqual({
      role: 'assistant',
      content: 'Hi there',
      name: 'Assistant',
    });
  });

  it('formats system message', () => {
    const input = {
      message: {
        role: 'system',
        text: 'Hi there',
      },
    };
    const result = formatMessage(input);
    expect(result).toEqual({
      role: 'system',
      content: 'Hi there',
    });
  });

  it('formats user message with langChain', () => {
    const input = {
      message: {
        sender: 'user',
        text: 'Hello',
      },
      userName: 'John',
      langChain: true,
    };
    const result = formatMessage(input);
    expect(result).toBeInstanceOf(HumanMessage);
    expect(result.lc_kwargs.content).toEqual(input.message.text);
    expect(result.lc_kwargs.name).toEqual(input.userName);
  });

  it('formats assistant message with langChain', () => {
    const input = {
      message: {
        sender: 'assistant',
        text: 'Hi there',
      },
      assistantName: 'Assistant',
      langChain: true,
    };
    const result = formatMessage(input);
    expect(result).toBeInstanceOf(AIMessage);
    expect(result.lc_kwargs.content).toEqual(input.message.text);
    expect(result.lc_kwargs.name).toEqual(input.assistantName);
  });

  it('formats system message with langChain', () => {
    const input = {
      message: {
        role: 'system',
        text: 'This is a system message.',
      },
      langChain: true,
    };
    const result = formatMessage(input);
    expect(result).toBeInstanceOf(SystemMessage);
    expect(result.lc_kwargs.content).toEqual(input.message.text);
  });

  it('formats langChain messages into OpenAI payload format', () => {
    const human = {
      message: new HumanMessage({
        content: 'Hello',
      }),
    };
    const system = {
      message: new SystemMessage({
        content: 'Hello',
      }),
    };
    const ai = {
      message: new AIMessage({
        content: 'Hello',
      }),
    };
    const humanResult = formatMessage(human);
    const systemResult = formatMessage(system);
    const aiResult = formatMessage(ai);
    expect(humanResult).toEqual({
      role: 'user',
      content: 'Hello',
    });
    expect(systemResult).toEqual({
      role: 'system',
      content: 'Hello',
    });
    expect(aiResult).toEqual({
      role: 'assistant',
      content: 'Hello',
    });
  });
});

describe('formatMediaMessage', () => {
  it('formats message with images for default provider', () => {
    const message = {
      role: 'user',
      content: 'Check out this image',
      name: 'John',
    };
    const mediaParts = [
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/image1.jpg' },
      },
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/image2.jpg' },
      },
    ];

    const result = formatMediaMessage({ message, mediaParts });

    expect(result.role).toBe('user');
    expect(result.name).toBe('John');
    expect(Array.isArray(result.content)).toBe(true);
    expect(result.content).toHaveLength(3);
    expect(result.content[0]).toEqual({
      type: 'text',
      text: 'Check out this image',
    });
    expect(result.content[1]).toEqual(mediaParts[0]);
    expect(result.content[2]).toEqual(mediaParts[1]);
  });

  it('formats message with images for Anthropic (media first)', () => {
    const message = {
      role: 'user',
      content: 'Check out this image',
    };
    const mediaParts = [
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/image.jpg' },
      },
    ];

    const result = formatMediaMessage({
      message,
      mediaParts,
      endpoint: Providers.ANTHROPIC,
    });

    expect(result.content).toHaveLength(2);
    expect(result.content[0]).toEqual(mediaParts[0]);
    expect(result.content[1]).toEqual({
      type: 'text',
      text: 'Check out this image',
    });
  });

  it('formats message with multiple media types', () => {
    const message = {
      role: 'user',
      content: 'Check out these files',
    };
    const mediaParts = [
      { type: 'document', document: { url: 'https://example.com/doc.pdf' } },
      { type: 'video', video: { url: 'https://example.com/video.mp4' } },
      { type: 'audio', audio: { url: 'https://example.com/audio.mp3' } },
      {
        type: 'image_url',
        image_url: { url: 'https://example.com/image.jpg' },
      },
    ];

    const result = formatMediaMessage({ message, mediaParts });

    expect(result.content).toHaveLength(5);
    expect(result.content[0]).toEqual({
      type: 'text',
      text: 'Check out these files',
    });
    expect(result.content[1]).toEqual(mediaParts[0]);
    expect(result.content[2]).toEqual(mediaParts[1]);
    expect(result.content[3]).toEqual(mediaParts[2]);
    expect(result.content[4]).toEqual(mediaParts[3]);
  });
});

describe('formatMessage with media', () => {
  it('formats user message with image_urls (backward compatibility)', () => {
    const input = {
      message: {
        sender: 'user',
        text: 'Check out this image',
        image_urls: [
          {
            type: 'image_url' as const,
            image_url: { url: 'https://example.com/image.jpg' },
          },
        ],
      },
      userName: 'John',
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.role).toBe('user');
      expect(result.name).toBe('John');
      expect(Array.isArray(result.content)).toBe(true);
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: 'text',
        text: 'Check out this image',
      });
      expect(result.content[1]).toEqual(input.message.image_urls[0]);
    }
  });

  it('formats user message with documents', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Review this document',
        documents: [
          {
            type: 'document',
            document: { url: 'https://example.com/report.pdf' },
          },
        ],
      },
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.role).toBe('user');
      expect(Array.isArray(result.content)).toBe(true);
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: 'text',
        text: 'Review this document',
      });
      expect(result.content[1]).toEqual(input.message.documents[0]);
    }
  });

  it('formats user message with videos', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Watch this video',
        videos: [
          { type: 'video', video: { url: 'https://example.com/demo.mp4' } },
        ],
      },
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.role).toBe('user');
      expect(Array.isArray(result.content)).toBe(true);
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: 'text',
        text: 'Watch this video',
      });
      expect(result.content[1]).toEqual(input.message.videos[0]);
    }
  });

  it('formats user message with audios', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Listen to this',
        audios: [
          { type: 'audio', audio: { url: 'https://example.com/podcast.mp3' } },
        ],
      },
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.role).toBe('user');
      expect(Array.isArray(result.content)).toBe(true);
      expect(result.content).toHaveLength(2);
      expect(result.content[0]).toEqual({
        type: 'text',
        text: 'Listen to this',
      });
      expect(result.content[1]).toEqual(input.message.audios[0]);
    }
  });

  it('formats user message with all media types in correct order', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Check out all these files',
        documents: [
          {
            type: 'document',
            document: { url: 'https://example.com/doc.pdf' },
          },
        ],
        videos: [
          { type: 'video', video: { url: 'https://example.com/video.mp4' } },
        ],
        audios: [
          { type: 'audio', audio: { url: 'https://example.com/audio.mp3' } },
        ],
        image_urls: [
          {
            type: 'image_url' as const,
            image_url: { url: 'https://example.com/image.jpg' },
          },
        ],
      },
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.role).toBe('user');
      expect(Array.isArray(result.content)).toBe(true);
      expect(result.content).toHaveLength(5);
      // Text first
      expect(result.content[0]).toEqual({
        type: 'text',
        text: 'Check out all these files',
      });
      // Then documents, videos, audios, images
      expect(result.content[1]).toEqual(input.message.documents[0]);
      expect(result.content[2]).toEqual(input.message.videos[0]);
      expect(result.content[3]).toEqual(input.message.audios[0]);
      expect(result.content[4]).toEqual(input.message.image_urls[0]);
    }
  });

  it('formats user message with multiple files of the same type', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Review these documents',
        documents: [
          {
            type: 'document',
            document: { url: 'https://example.com/doc1.pdf' },
          },
          {
            type: 'document',
            document: { url: 'https://example.com/doc2.pdf' },
          },
          {
            type: 'document',
            document: { url: 'https://example.com/doc3.pdf' },
          },
        ],
      },
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.content).toHaveLength(4);
      expect(result.content[0].type).toBe('text');
      expect(result.content[1]).toEqual(input.message.documents[0]);
      expect(result.content[2]).toEqual(input.message.documents[1]);
      expect(result.content[3]).toEqual(input.message.documents[2]);
    }
  });

  it('respects Anthropic provider ordering (media before text)', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Check this out',
        documents: [
          {
            type: 'document',
            document: { url: 'https://example.com/doc.pdf' },
          },
        ],
        image_urls: [
          {
            type: 'image_url' as const,
            image_url: { url: 'https://example.com/image.jpg' },
          },
        ],
      },
      endpoint: Providers.ANTHROPIC,
    };

    const result = formatMessage(input);

    expect(isFormattedMediaMessage(result)).toBe(true);
    if (isFormattedMediaMessage(result)) {
      expect(result.content).toHaveLength(3);
      // Media first for Anthropic
      expect(result.content[0]).toEqual(input.message.documents[0]);
      expect(result.content[1]).toEqual(input.message.image_urls[0]);
      expect(result.content[2]).toEqual({
        type: 'text',
        text: 'Check this out',
      });
    }
  });

  it('does not format media for assistant messages', () => {
    const input = {
      message: {
        role: 'assistant',
        content: 'Here is a response',
        documents: [
          {
            type: 'document',
            document: { url: 'https://example.com/doc.pdf' },
          },
        ],
      },
    };

    const result = formatMessage(input);

    expect(result).toMatchObject({
      role: 'assistant',
      content: 'Here is a response',
    });
  });

  it('handles empty media arrays gracefully', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Just text',
        documents: [],
        videos: [],
        audios: [],
        image_urls: [],
      },
    };

    const result = formatMessage(input);

    expect(result).toMatchObject({
      role: 'user',
      content: 'Just text',
    });
  });

  it('formats media with langChain flag', () => {
    const input = {
      message: {
        role: 'user',
        content: 'Check this image',
        image_urls: [
          {
            type: 'image_url' as const,
            image_url: { url: 'https://example.com/image.jpg' },
          },
        ],
      },
      langChain: true,
    };

    const result = formatMessage(input);

    expect(result).toBeInstanceOf(HumanMessage);
    expect(Array.isArray(result.lc_kwargs.content)).toBe(true);
    expect(result.lc_kwargs.content).toHaveLength(2);
  });
});

describe('formatLangChainMessages', () => {
  it('formats an array of messages for LangChain', () => {
    const messages = [
      {
        role: 'system',
        content: 'This is a system message',
      },
      {
        sender: 'user',
        text: 'Hello',
      },
      {
        sender: 'assistant',
        text: 'Hi there',
      },
    ];
    const formatOptions = {
      userName: 'John',
      assistantName: 'Assistant',
    };
    const result = formatLangChainMessages(messages, formatOptions);
    expect(result).toHaveLength(3);
    expect(result[0]).toBeInstanceOf(SystemMessage);
    expect(result[1]).toBeInstanceOf(HumanMessage);
    expect(result[2]).toBeInstanceOf(AIMessage);

    expect(result[0].lc_kwargs.content).toEqual(messages[0].content);
    expect(result[1].lc_kwargs.content).toEqual(messages[1].text);
    expect(result[2].lc_kwargs.content).toEqual(messages[2].text);

    expect(result[1].lc_kwargs.name).toEqual(formatOptions.userName);
    expect(result[2].lc_kwargs.name).toEqual(formatOptions.assistantName);
  });

  describe('formatFromLangChain', () => {
    it('should merge kwargs and additional_kwargs', () => {
      const message = {
        kwargs: {
          content: 'some content',
          name: 'dan',
          additional_kwargs: {
            function_call: {
              name: 'dall-e',
              arguments: '{\n  "input": "Subject: hedgehog, Style: cute"\n}',
            },
          },
        },
      };

      const expected = {
        content: 'some content',
        name: 'dan',
        function_call: {
          name: 'dall-e',
          arguments: '{\n  "input": "Subject: hedgehog, Style: cute"\n}',
        },
      };

      expect(formatFromLangChain(message)).toEqual(expected);
    });

    it('should handle messages without additional_kwargs', () => {
      const message = {
        kwargs: {
          content: 'some content',
          name: 'dan',
        },
      };

      const expected = {
        content: 'some content',
        name: 'dan',
      };

      expect(formatFromLangChain(message)).toEqual(expected);
    });

    it('should handle empty messages', () => {
      const message = {
        kwargs: {},
      };

      const expected = {};

      expect(formatFromLangChain(message)).toEqual(expected);
    });
  });
});
