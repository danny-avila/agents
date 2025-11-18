/* eslint-disable @typescript-eslint/no-explicit-any */
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { ChatVertexAI } from '@/llm/vertexai';

describe('VertexAI Multimodal Content Transformation', () => {
  let model: ChatVertexAI;

  beforeEach(() => {
    model = new ChatVertexAI({
      model: 'gemini-2.0-flash-exp',
      temperature: 0,
    });
  });

  describe('transformMessageContent', () => {
    it('should transform document type to media type', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Analyze this document' },
            {
              type: 'document',
              mimeType: 'application/pdf',
              data: 'base64encodeddata...',
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      expect(transformed).toHaveLength(1);
      expect(transformed[0]).toBeInstanceOf(HumanMessage);
      expect(Array.isArray(transformed[0].content)).toBe(true);

      const content = transformed[0].content as any[];
      expect(content).toHaveLength(2);
      expect(content[0]).toEqual({
        type: 'text',
        text: 'Analyze this document',
      });
      expect(content[1]).toEqual({
        type: 'media',
        mimeType: 'application/pdf',
        data: 'base64encodeddata...',
      });
    });

    it('should transform audio type to media type', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Transcribe this audio' },
            {
              type: 'audio',
              mimeType: 'audio/mpeg',
              data: 'base64audidata...',
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);
      const content = transformed[0].content as any[];

      expect(content[1]).toEqual({
        type: 'media',
        mimeType: 'audio/mpeg',
        data: 'base64audidata...',
      });
    });

    it('should transform video type to media type', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Analyze this video' },
            {
              type: 'video',
              mimeType: 'video/mp4',
              data: 'base64videodata...',
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);
      const content = transformed[0].content as any[];

      expect(content[1]).toEqual({
        type: 'media',
        mimeType: 'video/mp4',
        data: 'base64videodata...',
      });
    });

    it('should not transform messages without multimodal content', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Hello' },
            { type: 'text', text: 'World' },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      // Should return the same message untouched
      expect(transformed[0]).toBe(messages[0]);
    });

    it('should not transform text-only string content', async () => {
      const messages = [
        new HumanMessage({
          content: 'Simple text message',
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      // Should return the same message untouched
      expect(transformed[0]).toBe(messages[0]);
      expect(transformed[0].content).toBe('Simple text message');
    });

    it('should preserve image_url types without transformation', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Describe this image' },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,...' },
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);
      const content = transformed[0].content as any[];

      // image_url should pass through unchanged (it's already supported by LangChain)
      expect(content[1].type).toBe('image_url');
    });

    it('should handle mixed content with multiple file types', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Analyze these files' },
            {
              type: 'document',
              mimeType: 'application/pdf',
              data: 'pdfdata...',
            },
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,...' },
            },
            {
              type: 'audio',
              mimeType: 'audio/wav',
              data: 'audiodata...',
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);
      const content = transformed[0].content as any[];

      expect(content).toHaveLength(4);
      expect(content[0].type).toBe('text');
      expect(content[1].type).toBe('media');
      expect(content[1].mimeType).toBe('application/pdf');
      expect(content[2].type).toBe('image_url');
      expect(content[3].type).toBe('media');
      expect(content[3].mimeType).toBe('audio/wav');
    });

    it('should preserve message metadata during transformation', async () => {
      const messages = [
        new HumanMessage({
          content: [
            {
              type: 'document',
              mimeType: 'application/pdf',
              data: 'data...',
            },
          ],
          additional_kwargs: { custom: 'metadata' },
          id: 'message-123',
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      expect(transformed[0].additional_kwargs).toEqual({ custom: 'metadata' });
      expect(transformed[0].id).toBe('message-123');
    });

    it('should handle non-HumanMessage types without modification', async () => {
      const messages = [
        new AIMessage({
          content: 'AI response',
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      // AI messages should pass through unchanged
      expect(transformed[0]).toBe(messages[0]);
    });

    it('should handle empty content arrays', async () => {
      const messages = [
        new HumanMessage({
          content: [],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      // Empty array should pass through unchanged
      expect(transformed[0]).toBe(messages[0]);
    });

    it('should only transform when multimodal content is present', async () => {
      const textOnlyMessages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Generate a title for this conversation' },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(
        textOnlyMessages
      );

      // Text-only messages (like titleConvo) should not be transformed
      expect(transformed[0]).toBe(textOnlyMessages[0]);
    });

    it('should transform multiple messages in conversation', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'First message with doc' },
            {
              type: 'document',
              mimeType: 'application/pdf',
              data: 'doc1...',
            },
          ],
        }),
        new AIMessage({
          content: 'Response',
        }),
        new HumanMessage({
          content: [
            { type: 'text', text: 'Second message with audio' },
            {
              type: 'audio',
              mimeType: 'audio/mpeg',
              data: 'audio1...',
            },
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);

      expect(transformed).toHaveLength(3);
      expect(transformed[0].content[1].type).toBe('media');
      expect(transformed[1]).toBe(messages[1]); // AI message unchanged
      expect(transformed[2].content[1].type).toBe('media');
    });

    it('should handle parts without type property gracefully', async () => {
      const messages = [
        new HumanMessage({
          content: [
            { type: 'text', text: 'Normal text' },
            { someOtherProperty: 'value' } as any,
          ],
        }),
      ];

      const transformed = (model as any).transformMessageContent(messages);
      const content = transformed[0].content as any[];

      // Parts without 'type' should pass through unchanged
      expect(content[1]).toEqual({ someOtherProperty: 'value' });
    });
  });

  describe('Integration with invoke/generate', () => {
    it('should call transformMessageContent before invoke with array input', () => {
      const transformSpy = jest.spyOn(model as any, 'transformMessageContent');
      const messages = [
        new HumanMessage({
          content: [
            {
              type: 'document',
              mimeType: 'application/pdf',
              data: 'data...',
            },
          ],
        }),
      ];

      jest
        .spyOn(Object.getPrototypeOf(Object.getPrototypeOf(model)), 'invoke')
        .mockResolvedValue({} as any);

      model.invoke(messages);

      expect(transformSpy).toHaveBeenCalledWith(messages);
      transformSpy.mockRestore();
    });

    it('should not transform string input for titleConvo', () => {
      const transformSpy = jest.spyOn(model as any, 'transformMessageContent');
      const stringInput = 'Generate a title for this conversation';

      jest
        .spyOn(Object.getPrototypeOf(Object.getPrototypeOf(model)), 'invoke')
        .mockResolvedValue({} as any);

      model.invoke(stringInput as any);

      // Should not call transformMessageContent for string input
      expect(transformSpy).not.toHaveBeenCalled();
      transformSpy.mockRestore();
    });
  });
});
