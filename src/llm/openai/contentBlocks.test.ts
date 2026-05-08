import { describe, expect, test } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  type ContentBlock,
} from '@langchain/core/messages';

describe('OpenAI content block translator compatibility', () => {
  describe('Chat Completions', () => {
    test('translates string content and tool calls to v1 content blocks', () => {
      const message = new AIMessage({
        content: 'Hello from OpenAI',
        tool_calls: [
          {
            id: 'call_123',
            name: 'get_weather',
            args: { location: 'San Francisco' },
          },
        ],
        response_metadata: { model_provider: 'openai' },
      });

      const expected: Array<ContentBlock.Standard> = [
        { type: 'text', text: 'Hello from OpenAI' },
        {
          type: 'tool_call',
          id: 'call_123',
          name: 'get_weather',
          args: { location: 'San Francisco' },
        },
      ];

      expect(message.contentBlocks).toEqual(expected);
      expect(message.content).not.toEqual(expected);

      const v1Message = new AIMessage({
        content: message.contentBlocks,
        response_metadata: { output_version: 'v1' },
      });
      expect(v1Message.contentBlocks).toEqual(expected);
      expect(v1Message.content).toEqual(expected);
    });

    test('does not include empty text block when content is empty string with tool calls', () => {
      const message = new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_123',
            name: 'get_value',
            args: { key: 'a' },
          },
          {
            id: 'call_456',
            name: 'get_value',
            args: { key: 'b' },
          },
        ],
        response_metadata: { model_provider: 'openai' },
      });

      expect(message.contentBlocks).toEqual([
        {
          type: 'tool_call',
          id: 'call_123',
          name: 'get_value',
          args: { key: 'a' },
        },
        {
          type: 'tool_call',
          id: 'call_456',
          name: 'get_value',
          args: { key: 'b' },
        },
      ]);
    });

    test('translates chat completion chunks with parsed tool call chunks', () => {
      const chunk1 = new AIMessageChunk({
        content: [{ type: 'text', text: 'Looking ', index: 0 }],
        response_metadata: { model_provider: 'openai' },
      });
      const chunk2 = new AIMessageChunk({
        content: [{ type: 'text', text: 'up.', index: 0 }],
        tool_call_chunks: [
          {
            type: 'tool_call_chunk',
            id: 'call_abc',
            name: 'search',
            args: '{"query":"weather"}',
            index: 0,
          },
        ],
        response_metadata: { model_provider: 'openai' },
      });

      expect(chunk1.concat(chunk2).contentBlocks).toEqual([
        { type: 'text', text: 'Looking up.', index: 0 },
        {
          type: 'tool_call',
          id: 'call_abc',
          name: 'search',
          args: { query: 'weather' },
        },
      ]);
    });
  });

  describe('Responses', () => {
    test('translates Responses messages to v1 content blocks', () => {
      const code = ['print(', 'hello', ')'].join(String.fromCharCode(39));
      const responseTextBlock = {
        type: 'text',
        text: 'Here is a result.',
        annotations: [
          {
            type: 'url_citation',
            url: 'https://example.com',
            title: 'Example',
            start_index: 0,
            end_index: 4,
          },
          {
            type: 'file_citation',
            file_id: 'file_123',
            filename: 'doc.pdf',
            index: 10,
          },
        ],
      } as ContentBlock.Text;
      const message = new AIMessage({
        content: [responseTextBlock],
        tool_calls: [
          { id: 'call_456', name: 'summarize', args: { length: 'short' } },
        ],
        additional_kwargs: {
          reasoning: { summary: [{ text: 'Thinking...' }, { text: ' Done.' }] },
          tool_outputs: [
            {
              id: 'call_456',
              type: 'code_interpreter_call',
              code,
              status: 'completed',
              outputs: [{ type: 'logs', logs: 'hello' }],
            },
          ],
        },
        response_metadata: { model_provider: 'openai' },
      });

      const expected: Array<ContentBlock.Standard> = [
        { type: 'reasoning', reasoning: 'Thinking... Done.' },
        {
          type: 'text',
          text: 'Here is a result.',
          annotations: [
            {
              type: 'citation',
              url: 'https://example.com',
              title: 'Example',
              startIndex: 0,
              endIndex: 4,
            },
            {
              type: 'citation',
              title: 'doc.pdf',
              startIndex: 10,
              endIndex: 10,
              fileId: 'file_123',
            },
          ],
        },
        {
          type: 'tool_call',
          id: 'call_456',
          name: 'summarize',
          args: { length: 'short' },
        },
        {
          type: 'server_tool_call',
          name: 'code_interpreter',
          id: 'call_456',
          args: { code },
        },
        {
          type: 'server_tool_call_result',
          toolCallId: 'call_456',
          status: 'success',
          output: {
            type: 'code_interpreter_output',
            returnCode: 0,
            stderr: undefined,
            stdout: 'hello',
          },
        },
      ];

      expect(message.contentBlocks).toEqual(expected);

      const v1Message = new AIMessage({
        content: message.contentBlocks,
        response_metadata: { output_version: 'v1' },
      });
      expect(v1Message.contentBlocks).toEqual(expected);
      expect(v1Message.content).toEqual(expected);
    });

    test('translates image_generation_call to image content block', () => {
      const message = new AIMessage({
        content: [{ type: 'text', text: 'Here is your image:' }],
        additional_kwargs: {
          tool_outputs: [
            {
              type: 'image_generation_call',
              id: 'ig_abc123',
              status: 'completed',
              result: 'base64ImageData',
              revised_prompt: 'A beautiful sunset over the ocean',
            },
          ],
        },
        response_metadata: { model_provider: 'openai' },
      });

      expect(message.contentBlocks).toEqual([
        { type: 'text', text: 'Here is your image:' },
        {
          type: 'image',
          mimeType: 'image/png',
          data: 'base64ImageData',
          id: 'ig_abc123',
          metadata: {
            status: 'completed',
          },
        },
        {
          type: 'non_standard',
          value: {
            type: 'image_generation_call',
            id: 'ig_abc123',
            status: 'completed',
            result: 'base64ImageData',
            revised_prompt: 'A beautiful sunset over the ocean',
          },
        },
      ]);
    });

    test('translates web_search_call and file_search_call to server tool blocks', () => {
      const message = new AIMessage({
        content: [{ type: 'text', text: 'Search results:' }],
        additional_kwargs: {
          tool_outputs: [
            {
              type: 'web_search_call',
              id: 'ws_abc456',
              status: 'completed',
              action: {
                type: 'search',
                query: 'melbourne australia news today',
                sources: [{ type: 'url', url: 'https://example.com/news' }],
              },
            },
            {
              type: 'file_search_call',
              id: 'fs_abc123',
              status: 'completed',
              queries: ['quarterly report', 'revenue 2025'],
              results: [
                {
                  file_id: 'file_001',
                  filename: 'report.pdf',
                  score: 0.95,
                  text: 'Revenue grew 15% in Q3...',
                },
              ],
            },
          ],
        },
        response_metadata: { model_provider: 'openai' },
      });

      expect(message.contentBlocks).toEqual([
        { type: 'text', text: 'Search results:' },
        {
          type: 'server_tool_call',
          name: 'web_search',
          id: 'ws_abc456',
          args: { query: 'melbourne australia news today' },
        },
        {
          type: 'server_tool_call_result',
          toolCallId: 'ws_abc456',
          status: 'success',
          output: {
            action: {
              type: 'search',
              query: 'melbourne australia news today',
              sources: [{ type: 'url', url: 'https://example.com/news' }],
            },
          },
        },
        {
          type: 'server_tool_call',
          name: 'file_search',
          id: 'fs_abc123',
          args: { queries: ['quarterly report', 'revenue 2025'] },
        },
        {
          type: 'server_tool_call_result',
          toolCallId: 'fs_abc123',
          status: 'success',
          output: {
            results: [
              {
                file_id: 'file_001',
                filename: 'report.pdf',
                score: 0.95,
                text: 'Revenue grew 15% in Q3...',
              },
            ],
          },
        },
      ]);
    });

    test('moves phase into extras on text content blocks', () => {
      const textBlock = {
        type: 'text',
        text: 'The weather is sunny.',
        annotations: [],
        phase: 'final_answer',
      } as ContentBlock.Text & { phase: string };
      const message = new AIMessage({
        content: [textBlock],
        response_metadata: { model_provider: 'openai' },
      });

      const contentTextBlock = message.contentBlocks.find(
        (block): block is ContentBlock.Text => block.type === 'text'
      );
      expect(contentTextBlock).toBeDefined();
      expect(contentTextBlock?.extras).toEqual({ phase: 'final_answer' });
    });
  });
});
