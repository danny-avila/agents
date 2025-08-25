import {
  HumanMessage,
  AIMessageChunk,
  ToolMessage,
} from '@langchain/core/messages';
import { formatToolArrayContent } from './core';
import type { MessageContentComplex } from '@/types';

/**
 * Tests for formatToolArrayContent function which handles tool responses for OpenAI/Google providers.
 *
 * Note: There's also a formatAnthropicArtifactContent function that handles artifacts differently
 * for Anthropic providers. That function concatenates artifacts directly into ToolMessage content
 * rather than creating a HumanMessage.
 */
describe('formatToolArrayContent', () => {
  it('should not modify messages when last message is not a ToolMessage', () => {
    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: 'Hi there!',
      }),
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(2);
    expect(messages[1].content).toBe('Hi there!');
  });

  it('should not modify messages when ToolMessage has string content', () => {
    const toolMessage = new ToolMessage({
      content: 'Tool response as string',
      tool_call_id: 'call_123',
      name: 'test_tool',
    });

    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_123',
            name: 'test_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      toolMessage,
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(3);
    expect(toolMessage.content).toBe('Tool response as string');
  });

  it('should convert ToolMessage with array content to HumanMessage', () => {
    const arrayContent: MessageContentComplex[] = [
      { type: 'text', text: 'First response' },
      { type: 'text', text: 'Second response' },
    ];

    const toolMessage = new ToolMessage({
      content: arrayContent,
      tool_call_id: 'call_123',
      name: 'test_tool',
    });

    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_123',
            name: 'test_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      toolMessage,
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(4);
    expect(toolMessage.content).toBe(
      'Tool response is included in the next message as a Human message'
    );

    const lastMessage = messages[3] as HumanMessage;
    expect(lastMessage).toBeInstanceOf(HumanMessage);
    expect(lastMessage.content).toEqual(arrayContent);
  });

  it('should NOT include artifact content in HumanMessage', () => {
    const arrayContent: MessageContentComplex[] = [
      { type: 'text', text: 'Actual tool response' },
    ];

    const artifactContent: MessageContentComplex[] = [
      { type: 'text', text: 'This should NOT be included' },
    ];

    const toolMessage = new ToolMessage({
      content: arrayContent,
      tool_call_id: 'call_123',
      name: 'test_tool',
      artifact: {
        content: artifactContent,
      },
    });

    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_123',
            name: 'test_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      toolMessage,
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(4);

    const lastMessage = messages[3] as HumanMessage;
    expect(lastMessage).toBeInstanceOf(HumanMessage);
    expect(lastMessage.content).toEqual(arrayContent);
    expect(lastMessage.content).not.toContain(artifactContent[0]);

    // Verify artifact remains in ToolMessage
    expect(toolMessage.artifact).toBeDefined();
    expect(toolMessage.artifact.content).toEqual(artifactContent);
  });

  it('should handle multiple ToolMessages with array content', () => {
    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_123',
            name: 'tool1',
            args: {} as Record<string, unknown>,
          },
          {
            id: 'call_456',
            name: 'tool2',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'Response 1' }],
        tool_call_id: 'call_123',
        name: 'tool1',
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'Response 2' }],
        tool_call_id: 'call_456',
        name: 'tool2',
      }),
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(5);

    const humanMessage = messages[4] as HumanMessage;
    expect(humanMessage).toBeInstanceOf(HumanMessage);
    expect(humanMessage.content).toEqual([
      { type: 'text', text: 'Response 1' },
      { type: 'text', text: 'Response 2' },
    ]);
  });

  it('should not process ToolMessages that do not belong to the latest AI message', () => {
    const toolMessage = new ToolMessage({
      content: [{ type: 'text', text: 'Should not be processed' }],
      tool_call_id: 'old_call',
      name: 'old_tool',
    });

    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'old_call',
            name: 'old_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      toolMessage,
      new HumanMessage('Another question'),
      new AIMessageChunk({
        content: 'Let me help',
        tool_calls: [
          {
            id: 'new_call',
            name: 'new_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      new ToolMessage({
        content: 'String response',
        tool_call_id: 'new_call',
        name: 'new_tool',
      }),
    ];

    formatToolArrayContent(messages);

    // Should not add any HumanMessage since the last ToolMessage has string content
    expect(messages).toHaveLength(6);
    expect(toolMessage.content).toEqual([
      { type: 'text', text: 'Should not be processed' },
    ]);
  });

  it('should only process ToolMessages after the latest AIMessage with tool_calls', () => {
    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'call_1', name: 'tool1', args: {} as Record<string, unknown> },
        ],
      }),
      new ToolMessage({
        content: 'String content 1',
        tool_call_id: 'call_1',
        name: 'tool1',
      }),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          { id: 'call_2', name: 'tool2', args: {} as Record<string, unknown> },
        ],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'Array content' }],
        tool_call_id: 'call_2',
        name: 'tool2',
      }),
    ];

    formatToolArrayContent(messages);

    expect(messages).toHaveLength(6);

    const humanMessage = messages[5] as HumanMessage;
    expect(humanMessage).toBeInstanceOf(HumanMessage);
    expect(humanMessage.content).toEqual([
      { type: 'text', text: 'Array content' },
    ]);
  });
});
