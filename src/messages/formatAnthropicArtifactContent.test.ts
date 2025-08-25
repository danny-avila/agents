import {
  HumanMessage,
  AIMessageChunk,
  ToolMessage,
} from '@langchain/core/messages';
import { formatAnthropicArtifactContent } from './core';
import type { MessageContentComplex } from '@/types';

/**
 * Tests for formatAnthropicArtifactContent function which handles tool responses for Anthropic provider.
 *
 * This function is now a no-op to prevent artifact injection vulnerabilities.
 * These tests ensure that artifacts are NOT concatenated into ToolMessage content.
 */
describe('formatAnthropicArtifactContent', () => {
  it('should not modify any messages (no-op behavior)', () => {
    const arrayContent: MessageContentComplex[] = [
      { type: 'text', text: 'Tool response' },
    ];

    const artifactContent: MessageContentComplex[] = [
      { type: 'text', text: 'This artifact should NOT be added to content' },
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

    // Create a deep copy to check if messages are modified
    const originalContent = [...toolMessage.content];

    formatAnthropicArtifactContent(messages);

    // Verify nothing was modified
    expect(messages).toHaveLength(3);
    expect(toolMessage.content).toEqual(originalContent);
    expect(toolMessage.content).not.toContain(artifactContent[0]);

    // Verify artifact remains untouched in ToolMessage
    expect(toolMessage.artifact).toBeDefined();
    expect(toolMessage.artifact.content).toEqual(artifactContent);
  });

  it('should not modify messages when last message is not a ToolMessage', () => {
    const messages = [
      new HumanMessage('Hello'),
      new AIMessageChunk({
        content: 'Hi there!',
      }),
    ];

    const originalLength = messages.length;
    const originalContent0 = messages[0].content;
    const originalContent1 = messages[1].content;

    formatAnthropicArtifactContent(messages);

    expect(messages).toHaveLength(originalLength);
    expect(messages[0].content).toEqual(originalContent0);
    expect(messages[1].content).toEqual(originalContent1);
  });

  it('should not modify messages even with multiple tool calls', () => {
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
        artifact: {
          content: [{ type: 'text', text: 'Artifact 1' }],
        },
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'Response 2' }],
        tool_call_id: 'call_456',
        name: 'tool2',
        artifact: {
          content: [{ type: 'text', text: 'Artifact 2' }],
        },
      }),
    ];

    const originalContent1 = [...(messages[2] as ToolMessage).content];
    const originalContent2 = [...(messages[3] as ToolMessage).content];

    formatAnthropicArtifactContent(messages);

    expect(messages).toHaveLength(4);
    expect((messages[2] as ToolMessage).content).toEqual(originalContent1);
    expect((messages[3] as ToolMessage).content).toEqual(originalContent2);

    // Verify artifacts are not in content
    expect((messages[2] as ToolMessage).content).not.toContainEqual({
      type: 'text',
      text: 'Artifact 1',
    });
    expect((messages[3] as ToolMessage).content).not.toContainEqual({
      type: 'text',
      text: 'Artifact 2',
    });
  });

  it('should preserve the original function signature but do nothing', () => {
    const messages = [
      new HumanMessage('Test'),
      new AIMessageChunk({
        content: '',
        tool_calls: [
          {
            id: 'call_789',
            name: 'test_tool',
            args: {} as Record<string, unknown>,
          },
        ],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'Original content' }],
        tool_call_id: 'call_789',
        name: 'test_tool',
        artifact: {
          content: [
            { type: 'text', text: 'Artifact that should not be injected' },
          ],
        },
      }),
    ];

    // The function should accept the messages array
    expect(() => formatAnthropicArtifactContent(messages)).not.toThrow();

    // And return void (undefined)
    const result = formatAnthropicArtifactContent(messages);
    expect(result).toBeUndefined();
  });
});
