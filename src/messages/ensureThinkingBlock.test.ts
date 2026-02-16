import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ExtendedMessageContent } from '@/types';
import { ensureThinkingBlockInMessages } from './format';
import { Providers, ContentTypes } from '@/common';

describe('ensureThinkingBlockInMessages', () => {
  describe('messages with thinking blocks (should not be modified)', () => {
    test('should not modify AI message that already has thinking block', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({
          content: [
            { type: ContentTypes.THINKING, thinking: 'Let me think...' },
            { type: 'text', text: 'Hi there!' },
          ],
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.THINKING
      );
    });

    test('should not modify AI message that has redacted_thinking block', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({
          content: [
            { type: 'redacted_thinking', data: 'redacted' },
            { type: 'text', text: 'Hi there!' },
          ],
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        'redacted_thinking'
      );
    });

    test('should not modify AI message with reasoning_content block and tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Calculate something' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'I need to use a calculator' },
            },
          ],
          tool_calls: [
            {
              id: 'call_456',
              name: 'calculator',
              args: { input: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          tool_call_id: 'call_456',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.REASONING_CONTENT
      );
    });

    test('should not modify AI message when reasoning_content is not the first block (Bedrock whitespace artifact)', () => {
      // Bedrock emits a "\n\n" text chunk before the thinking block,
      // pushing reasoning_content to content[1] instead of content[0].
      const messages = [
        new HumanMessage({ content: 'Do something' }),
        new AIMessage({
          content: [
            { type: 'text', text: '\n\n' },
            {
              type: ContentTypes.REASONING_CONTENT,
              reasoningText: { text: 'Let me think about this' },
            },
            { type: 'text', text: 'Let me help!' },
          ],
          tool_calls: [
            {
              id: 'call_bedrock',
              name: 'some_tool',
              args: { x: 1 },
              type: 'tool_call' as const,
            },
          ],
        }),
        new ToolMessage({
          content: 'tool result',
          tool_call_id: 'call_bedrock',
        }),
      ];

      const result = ensureThinkingBlockInMessages(messages, Providers.BEDROCK);

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      // The AI message should be preserved, not converted to a HumanMessage
      expect(result[1].content).toEqual(messages[1].content);
    });

    test('should not modify AI message with reasoning block and tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Calculate something' }),
        new AIMessage({
          content: [
            {
              type: ContentTypes.REASONING,
              reasoning: 'I need to use a calculator',
            },
          ],
          tool_calls: [
            {
              id: 'call_789',
              name: 'calculator',
              args: { input: '3+3' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '6',
          tool_call_id: 'call_789',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.VERTEXAI
      );

      expect(result).toHaveLength(3);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(ToolMessage);
      expect((result[1].content as ExtendedMessageContent[])[0].type).toBe(
        ContentTypes.REASONING
      );
    });
  });

  describe('messages with tool_calls (should be converted)', () => {
    test('should convert AI message with tool_calls to HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'What is the weather?' }),
        new AIMessage({
          content: 'Let me check the weather.',
          tool_calls: [
            {
              id: 'call_123',
              name: 'get_weather',
              args: { location: 'NYC' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Sunny, 75°F',
          tool_call_id: 'call_123',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should have 2 messages: HumanMessage + converted HumanMessage
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('What is the weather?');
      expect(result[1]).toBeInstanceOf(HumanMessage);

      // Check that the converted message includes the context prefix
      expect(result[1].content).toContain('[Previous agent context]');
      expect(result[1].content).toContain('Let me check the weather');
      expect(result[1].content).toContain('Sunny, 75°F');
    });

    test('should convert AI message with tool_use in content to HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'Search for something' }),
        new AIMessage({
          content: [
            { type: 'text', text: 'Searching...' },
            {
              type: 'tool_use',
              id: 'call_456',
              name: 'search',
              input: { query: 'test' },
            },
          ],
        }),
        new ToolMessage({
          content: 'Found results',
          tool_call_id: 'call_456',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(result[1].content).toContain('[Previous agent context]');
      expect(result[1].content).toContain('Searching...');
      expect(result[1].content).toContain('Found results');
    });

    test('should handle multiple tool messages in sequence', () => {
      const messages = [
        new HumanMessage({ content: 'Do multiple things' }),
        new AIMessage({
          content: 'I will perform multiple actions.',
          tool_calls: [
            {
              id: 'call_1',
              name: 'action1',
              args: { param: 'a' },
              type: 'tool_call',
            },
            {
              id: 'call_2',
              name: 'action2',
              args: { param: 'b' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 1',
          tool_call_id: 'call_1',
        }),
        new ToolMessage({
          content: 'Result 2',
          tool_call_id: 'call_2',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should combine all tool messages into one HumanMessage
      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(result[1].content).toContain('Result 1');
      expect(result[1].content).toContain('Result 2');
    });
  });

  describe('messages without tool calls (should pass through)', () => {
    test('should not modify AI message without tool calls', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({ content: 'Hi there, how can I help?' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('Hello');
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[1].content).toBe('Hi there, how can I help?');
    });

    test('should preserve HumanMessages and other message types', () => {
      const messages = [
        new HumanMessage({ content: 'Question 1' }),
        new AIMessage({ content: 'Answer 1' }),
        new HumanMessage({ content: 'Question 2' }),
        new AIMessage({ content: 'Answer 2' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(4);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(HumanMessage);
      expect(result[3]).toBeInstanceOf(AIMessage);
    });
  });

  describe('mixed scenarios', () => {
    test('should handle mix of normal and tool-using messages', () => {
      const messages = [
        new HumanMessage({ content: 'First question' }),
        new AIMessage({ content: 'First answer without tools' }),
        new HumanMessage({ content: 'Second question' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            {
              id: 'call_abc',
              name: 'some_tool',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Tool result',
          tool_call_id: 'call_abc',
        }),
        new HumanMessage({ content: 'Third question' }),
        new AIMessage({ content: 'Third answer without tools' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Original message 1: HumanMessage (preserved)
      // Original message 2: AIMessage without tools (preserved)
      // Original message 3: HumanMessage (preserved)
      // Original messages 4-5: AIMessage with tool + ToolMessage (converted to 1 HumanMessage)
      // Original message 6: HumanMessage (preserved)
      // Original message 7: AIMessage without tools (preserved)
      expect(result).toHaveLength(6);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(AIMessage);
      expect(result[2]).toBeInstanceOf(HumanMessage);
      expect(result[3]).toBeInstanceOf(HumanMessage); // Converted
      expect(result[4]).toBeInstanceOf(HumanMessage);
      expect(result[5]).toBeInstanceOf(AIMessage);
    });

    test('should handle multiple tool-using sequences', () => {
      const messages = [
        new HumanMessage({ content: 'Do task 1' }),
        new AIMessage({
          content: 'Doing task 1',
          tool_calls: [
            {
              id: 'call_1',
              name: 'tool1',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 1',
          tool_call_id: 'call_1',
        }),
        new HumanMessage({ content: 'Do task 2' }),
        new AIMessage({
          content: 'Doing task 2',
          tool_calls: [
            {
              id: 'call_2',
              name: 'tool2',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result 2',
          tool_call_id: 'call_2',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Each tool sequence should be converted to a HumanMessage
      expect(result).toHaveLength(4);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('Do task 1');
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(result[1].content).toContain('Doing task 1');
      expect(result[2]).toBeInstanceOf(HumanMessage);
      expect(result[2].content).toBe('Do task 2');
      expect(result[3]).toBeInstanceOf(HumanMessage);
      expect(result[3].content).toContain('Doing task 2');
    });
  });

  describe('fast exit when last message is HumanMessage', () => {
    test('should return messages as-is when last message is a HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'first request' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
        new HumanMessage({ content: 'second request' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should return the exact same array reference — no processing done
      expect(result).toBe(messages);
      expect(result).toHaveLength(4);
    });

    test('should return messages as-is when only message is a HumanMessage', () => {
      const messages = [new HumanMessage({ content: 'hello' })];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toBe(messages);
    });

    test('should still process when last message is not a HumanMessage', () => {
      const messages = [
        new HumanMessage({ content: 'do something' }),
        new AIMessage({
          content: 'Using a tool',
          tool_calls: [
            { id: 'c1', name: 'tool1', args: {}, type: 'tool_call' as const },
          ],
        }),
        new ToolMessage({ content: 'r1', tool_call_id: 'c1' }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should process — last message is ToolMessage, not HumanMessage
      expect(result).not.toBe(messages);
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[0].content).toBe('do something');
      expect(result[1]).toBeInstanceOf(HumanMessage);
      expect(result[1].content).toContain('[Previous agent context]');
    });
  });

  describe('edge cases', () => {
    test('should handle empty messages array', () => {
      const messages: never[] = [];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(0);
    });

    test('should handle AI message with empty content array', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new AIMessage({ content: [] }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      expect(result).toHaveLength(2);
      expect(result[1]).toBeInstanceOf(AIMessage);
    });

    test('should work with different providers', () => {
      const messages = [
        new AIMessage({
          content: 'Using tool',
          tool_calls: [
            {
              id: 'call_x',
              name: 'test',
              args: {},
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: 'Result',
          tool_call_id: 'call_x',
        }),
      ];

      // Test with Anthropic
      const resultAnthropic = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );
      expect(resultAnthropic).toHaveLength(1);
      expect(resultAnthropic[0]).toBeInstanceOf(HumanMessage);

      // Test with Bedrock
      const resultBedrock = ensureThinkingBlockInMessages(
        messages,
        Providers.BEDROCK
      );
      expect(resultBedrock).toHaveLength(1);
      expect(resultBedrock[0]).toBeInstanceOf(HumanMessage);
    });

    test('should handle tool message without preceding AI message', () => {
      const messages = [
        new HumanMessage({ content: 'Hello' }),
        new ToolMessage({
          content: 'Unexpected tool result',
          tool_call_id: 'call_orphan',
        }),
      ];

      const result = ensureThinkingBlockInMessages(
        messages,
        Providers.ANTHROPIC
      );

      // Should preserve both messages as-is since tool message has no preceding AI message with tools
      expect(result).toHaveLength(2);
      expect(result[0]).toBeInstanceOf(HumanMessage);
      expect(result[1]).toBeInstanceOf(ToolMessage);
    });
  });
});
