import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { MessageContentComplex, TPayload } from '@/types';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

describe('formatAgentMessages', () => {
  it('should format simple user and AI messages', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
  });

  it('preserves source messageId on formatted messages', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'msg_assistant_1',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
        ],
      },
      { role: 'user', messageId: 'msg_user_1', content: 'thanks' },
    ];

    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(HumanMessage);
    expect(result.messages[0].id).toBe('msg_assistant_1');
    expect(result.messages[1].id).toBe('msg_assistant_1');
    expect(result.messages[2].id).toBe('msg_user_1');
  });

  it('should handle system messages', () => {
    const payload = [
      { role: 'system', content: 'You are a helpful assistant.' },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(SystemMessage);
  });

  it('should prepend the latest summary and trim context before its boundary', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Old user message' },
      { role: 'assistant', content: 'Old assistant message' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Covered by summary' },
          {
            type: ContentTypes.SUMMARY,
            text: 'Conversation summary',
            tokenCount: 12,
          },
          { type: ContentTypes.TEXT, text: 'Preserved tail' },
        ],
      },
      { role: 'user', content: 'Latest user message' },
    ];

    const result = formatAgentMessages(payload, {
      0: 5,
      1: 6,
      2: 18,
      3: 4,
    });

    expect(result.messages).toHaveLength(2);
    expect(result.summary).toBeDefined();
    expect(result.summary!.text).toBe('Conversation summary');
    expect(result.summary!.tokenCount).toBe(12);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(HumanMessage);
    expect(
      (result.messages[0].content as MessageContentComplex[])[0]
    ).toMatchObject({
      type: ContentTypes.TEXT,
      text: 'Preserved tail',
    });
    expect(result.indexTokenCountMap?.[0]).toBe(18);
    expect(result.indexTokenCountMap?.[1]).toBe(4);
  });

  it('should apply last-summary-wins when multiple summary blocks exist', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.SUMMARY, text: 'Old summary', tokenCount: 3 },
          { type: ContentTypes.TEXT, text: 'Old tail' },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Drop this part' },
          { type: ContentTypes.SUMMARY, text: 'Newest summary', tokenCount: 9 },
          { type: ContentTypes.TEXT, text: 'Keep this part' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.summary).toBeDefined();
    expect(result.summary!.text).toBe('Newest summary');
    expect(result.summary!.tokenCount).toBe(9);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(
      (result.messages[0].content as MessageContentComplex[])[0]
    ).toMatchObject({
      type: ContentTypes.TEXT,
      text: 'Keep this part',
    });
  });

  it('should format messages with content arrays', () => {
    const payload = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello' }],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
  });

  it('should handle tool calls and create ToolMessages', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: '{"query":"weather"}',
              output: 'The weather is sunny.',
            },
          },
        ],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('123');
  });

  it('should handle malformed tool call entries with missing tool_call property', () => {
    const tools = new Set(['search']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that.',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            // Missing tool_call property - should not crash
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Result',
            },
          },
        ],
      },
    ];
    // Should not throw error
    const result = formatAgentMessages(payload, undefined, tools);
    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it('should handle malformed tool call entries with missing name', () => {
    const tools = new Set(['search']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Checking...',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '456',
              // Missing name property
              args: '{}',
            },
          },
        ],
      },
    ];
    // Should not throw error
    const result = formatAgentMessages(payload, undefined, tools);
    expect(result.messages).toBeDefined();
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it('should handle multiple content parts in assistant messages', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Part 1' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Part 2' },
        ],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toHaveLength(2);
  });

  it('should heal invalid tool call structure by creating a preceding AIMessage', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: '{"query":"weather"}',
              output: 'The weather is sunny.',
            },
          },
        ],
      },
    ];
    const result = formatAgentMessages(payload);

    // Should have 2 messages: an AIMessage and a ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have an empty content and the tool_call
    expect(result.messages[0].content).toBe('');
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: '123',
      name: 'search',
      args: { query: 'weather' },
    });

    // The ToolMessage should have the correct properties
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('123');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('The weather is sunny.');
  });

  it('should handle tool calls with non-JSON args', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Checking...',
            tool_call_ids: ['123'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: '123',
              name: 'search',
              args: 'non-json-string',
              output: 'Result',
            },
          },
        ],
      },
    ];
    const result = formatAgentMessages(payload);
    expect(result.messages).toHaveLength(2);
    expect(
      (result.messages[0] as AIMessage).tool_calls?.[0].args
    ).toStrictEqual({ input: 'non-json-string' });
  });

  it('should handle complex tool calls with multiple steps', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for that information.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'search',
              args: '{"query":"weather in New York"}',
              output:
                'The weather in New York is currently sunny with a temperature of 75°F.',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now, I\'ll convert the temperature.',
            tool_call_ids: ['convert_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'convert_1',
              name: 'convert_temperature',
              args: '{"temperature": 75, "from": "F", "to": "C"}',
              output: '23.89°C',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here\'s your answer.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(5);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);
    expect(result.messages[3]).toBeInstanceOf(ToolMessage);
    expect(result.messages[4]).toBeInstanceOf(AIMessage);

    // Check first AIMessage
    expect(result.messages[0].content).toBe(
      'I\'ll search for that information.'
    );
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'search_1',
      name: 'search',
      args: { query: 'weather in New York' },
    });

    // Check first ToolMessage
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('search_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe(
      'The weather in New York is currently sunny with a temperature of 75°F.'
    );

    // Check second AIMessage
    expect(result.messages[2].content).toBe(
      'Now, I\'ll convert the temperature.'
    );
    expect((result.messages[2] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[2] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'convert_1',
      name: 'convert_temperature',
      args: { temperature: 75, from: 'F', to: 'C' },
    });

    // Check second ToolMessage
    expect((result.messages[3] as ToolMessage).tool_call_id).toBe('convert_1');
    expect(result.messages[3].name).toBe('convert_temperature');
    expect(result.messages[3].content).toBe('23.89°C');

    // Check final AIMessage
    expect(result.messages[4].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Here\'s your answer.', type: ContentTypes.TEXT },
    ]);
  });

  it('should dynamically discover tools from tool_search output and keep their tool calls', () => {
    const tools = new Set(['tool_search', 'calculator']);
    const payload = [
      {
        role: 'user',
        content: 'Search for commits and list them',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for tools first.',
            tool_call_ids: ['ts_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'ts_1',
              name: 'tool_search',
              args: '{"query":"commits"}',
              output: '{"found": 1, "tools": [{"name": "list_commits"}]}',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now listing commits.',
            tool_call_ids: ['lc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_1',
              name: 'list_commits',
              args: '{"repo":"test"}',
              output: '[{"sha":"abc123"}]',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here are the results.',
          },
        ],
      },
      {
        role: 'user',
        content: 'Thanks!',
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /**
     * Since tool_search discovered list_commits, both should be kept.
     * The dynamic discovery adds list_commits to the valid tools set.
     */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(2);

    const toolNames = toolMessages.map((m) => m.name).sort();
    expect(toolNames).toEqual(['list_commits', 'tool_search']);
  });

  it('should filter out tool calls not in set and not discovered by tool_search', () => {
    const tools = new Set(['tool_search', 'calculator']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll call an unknown tool.',
            tool_call_ids: ['uk_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'uk_1',
              name: 'unknown_tool',
              args: '{}',
              output: 'result',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Done.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** unknown_tool should be filtered out since it's not in tools set and not discovered */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(0);
  });

  it('should keep all tool calls when all are in the tools set', () => {
    const tools = new Set(['search', 'calculator']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me help.',
            tool_call_ids: ['s1', 'c1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 's1',
              name: 'search',
              args: '{"q":"test"}',
              output: 'Search results',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'c1',
              name: 'calculator',
              args: '{"expr":"2+2"}',
              output: '4',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];
    expect(toolMessages.length).toBe(2);
    expect(toolMessages.map((m) => m.name).sort()).toEqual([
      'calculator',
      'search',
    ]);
  });

  it('should preserve discovered tools across multiple assistant messages', () => {
    /**
     * This test verifies that once tool_search discovers a tool, it remains valid
     * for all subsequent messages in the conversation, not just the current message.
     */
    const tools = new Set(['tool_search']);
    const payload = [
      {
        role: 'user',
        content: 'Find me a tool to list commits and use it',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me search for that tool.',
            tool_call_ids: ['ts_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'ts_1',
              name: 'tool_search',
              args: '{"query":"commits"}',
              output:
                '{"found": 1, "tools": [{"name": "list_commits_mcp_github"}]}',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now using the discovered tool.',
            tool_call_ids: ['lc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_1',
              name: 'list_commits_mcp_github',
              args: '{"repo":"test"}',
              output: '[{"sha":"abc123","message":"Initial commit"}]',
            },
          },
        ],
      },
      {
        role: 'user',
        content: 'Show me more commits',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Fetching more commits.',
            tool_call_ids: ['lc_2'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'lc_2',
              name: 'list_commits_mcp_github',
              args: '{"repo":"test","page":2}',
              output: '[{"sha":"def456","message":"Second commit"}]',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** All three tool calls should be preserved as ToolMessages */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];

    expect(toolMessages.length).toBe(3);
    expect(toolMessages[0].name).toBe('tool_search');
    expect(toolMessages[1].name).toBe('list_commits_mcp_github');
    expect(toolMessages[2].name).toBe('list_commits_mcp_github');
  });

  it('should convert invalid tools to string while keeping valid tools as ToolMessages', () => {
    /**
     * This test documents the hybrid behavior:
     * - Valid tools remain as proper AIMessage + ToolMessage structures
     * - Invalid tools are converted to string and appended to text content
     *   (preserving context without losing information)
     */
    const tools = new Set(['calculator']);
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will use two tools.',
            tool_call_ids: ['calc_1', 'unknown_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expr":"2+2"}',
              output: '4',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'unknown_1',
              name: 'some_unknown_tool',
              args: '{"query":"test"}',
              output: 'This is the result from unknown tool',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** Should have AIMessage + ToolMessage for calculator */
    expect(result.messages.length).toBe(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    /** The valid tool should be kept */
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );
    expect((result.messages[1] as ToolMessage).name).toBe('calculator');

    /** The invalid tool should be converted to string in the content */
    const aiContent = result.messages[0].content;
    const aiContentStr =
      typeof aiContent === 'string' ? aiContent : JSON.stringify(aiContent);
    expect(aiContentStr).toContain('some_unknown_tool');
    expect(aiContentStr).toContain('This is the result from unknown tool');
  });

  it('should simulate realistic deferred tools flow with tool_search', () => {
    /**
     * This test simulates the real-world use case:
     * 1. Agent only has tool_search initially (deferred tools not in set)
     * 2. User asks to do something that requires a deferred tool
     * 3. Agent uses tool_search to discover the tool
     * 4. Agent then uses the discovered tool
     * 5. On subsequent conversation turns, both tool calls should be valid
     */
    const tools = new Set(['tool_search', 'execute_code']);
    const payload = [
      { role: 'user', content: 'List the recent commits from the repo' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]:
              'I need to find a tool for listing commits. Let me search.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'tool_search',
              args: '{"query":"git commits list"}',
              output:
                '{\n  "found": 1,\n  "tools": [\n    {\n      "name": "list_commits_mcp_github",\n      "score": 0.95,\n      "matched_in": "name",\n      "snippet": "Lists commits from a GitHub repository"\n    }\n  ],\n  "total_searched": 50,\n  "query": "git commits list"\n}',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Found the tool! Now I will list the commits.',
            tool_call_ids: ['commits_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'commits_1',
              name: 'list_commits_mcp_github',
              args: '{"owner":"librechat","repo":"librechat"}',
              output:
                '[{"sha":"abc123","message":"feat: add deferred tools"},{"sha":"def456","message":"fix: tool loading"}]',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload, undefined, tools);

    /** Both tool_search and list_commits_mcp_github should be preserved */
    const toolMessages = result.messages.filter(
      (m) => m._getType() === 'tool'
    ) as ToolMessage[];

    expect(toolMessages.length).toBe(2);
    expect(toolMessages[0].name).toBe('tool_search');
    expect(toolMessages[1].name).toBe('list_commits_mcp_github');

    /** The AI messages should have proper tool_calls */
    const aiMessages = result.messages.filter(
      (m) => m._getType() === 'ai'
    ) as AIMessage[];

    const toolCallNames = aiMessages.flatMap(
      (m) => m.tool_calls?.map((tc) => tc.name) ?? []
    );
    expect(toolCallNames).toContain('tool_search');
    expect(toolCallNames).toContain('list_commits_mcp_github');
  });

  it.skip('should not produce two consecutive assistant messages and format content correctly', () => {
    const payload = [
      { role: 'user', content: 'Hello' },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hi there!' },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'How can I help you?',
          },
        ],
      },
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here\'s the weather information.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Check correct message count and types
    expect(result.messages).toHaveLength(6);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(HumanMessage);
    expect(result.messages[3]).toBeInstanceOf(AIMessage);
    expect(result.messages[4]).toBeInstanceOf(ToolMessage);
    expect(result.messages[5]).toBeInstanceOf(AIMessage);

    // Check content of messages
    expect(result.messages[0].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Hello', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[1].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'Hi there!', type: ContentTypes.TEXT },
      { [ContentTypes.TEXT]: 'How can I help you?', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[2].content).toStrictEqual([
      { [ContentTypes.TEXT]: 'What\'s the weather?', type: ContentTypes.TEXT },
    ]);
    expect(result.messages[3].content).toBe('Let me check that for you.');
    expect(result.messages[4].content).toBe('Sunny, 75°F');
    expect(result.messages[5].content).toStrictEqual([
      {
        [ContentTypes.TEXT]: 'Here\'s the weather information.',
        type: ContentTypes.TEXT,
      },
    ]);

    // Check that there are no consecutive AIMessages
    const messageTypes = result.messages.map((message) => message.constructor);
    for (let i = 0; i < messageTypes.length - 1; i++) {
      expect(
        messageTypes[i] === AIMessage && messageTypes[i + 1] === AIMessage
      ).toBe(false);
    }

    // Additional check to ensure the consecutive assistant messages were combined
    expect(result.messages[1].content).toHaveLength(2);
  });

  it('should strip THINK content and join TEXT parts as string', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Initial response' },
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Reasoning about the problem...',
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toEqual(
      'Initial response\nFinal answer'
    );
  });

  it('should join TEXT content as string when THINK content type is present', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Analyzing the problem...',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'First part of response',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Second part of response',
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Final part of response',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(typeof result.messages[0].content).toBe('string');
    expect(result.messages[0].content).toBe(
      'First part of response\nSecond part of response\nFinal part of response'
    );
    expect(result.messages[0].content).not.toContain(
      'Analyzing the problem...'
    );
  });

  it('should exclude ERROR type content parts', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello there' },
          {
            type: ContentTypes.ERROR,
            [ContentTypes.ERROR]:
              'An error occurred while processing the request: Something went wrong',
          },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[0].content).toEqual([
      { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Hello there' },
      { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Final answer' },
    ]);

    const hasErrorContent =
      Array.isArray(result.messages[0].content) &&
      result.messages[0].content.some(
        (item) =>
          item.type === ContentTypes.ERROR ||
          JSON.stringify(item).includes('An error occurred')
      );
    expect(hasErrorContent).toBe(false);
  });
  it('should handle indexTokenCountMap and return updated map', () => {
    const payload = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ];

    const indexTokenCountMap = {
      0: 5, // 5 tokens for "Hello"
      1: 10, // 10 tokens for "Hi there!"
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    expect(result.messages).toHaveLength(2);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(result.indexTokenCountMap?.[0]).toBe(5);
    expect(result.indexTokenCountMap?.[1]).toBe(10);
  });

  it('should handle complex message transformations with indexTokenCountMap', () => {
    const payload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check that for you.',
            tool_call_ids: ['weather_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'weather_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 10, // 10 tokens for "What's the weather?"
      1: 50, // 50 tokens for the assistant message with tool call
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // The original message at index 1 should be split into two messages
    expect(result.messages).toHaveLength(3);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(result.indexTokenCountMap?.[0]).toBe(10); // User message stays the same

    // The assistant message tokens should be distributed across the resulting messages
    const totalAssistantTokens =
      Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, count) => sum + count,
        0
      ) - 10; // Subtract user message tokens

    expect(totalAssistantTokens).toBe(50); // Should match the original token count
  });

  it('should handle one-to-many message expansion with tool calls', () => {
    // One message with multiple tool calls expands to multiple messages
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'First tool call:',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"test"}',
              output: 'Search result',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Second tool call:',
            tool_call_ids: ['tool_2'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_2',
              name: 'calculate',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Final response',
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 100, // 100 tokens for the complex assistant message
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // One message expands to 5 messages (2 tool calls + text before, between, and after)
    expect(result.messages).toHaveLength(5);
    expect(result.indexTokenCountMap).toBeDefined();

    // The sum of all token counts should equal the original
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(100);

    // Check that each resulting message has a token count
    for (let i = 0; i < result.messages.length; i++) {
      expect(result.indexTokenCountMap?.[i]).toBeDefined();
    }
  });

  it('should handle content filtering that reduces message count', () => {
    // Message with THINK and ERROR parts that get filtered out
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.THINK, [ContentTypes.THINK]: 'Thinking...' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'Visible response' },
          { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'Error occurred' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 60, // 60 tokens for the message with filtered content
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // Only one message should remain after filtering
    expect(result.messages).toHaveLength(1);
    expect(result.indexTokenCountMap).toBeDefined();

    // All tokens should be assigned to the remaining message
    expect(result.indexTokenCountMap?.[0]).toBe(60);
  });

  it('should handle empty result after content filtering', () => {
    // Message with only THINK and ERROR parts that all get filtered out
    const payload = [
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.THINK, [ContentTypes.THINK]: 'Thinking...' },
          { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'Error occurred' },
          { type: ContentTypes.AGENT_UPDATE, update: 'Processing...' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 40, // 40 tokens for the message with filtered content
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // No messages should remain after filtering
    expect(result.messages).toHaveLength(0);
    expect(result.indexTokenCountMap).toBeDefined();

    // The token count map should be empty since there are no messages
    expect(Object.keys(result.indexTokenCountMap || {})).toHaveLength(0);
  });

  it('should demonstrate how 2 input messages can become more than 2 output messages', () => {
    // Two input messages where one contains tool calls
    const payload = [
      { role: 'user', content: 'Can you help me with something?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll help you with that.',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"help topics"}',
              output: 'Found several help topics.',
            },
          },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 15, // 15 tokens for the user message
      1: 45, // 45 tokens for the assistant message with tool call
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // 2 input messages become 3 output messages (user + assistant + tool)
    expect(payload).toHaveLength(2);
    expect(result.messages).toHaveLength(3);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(Object.keys(result.indexTokenCountMap ?? {}).length).toBe(3);

    // Check message types
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);

    // The sum of all token counts should equal the original total
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(60); // 15 + 45
  });

  it('should handle an AI message with 5 tool calls in a single message', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll perform multiple operations for you.',
            tool_call_ids: ['tool_1', 'tool_2', 'tool_3', 'tool_4', 'tool_5'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"latest news"}',
              output: 'Found several news articles.',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_2',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_3',
              name: 'calculate',
              args: '{"expression":"356 * 24"}',
              output: '8544',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_4',
              name: 'translate',
              args: '{"text":"Hello world","source":"en","target":"fr"}',
              output: 'Bonjour le monde',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_5',
              name: 'fetch_data',
              args: '{"endpoint":"/api/users","params":{"limit":5}}',
              output:
                '{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"},{"id":3,"name":"Charlie"},{"id":4,"name":"David"},{"id":5,"name":"Eve"}]}',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 6 messages: 1 AIMessage and 5 ToolMessages
    expect(result.messages).toHaveLength(6);

    // Check message types in the correct sequence
    expect(result.messages[0]).toBeInstanceOf(AIMessage); // Initial message with all tool calls
    expect(result.messages[1]).toBeInstanceOf(ToolMessage); // Tool 1 response
    expect(result.messages[2]).toBeInstanceOf(ToolMessage); // Tool 2 response
    expect(result.messages[3]).toBeInstanceOf(ToolMessage); // Tool 3 response
    expect(result.messages[4]).toBeInstanceOf(ToolMessage); // Tool 4 response
    expect(result.messages[5]).toBeInstanceOf(ToolMessage); // Tool 5 response

    // Check AIMessage has all 5 tool calls
    expect(result.messages[0].content).toBe(
      'I\'ll perform multiple operations for you.'
    );
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(5);

    // Verify each tool call in the AIMessage
    expect((result.messages[0] as AIMessage).tool_calls?.[0]).toEqual({
      id: 'tool_1',
      name: 'search',
      args: { query: 'latest news' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[1]).toEqual({
      id: 'tool_2',
      name: 'check_weather',
      args: { location: 'New York' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[2]).toEqual({
      id: 'tool_3',
      name: 'calculate',
      args: { expression: '356 * 24' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[3]).toEqual({
      id: 'tool_4',
      name: 'translate',
      args: { text: 'Hello world', source: 'en', target: 'fr' },
    });

    expect((result.messages[0] as AIMessage).tool_calls?.[4]).toEqual({
      id: 'tool_5',
      name: 'fetch_data',
      args: { endpoint: '/api/users', params: { limit: 5 } },
    });

    // Check each ToolMessage
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('Found several news articles.');

    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('tool_2');
    expect(result.messages[2].name).toBe('check_weather');
    expect(result.messages[2].content).toBe('Sunny, 75°F');

    expect((result.messages[3] as ToolMessage).tool_call_id).toBe('tool_3');
    expect(result.messages[3].name).toBe('calculate');
    expect(result.messages[3].content).toBe('8544');

    expect((result.messages[4] as ToolMessage).tool_call_id).toBe('tool_4');
    expect(result.messages[4].name).toBe('translate');
    expect(result.messages[4].content).toBe('Bonjour le monde');

    expect((result.messages[5] as ToolMessage).tool_call_id).toBe('tool_5');
    expect(result.messages[5].name).toBe('fetch_data');
    expect(result.messages[5].content).toBe(
      '{"users":[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"},{"id":3,"name":"Charlie"},{"id":4,"name":"David"},{"id":5,"name":"Eve"}]}'
    );
  });

  it('should heal tool call structure with thinking content', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]:
              'I\'ll add this agreement as an observation to our existing troubleshooting task in the project memory system.',
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tooluse_Zz-mw_wHTrWTvDHaCbfaZg',
              name: 'add_observations_mcp_project-memory',
              args: '{"observations":[{"entityName":"MCP_Tool_Error_Troubleshooting","contents":["Agreement established: Document all future tests in the project memory system to maintain a comprehensive troubleshooting log","This will provide a structured record of the entire troubleshooting process and help identify patterns in the error behavior"]}]}',
              type: 'tool_call',
              progress: 1,
              output:
                '[\n  {\n    "entityName": "MCP_Tool_Error_Troubleshooting",\n    "addedObservations": [\n      {\n        "content": "Agreement established: Document all future tests in the project memory system to maintain a comprehensive troubleshooting log",\n        "timestamp": "2025-03-26T00:46:42.154Z"\n      },\n      {\n        "content": "This will provide a structured record of the entire troubleshooting process and help identify patterns in the error behavior",\n        "timestamp": "2025-03-26T00:46:42.154Z"\n      }\n    ]\n  }\n]',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]:
              '\n\nI\'ve successfully added our agreement to the project memory system. The observation has been recorded in the "MCP_Tool_Error_Troubleshooting" entity with the current timestamp.\n\nGoing forward, I will:\n\n1. Document each test we perform\n2. Record the methodology and results\n3. Update the project memory with our findings\n4. Establish appropriate relationships between tests and related components\n5. Provide a summary of what we\'ve learned from each test\n\nThis structured approach will help us build a comprehensive knowledge base of the error behavior and our troubleshooting process, which may prove valuable for resolving similar issues in the future or for other developers facing similar challenges.\n\nWhat test would you like to perform next in our troubleshooting process?',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 3 messages: an AIMessage with empty content, a ToolMessage, and a final AIMessage with the text
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);
    expect(result.messages[2]).toBeInstanceOf(AIMessage);

    // The first AIMessage should have an empty content and the tool_call
    expect(result.messages[0].content).toBe('');
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'add_observations_mcp_project-memory'
    );

    // The ToolMessage should have the correct properties
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe(
      'tooluse_Zz-mw_wHTrWTvDHaCbfaZg'
    );
    expect(result.messages[1].name).toBe('add_observations_mcp_project-memory');
    expect(result.messages[1].content).toContain(
      'MCP_Tool_Error_Troubleshooting'
    );

    // The final AIMessage should contain the text response
    expect(typeof result.messages[2].content).toBe('string');
    expect((result.messages[2].content as string).trim()).toContain(
      'I\'ve successfully added our agreement to the project memory system'
    );
  });

  it('should demonstrate how messages can be filtered out, reducing count', () => {
    // Two input messages where one gets completely filtered out
    const payload = [
      { role: 'user', content: 'Hello there' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.THINK,
            [ContentTypes.THINK]: 'Thinking about response...',
          },
          {
            type: ContentTypes.ERROR,
            [ContentTypes.ERROR]: 'Error in processing',
          },
          { type: ContentTypes.AGENT_UPDATE, update: 'Working on it...' },
        ],
      },
    ];

    const indexTokenCountMap = {
      0: 10, // 10 tokens for the user message
      1: 30, // 30 tokens for the assistant message that will be filtered out
    };

    const result = formatAgentMessages(payload, indexTokenCountMap);

    // 2 input messages become 1 output message (only the user message remains)
    expect(payload).toHaveLength(2);
    expect(result.messages).toHaveLength(1);
    expect(result.indexTokenCountMap).toBeDefined();
    expect(Object.keys(result.indexTokenCountMap ?? {}).length).toBe(1);

    // Check message type
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);

    // Only the user message tokens should remain
    expect(result.indexTokenCountMap?.[0]).toBe(10);

    // The total tokens should be just the user message tokens
    const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
      (sum, count) => sum + count,
      0
    );

    expect(totalTokens).toBe(10);
  });

  it('should skip invalid tool calls with no name AND no output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me help you with that.',
            tool_call_ids: ['valid_tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'invalid_tool_1',
              name: '',
              args: '{"query":"test"}',
              output: '',
            },
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'valid_tool_1',
              name: 'search',
              args: '{"query":"weather"}',
              output: 'The weather is sunny.',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage (invalid tool call is skipped)
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should only have 1 tool call (the valid one)
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );
    expect((result.messages[0] as AIMessage).tool_calls?.[0].id).toBe(
      'valid_tool_1'
    );

    // The ToolMessage should be for the valid tool call
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe(
      'valid_tool_1'
    );
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('The weather is sunny.');
  });

  it('should skip tool calls with no name AND null output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'invalid_tool_1',
              name: '',
              args: '{"query":"test"}',
              output: null,
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Here is the information.',
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 1 message: AIMessage (invalid tool call is skipped)
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool calls or an empty array
    const toolCalls = (result.messages[0] as AIMessage).tool_calls;
    expect(toolCalls === undefined || toolCalls.length === 0).toBe(true);
    expect(result.messages[0].content).toStrictEqual([
      {
        type: ContentTypes.TEXT,
        [ContentTypes.TEXT]: 'Here is the information.',
      },
    ]);
  });

  it('should NOT skip tool calls with no name but valid output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: '',
              args: '{"query":"test"}',
              output: 'Valid output despite missing name',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have 1 tool call
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);

    // The ToolMessage should have the output
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].content).toBe(
      'Valid output despite missing name'
    );
  });

  it('should NOT skip tool calls with valid name but no output', () => {
    const payload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"test"}',
              output: '',
            },
          },
        ],
      },
    ];

    const result = formatAgentMessages(payload);

    // Should have 2 messages: AIMessage and ToolMessage
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);
    expect(result.messages[1]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have 1 tool call
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[0] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );

    // The ToolMessage should have empty content
    expect((result.messages[1] as ToolMessage).tool_call_id).toBe('tool_1');
    expect(result.messages[1].name).toBe('search');
    expect(result.messages[1].content).toBe('');
  });

  describe('proportional token distribution', () => {
    it('should distribute tokens proportionally based on content length', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Short text',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'search',
                args: '{"query":"test"}',
                output:
                  'A much longer tool result that contains significantly more content than the original message text',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(100);
      expect(toolTokens).toBeGreaterThan(aiTokens);
    });

    it('should give the vast majority of tokens to a large tool result vs tiny AI message', () => {
      const bigOutput = 'x'.repeat(10000);
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'ok',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'snapshot',
                args: '{}',
                output: bigOutput,
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 5000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(5000);
      expect(toolTokens).toBeGreaterThan(4900);
      expect(aiTokens).toBeLessThan(100);
    });

    it('should fall back to even distribution when all content lengths are zero', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: '',
              tool_call_ids: ['tool_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'noop',
                args: '{}',
                output: '',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 20 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(2);
      const aiTokens = result.indexTokenCountMap?.[0] ?? 0;
      const toolTokens = result.indexTokenCountMap?.[1] ?? 0;
      expect(aiTokens + toolTokens).toBe(20);
      expect(aiTokens).toBeGreaterThanOrEqual(0);
      expect(toolTokens).toBeGreaterThanOrEqual(0);
    });

    it('should handle odd token counts without losing remainder', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'abc',
              tool_call_ids: ['tool_1', 'tool_2', 'tool_3'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_1',
                name: 'a',
                args: '{}',
                output: 'abc',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_2',
                name: 'b',
                args: '{}',
                output: 'abc',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'tool_3',
                name: 'c',
                args: '{}',
                output: 'abc',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 7 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(4);
      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(7);
      for (let i = 0; i < result.messages.length; i++) {
        expect(result.indexTokenCountMap?.[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it('should never produce negative token counts', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'a',
              tool_call_ids: ['t1', 't2', 't3', 't4', 't5'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'x', args: '{}', output: 'b' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't2', name: 'x', args: '{}', output: 'c' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't3', name: 'x', args: '{}', output: 'd' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't4', name: 'x', args: '{}', output: 'e' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't5', name: 'x', args: '{}', output: 'f' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 3 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(3);
      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle single token budget distributed across many messages', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'hello',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'a',
                args: '{}',
                output: 'result one',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'b',
                args: '{}',
                output: 'result two',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 1 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(1);
      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle zero token budget', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'hello',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'world' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 0 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(0);
    });

    it('should distribute tokens proportionally with 5 tool calls of varying sizes', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'I will perform multiple operations.',
              tool_call_ids: ['t1', 't2', 't3', 't4', 't5'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'navigate',
                args: '{"url":"https://example.com"}',
                output: 'Navigated successfully.',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'snapshot',
                args: '{}',
                output: 'x'.repeat(5000),
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't3',
                name: 'click',
                args: '{"selector":"#btn"}',
                output: 'Clicked.',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't4',
                name: 'snapshot',
                args: '{}',
                output: 'y'.repeat(8000),
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't5',
                name: 'extract',
                args: '{"selector":"h1"}',
                output: 'Page Title',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 3000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(6);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(3000);

      const snapshotIdx1 = 2;
      const snapshotIdx2 = 4;
      const bigSnapshotTokens =
        (result.indexTokenCountMap?.[snapshotIdx1] ?? 0) +
        (result.indexTokenCountMap?.[snapshotIdx2] ?? 0);
      expect(bigSnapshotTokens).toBeGreaterThan(2500);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle HN-like payload: AI with 18 tool calls and large snapshot results', () => {
      const smallOutput = 'Successfully navigated to page.';
      const hugeSnapshot = 'uid=8_0 RootWebArea ' + 'x'.repeat(20000);

      const toolCalls: Array<{
        type: string;
        tool_call: { id: string; name: string; args: string; output: string };
      }> = [];
      const toolCallIds: string[] = [];

      for (let i = 0; i < 18; i++) {
        const id = `tool_${i}`;
        toolCallIds.push(id);
        const isSnapshot = i % 3 === 1;
        toolCalls.push({
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id,
            name: isSnapshot ? 'take_snapshot' : 'navigate_page',
            args: isSnapshot ? '{}' : `{"url":"https://example.com/${i}"}`,
            output: isSnapshot ? hugeSnapshot : smallOutput,
          },
        });
      }

      const payload = [
        {
          role: 'user',
          content: 'Look up top 5 posts on HN',
        },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: '',
              tool_call_ids: toolCallIds,
            },
            ...toolCalls,
          ],
        },
      ];

      const indexTokenCountMap = { 0: 20, 1: 10000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages.length).toBeGreaterThan(2);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(10020);

      expect(result.indexTokenCountMap?.[0]).toBe(20);

      let snapshotTokenTotal = 0;
      let navTokenTotal = 0;
      for (let i = 1; i < result.messages.length; i++) {
        const tokens = result.indexTokenCountMap?.[i] ?? 0;
        expect(tokens).toBeGreaterThanOrEqual(0);

        if (result.messages[i] instanceof ToolMessage) {
          const content = result.messages[i].content;
          if (typeof content === 'string' && content.length > 1000) {
            snapshotTokenTotal += tokens;
          } else {
            navTokenTotal += tokens;
          }
        }
      }

      expect(snapshotTokenTotal).toBeGreaterThan(navTokenTotal);
    });

    it('should complete proportional distribution within reasonable time for large payloads', () => {
      const toolCalls: Array<{
        type: string;
        tool_call: { id: string; name: string; args: string; output: string };
      }> = [];
      const toolCallIds: string[] = [];

      for (let i = 0; i < 50; i++) {
        const id = `tool_${i}`;
        toolCallIds.push(id);
        toolCalls.push({
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id,
            name: `tool_${i}`,
            args: JSON.stringify({ data: 'x'.repeat(100) }),
            output: 'y'.repeat(Math.floor(Math.random() * 10000)),
          },
        });
      }

      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Processing...',
              tool_call_ids: toolCallIds,
            },
            ...toolCalls,
          ],
        },
      ];

      const indexTokenCountMap = { 0: 50000 };

      const start = performance.now();
      const result = formatAgentMessages(payload, indexTokenCountMap);
      const elapsed = performance.now() - start;

      expect(elapsed).toBeLessThan(500);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(50000);
    });

    it('should always preserve total token count across multiple original messages', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Let me search.',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{"q":"test"}',
                output:
                  'Found 10 results with detailed descriptions: ' +
                  'z'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: 'Thanks' },
        { role: 'assistant', content: 'You are welcome!' },
      ];

      const indexTokenCountMap = { 0: 5, 1: 200, 2: 3, 3: 8 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(216);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(Number.isInteger(val)).toBe(true);
      }
    });

    it('should produce integer token counts (no floating point)', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'abc',
              tool_call_ids: ['t1', 't2', 't3'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'defgh' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't2', name: 'b', args: '{}', output: 'ij' },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't3',
                name: 'c',
                args: '{}',
                output: 'klmnopqrst',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 97 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(Number.isInteger(val)).toBe(true);
      }

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(97);
    });

    it('should account for tool call args in content length calculation', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'x',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'tiny_tool',
                args: '{}',
                output: 'small',
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'big_args_tool',
                args: JSON.stringify({ data: 'a'.repeat(5000) }),
                output: 'small',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 1000 };
      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.messages).toHaveLength(3);

      const total = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, v) => sum + v,
        0
      );
      expect(total).toBe(1000);

      for (const val of Object.values(result.indexTokenCountMap || {})) {
        expect(val).toBeGreaterThanOrEqual(0);
      }
    });

    it('should not throw when indexTokenCountMap has undefined values for some indices', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'response',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: 'result',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap: Record<number, number | undefined> = {
        0: undefined,
        1: 50,
      };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(50);
      }).not.toThrow();
    });

    it('should not throw when indexTokenCountMap is sparse (missing indices)', () => {
      const payload = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'World' },
        { role: 'user', content: 'Bye' },
      ];

      const indexTokenCountMap = { 0: 5, 2: 3 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        expect(result.indexTokenCountMap?.[0]).toBe(5);
        expect(result.indexTokenCountMap?.[2]).toBe(3);
      }).not.toThrow();
    });

    it('should not throw when indexTokenCountMap has extra indices beyond payload', () => {
      const payload = [{ role: 'user', content: 'Hello' }];

      const indexTokenCountMap = { 0: 5, 1: 10, 2: 15, 99: 999 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap?.[0]).toBe(5);
      }).not.toThrow();
    });

    it('should not throw with empty payload and non-empty indexTokenCountMap', () => {
      const payload: Array<{ role: string; content: string }> = [];
      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.messages).toHaveLength(0);
      }).not.toThrow();
    });

    it('should not throw when assistant message content is empty array', () => {
      const payload = [
        {
          role: 'assistant',
          content: [] as Array<{ type: string; text?: string }>,
        },
      ];

      const indexTokenCountMap = { 0: 50 };

      expect(() => {
        formatAgentMessages(payload, indexTokenCountMap);
      }).not.toThrow();
    });

    it('should not throw when tool call output is null or undefined', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'calling tools',
              tool_call_ids: ['t1', 't2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: null as unknown as string,
              },
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'fetch',
                args: '{}',
                output: undefined as unknown as string,
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 30 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(30);
      }).not.toThrow();
    });

    it('should not throw when tool call args are deeply nested objects', () => {
      const deepArgs = { a: { b: { c: { d: { e: { f: 'deep' } } } } } };
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'deep call',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'deep_tool',
                args: JSON.stringify(deepArgs),
                output: 'done',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(100);
      }).not.toThrow();
    });

    it('should not throw when tool call args are not valid JSON strings', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'bad args',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'tool',
                args: '{not valid json!!!',
                output: 'output',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 40 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(40);
      }).not.toThrow();
    });

    it('should not throw when content array has mixed types including unexpected values', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'hello' },
            null as unknown as { type: string },
            undefined as unknown as { type: string },
            { type: 'unknown_type', something: 'weird' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 25 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap?.[0]).toBe(25);
      }).not.toThrow();
    });

    it('should not throw when tool call has empty name and empty args', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'test',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: '',
                args: '',
                output: 'some output',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 50 };

      expect(() => {
        formatAgentMessages(payload, indexTokenCountMap);
      }).not.toThrow();
    });

    it('should not throw when all content parts are filtered out (THINK + ERROR only)', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.THINK, [ContentTypes.THINK]: 'thinking...' },
            { type: ContentTypes.ERROR, [ContentTypes.ERROR]: 'error...' },
          ],
        },
      ];

      const indexTokenCountMap = { 0: 100 };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(Object.keys(result.indexTokenCountMap || {}).length).toBe(0);
      }).not.toThrow();
    });

    it('should not throw with very large token count values', () => {
      const payload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'big tokens',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: { id: 't1', name: 'a', args: '{}', output: 'b' },
            },
          ],
        },
      ];

      const indexTokenCountMap = { 0: Number.MAX_SAFE_INTEGER };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        const total = Object.values(result.indexTokenCountMap || {}).reduce(
          (sum, v) => sum + v,
          0
        );
        expect(total).toBe(Number.MAX_SAFE_INTEGER);
      }).not.toThrow();
    });

    it('should not throw when multiple payload messages expand and some have undefined token counts', () => {
      const payload = [
        { role: 'user', content: 'msg1' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'response with tool',
              tool_call_ids: ['t1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't1',
                name: 'search',
                args: '{}',
                output: 'found',
              },
            },
          ],
        },
        { role: 'user', content: 'msg2' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'another response',
              tool_call_ids: ['t2'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 't2',
                name: 'fetch',
                args: '{}',
                output: 'data',
              },
            },
          ],
        },
      ];

      const indexTokenCountMap: Record<number, number | undefined> = {
        0: 5,
        1: undefined,
        2: 3,
        3: 80,
      };

      expect(() => {
        const result = formatAgentMessages(payload, indexTokenCountMap);
        expect(result.indexTokenCountMap).toBeDefined();
        expect(result.indexTokenCountMap?.[0]).toBe(5);
      }).not.toThrow();
    });
  });

  describe('cross-run summary token accounting', () => {
    it('should conserve tokens: summary boundary excludes pre-boundary messages from the map', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Old question' },
        { role: 'assistant', content: 'Old answer' },
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'Text before summary' },
            {
              type: ContentTypes.SUMMARY,
              text: 'This is a conversation summary capturing prior context.',
              tokenCount: 25,
            },
            { type: ContentTypes.TEXT, text: 'Text after summary' },
          ],
        },
        { role: 'user', content: 'New question after summary' },
        { role: 'assistant', content: 'New answer after summary' },
      ];

      const indexTokenCountMap = {
        0: 8,
        1: 12,
        2: 60,
        3: 10,
        4: 15,
      };

      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();
      expect(result.summary!.text).toBe(
        'This is a conversation summary capturing prior context.'
      );
      expect(result.summary!.tokenCount).toBe(25);

      const preBoundaryTokensPresent = Object.values(
        result.indexTokenCountMap || {}
      ).some((v) => v === 8 || v === 12);
      expect(preBoundaryTokensPresent).toBe(false);

      const postBoundaryTotal = Object.values(
        result.indexTokenCountMap || {}
      ).reduce((sum, v) => sum + v, 0);

      expect(postBoundaryTotal).toBe(60 + 10 + 15);
    });

    it('should preserve summary token at index 0 when tool calls expand post-boundary messages', () => {
      const payload: TPayload = [
        { role: 'user', content: 'Summarized away' },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'Summary of the conversation so far.',
              tokenCount: 20,
            },
          ],
        },
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'Let me compute that.',
              tool_call_ids: ['calc_1'],
            },
            {
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: 'calc_1',
                name: 'calculator',
                args: '{"expr":"2+2"}',
                output: '4',
              },
            },
            {
              type: ContentTypes.TEXT,
              [ContentTypes.TEXT]: 'The answer is 4.',
            },
          ],
        },
        { role: 'user', content: 'Thanks!' },
      ];

      const indexTokenCountMap = {
        0: 5,
        1: 30,
        2: 80,
        3: 6,
      };

      const result = formatAgentMessages(payload, indexTokenCountMap);

      expect(result.summary).toBeDefined();
      expect(result.summary!.text).toBe('Summary of the conversation so far.');
      expect(result.summary!.tokenCount).toBe(20);

      const totalTokens = Object.values(result.indexTokenCountMap || {}).reduce(
        (sum, count) => sum + count,
        0
      );
      expect(totalTokens).toBe(80 + 6);
    });

    it('should produce correct maps across a simulated multi-run lifecycle', () => {
      const run1Payload: TPayload = [
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: 'The answer is 4.' },
      ];
      const run1Map = { 0: 10, 1: 12 };

      const run1Result = formatAgentMessages(run1Payload, run1Map);
      expect(run1Result.messages).toHaveLength(2);
      expect(run1Result.indexTokenCountMap?.[0]).toBe(10);
      expect(run1Result.indexTokenCountMap?.[1]).toBe(12);

      const run2Payload: TPayload = [
        ...run1Payload,
        { role: 'user', content: 'Now multiply 4 by 10.' },
        {
          role: 'assistant',
          content: [
            { type: ContentTypes.TEXT, text: 'Sure, the answer is 40.' },
            {
              type: ContentTypes.SUMMARY,
              text: 'User asked basic arithmetic: 2+2=4, then 4*10=40.',
              tokenCount: 18,
            },
          ],
        },
      ];
      const run2Map = { 0: 10, 1: 12, 2: 14, 3: 50 };

      const run2Result = formatAgentMessages(run2Payload, run2Map);
      expect(run2Result.summary).toBeDefined();
      expect(run2Result.summary!.text).toBe(
        'User asked basic arithmetic: 2+2=4, then 4*10=40.'
      );
      expect(run2Result.summary!.tokenCount).toBe(18);

      const run2TotalPostBoundary = Object.values(
        run2Result.indexTokenCountMap || {}
      ).reduce((sum, v) => sum + v, 0);
      expect(run2TotalPostBoundary).toBe(0);

      const run3Payload: TPayload = [
        {
          role: 'assistant',
          content: [
            {
              type: ContentTypes.SUMMARY,
              text: 'User asked basic arithmetic: 2+2=4, then 4*10=40.',
              tokenCount: 18,
            },
          ],
        },
        { role: 'user', content: 'What is the square root of 40?' },
        {
          role: 'assistant',
          content: 'The square root of 40 is approximately 6.32.',
        },
      ];
      const run3Map = { 0: 18, 1: 15, 2: 20 };

      const run3Result = formatAgentMessages(run3Payload, run3Map);
      expect(run3Result.summary).toBeDefined();
      expect(run3Result.summary!.text).toBe(
        'User asked basic arithmetic: 2+2=4, then 4*10=40.'
      );
      expect(run3Result.summary!.tokenCount).toBe(18);

      const run3Total = Object.values(
        run3Result.indexTokenCountMap || {}
      ).reduce((sum, count) => sum + count, 0);
      expect(run3Total).toBe(15 + 20);
    });
  });
});
