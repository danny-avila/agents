import { HumanMessage, AIMessage, ToolMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

describe('formatAgentMessages with tools parameter', () => {
  it('should process messages normally when tools is not provided', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Hello' },
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

    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('123');
  });

  it('should filter out all tool calls when tools set is empty', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
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

    // Provide an empty set of allowed tools
    const allowedTools = new Set<string>();

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should filter out the tool call, keeping only text content
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool_calls (they were filtered out)
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(0);
  });

  it('should filter out tool calls not in the allowed set', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
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

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should filter out the invalid tool call, keeping text content
    expect(result.messages).toHaveLength(2);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool_calls (check_weather was filtered out)
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(0);
  });

  it('should not convert tool messages when tool is in the allowed set', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
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

    // Provide a set of allowed tools that includes 'check_weather'
    const allowedTools = new Set(['check_weather', 'search']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should keep the original structure
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);
  });

  it('should handle multiple tool calls with mixed allowed/disallowed tools', () => {
    const payload: TPayload = [
      {
        role: 'user',
        content: 'Tell me about the weather and calculate something',
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather first.',
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
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now let me calculate something for you.',
            tool_call_ids: ['calc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
        ],
      },
    ];

    // Allow calculator but not check_weather
    const allowedTools = new Set(['calculator', 'search']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should keep valid tool (calculator) and convert invalid (check_weather) to string
    expect(result.messages).toHaveLength(3);
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage);
    expect(result.messages[2]).toBeInstanceOf(ToolMessage);

    // The AIMessage should have the calculator tool_call
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );

    // The content should include invalid tool as string
    expect(result.messages[1].content).toContain('check_weather');
    expect(result.messages[1].content).toContain('Sunny, 75°F');

    // The ToolMessage should be for calculator
    expect((result.messages[2] as ToolMessage).name).toBe('calculator');
    expect(result.messages[2].content).toBe('2');
  });

  it('should update indexTokenCountMap correctly when converting tool messages', () => {
    const payload: TPayload = [
      { role: 'user', content: 'What\'s the weather?' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Let me check the weather for you.',
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
      0: 10, // 10 tokens for user message
      1: 40, // 40 tokens for assistant message with tool call
    };

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(
      payload,
      indexTokenCountMap,
      allowedTools
    );

    // Should have 2 messages and 2 entries in the token count map
    expect(result.messages).toHaveLength(2);
    expect(Object.keys(result.indexTokenCountMap || {}).length).toBe(2);

    // User message token count should be unchanged
    expect(result.indexTokenCountMap?.[0]).toBe(10);

    // All assistant message tokens should be assigned to the single AIMessage
    expect(result.indexTokenCountMap?.[1]).toBe(40);
  });

  it('should convert invalid tool to text content when no other content exists', () => {
    const payload: TPayload = [
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'check_weather',
              args: '{"location":"New York"}',
              output: 'Sunny, 75°F',
            },
          },
        ],
      },
    ];

    // Provide a set of allowed tools that doesn't include 'check_weather'
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // Should create an AIMessage with the invalid tool converted to text
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0]).toBeInstanceOf(AIMessage);

    // The AIMessage should have no tool_calls (all were invalid)
    expect((result.messages[0] as AIMessage).tool_calls).toHaveLength(0);

    // The content should contain the invalid tool info
    const content = result.messages[0].content;
    const contentStr =
      typeof content === 'string' ? content : JSON.stringify(content);
    expect(contentStr).toContain('check_weather');
    expect(contentStr).toContain('Sunny, 75°F');
  });

  it('should handle complex sequences with multiple tool calls', () => {
    const payload: TPayload = [
      { role: 'user', content: 'Help me with a complex task' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I\'ll search for information first.',
            tool_call_ids: ['search_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'search_1',
              name: 'search',
              args: '{"query":"complex task"}',
              output: 'Found information about complex tasks.',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Now I\'ll check the weather.',
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
            [ContentTypes.TEXT]: 'Finally, I\'ll calculate something.',
            tool_call_ids: ['calc_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'calc_1',
              name: 'calculator',
              args: '{"expression":"1+1"}',
              output: '2',
            },
          },
        ],
      },
      {
        role: 'assistant',
        content: 'Here\'s your answer based on all that information.',
      },
    ];

    // Allow search and calculator but not check_weather
    const allowedTools = new Set(['search', 'calculator']);

    const result = formatAgentMessages(payload, undefined, allowedTools);

    // With selective filtering: valid tools are kept, invalid tools are converted to string
    // 1. HumanMessage
    // 2. AIMessage (search tool_call)
    // 3. ToolMessage (search result)
    // 4. AIMessage (text + invalid weather tool as string, no tool_calls)
    // 5. AIMessage (calculator tool_call)
    // 6. ToolMessage (calculator result)
    // 7. AIMessage (final text)
    expect(result.messages).toHaveLength(7);

    // Check the types of messages
    expect(result.messages[0]).toBeInstanceOf(HumanMessage);
    expect(result.messages[1]).toBeInstanceOf(AIMessage); // Search message
    expect(result.messages[2]).toBeInstanceOf(ToolMessage); // Search tool response
    expect(result.messages[3]).toBeInstanceOf(AIMessage); // Weather message (tool converted to string)
    expect(result.messages[4]).toBeInstanceOf(AIMessage); // Calculator message
    expect(result.messages[5]).toBeInstanceOf(ToolMessage); // Calculator tool response
    expect(result.messages[6]).toBeInstanceOf(AIMessage); // Final message

    // Check that search tool was kept
    expect((result.messages[1] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[1] as AIMessage).tool_calls?.[0].name).toBe(
      'search'
    );

    // Check that weather message has no tool_calls but contains the invalid tool as text
    expect((result.messages[3] as AIMessage).tool_calls).toHaveLength(0);
    const weatherContent = result.messages[3].content;
    const weatherContentStr =
      typeof weatherContent === 'string'
        ? weatherContent
        : JSON.stringify(weatherContent);
    expect(weatherContentStr).toContain('check_weather');
    expect(weatherContentStr).toContain('Sunny');

    // Check that calculator tool was kept
    expect((result.messages[4] as AIMessage).tool_calls).toHaveLength(1);
    expect((result.messages[4] as AIMessage).tool_calls?.[0].name).toBe(
      'calculator'
    );
  });
});
