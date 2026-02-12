// src/specs/thinking-prune.test.ts
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { createPruneMessages } from '@/messages/prune';

// Create a simple token counter for testing
const createTestTokenCounter = (): t.TokenCounter => {
  return (message: BaseMessage): number => {
    // Use type assertion to help TypeScript understand the type
    const content = message.content as
      | string
      | Array<t.MessageContentComplex | string>
      | undefined;

    // Handle string content
    if (typeof content === 'string') {
      return content.length;
    }

    // Handle array content
    if (Array.isArray(content)) {
      let totalLength = 0;

      for (const item of content) {
        if (typeof item === 'string') {
          totalLength += item.length;
        } else if (typeof item === 'object') {
          if (item.type === 'thinking' && typeof item.thinking === 'string') {
            totalLength += item.thinking.length;
          } else if ('text' in item && typeof item.text === 'string') {
            totalLength += item.text.length;
          } else if ('input' in item && typeof item.input === 'string') {
            totalLength += item.input.length;
          }
        }
      }

      return totalLength;
    }

    // Default case - if content is null, undefined, or any other type
    return 0;
  };
};

describe('Prune Messages with Thinking Mode Tests', () => {
  jest.setTimeout(30000);

  it('should preserve thinking blocks when pruning with thinking mode enabled', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages based on the example provided
    const userMessage = new HumanMessage({
      content: [
        {
          type: 'text',
          text: '/home/danny/LibreChat/gistfile1.txt\n\nread it 200 lines at a time\n\nthere are 5000 lines\n\ndo not stop until done',
        },
      ],
    });

    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking:
            'The user is asking me to read a file located at `/home/danny/LibreChat/gistfile1.txt` in chunks of 200 lines at a time, mentioning that the file has 5000 lines total. They want me to continue reading through the entire file without stopping.\n\nI\'ll need to use the text editor tool to view the file in chunks of 200 lines each. Since the file has 5000 lines, I\'ll need to view it in 25 chunks (5000 ÷ 200 = 25).\n\nI\'ll need to make multiple calls to the text editor with the `view` command, specifying different line ranges for each call.\n\nLet me plan out the approach:\n1. Start with lines 1-200\n2. Then 201-400\n3. Then 401-600\n4. And so on until I reach 4801-5000\n\nFor each call, I\'ll use the `view` command with the specific line range in the `view_range` parameter. I\'ll continue until I\'ve shown all 5000 lines as requested.',
        },
        {
          type: 'text',
          text: 'I\'ll read the file for you in chunks of 200 lines at a time until all 5000 lines are displayed.\n\nLet me start with the first 200 lines:',
        },
        {
          type: 'tool_use',
          id: 'toolu_01YApWuFsEQCuBFDgYXmcmeZ',
          name: 'text_editor_mcp_textEditor',
          input:
            '{"command": "view", "path": "/home/danny/LibreChat/gistfile1.txt", "description": "Viewing lines 1-200 of the file", "view_range": [1,200]}',
        },
      ],
    });

    const toolResponseMessage1 = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content (truncated):"',
        },
      ],
      tool_call_id: 'toolu_01YApWuFsEQCuBFDgYXmcmeZ',
      name: 'text_editor_mcp_textEditor',
    });

    const assistantMessage2 = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'It looks like the file content is clipped in the response. I\'ll continue with the next chunks:',
        },
        {
          type: 'tool_use',
          id: 'toolu_01VnyMQ4CvEd6zLDxxtTd6d4',
          name: 'text_editor_mcp_textEditor',
          input:
            '{"command": "view", "path": "/home/danny/LibreChat/gistfile1.txt", "description": "Viewing lines 201-400 of the file", "view_range": [201,400]}',
        },
      ],
    });

    const toolResponseMessage2 = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content (truncated):"',
        },
      ],
      tool_call_id: 'toolu_01VnyMQ4CvEd6zLDxxtTd6d4',
      name: 'text_editor_mcp_textEditor',
    });

    const assistantMessage3 = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'Continuing with the next chunk:',
        },
        {
          type: 'tool_use',
          id: 'toolu_01TZKs4nnBc58BYXKz1Mw4fp',
          name: 'text_editor_mcp_textEditor',
          input:
            '{"command": "view", "path": "/home/danny/LibreChat/gistfile1.txt", "description": "Viewing lines 401-600 of the file", "view_range": [401,600]}',
        },
      ],
    });

    const toolResponseMessage3 = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content (truncated):"',
        },
      ],
      tool_call_id: 'toolu_01TZKs4nnBc58BYXKz1Mw4fp',
      name: 'text_editor_mcp_textEditor',
    });

    const assistantMessage4 = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'Let\'s continue with the next chunk:',
        },
        {
          type: 'tool_use',
          id: 'toolu_01TZgBacNxjx1QNUpJg9hca5',
          name: 'text_editor_mcp_textEditor',
          input:
            '{"command": "view", "path": "/home/danny/LibreChat/gistfile1.txt", "description": "Viewing lines 601-800 of the file", "view_range": [601,800]}',
        },
      ],
    });

    const toolResponseMessage4 = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content (truncated):"',
        },
      ],
      tool_call_id: 'toolu_01TZgBacNxjx1QNUpJg9hca5',
      name: 'text_editor_mcp_textEditor',
    });

    const messages = [
      userMessage,
      assistantMessageWithThinking,
      toolResponseMessage1,
      assistantMessage2,
      toolResponseMessage2,
      assistantMessage3,
      toolResponseMessage3,
      assistantMessage4,
      toolResponseMessage4,
    ];

    // Create indexTokenCountMap based on the example provided
    const indexTokenCountMap = {
      '0': 617, // userMessage
      '1': 52, // assistantMessageWithThinking
      '2': 4995, // toolResponseMessage1
      '3': 307, // assistantMessage2
      '4': 9359, // toolResponseMessage2
      '5': 178, // assistantMessage3
      '6': 5463, // toolResponseMessage3
      '7': 125, // assistantMessage4
      '8': 4264, // toolResponseMessage4
    };

    // Create pruneMessages function with thinking mode enabled
    const pruneMessages = createPruneMessages({
      maxTokens: 19800,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({
      messages,
      usageMetadata: {
        input_tokens: 25254,
        output_tokens: 106,
        total_tokens: 25360,
        input_token_details: {
          cache_read: 0,
          cache_creation: 0,
        },
      },
      startType: 'human',
    });

    // Verify that the first assistant message in the pruned context has a thinking block
    expect(result.context.length).toBeGreaterThan(0);

    // Find the first assistant message in the pruned context
    const firstAssistantIndex = result.context.findIndex(
      (msg) => msg.getType() === 'ai'
    );
    expect(firstAssistantIndex).toBe(0);

    const firstAssistantMsg = result.context[firstAssistantIndex];
    expect(Array.isArray(firstAssistantMsg.content)).toBe(true);

    // Verify that the first assistant message has a thinking block
    const hasThinkingBlock = (
      firstAssistantMsg.content as t.MessageContentComplex[]
    ).some(
      (item: t.MessageContentComplex) =>
        typeof item === 'object' && item.type === 'thinking'
    );
    expect(hasThinkingBlock).toBe(true);

    // Verify that the thinking block is from the original assistant message
    const thinkingBlock = (
      firstAssistantMsg.content as t.MessageContentComplex[]
    ).find(
      (item: t.MessageContentComplex) =>
        typeof item === 'object' && item.type === 'thinking'
    );
    expect(thinkingBlock).toBeDefined();
    expect((thinkingBlock as t.ThinkingContentText).thinking).toContain(
      'The user is asking me to read a file'
    );
  });

  it('should handle token recalculation when inserting thinking blocks', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create a message with thinking block
    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a thinking block',
        },
        {
          type: 'text',
          text: 'Response with thinking',
        },
      ],
    });

    // Create a message without thinking block
    const assistantMessageWithoutThinking = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'Response without thinking',
        },
      ],
    });

    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      assistantMessageWithThinking,
      new HumanMessage('Next message'),
      assistantMessageWithoutThinking,
    ];

    // Calculate token counts for each message
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Create pruneMessages function with thinking mode enabled
    const pruneMessages = createPruneMessages({
      maxTokens: 50, // Set a low token limit to force pruning
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({ messages });

    // Verify that the pruned context has fewer messages than the original
    expect(result.context.length).toBeLessThan(messages.length);
  });

  it('should not modify messages when under token limit', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create a message with thinking block
    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a thinking block',
        },
        {
          type: 'text',
          text: 'Response with thinking',
        },
      ],
    });

    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      assistantMessageWithThinking,
    ];

    // Calculate token counts for each message
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Create pruneMessages function with thinking mode enabled
    const pruneMessages = createPruneMessages({
      maxTokens: 1000, // Set a high token limit to avoid pruning
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({ messages });

    // Verify that all messages are preserved
    expect(result.context.length).toBe(messages.length);
    expect(result.context).toEqual(messages);
  });

  it('should handle the case when no thinking blocks are found', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages without thinking blocks
    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      new AIMessage('Response without thinking'),
      new HumanMessage('Next message'),
      new AIMessage('Another response without thinking'),
    ];

    // Calculate token counts for each message
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Create pruneMessages function with thinking mode enabled
    const pruneMessages = createPruneMessages({
      maxTokens: 50, // Set a low token limit to force pruning
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({ messages });

    // Verify that the pruned context has fewer messages than the original
    expect(result.context.length).toBeLessThan(messages.length);

    // The function should not throw an error even though no thinking blocks are found
    expect(() => pruneMessages({ messages })).not.toThrow();
  });

  it('should preserve AI <--> tool message correspondences when pruning', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages with tool calls
    const assistantMessageWithToolCall = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'Let me check that file:',
        },
        {
          type: 'tool_use',
          id: 'tool123',
          name: 'text_editor_mcp_textEditor',
          input: '{"command": "view", "path": "/path/to/file.txt"}',
        },
      ],
    });

    const toolResponseMessage = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content"}',
        },
      ],
      tool_call_id: 'tool123',
      name: 'text_editor_mcp_textEditor',
    });

    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a thinking block',
        },
        {
          type: 'text',
          text: 'Response with thinking',
        },
      ],
    });

    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      assistantMessageWithToolCall,
      toolResponseMessage,
      new HumanMessage('Next message'),
      assistantMessageWithThinking,
    ];

    // Calculate token counts for each message
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Create pruneMessages function with thinking mode enabled and a low token limit
    const pruneMessages = createPruneMessages({
      maxTokens: 100, // Set a low token limit to force pruning
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({ messages });

    // Find assistant message with tool call and its corresponding tool message in the pruned context
    const assistantIndex = result.context.findIndex(
      (msg) =>
        msg.getType() === 'ai' &&
        Array.isArray(msg.content) &&
        msg.content.some(
          (item) =>
            typeof item === 'object' &&
            item.type === 'tool_use' &&
            item.id === 'tool123'
        )
    );

    // If the assistant message with tool call is in the context, its corresponding tool message should also be there
    if (assistantIndex !== -1) {
      const toolIndex = result.context.findIndex(
        (msg) =>
          msg.getType() === 'tool' &&
          'tool_call_id' in msg &&
          msg.tool_call_id === 'tool123'
      );

      expect(toolIndex).not.toBe(-1);
    }

    // If the tool message is in the context, its corresponding assistant message should also be there
    const toolIndex = result.context.findIndex(
      (msg) =>
        msg.getType() === 'tool' &&
        'tool_call_id' in msg &&
        msg.tool_call_id === 'tool123'
    );

    if (toolIndex !== -1) {
      const assistantWithToolIndex = result.context.findIndex(
        (msg) =>
          msg.getType() === 'ai' &&
          Array.isArray(msg.content) &&
          msg.content.some(
            (item) =>
              typeof item === 'object' &&
              item.type === 'tool_use' &&
              item.id === 'tool123'
          )
      );

      expect(assistantWithToolIndex).not.toBe(-1);
    }
  });

  it('should ensure an assistant message with thinking appears in the latest sequence of assistant/tool messages', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages with the latest message being an assistant message with thinking
    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a thinking block',
        },
        {
          type: 'text',
          text: 'Response with thinking',
        },
      ],
    });

    // Create an assistant message with tool use
    const assistantMessageWithToolUse = new AIMessage({
      content: [
        {
          type: 'text',
          text: 'Let me check that file:',
        },
        {
          type: 'tool_use',
          id: 'tool123',
          name: 'text_editor_mcp_textEditor',
          input: '{"command": "view", "path": "/path/to/file.txt"}',
        },
      ],
    });

    // Create a tool response message
    const toolResponseMessage = new ToolMessage({
      content: [
        {
          type: 'text',
          text: '{"success":true,"message":"File content"}',
        },
      ],
      tool_call_id: 'tool123',
      name: 'text_editor_mcp_textEditor',
    });

    // Test case without system message
    const messagesWithoutSystem = [
      new HumanMessage('Hello'),
      assistantMessageWithToolUse,
      toolResponseMessage,
      new HumanMessage('Next message'),
      assistantMessageWithThinking, // Latest message is an assistant message with thinking
    ];

    // Calculate token counts for each message
    const indexTokenCountMapWithoutSystem: Record<string, number> = {};
    for (let i = 0; i < messagesWithoutSystem.length; i++) {
      indexTokenCountMapWithoutSystem[i] = tokenCounter(
        messagesWithoutSystem[i]
      );
    }

    // Create pruneMessages function with thinking mode enabled
    const pruneMessagesWithoutSystem = createPruneMessages({
      maxTokens: 100, // Set a token limit to force some pruning
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMapWithoutSystem },
      thinkingEnabled: true,
    });

    // Prune messages
    const resultWithoutSystem = pruneMessagesWithoutSystem({
      messages: messagesWithoutSystem,
    });

    // Verify that the pruned context contains at least one message
    expect(resultWithoutSystem.context.length).toBeGreaterThan(0);

    // Find all assistant messages in the latest sequence (after the last human message)
    const lastHumanIndex = resultWithoutSystem.context
      .map((msg) => msg.getType())
      .lastIndexOf('human');
    const assistantMessagesAfterLastHuman = resultWithoutSystem.context
      .slice(lastHumanIndex + 1)
      .filter((msg) => msg.getType() === 'ai');

    // Verify that at least one assistant message exists in the latest sequence
    expect(assistantMessagesAfterLastHuman.length).toBeGreaterThan(0);

    // Verify that at least one of these assistant messages has a thinking block
    const hasThinkingBlock = assistantMessagesAfterLastHuman.some((msg) => {
      const content = msg.content as t.MessageContentComplex[];
      return (
        Array.isArray(content) &&
        content.some(
          (item) => typeof item === 'object' && item.type === 'thinking'
        )
      );
    });
    expect(hasThinkingBlock).toBe(true);

    // Test case with system message
    const messagesWithSystem = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      assistantMessageWithToolUse,
      toolResponseMessage,
      new HumanMessage('Next message'),
      assistantMessageWithThinking, // Latest message is an assistant message with thinking
    ];

    // Calculate token counts for each message
    const indexTokenCountMapWithSystem: Record<string, number> = {};
    for (let i = 0; i < messagesWithSystem.length; i++) {
      indexTokenCountMapWithSystem[i] = tokenCounter(messagesWithSystem[i]);
    }

    // Create pruneMessages function with thinking mode enabled
    const pruneMessagesWithSystem = createPruneMessages({
      maxTokens: 120, // Set a token limit to force some pruning but keep system message
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMapWithSystem },
      thinkingEnabled: true,
    });

    // Prune messages
    const resultWithSystem = pruneMessagesWithSystem({
      messages: messagesWithSystem,
    });

    // Verify that the system message remains first
    expect(resultWithSystem.context.length).toBeGreaterThan(1);
    expect(resultWithSystem.context[0].getType()).toBe('system');

    // Find all assistant messages in the latest sequence (after the last human message)
    const lastHumanIndexWithSystem = resultWithSystem.context
      .map((msg) => msg.getType())
      .lastIndexOf('human');
    const assistantMessagesAfterLastHumanWithSystem = resultWithSystem.context
      .slice(lastHumanIndexWithSystem + 1)
      .filter((msg) => msg.getType() === 'ai');

    // Verify that at least one assistant message exists in the latest sequence
    expect(assistantMessagesAfterLastHumanWithSystem.length).toBeGreaterThan(0);

    // Verify that at least one of these assistant messages has a thinking block
    const hasThinkingBlockWithSystem =
      assistantMessagesAfterLastHumanWithSystem.some((msg) => {
        const content = msg.content as t.MessageContentComplex[];
        return (
          Array.isArray(content) &&
          content.some(
            (item) => typeof item === 'object' && item.type === 'thinking'
          )
        );
      });
    expect(hasThinkingBlockWithSystem).toBe(true);
  });

  it('should look for thinking blocks starting from the most recent messages', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages with multiple thinking blocks
    const olderAssistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is an older thinking block',
        },
        {
          type: 'text',
          text: 'Older response with thinking',
        },
      ],
    });

    const newerAssistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a newer thinking block',
        },
        {
          type: 'text',
          text: 'Newer response with thinking',
        },
      ],
    });

    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      olderAssistantMessageWithThinking,
      new HumanMessage('Next message'),
      newerAssistantMessageWithThinking,
    ];

    // Calculate token counts for each message
    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    // Create pruneMessages function with thinking mode enabled
    // Set a token limit that will force pruning of the older assistant message
    const pruneMessages = createPruneMessages({
      maxTokens: 80,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    // Prune messages
    const result = pruneMessages({ messages });

    // Find the first assistant message in the pruned context
    const firstAssistantIndex = result.context.findIndex(
      (msg) => msg.getType() === 'ai'
    );
    expect(firstAssistantIndex).not.toBe(-1);

    const firstAssistantMsg = result.context[firstAssistantIndex];
    expect(Array.isArray(firstAssistantMsg.content)).toBe(true);

    // Verify that the first assistant message has a thinking block
    const thinkingBlock = (
      firstAssistantMsg.content as t.MessageContentComplex[]
    ).find((item) => typeof item === 'object' && item.type === 'thinking');
    expect(thinkingBlock).toBeDefined();

    // Verify that it's the newer thinking block
    expect((thinkingBlock as t.ThinkingContentText).thinking).toContain(
      'newer thinking block'
    );
  });

  it('should throw descriptive error when aggressive pruning removes all AI messages', () => {
    const tokenCounter = createTestTokenCounter();

    const assistantMessageWithThinking = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'This is a thinking block that will be pruned',
        },
        {
          type: 'text',
          text: 'Response with thinking',
        },
        {
          type: 'tool_use',
          id: 'tool123',
          name: 'large_tool',
          input: '{"query": "test"}',
        },
      ],
    });

    const largeToolResponse = new ToolMessage({
      content: 'A'.repeat(10000),
      tool_call_id: 'tool123',
      name: 'large_tool',
    });

    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Hello'),
      assistantMessageWithThinking,
      largeToolResponse,
    ];

    const indexTokenCountMap: Record<string, number> = {
      '0': 17,
      '1': 5,
      '2': 100,
      '3': 10000,
    };

    const pruneMessages = createPruneMessages({
      maxTokens: 50,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      thinkingEnabled: true,
    });

    expect(() => pruneMessages({ messages })).toThrow(
      /Context window exceeded/
    );
  });
});
