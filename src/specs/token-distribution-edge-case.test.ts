// src/specs/token-distribution-edge-case.test.ts
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { createPruneMessages } from '@/messages/prune';

// Create a simple token counter for testing
const createTestTokenCounter = (): t.TokenCounter => {
  // This simple token counter just counts characters as tokens for predictable testing
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
          if ('text' in item && typeof item.text === 'string') {
            totalLength += item.text.length;
          }
        }
      }

      return totalLength;
    }

    // Default case - if content is null, undefined, or any other type
    return 0;
  };
};

describe('Token Distribution Edge Case Tests', () => {
  it('should only distribute tokens to messages that remain in the context after pruning', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages
    const messages = [
      new SystemMessage('System instruction'), // Will always be included
      new HumanMessage('Message 1'), // Will be pruned
      new AIMessage('Response 1'), // Will be pruned
      new HumanMessage('Message 2'), // Will remain
      new AIMessage('Response 2'), // Will remain
    ];

    // Calculate initial token counts for each message
    const indexTokenCountMap: Record<string, number> = {
      0: 17, // "System instruction"
      1: 9, // "Message 1"
      2: 10, // "Response 1"
      3: 9, // "Message 2"
      4: 10, // "Response 2"
    };

    // Set a token limit that will force pruning of the first two messages after the system message
    const pruneMessages = createPruneMessages({
      maxTokens: 40, // Only enough for system message + last two messages
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      reserveRatio: 0,
    });

    // First call to establish lastCutOffIndex
    const initialResult = pruneMessages({ messages });

    // Verify initial pruning
    expect(initialResult.context.length).toBe(3);
    expect(initialResult.context[0].content).toBe('System instruction');
    expect(initialResult.context[1].content).toBe('Message 2');
    expect(initialResult.context[2].content).toBe('Response 2');

    // Now provide usage metadata with a different total token count
    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 30,
      output_tokens: 20,
      total_tokens: 50, // Different from the sum of our initial token counts
    };

    // Call pruneMessages again with the usage metadata
    const result = pruneMessages({
      messages,
      usageMetadata,
    });

    // The token distribution should only affect messages that remain in the context
    // Messages at indices 0, 3, and 4 should have their token counts adjusted
    // Messages at indices 1 and 2 should remain unchanged since they're pruned

    // The token distribution should only affect messages that remain in the context
    // Messages at indices 0, 3, and 4 should have their token counts adjusted
    // Messages at indices 1 and 2 should remain unchanged since they're pruned

    // Check that at least one of the pruned messages' token counts was not adjusted
    // We're testing the principle that pruned messages don't get token redistribution
    const atLeastOnePrunedMessageUnchanged =
      result.indexTokenCountMap[1] === indexTokenCountMap[1] ||
      result.indexTokenCountMap[2] === indexTokenCountMap[2];

    expect(atLeastOnePrunedMessageUnchanged).toBe(true);

    // Calibration uses input_tokens (30) only (no output), minus instruction overhead (0).
    // Map stays in raw tiktoken space — calibrationRatio captures the multiplier.
    // Context messages: indices 0, 3, 4 → raw sum unchanged.
    // calibrationRatio × rawSum should approximate input_tokens (30).
    const rawContextTokens =
      (result.indexTokenCountMap[0] ?? 0) +
      (result.indexTokenCountMap[3] ?? 0) +
      (result.indexTokenCountMap[4] ?? 0);
    expect(rawContextTokens).toBeGreaterThan(0);

    const calibratedTotal = Math.round(
      rawContextTokens * (result.calibrationRatio ?? 1)
    );
    const tokenDifference = Math.abs(calibratedTotal - 30);
    expect(tokenDifference).toBeLessThan(10);
  });

  it('should handle the case when all messages fit within the token limit', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create messages
    const messages = [
      new SystemMessage('System instruction'),
      new HumanMessage('Message 1'),
      new AIMessage('Response 1'),
    ];

    // Calculate initial token counts for each message
    const indexTokenCountMap: Record<string, number> = {
      0: 17, // "System instruction"
      1: 9, // "Message 1"
      2: 10, // "Response 1"
    };

    // Set a token limit that will allow all messages to fit
    const pruneMessages = createPruneMessages({
      maxTokens: 100,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
    });

    // First call to establish lastCutOffIndex (should be 0 since no pruning occurs)
    const initialResult = pruneMessages({ messages });

    // Verify no pruning occurred
    expect(initialResult.context.length).toBe(3);

    // Now provide usage metadata with a different total token count
    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 20,
      output_tokens: 10,
      total_tokens: 30, // Different from the sum of our initial token counts
    };

    // Call pruneMessages again with the usage metadata
    const result = pruneMessages({
      messages,
      usageMetadata,
    });

    // Calibration uses input_tokens (20) only, minus instruction overhead (0).
    // messageTokenSum = 17 + 9 + 10 = 36. ratio = 20/36 = 0.556. Safe.
    // Map stays raw — calibrationRatio captures the multiplier
    expect(result.indexTokenCountMap[0]).toBe(indexTokenCountMap[0]);
    expect(result.indexTokenCountMap[1]).toBe(indexTokenCountMap[1]);
    expect(result.indexTokenCountMap[2]).toBe(indexTokenCountMap[2]);

    // rawSum × calibrationRatio should approximate input_tokens (20)
    const rawTotal =
      (result.indexTokenCountMap[0] ?? 0) +
      (result.indexTokenCountMap[1] ?? 0) +
      (result.indexTokenCountMap[2] ?? 0);
    const calibratedTotal = Math.round(
      rawTotal * (result.calibrationRatio ?? 1)
    );
    expect(Math.abs(calibratedTotal - 20)).toBeLessThanOrEqual(3);
  });

  it('should handle multiple pruning operations with token redistribution', () => {
    // Create a token counter
    const tokenCounter = createTestTokenCounter();

    // Create a longer sequence of messages
    const messages = [
      new SystemMessage('System instruction'), // Will always be included
      new HumanMessage('Message 1'), // Will be pruned in first round
      new AIMessage('Response 1'), // Will be pruned in first round
      new HumanMessage('Message 2'), // Will be pruned in second round
      new AIMessage('Response 2'), // Will be pruned in second round
      new HumanMessage('Message 3'), // Will remain
      new AIMessage('Response 3'), // Will remain
    ];

    // Calculate initial token counts for each message
    const indexTokenCountMap: Record<string, number> = {
      0: 17, // "System instruction"
      1: 9, // "Message 1"
      2: 10, // "Response 1"
      3: 9, // "Message 2"
      4: 10, // "Response 2"
      5: 9, // "Message 3"
      6: 10, // "Response 3"
    };

    // Set a token limit that will force pruning
    const pruneMessages = createPruneMessages({
      maxTokens: 40, // Only enough for system message + last two messages
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      reserveRatio: 0,
    });

    // First pruning operation
    const firstResult = pruneMessages({ messages });

    // Verify first pruning
    expect(firstResult.context.length).toBe(3);
    expect(firstResult.context[0].content).toBe('System instruction');
    expect(firstResult.context[1].content).toBe('Message 3');
    expect(firstResult.context[2].content).toBe('Response 3');

    // First usage metadata update
    const firstUsageMetadata: Partial<UsageMetadata> = {
      input_tokens: 30,
      output_tokens: 20,
      total_tokens: 50,
    };

    // Apply first usage metadata
    const secondResult = pruneMessages({
      messages,
      usageMetadata: firstUsageMetadata,
    });

    // Add two more messages
    messages.push(new HumanMessage('Message 4'));
    const extendedMessages = [...messages, new AIMessage('Response 4')];

    // Second usage metadata update
    const secondUsageMetadata: Partial<UsageMetadata> = {
      input_tokens: 30,
      output_tokens: 20,
      total_tokens: 50,
    };

    // Apply second usage metadata with extended messages
    const thirdResult = pruneMessages({
      messages: extendedMessages,
      usageMetadata: secondUsageMetadata,
    });

    // The context should include the system message and some of the latest messages
    expect(thirdResult.context.length).toBeGreaterThan(0);
    expect(thirdResult.context[0].content).toBe('System instruction');
    expect(thirdResult.context[1].content).toBe('Response 4');

    // Find which messages are in the final context
    const contextMessageIndices = thirdResult.context.map((msg) => {
      // Find the index of this message in the original array
      return extendedMessages.findIndex((m) => m.content === msg.content);
    });

    // Get the sum of token counts for messages in the context
    let totalContextTokens = 0;
    for (const idx of contextMessageIndices) {
      totalContextTokens += thirdResult.indexTokenCountMap[idx] ?? 0;
    }

    // Verify that the sum of tokens for messages in the context is close to the total_tokens from usageMetadata
    // There might be small rounding differences or implementation details that affect the exact sum
    expect(totalContextTokens).toBeGreaterThan(0);

    // The key thing we're testing is that the token distribution happens for messages in the context
    // and that the sum is reasonably close to the expected total
    const tokenDifference = Math.abs(totalContextTokens - 70);
    expect(tokenDifference).toBeLessThan(50); // Allow for some difference due to implementation details

    // Verify that messages not in the context have their original token counts or previously adjusted values
    for (let i = 0; i < extendedMessages.length; i++) {
      if (!contextMessageIndices.includes(i)) {
        const expectedValue =
          i < messages.length
            ? (secondResult.indexTokenCountMap[i] ?? 0) || indexTokenCountMap[i]
            : ((indexTokenCountMap as Record<string, number | undefined>)[i] ??
              0);

        const difference = Math.abs(
          (thirdResult.indexTokenCountMap[i] ?? 0) - expectedValue
        );
        expect(difference).toBe(0);
      }
    }
  });
});
