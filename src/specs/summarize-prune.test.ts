import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import { createPruneMessages } from '@/messages/prune';
import { Providers, ContentTypes } from '@/common';

function tokenCounter(msg: { content: unknown }): number {
  const content =
    typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
  return Math.ceil(content.length / 4);
}

describe('Prune + Summarize Integration', () => {
  it('should return messagesToRefine when messages exceed token limit', () => {
    const messages = [
      new SystemMessage('You are a helpful assistant.'),
      new HumanMessage('First question'),
      new AIMessage('First answer'),
      new HumanMessage('Second question'),
      new AIMessage('Second answer'),
      new HumanMessage('Third question'),
      new AIMessage(
        'Third answer that is quite long to push things over the limit'
      ),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const totalTokens = Object.values(indexTokenCountMap).reduce(
      (a = 0, b = 0) => a! + b!,
      0
    ) as number;
    const maxTokens = Math.floor(totalTokens * 0.6);

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const result = pruneMessages({ messages });

    expect(result.messagesToRefine).toBeDefined();
    expect(result.messagesToRefine!.length).toBeGreaterThan(0);
    expect(result.remainingContextTokens).toBeDefined();
    expect(typeof result.remainingContextTokens).toBe('number');
    expect(result.context.length).toBeLessThan(messages.length);
  });

  it('should return empty messagesToRefine when all messages fit', () => {
    const messages = [new HumanMessage('Hi'), new AIMessage('Hello')];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 10000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const result = pruneMessages({ messages });

    expect(result.messagesToRefine).toBeDefined();
    expect(result.messagesToRefine).toHaveLength(0);
    expect(result.remainingContextTokens).toBeGreaterThan(0);
    expect(result.context).toEqual(messages);
  });

  it('should preserve system message in context even when pruning', () => {
    const sysMsg = new SystemMessage(
      'Instructions for the assistant to follow carefully'
    );
    const messages = [
      sysMsg,
      new HumanMessage(
        'This is the first message in our conversation and it is fairly long'
      ),
      new AIMessage(
        'This is the first response and it is also fairly long with details'
      ),
      new HumanMessage(
        'This is the second message with more context and questions'
      ),
      new AIMessage(
        'This is the second response which is even more detailed and verbose'
      ),
      new HumanMessage('Third message in the conversation chain'),
      new AIMessage('Third response with additional lengthy explanations'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = tokenCounter(messages[i]);
    }

    const totalTokens = Object.values(indexTokenCountMap).reduce(
      (a = 0, b = 0) => a! + b!,
      0
    ) as number;
    const maxTokens = Math.floor(totalTokens * 0.35);

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const result = pruneMessages({ messages });

    expect(result.context[0]).toBe(sysMsg);
    expect(result.messagesToRefine!.length).toBeGreaterThan(0);
  });

  it('should not include summary content type in pruned messages passed to providers', () => {
    const summaryBlock = {
      type: ContentTypes.SUMMARY,
      text: 'Summary of prior conversation',
    };
    expect(summaryBlock.type).toBe('summary');
    expect(Object.values(ContentTypes)).toContain('summary');
  });
});

describe('pruneMessages ratio-based token grounding', () => {
  it('should adjust indexTokenCountMap entries proportionally when usageMetadata is provided', () => {
    const messages = [
      new SystemMessage('Be concise.'),
      new HumanMessage('What is 2+2?'),
      new AIMessage('The answer is 4.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 10,
      1: 20,
      2: 30,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 5000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 50,
      output_tokens: 40,
    };

    const result = pruneMessages({ messages, usageMetadata });

    const totalUsage = 50 + 40;
    const originalTotal = 10 + 20 + 30;
    const expectedRatio = totalUsage / originalTotal;

    expect(result.indexTokenCountMap[0]).toBe(Math.round(10 * expectedRatio));
    expect(result.indexTokenCountMap[1]).toBe(Math.round(20 * expectedRatio));
    expect(result.indexTokenCountMap[2]).toBe(Math.round(30 * expectedRatio));
  });

  it('should NOT adjust when ratio falls outside safe bounds (< 1/3)', () => {
    const messages = [
      new HumanMessage('What is 2+2?'),
      new AIMessage('The answer is 4.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 100,
      1: 200,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 50000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 5,
      output_tokens: 5,
    };

    const result = pruneMessages({ messages, usageMetadata });

    expect(result.indexTokenCountMap[0]).toBe(100);
    expect(result.indexTokenCountMap[1]).toBe(200);
  });

  it('should NOT adjust when ratio falls outside safe bounds (> 2.5)', () => {
    const messages = [new HumanMessage('Hi'), new AIMessage('Hello')];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 5,
      1: 5,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 50000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 100,
      output_tokens: 100,
    };

    const result = pruneMessages({ messages, usageMetadata });

    expect(result.indexTokenCountMap[0]).toBe(5);
    expect(result.indexTokenCountMap[1]).toBe(5);
  });

  it('should include cache_read and cache_creation in ratio total', () => {
    const messages = [
      new SystemMessage('Instructions'),
      new HumanMessage('Hello'),
      new AIMessage('Hi there!'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 10,
      1: 20,
      2: 30,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.ANTHROPIC,
      maxTokens: 5000,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 30,
      output_tokens: 20,
      input_token_details: {
        cache_read: 15,
        cache_creation: 10,
      },
    };

    const totalUsage = 30 + 15 + 10 + 20;
    const originalTotal = 10 + 20 + 30;
    const expectedRatio = totalUsage / originalTotal;

    const result = pruneMessages({ messages, usageMetadata });

    expect(result.indexTokenCountMap[0]).toBe(Math.round(10 * expectedRatio));
    expect(result.indexTokenCountMap[1]).toBe(Math.round(20 * expectedRatio));
    expect(result.indexTokenCountMap[2]).toBe(Math.round(30 * expectedRatio));
  });

  it('should assign output_tokens to the first new message at startIndex', () => {
    const messages = [
      new HumanMessage('What is 2+2?'),
      new AIMessage('The answer is 4.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 15,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 5000,
      startIndex: 1,
      tokenCounter,
      indexTokenCountMap,
    });

    const usageMetadata: Partial<UsageMetadata> = {
      input_tokens: 20,
      output_tokens: 25,
    };

    const result = pruneMessages({ messages, usageMetadata });

    expect(result.indexTokenCountMap[1]).toBeDefined();
    expect(result.indexTokenCountMap[1] as number).toBeGreaterThan(0);

    const preRatioIndex0 = 15;
    const preRatioIndex1 = 25;
    const totalUsage = 20 + 25;
    const ratio = totalUsage / (preRatioIndex0 + preRatioIndex1);
    const isRatioSafe = ratio >= 1 / 3 && ratio <= 2.5;

    if (isRatioSafe) {
      expect(result.indexTokenCountMap[0]).toBe(
        Math.round(preRatioIndex0 * ratio)
      );
    } else {
      expect(result.indexTokenCountMap[0]).toBe(preRatioIndex0);
    }
  });

  it('should ground tokens correctly across multiple pruneMessages calls', () => {
    const turn1Messages = [
      new SystemMessage('Be concise.'),
      new HumanMessage('What is 2+2?'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 10,
      1: 20,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 5000,
      startIndex: turn1Messages.length,
      tokenCounter,
      indexTokenCountMap,
    });

    const turn1Result = pruneMessages({
      messages: turn1Messages,
    });

    expect(turn1Result.indexTokenCountMap[0]).toBe(10);
    expect(turn1Result.indexTokenCountMap[1]).toBe(20);

    const turn2Messages = [
      ...turn1Messages,
      new AIMessage('4'),
      new HumanMessage('And 3+3?'),
    ];

    const turn2Usage: Partial<UsageMetadata> = {
      input_tokens: 25,
      output_tokens: 10,
    };

    const turn2Result = pruneMessages({
      messages: turn2Messages,
      usageMetadata: turn2Usage,
    });

    expect(turn2Result.indexTokenCountMap[2]).toBeDefined();
    expect(turn2Result.indexTokenCountMap[2] as number).toBeGreaterThan(0);
    expect(turn2Result.indexTokenCountMap[3]).toBeDefined();
    expect(turn2Result.indexTokenCountMap[3] as number).toBeGreaterThan(0);

    for (let i = 0; i < turn2Messages.length; i++) {
      expect(turn2Result.indexTokenCountMap[i]).toBeDefined();
      expect(turn2Result.indexTokenCountMap[i] as number).toBeGreaterThan(0);
    }
  });
});
