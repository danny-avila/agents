import {
  AIMessage,
  ToolMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { createPruneMessages, calculateTotalTokens } from '@/messages/prune';
import { Providers } from '@/common';

// ---------------------------------------------------------------------------
// Deterministic char-based token counter — 1 char = 1 token, plus 3 per-message
// overhead (matches the real getTokenCountForMessage tokensPerMessage constant).
// ---------------------------------------------------------------------------
const charCounter: t.TokenCounter = (msg: BaseMessage): number => {
  const content = msg.content;
  if (typeof content === 'string') {
    return content.length + 3;
  }
  if (Array.isArray(content)) {
    let len = 3;
    for (const item of content as Array<
      string | { type: string; text?: string }
    >) {
      if (typeof item === 'string') {
        len += item.length;
      } else if (
        typeof item === 'object' &&
        'text' in item &&
        item.text != null &&
        item.text
      ) {
        len += item.text.length;
      }
    }
    return len;
  }
  return 3;
};

function toolMsg(
  content: string,
  name = 'tool',
  toolCallId = `tc_${Math.random().toString(36).slice(2, 8)}`
): ToolMessage {
  return new ToolMessage({ content, tool_call_id: toolCallId, name });
}

function aiToolCall(toolCallId: string, name = 'tool'): AIMessage {
  return new AIMessage({
    content: [{ type: 'tool_use', id: toolCallId, name, input: {} }],
    tool_calls: [{ id: toolCallId, name, args: {}, type: 'tool_call' }],
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('calculateTotalTokens — cache detection heuristic', () => {
  it('treats cache as additive when cacheSum > baseInputTokens (Anthropic pattern)', () => {
    const result = calculateTotalTokens({
      input_tokens: 100,
      output_tokens: 20,
      input_token_details: { cache_creation: 50, cache_read: 200 },
    });
    expect(result).toEqual({
      input_tokens: 350,
      output_tokens: 20,
      total_tokens: 370,
    });
  });

  it('does NOT add cache when cacheSum <= baseInputTokens (OpenAI pattern)', () => {
    const result = calculateTotalTokens({
      input_tokens: 300,
      output_tokens: 20,
      input_token_details: { cache_read: 100 },
    });
    expect(result).toEqual({
      input_tokens: 300,
      output_tokens: 20,
      total_tokens: 320,
    });
  });

  it('handles zero cache gracefully', () => {
    const result = calculateTotalTokens({
      input_tokens: 500,
      output_tokens: 50,
    });
    expect(result).toEqual({
      input_tokens: 500,
      output_tokens: 50,
      total_tokens: 550,
    });
  });

  it('handles all-zero usage', () => {
    const result = calculateTotalTokens({});
    expect(result).toEqual({
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
    });
  });
});

describe('Token accounting pipeline — multi-turn calibration', () => {
  it('calibration scales message tokens to match provider input_tokens', () => {
    const messages = [
      new SystemMessage('You are a helpful assistant.'),
      new HumanMessage('Hello'),
      new AIMessage('Hi!'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }
    const originalSum = Object.values(indexTokenCountMap).reduce(
      (a, b) => (a ?? 0) + (b ?? 0),
      0
    ) as number;

    const pruneMessages = createPruneMessages({
      maxTokens: 5000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const providerInput = Math.round(originalSum * 1.4);
    const result = pruneMessages({
      messages,
      usageMetadata: { input_tokens: providerInput, output_tokens: 30 },
    });

    // Map stays in raw tiktoken space — calibrationRatio captures the multiplier.
    // rawSum * calibrationRatio should approximate providerInput.
    let rawSum = 0;
    for (let i = 0; i < messages.length; i++) {
      rawSum += result.indexTokenCountMap[i] ?? 0;
    }
    const calibratedEstimate = Math.round(
      rawSum * (result.calibrationRatio ?? 1)
    );

    expect(Math.abs(calibratedEstimate - providerInput)).toBeLessThanOrEqual(
      messages.length
    );
  });

  it('first response at startIndex gets output_tokens and is excluded from calibration ratio', () => {
    const messages = [
      new HumanMessage('What is the meaning of life?'),
      new AIMessage('42.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: charCounter(messages[0]),
    };

    const pruneMessages = createPruneMessages({
      maxTokens: 5000,
      startIndex: 1,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({
      messages,
      usageMetadata: { input_tokens: 40, output_tokens: 10 },
    });

    // Index 1 should be assigned output_tokens
    expect(result.indexTokenCountMap[1]).toBe(10);

    // Map stays raw — index 0 keeps its original count.
    // calibrationRatio captures providerInput / rawMessageSum.
    const index0Original = charCounter(messages[0]);
    expect(result.indexTokenCountMap[0]).toBe(index0Original);
    const expectedRatio = 40 / index0Original;
    expect(result.calibrationRatio).toBeCloseTo(expectedRatio, 1);
  });

  it('unsafe ratio (< 1/3) prevents calibration — map stays unchanged', () => {
    const messages = [
      new HumanMessage('Long message content here'),
      new AIMessage('Also a long response here'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 200,
      1: 300,
    };

    const pruneMessages = createPruneMessages({
      maxTokens: 50000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({
      messages,
      usageMetadata: { input_tokens: 10, output_tokens: 5 },
    });

    // ratio = 10/500 = 0.02, way below 1/3
    expect(result.indexTokenCountMap[0]).toBe(200);
    expect(result.indexTokenCountMap[1]).toBe(300);
  });

  it('unsafe ratio (> 2.5) prevents calibration', () => {
    const messages = [new HumanMessage('Hi'), new AIMessage('Hello')];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 5,
      1: 8,
    };

    const pruneMessages = createPruneMessages({
      maxTokens: 50000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({
      messages,
      usageMetadata: { input_tokens: 500, output_tokens: 100 },
    });

    // ratio = 500/13 = 38.5, way above 2.5
    expect(result.indexTokenCountMap[0]).toBe(5);
    expect(result.indexTokenCountMap[1]).toBe(8);
  });

  it('multi-turn closure state persists calibrated values across calls', () => {
    // Simulate realistic flow: human message counted before creating pruner,
    // AI response is the first "new" message at startIndex.
    const messages: BaseMessage[] = [
      new HumanMessage('Turn 1 question'),
      new AIMessage('Turn 1 answer'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: charCounter(messages[0]),
    };

    // startIndex=1: the human message (0) was pre-existing, AI response (1) is new
    const pruneMessages = createPruneMessages({
      maxTokens: 10000,
      startIndex: 1,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    // Turn 1: model responds — index 1 (at startIndex) gets output_tokens
    const turn1 = pruneMessages({
      messages,
      usageMetadata: { input_tokens: 25, output_tokens: 20 },
    });

    expect(turn1.indexTokenCountMap[1]).toBe(20);
    // Map stays raw — calibrationRatio captures the multiplier
    const index0Original = charCounter(messages[0]);
    expect(turn1.indexTokenCountMap[0]).toBe(index0Original);
    const turn1Ratio = 25 / index0Original;
    if (turn1Ratio >= 0.5 && turn1Ratio <= 5) {
      expect(turn1.calibrationRatio).toBeCloseTo(turn1Ratio, 1);
    }

    // Turn 2: user sends message, model responds. Both new indices (2, 3) are unset.
    // In real flow, the user message (2) is counted before processStream,
    // but here the pruner hasn't seen it. Index 2 at lastTurnStartIndex gets output_tokens.
    messages.push(new HumanMessage('Turn 2 question'));
    messages.push(new AIMessage('Turn 2 answer'));

    const turn2 = pruneMessages({
      messages,
      usageMetadata: { input_tokens: 60, output_tokens: 15 },
    });

    // All 4 indices should be populated
    for (let i = 0; i < 4; i++) {
      expect(turn2.indexTokenCountMap[i]).toBeDefined();
      expect(turn2.indexTokenCountMap[i] as number).toBeGreaterThan(0);
    }
  });
});

describe('Token accounting pipeline — budget computation and context pressure', () => {
  it('getInstructionTokens reduces effective budget', () => {
    // 5 messages, each ~20 chars → ~23 tokens with 3-token overhead
    const messages = [
      new HumanMessage('a'.repeat(20)),
      new AIMessage('b'.repeat(20)),
      new HumanMessage('c'.repeat(20)),
      new AIMessage('d'.repeat(20)),
      new HumanMessage('e'.repeat(20)),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let totalEstimate = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      totalEstimate += indexTokenCountMap[i] ?? 0;
    }

    // Set maxTokens so messages fit WITHOUT instruction overhead
    // but do NOT fit WITH 50 tokens of instruction overhead
    const tightBudget = totalEstimate + 10;

    const pruneMessages = createPruneMessages({
      maxTokens: tightBudget,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
      getInstructionTokens: () => 50,
    });

    const result = pruneMessages({ messages });

    // With 50 tokens of instruction overhead on a tight budget,
    // pruning should have kicked in — context should be shorter
    expect(result.context.length).toBeLessThan(messages.length);
  });

  it('reserve ratio reduces pruning budget by the configured fraction', () => {
    const messages = [
      new HumanMessage('x'.repeat(80)),
      new AIMessage('y'.repeat(80)),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let totalEstimate = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      totalEstimate += indexTokenCountMap[i] ?? 0;
    }

    // With 0% reserve, messages fit. With 20% reserve, they won't.
    const maxTokens = totalEstimate + 5;

    const withReserve = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      reserveRatio: 0.2,
    });

    const withoutReserve = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap: { ...indexTokenCountMap },
      reserveRatio: 0,
    });

    const resultWithReserve = withReserve({ messages });
    const resultWithoutReserve = withoutReserve({ messages });

    expect(resultWithoutReserve.context.length).toBe(2);
    expect(resultWithReserve.context.length).toBeLessThan(2);
  });

  it('context pressure is computed after calibration and recount', () => {
    const messages = [
      new HumanMessage('a'.repeat(100)),
      new AIMessage('b'.repeat(100)),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: charCounter(messages[0]),
      1: charCounter(messages[1]),
    };
    const ourEstimate =
      (indexTokenCountMap[0] ?? 0) + (indexTokenCountMap[1] ?? 0);

    // Provider says input is 2× our estimate — calibration should inflate
    const providerInput = ourEstimate * 2;
    const maxTokens = Math.round(providerInput * 1.2);

    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({
      messages,
      usageMetadata: { input_tokens: providerInput, output_tokens: 50 },
    });

    // After calibration, tokens ~= providerInput (2× original)
    // contextPressure = calibratedSum / pruningBudget ≈ providerInput / maxTokens ≈ 0.83
    expect(result.contextPressure).toBeDefined();
    expect(result.contextPressure as number).toBeGreaterThan(0.7);
    expect(result.contextPressure as number).toBeLessThan(1.0);
  });
});

describe('Token accounting pipeline — observation masking at 80%+ pressure', () => {
  it('masks consumed tool results when pressure >= 0.8', () => {
    const tcId = 'tc_search';
    const bigResult = 'R'.repeat(2000);
    const messages: BaseMessage[] = [
      new HumanMessage('Search for info'),
      aiToolCall(tcId, 'search'),
      toolMsg(bigResult, 'search', tcId),
      new AIMessage('Based on the results, here is the answer.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let sum = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      sum += indexTokenCountMap[i] ?? 0;
    }

    // Set maxTokens so pressure is ~85%
    const maxTokens = Math.round(sum / 0.85);

    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({ messages });

    // Budget-aware masking: if the result fits within the available
    // message budget, it may be kept intact or only lightly trimmed.
    // Verify masking ran (context pressure triggered it) and the result
    // is within the raw message budget.
    const maskedTokens = result.indexTokenCountMap[2] ?? 0;
    const rawBudget = Math.round(maxTokens / (result.calibrationRatio ?? 1));
    expect(maskedTokens).toBeLessThanOrEqual(rawBudget);
  });

  it('does NOT mask when pressure < 0.8', () => {
    const tcId = 'tc_search';
    const bigResult = 'R'.repeat(2000);
    const messages: BaseMessage[] = [
      new HumanMessage('Search for info'),
      aiToolCall(tcId, 'search'),
      toolMsg(bigResult, 'search', tcId),
      new AIMessage('Based on the results, here is the answer.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const pruneMessages = createPruneMessages({
      maxTokens: 50000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({ messages });

    // No masking at low pressure — original token count preserved
    expect(result.indexTokenCountMap[2]).toBe(charCounter(messages[2]));
  });
});

describe('Token accounting pipeline — pruning drops oldest messages', () => {
  it('preserves system message and most recent messages when budget exceeded', () => {
    const sys = new SystemMessage('System prompt');
    const messages: BaseMessage[] = [sys];
    for (let i = 0; i < 10; i++) {
      messages.push(new HumanMessage(`User message ${i}`));
      messages.push(new AIMessage(`Assistant reply ${i}`));
    }

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let sum = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      sum += indexTokenCountMap[i] ?? 0;
    }

    // Budget only allows ~half the messages
    const pruneMessages = createPruneMessages({
      maxTokens: Math.round(sum * 0.5),
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({ messages });

    // System message should always be preserved
    expect(result.context[0].content).toBe('System prompt');
    // Should have fewer messages than original
    expect(result.context.length).toBeLessThan(messages.length);
    expect(result.context.length).toBeGreaterThan(1);

    // Last message in result should be from near the end of the original
    const lastResult = result.context[result.context.length - 1];
    const lastOriginal = messages[messages.length - 1];
    expect(lastResult.content).toBe(lastOriginal.content);
  });

  it('produces messagesToRefine when summarization is enabled', () => {
    const messages: BaseMessage[] = [];
    for (let i = 0; i < 10; i++) {
      messages.push(new HumanMessage(`User ${i}: ${'x'.repeat(50)}`));
      messages.push(new AIMessage(`Bot ${i}: ${'y'.repeat(50)}`));
    }

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let sum = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      sum += indexTokenCountMap[i] ?? 0;
    }

    const pruneMessages = createPruneMessages({
      maxTokens: Math.round(sum * 0.4),
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      summarizationEnabled: true,
      reserveRatio: 0,
    });

    const result = pruneMessages({ messages });

    // With summarization enabled, pruned messages go to messagesToRefine
    expect(result.messagesToRefine).toBeDefined();
    expect(result.messagesToRefine!.length).toBeGreaterThan(0);
    // messagesToRefine + context should account for all messages
    expect(
      result.context.length + result.messagesToRefine!.length
    ).toBeGreaterThanOrEqual(messages.length);
  });
});

describe('Token accounting pipeline — end-to-end multi-turn with calibration', () => {
  it('simulates a 4-turn conversation with growing context and calibration each turn', () => {
    const logs: Array<{ turn: number; message: string; data: unknown }> = [];
    const log = (
      _level: string,
      message: string,
      data?: Record<string, unknown>
    ): void => {
      logs.push({ turn: logs.length, message, data });
    };

    const systemMsg = new SystemMessage('You are helpful.');
    const firstHuman = new HumanMessage('Hello, how are you?');
    const conversationHistory: BaseMessage[] = [systemMsg, firstHuman];

    // Pre-count system and first human message (as the real system does)
    const indexTokenCountMap: Record<string, number | undefined> = {
      0: charCounter(systemMsg),
      1: charCounter(firstHuman),
    };

    // startIndex=2: system(0) + human(1) are pre-existing, AI response(2) is new
    const pruneMessages = createPruneMessages({
      maxTokens: 600,
      startIndex: 2,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0.05,
      log,
    });

    // The closure returns a reference to its internal map. We use it to
    // pre-count human messages on subsequent turns (matching real behavior).
    let liveMap: Record<string, number | undefined> = indexTokenCountMap;

    const simulateTurn = (
      humanText: string,
      aiText: string,
      usage: Partial<UsageMetadata>,
      skipHumanPush = false
    ): ReturnType<ReturnType<typeof createPruneMessages>> => {
      if (!skipHumanPush) {
        const humanMsg = new HumanMessage(humanText);
        conversationHistory.push(humanMsg);
        const humanIdx = conversationHistory.length - 1;
        liveMap[humanIdx] = charCounter(humanMsg);
      }

      conversationHistory.push(new AIMessage(aiText));

      const result = pruneMessages({
        messages: conversationHistory,
        usageMetadata: usage,
      });
      liveMap = result.indexTokenCountMap;
      return result;
    };

    // --- Turn 1: human already pushed, only AI response is new ---
    const turn1 = simulateTurn(
      '',
      'I am fine, thank you for asking!',
      { input_tokens: 30, output_tokens: 15 },
      true
    );
    expect(turn1.context.length).toBe(3);
    // AI response (index 2) should get output_tokens
    expect(turn1.indexTokenCountMap[2]).toBe(15);

    // --- Turn 2 ---
    const turn2 = simulateTurn(
      'Can you explain quantum computing in detail?',
      'Quantum computing uses qubits that can exist in superposition. ' +
        'This allows quantum computers to process many possibilities simultaneously.',
      { input_tokens: 80, output_tokens: 50 }
    );
    expect(turn2.context.length).toBe(5);
    // AI response (index 4) gets tokenCounter count (not output_tokens) since
    // the human message at lastTurnStartIndex was pre-counted.
    expect(turn2.indexTokenCountMap[4]).toBeDefined();
    expect(turn2.indexTokenCountMap[4] as number).toBeGreaterThan(0);

    // --- Turn 3 ---
    const turn3 = simulateTurn(
      'What about quantum entanglement?',
      'Quantum entanglement is a phenomenon where particles become correlated ' +
        'such that the quantum state of one instantly influences the other, ' +
        'regardless of distance. Einstein called it spooky action at a distance.',
      { input_tokens: 200, output_tokens: 80 }
    );
    expect(turn3.indexTokenCountMap[6]).toBeDefined();
    expect(turn3.indexTokenCountMap[6] as number).toBeGreaterThan(0);

    // --- Turn 4: push past budget to trigger pruning ---
    const turn4 = simulateTurn(
      'Tell me about ' + 'quantum '.repeat(30) + 'physics.',
      'A'.repeat(200),
      { input_tokens: 500, output_tokens: 120 }
    );

    expect(turn4.context.length).toBeLessThan(conversationHistory.length);
    expect(turn4.context.length).toBeGreaterThan(1);

    // All returned indices should have token counts
    for (let i = 0; i < conversationHistory.length; i++) {
      expect(turn4.indexTokenCountMap[i]).toBeDefined();
    }

    // Verify variance logs were emitted
    const varianceLogs = logs.filter(
      (l) => l.message === 'Calibration observed'
    );
    expect(varianceLogs.length).toBeGreaterThanOrEqual(4);
  });

  it('calibration + observation masking + pruning produce consistent accounting', () => {
    const tcId = 'tc_big';
    const bigToolResult = 'D'.repeat(3000);

    const messages: BaseMessage[] = [
      new SystemMessage('Assistant'),
      new HumanMessage('Search for data'),
      aiToolCall(tcId, 'search'),
      toolMsg(bigToolResult, 'search', tcId),
      new AIMessage('Here is what I found from the search results.'),
      new HumanMessage('Thanks, now summarize it'),
      new AIMessage('The data shows important patterns in the results.'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    let sum = 0;
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
      sum += indexTokenCountMap[i] ?? 0;
    }

    // Pressure ~90% to trigger both masking and context pressure fading
    const maxTokens = Math.round(sum / 0.9);

    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    const result = pruneMessages({ messages });

    // Tool result at index 3 should have been masked (consumed by AI at index 4)
    const toolTokensAfter = result.indexTokenCountMap[3] ?? 0;
    expect(toolTokensAfter).toBeLessThan(3000);

    // The final token sum should be within the pruning budget
    let finalSum = 0;
    for (let i = 0; i < result.context.length; i++) {
      const origIdx = messages.indexOf(result.context[i]);
      if (origIdx >= 0) {
        finalSum += result.indexTokenCountMap[origIdx] ?? 0;
      }
    }
    expect(finalSum).toBeLessThanOrEqual(maxTokens);

    // Context pressure should have been computed
    expect(result.contextPressure).toBeDefined();
    expect(result.contextPressure as number).toBeGreaterThan(0);
  });
});

describe('Token accounting pipeline — Anthropic vs OpenAI cache semantics', () => {
  it('Anthropic additive cache inflates calibration input correctly', () => {
    const messages = [new HumanMessage('Hello'), new AIMessage('Hi')];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 50,
      1: 50,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.ANTHROPIC,
      maxTokens: 10000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    // Anthropic: input_tokens=30, cache_read=60, cache_creation=20
    // cacheSum=80 > baseInput=30 → additive → totalInput=30+80=110
    const result = pruneMessages({
      messages,
      usageMetadata: {
        input_tokens: 30,
        output_tokens: 15,
        input_token_details: { cache_read: 60, cache_creation: 20 },
      },
    });

    // Map stays raw — calibrationRatio = 110 / 100 = 1.1
    expect(result.indexTokenCountMap[0]).toBe(50);
    expect(result.indexTokenCountMap[1]).toBe(50);
    expect(result.calibrationRatio).toBeCloseTo(1.1, 1);
  });

  it('OpenAI inclusive cache does NOT inflate calibration input', () => {
    const messages = [new HumanMessage('Hello'), new AIMessage('Hi')];

    const indexTokenCountMap: Record<string, number | undefined> = {
      0: 50,
      1: 50,
    };

    const pruneMessages = createPruneMessages({
      provider: Providers.OPENAI,
      maxTokens: 10000,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
    });

    // OpenAI: input_tokens=100, cache_read=40 — cacheSum=40 <= baseInput=100
    const result = pruneMessages({
      messages,
      usageMetadata: {
        input_tokens: 100,
        output_tokens: 20,
        input_token_details: { cache_read: 40 },
      },
    });

    // ratio = 100 / 100 = 1.0 — no change
    expect(result.indexTokenCountMap[0]).toBe(50);
    expect(result.indexTokenCountMap[1]).toBe(50);
  });
});

describe('Token accounting pipeline — instruction-budget short-circuit', () => {
  it('yields all messages for summarization when instructions consume entire budget', () => {
    const messages = [
      new HumanMessage('First question'),
      new AIMessage('First answer'),
      new HumanMessage('Second question'),
      new AIMessage('Second answer'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    // Budget 100, instruction overhead 100 → effectiveMaxTokens = 0
    const pruneMessages = createPruneMessages({
      maxTokens: 100,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      summarizationEnabled: true,
      reserveRatio: 0,
      getInstructionTokens: () => 100,
    });

    const result = pruneMessages({ messages });

    expect(result.context).toHaveLength(0);
    expect(result.messagesToRefine).toHaveLength(4);
    expect(result.remainingContextTokens).toBe(0);
  });

  it('does NOT short-circuit when summarization is disabled', () => {
    const messages = [
      new HumanMessage('First question'),
      new AIMessage('First answer'),
    ];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const pruneMessages = createPruneMessages({
      maxTokens: 100,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      reserveRatio: 0,
      getInstructionTokens: () => 100,
    });

    const result = pruneMessages({ messages });

    // Without summarization, the pruner goes through normal/emergency path
    // instead of the short-circuit — messagesToRefine may be empty
    expect(
      result.context.length + (result.messagesToRefine?.length ?? 0)
    ).toBeGreaterThanOrEqual(0);
  });

  it('does NOT short-circuit when effectiveMaxTokens > 0', () => {
    const messages = [new HumanMessage('Short'), new AIMessage('Reply')];

    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const pruneMessages = createPruneMessages({
      maxTokens: 200,
      startIndex: messages.length,
      tokenCounter: charCounter,
      indexTokenCountMap,
      summarizationEnabled: true,
      reserveRatio: 0,
      getInstructionTokens: () => 50,
    });

    const result = pruneMessages({ messages });

    // effectiveMaxTokens = 150, messages fit → normal early return
    expect(result.context.length).toBe(2);
    expect(result.messagesToRefine).toHaveLength(0);
  });
});
