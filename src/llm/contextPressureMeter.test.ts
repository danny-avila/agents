import { HumanMessage } from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';

import { stampSyntheticProviderMessage } from '@/messages';
import { createContextPressureMeter } from './contextPressureMeter';

function contentLength(message: BaseMessage): number {
  return typeof message.content === 'string'
    ? message.content.length
    : JSON.stringify(message.content).length;
}

describe('createContextPressureMeter', () => {
  it('preserves provider-grounded attribution while caching exact counts', () => {
    const source = [
      new HumanMessage({ id: 'first', content: 'a' }),
      new HumanMessage({ id: 'second', content: 'bb' }),
    ];
    const tokenCounter = jest.fn(contentLength);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: source,
      retainedMessages: source,
      indexTokenCountMap: { 0: 10, 1: 20 },
      contextUsage: {
        contextBudget: 100,
        effectiveInstructionTokens: 10,
        remainingContextTokens: 60,
        calibrationRatio: 1,
      },
      instructionTokens: 10,
      calibrationRatio: 1,
    });
    const projected = [
      new HumanMessage({ id: 'first', content: 'aaaa' }),
      source[1],
    ];
    meter.trackProjection(source, projected);

    expect(meter.measure(projected)).toEqual({
      fits: true,
      projectedMessageTokens: 33,
      availableMessageTokens: 90,
      contextBudget: 100,
      effectiveInstructionTokens: 10,
    });
    expect(meter.measure(projected).projectedMessageTokens).toBe(33);
    expect(tokenCounter).toHaveBeenCalledTimes(3);
  });

  it('tokenizes only changed candidates across a compaction search', () => {
    const retained = Array.from(
      { length: 100 },
      (_, index) =>
        new HumanMessage({ id: `message-${index}`, content: 'retained' })
    );
    const tokenCounter = jest.fn(contentLength);
    const meter = createContextPressureMeter({
      tokenCounter,
      sourceMessages: retained,
      retainedMessages: retained,
      indexTokenCountMap: Object.fromEntries(
        retained.map((_, index) => [index, 8])
      ),
      contextUsage: {
        contextBudget: 10_000,
        effectiveInstructionTokens: 100,
        remainingContextTokens: 9_000,
        calibrationRatio: 1,
      },
      instructionTokens: 100,
      calibrationRatio: 1,
    });

    for (let i = 0; i < 13; i++) {
      meter.measure([
        ...retained,
        stampSyntheticProviderMessage(
          new HumanMessage({ content: 'x'.repeat(i + 1) })
        ),
      ]);
    }

    expect(tokenCounter).toHaveBeenCalledTimes(113);
  });

  it('uses a conservative ratio for fallback payloads', () => {
    const message = new HumanMessage('1234567890');
    const meter = createContextPressureMeter({
      tokenCounter: contentLength,
      sourceMessages: [message],
      retainedMessages: [message],
      indexTokenCountMap: { 0: 10 },
      instructionTokens: 5,
      calibrationRatio: 0.5,
    });

    expect(
      meter.measure([message], {
        contextBudget: 17,
        forceRawRecount: true,
      })
    ).toEqual({
      fits: false,
      projectedMessageTokens: 13,
      availableMessageTokens: 12,
      contextBudget: 17,
      effectiveInstructionTokens: 5,
    });
  });
});
