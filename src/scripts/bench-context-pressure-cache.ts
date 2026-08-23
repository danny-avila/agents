import { performance } from 'node:perf_hooks';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';

import {
  createContextPressureMeter,
  createExactTokenCountCache,
} from '@/llm/contextPressureMeter';
import { createTokenCounter } from '@/utils/tokens';

interface BenchmarkResult {
  elapsedMs: number;
  tokenizations: number;
  checksum: number;
}

const ITERATIONS = 10;
const SAMPLE_COUNT = 5;

function createHistory(messageCount: number): BaseMessage[] {
  return Array.from({ length: messageCount }, (_, index) => {
    const content = `Message ${index}: ${'retained conversation context '.repeat(32)}`;
    return index % 2 === 0
      ? new HumanMessage({ id: `human-${index}`, content })
      : new AIMessage({ id: `assistant-${index}`, content });
  });
}

async function runBenchmark(
  messageCount: number,
  useSharedCache: boolean
): Promise<BenchmarkResult> {
  const history = createHistory(messageCount);
  const exactCounter = await createTokenCounter();
  let tokenizations = 0;
  const tokenCounter = (message: BaseMessage): number => {
    tokenizations++;
    return exactCounter(message);
  };
  const tokenCountCache = useSharedCache
    ? createExactTokenCountCache(tokenCounter)
    : undefined;
  const indexTokenCountMap: Record<string, number> = Object.create(null);
  let checksum = 0;
  const startedAt = performance.now();

  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const meter = createContextPressureMeter({
      tokenCounter,
      tokenCountCache,
      sourceMessages: history,
      retainedMessages: history,
      indexTokenCountMap,
      contextUsage: {
        contextBudget: 1_000_000,
        effectiveInstructionTokens: 1_000,
        remainingContextTokens: 900_000,
        calibrationRatio: 1,
      },
      instructionTokens: 1_000,
      calibrationRatio: 1,
    });
    checksum += meter.measure(history).projectedMessageTokens ?? 0;
  }

  return {
    elapsedMs: performance.now() - startedAt,
    tokenizations,
    checksum,
  };
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

for (const messageCount of [100, 500]) {
  const beforeSamples: BenchmarkResult[] = [];
  const afterSamples: BenchmarkResult[] = [];

  await runBenchmark(messageCount, false);
  await runBenchmark(messageCount, true);
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const firstUsesCache = sample % 2 === 0;
    const first = await runBenchmark(messageCount, firstUsesCache);
    const second = await runBenchmark(messageCount, !firstUsesCache);
    (firstUsesCache ? afterSamples : beforeSamples).push(first);
    (firstUsesCache ? beforeSamples : afterSamples).push(second);
  }

  const beforeMs = median(beforeSamples.map(({ elapsedMs }) => elapsedMs));
  const afterMs = median(afterSamples.map(({ elapsedMs }) => elapsedMs));
  const before = beforeSamples[0];
  const after = afterSamples[0];
  if (before.checksum !== after.checksum) {
    throw new Error('Cached and uncached context measurements diverged');
  }

  console.log(
    JSON.stringify({
      messages: messageCount,
      iterations: ITERATIONS,
      beforeMs: Number(beforeMs.toFixed(2)),
      afterMs: Number(afterMs.toFixed(2)),
      speedup: Number((beforeMs / afterMs).toFixed(2)),
      tokenizationsBefore: before.tokenizations,
      tokenizationsAfter: after.tokenizations,
    })
  );
}
