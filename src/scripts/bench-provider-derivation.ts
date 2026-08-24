/* eslint-disable no-console */
import { performance } from 'node:perf_hooks';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';
import { projectToolStreamContentForProvider } from '@/messages/core';
import {
  projectToolCallInputs,
  projectToolMessagesForProvider,
} from '@/messages/prune';
import { serializeMessage } from '@/session/messageSerialization';

interface BenchmarkResult {
  elapsedMs: number;
  checksum: number;
}

const ITERATIONS = 250;
const SAMPLE_COUNT = 7;
const MAX_TOOL_INPUT_CHARS = 800;

function createTextHistory(messageCount: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let index = 0; index < messageCount; index++) {
    messages.push(
      new HumanMessage(
        `Message ${index}: ${'retained conversation context '.repeat(24)}`
      )
    );
  }
  return messages;
}

function createToolHistory(turnCount: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let index = 0; index < turnCount; index++) {
    const callId = `call-${index}`;
    const query = `query-${index}-${'x'.repeat(2_000)}`;
    messages.push(
      new HumanMessage(`Find context for turn ${index}.`),
      new AIMessageChunk({
        content: [
          { type: 'text', text: '' },
          {
            type: 'tool_use',
            id: callId,
            name: 'search',
            input: { query },
          },
        ],
        tool_calls: [
          {
            id: callId,
            name: 'search',
            args: { query },
          },
        ],
      })
    );
  }
  return messages;
}

function serialize(messages: BaseMessage[]): string {
  return JSON.stringify(messages.map(serializeMessage));
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function runBefore(messages: BaseMessage[]): BenchmarkResult {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    checksum += projectToolCallInputs(
      projectToolStreamContentForProvider(messages),
      MAX_TOOL_INPUT_CHARS
    ).length;
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

function runAfter(messages: BaseMessage[]): BenchmarkResult {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    checksum += projectToolMessagesForProvider(
      messages,
      MAX_TOOL_INPUT_CHARS
    ).length;
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

for (const scenario of [
  { name: 'text-500', messages: createTextHistory(500) },
  { name: 'tools-100', messages: createToolHistory(100) },
]) {
  const beforeProjection = projectToolCallInputs(
    projectToolStreamContentForProvider(scenario.messages),
    MAX_TOOL_INPUT_CHARS
  );
  const afterProjection = projectToolMessagesForProvider(
    scenario.messages,
    MAX_TOOL_INPUT_CHARS
  );
  if (serialize(beforeProjection) !== serialize(afterProjection)) {
    throw new Error(`Provider derivation changed ${scenario.name} output`);
  }

  runBefore(scenario.messages);
  runAfter(scenario.messages);
  const beforeSamples: BenchmarkResult[] = [];
  const afterSamples: BenchmarkResult[] = [];
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const beforeFirst = sample % 2 === 0;
    const first = beforeFirst
      ? runBefore(scenario.messages)
      : runAfter(scenario.messages);
    const second = beforeFirst
      ? runAfter(scenario.messages)
      : runBefore(scenario.messages);
    (beforeFirst ? beforeSamples : afterSamples).push(first);
    (beforeFirst ? afterSamples : beforeSamples).push(second);
  }
  const beforeMs = median(beforeSamples.map(({ elapsedMs }) => elapsedMs));
  const afterMs = median(afterSamples.map(({ elapsedMs }) => elapsedMs));
  if (
    beforeSamples[0].checksum !== afterSamples[0].checksum ||
    beforeSamples.some(
      (sample) => sample.checksum !== beforeSamples[0].checksum
    ) ||
    afterSamples.some((sample) => sample.checksum !== afterSamples[0].checksum)
  ) {
    throw new Error(
      `Provider derivation changed ${scenario.name} message count`
    );
  }
  console.log(
    JSON.stringify({
      scenario: scenario.name,
      iterations: ITERATIONS,
      beforeMs: Number(beforeMs.toFixed(2)),
      afterMs: Number(afterMs.toFixed(2)),
      speedup: Number((beforeMs / afterMs).toFixed(2)),
    })
  );
}
