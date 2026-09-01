/* eslint-disable no-console */
import { performance } from 'node:perf_hooks';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  inspectProviderMessageProjection,
  setProviderMessageProvenance,
} from '@/messages';
import { Providers } from '@/common';
import { prepareProviderRequest } from '@/llm/prepareProviderRequest';

interface BenchmarkResult {
  elapsedMs: number;
  checksum: number;
}

const ITERATIONS = 250;
const SAMPLE_COUNT = 7;

const benchmarkModel = {
  model: 'projection-benchmark',
  invoke: async (): Promise<AIMessageChunk> => new AIMessageChunk(''),
} as t.ChatModel;

function createTextHistory(messageCount: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let index = 0; index < messageCount; index++) {
    const content = `Message ${index}: ${'retained conversation context '.repeat(24)}`;
    const isUserMessage = index % 2 === 0;
    const sourceMessageId = isUserMessage
      ? `human-${index}`
      : `assistant-${index}`;
    const message = isUserMessage
      ? new HumanMessage({ id: sourceMessageId, content })
      : new AIMessage({ id: sourceMessageId, content });
    setProviderMessageProvenance(message, [
      {
        attribution: isUserMessage ? 'user' : 'model',
        sourceMessageId,
      },
    ]);
    messages.push(message);
  }
  return messages;
}

function createToolHistory(turnCount: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let index = 0; index < turnCount; index++) {
    const callId = `call-${index}`;
    const user = new HumanMessage({
      id: `human-${index}`,
      content: `Find context for turn ${index}.`,
    });
    const assistant = new AIMessage({
      id: `assistant-${index}`,
      content: '',
      tool_calls: [
        {
          id: callId,
          name: 'search',
          args: { query: `turn-${index}` },
          type: 'tool_call',
        },
      ],
    });
    const toolResult = new ToolMessage({
      id: `tool-${index}`,
      tool_call_id: callId,
      content: [
        {
          type: 'text',
          text: `Result ${index}: ${'structured provider-neutral tool output '.repeat(12)}`,
        },
      ],
    });
    setProviderMessageProvenance(user, [
      { attribution: 'user', sourceMessageId: `human-${index}` },
    ]);
    setProviderMessageProvenance(assistant, [
      { attribution: 'model', sourceMessageId: `assistant-${index}` },
    ]);
    setProviderMessageProvenance(toolResult, [
      { attribution: 'tool', sourceMessageId: `tool-${index}` },
    ]);
    messages.push(user, assistant, toolResult);
  }
  return messages;
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function runBenchmark(
  messages: BaseMessage[],
  provider: t.ProviderName,
  inspectInvariant = false
): BenchmarkResult {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < ITERATIONS; iteration++) {
    const request = prepareProviderRequest({
      model: benchmarkModel,
      messages,
      provider,
    });
    checksum += request.messages.length;
    if (inspectInvariant) {
      const report = inspectProviderMessageProjection(request.messages);
      if (!report.valid) {
        throw new Error('Benchmark projection has provenance gaps');
      }
      checksum += report.sourceBackedMessageCount;
    }
  }
  return {
    elapsedMs: performance.now() - startedAt,
    checksum,
  };
}

for (const scenario of [
  { name: 'text-100', messages: createTextHistory(100) },
  { name: 'text-500', messages: createTextHistory(500) },
  { name: 'tools-100', messages: createToolHistory(100) },
]) {
  for (const provider of [Providers.OPENAI, Providers.ANTHROPIC]) {
    for (const invariant of ['off', 'observe'] as const) {
      const inspectInvariant = invariant === 'observe';
      runBenchmark(scenario.messages, provider, inspectInvariant);
      const samples: BenchmarkResult[] = [];
      for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
        samples.push(
          runBenchmark(scenario.messages, provider, inspectInvariant)
        );
      }
      const checksum = samples[0].checksum;
      if (samples.some((sample) => sample.checksum !== checksum)) {
        throw new Error(
          `Provider projection output changed for ${scenario.name}`
        );
      }
      const medianMs = median(samples.map(({ elapsedMs }) => elapsedMs));
      console.log(
        JSON.stringify({
          scenario: scenario.name,
          provider,
          invariant,
          iterations: ITERATIONS,
          medianMs: Number(medianMs.toFixed(2)),
          messagesPerSecond: Number(
            (
              (scenario.messages.length * ITERATIONS * 1_000) /
              medianMs
            ).toFixed(0)
          ),
        })
      );
    }
  }
}
