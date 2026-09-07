/* eslint-disable no-console */
import { performance } from 'node:perf_hooks';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { createToolHistoryPreparation } from '@/messages/toolHistoryProjection';
import { projectToolStreamContentForProvider } from '@/messages/core';
import { foldToolBlocksForToollessAgent } from '@/messages/format';

const iterations = 100;
const samples = 5;

function prepare(messages: BaseMessage[], shared: boolean): number {
  const history = createToolHistoryPreparation();
  const folded = foldToolBlocksForToollessAgent(
    messages,
    undefined,
    false,
    history
  );
  const replayed = projectToolStreamContentForProvider(
    messages,
    'native',
    8000,
    shared ? history : createToolHistoryPreparation()
  );
  return folded.length + replayed.length;
}

function measure(messages: BaseMessage[], shared: boolean): number {
  const start = performance.now();
  let checksum = 0;
  for (let iteration = 0; iteration < iterations; iteration++) {
    checksum += prepare(messages, shared);
  }
  if (checksum !== messages.length * iterations * 2) {
    throw new Error('Unexpected projection output count');
  }
  return performance.now() - start;
}

function median(values: number[]): number {
  return [...values].sort((left, right) => left - right)[
    Math.floor(values.length / 2)
  ];
}

const plain = Array.from(
  { length: 500 },
  (_, index) => new HumanMessage(`plain text ${index}`)
);
const tools = Array.from(
  { length: 100 },
  (_, index) =>
    new AIMessage({
      content: `Answer ${index}`,
      response_metadata: {
        model_provider: 'openai',
        preempted: true,
        output: [
          {
            type: 'code_interpreter_call',
            id: `tool-${index}`,
            status: 'completed',
            outputs: [{ type: 'logs', logs: `result-${index}` }],
          },
          {
            type: 'message',
            id: `message-${index}`,
            content: [{ type: 'output_text', text: `Answer ${index}` }],
          },
        ],
      },
    })
);

for (const scenario of [
  { name: 'plain-500', messages: plain },
  { name: 'tools-100', messages: tools },
]) {
  measure(scenario.messages, false);
  measure(scenario.messages, true);
  const separate: number[] = [];
  const shared: number[] = [];
  for (let sample = 0; sample < samples; sample++) {
    if (sample % 2 === 0) {
      separate.push(measure(scenario.messages, false));
      shared.push(measure(scenario.messages, true));
    } else {
      shared.push(measure(scenario.messages, true));
      separate.push(measure(scenario.messages, false));
    }
  }
  const history = createToolHistoryPreparation();
  let projections = 0;
  for (const message of scenario.messages) {
    if (history.get(message) != null) {
      projections++;
    }
  }
  console.log(
    JSON.stringify({
      scenario: scenario.name,
      iterations,
      separateMs: Number(median(separate).toFixed(2)),
      sharedMs: Number(median(shared).toFixed(2)),
      projections,
      sourceArrayPreserved:
        foldToolBlocksForToollessAgent(scenario.messages) === scenario.messages,
    })
  );
}
