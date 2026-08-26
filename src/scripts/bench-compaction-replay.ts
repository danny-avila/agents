/* eslint-disable no-console */
import { performance } from 'node:perf_hooks';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';
import {
  createCompactionCacheNamespace,
  createCompactionReplayRecipe,
  inspectCompactionReplayEligibility,
} from '@/llm/compactionReplay';
import { setProviderMessageProvenance } from '@/messages/provenance';
import { Providers } from '@/common';

const CAPTURE_ITERATIONS = 100_000;
const INSPECTION_ITERATIONS = 2_000;

function stamp<T extends BaseMessage>(message: T, sourceId: string): T {
  setProviderMessageProvenance(message, [
    { attribution: 'user', sourceMessageId: sourceId },
  ]);
  return message;
}

function createToolHistory(
  toolSteps: number,
  priorCheckpoint: boolean
): BaseMessage[] {
  const messages: BaseMessage[] = [];
  if (priorCheckpoint) {
    messages.push(
      new HumanMessage({
        content: '<summary>prior checkpoint</summary>',
        additional_kwargs: { injected: true, source: 'summary' },
      })
    );
  }
  messages.push(stamp(new HumanMessage('complete the task'), 'user-0'));
  for (let i = 0; i < toolSteps; i++) {
    const callId = `call-${i}`;
    const assistant = new AIMessage({
      id: `assistant-${i}`,
      content: '',
      tool_calls: [
        {
          id: callId,
          name: 'exec_command',
          args: { cmd: `rg pattern-${i}` },
        },
      ],
    });
    const tool = new ToolMessage({
      id: `tool-${i}`,
      content: `result-${i}`,
      tool_call_id: callId,
      name: 'exec_command',
    });
    setProviderMessageProvenance(assistant, [
      { attribution: 'model', sourceMessageId: `assistant-${i}` },
    ]);
    setProviderMessageProvenance(tool, [
      { attribution: 'tool', sourceMessageId: `tool-${i}` },
    ]);
    messages.push(assistant, tool);
  }
  return messages;
}

function measure(iterations: number, operation: () => void): number {
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    operation();
  }
  return (performance.now() - start) / iterations;
}

for (const toolSteps of [20, 50, 100]) {
  for (const priorCheckpoint of [false, true]) {
    const messages = createToolHistory(toolSteps, priorCheckpoint);
    const cacheNamespace = createCompactionCacheNamespace(
      Providers.ANTHROPIC,
      {
        baseURL: 'https://benchmark.invalid',
      }
    );
    const createEnvelope = () =>
      createCompactionReplayRecipe({
        provider: Providers.ANTHROPIC,
        modelId: 'benchmark-model',
        projectionMode: 'chat-messages',
        cacheNamespace,
        systemRevision: 0,
        toolRevision: 0,
        messages,
      });
    const envelope = createEnvelope();
    const compactableMessages = messages.slice(0, messages.length - 4);
    const captureMs = measure(CAPTURE_ITERATIONS, () => {
      createEnvelope();
    });
    const inspectionMs = measure(INSPECTION_ITERATIONS, () => {
      inspectCompactionReplayEligibility(
        envelope,
        {
          provider: Providers.ANTHROPIC,
          modelId: 'benchmark-model',
          projectionMode: 'chat-messages',
          cacheNamespace,
          systemRevision: 0,
          toolRevision: 0,
          messages: compactableMessages,
          restoredToolSubstitution: false,
        }
      );
    });
    const eligibility = inspectCompactionReplayEligibility(
      envelope,
      {
        provider: Providers.ANTHROPIC,
        modelId: 'benchmark-model',
        projectionMode: 'chat-messages',
        cacheNamespace,
        systemRevision: 0,
        toolRevision: 0,
        messages: compactableMessages,
        restoredToolSubstitution: false,
      }
    );

    console.log(
      JSON.stringify({
        toolSteps,
        priorCheckpoint,
        requestMessages: messages.length,
        compactableMessages: compactableMessages.length,
        captureMicrosecondsPerRequest: Number((captureMs * 1_000).toFixed(3)),
        inspectionMicrosecondsPerCompaction: Number(
          (inspectionMs * 1_000).toFixed(3)
        ),
        eligibility,
      })
    );
  }
}
