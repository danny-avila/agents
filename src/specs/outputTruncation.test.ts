// src/specs/outputTruncation.test.ts
/**
 * A model turn cut off by the output token ceiling while producing plain
 * text/reasoning carries no tool call, so `toolsCondition` reads it as an
 * ordinary finished turn and routes to END exactly like a real completion.
 * `getOutputTruncated()` is the channel that lets a host tell the two apart
 * and persist the turn as unfinished instead of a silently truncated
 * "complete" answer. See `outputTruncatedIncomplete` on `StandardGraph`.
 */
import { HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { Run } from '@/run';

const streamConfig = {
  configurable: { thread_id: 'output-truncation' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

async function createPlainRun(runId: string): Promise<Run<t.IState>> {
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o-mini',
        apiKey: 'test-key',
      },
      instructions: 'Answer plainly.',
    },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  return run;
}

describe('output-token truncation without a tool call', () => {
  it('flags a plain-text turn cut off at the output token ceiling', async () => {
    const run = await createPlainRun('truncated-plain-text');
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['The channel posts about three times a day and the'],
      finalChunkGenerationInfo: { finish_reason: 'length' },
    }) as unknown as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('Analyze this channel')] },
      streamConfig
    );

    expect(run.getOutputTruncated()).toBe(true);
    expect(run.Graph?.outputTruncatedIncomplete).toBe(true);
    // Distinct from the preempt/seal machinery — this path never touches it.
    expect(run.getHaltReason()).toBeUndefined();
    expect(run.Graph?.preemptIncomplete).toBe(false);
  });

  it('does not flag a normal completed turn', async () => {
    const run = await createPlainRun('normal-completion');
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['The channel posts about three times a day.'],
      finalChunkGenerationInfo: { finish_reason: 'stop' },
    }) as unknown as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('Analyze this channel')] },
      streamConfig
    );

    expect(run.getOutputTruncated()).toBe(false);
    expect(run.Graph?.outputTruncatedIncomplete).toBe(false);
  });

  it('does not flag a turn with no finish-reason metadata at all', async () => {
    const run = await createPlainRun('no-metadata');
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['The channel posts about three times a day.'],
    }) as unknown as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('Analyze this channel')] },
      streamConfig
    );

    expect(run.getOutputTruncated()).toBe(false);
  });
});
