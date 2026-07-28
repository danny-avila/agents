// src/specs/preemptSeal.test.ts
/**
 * End-to-end cooperative seal flow through a real `Run`: fake model streams,
 * the host requests a preempt, the stream seals at a safe chunk, the
 * `PreemptBoundary` drain decides what happens next. Everything below runs
 * the dispatch-synchronous loop in `attemptInvoke` (no registered
 * CHAT_MODEL_STREAM handler), which is the only loop allowed to seal.
 */
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers } from '@/common';
import { HookRegistry } from '@/hooks/HookRegistry';
import type { HookCallback } from '@/hooks/types';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const FULL_RESPONSE = 'Alpha beta gamma delta epsilon zeta';
const RESUMED_RESPONSE = 'Continuing after the steer.';

const streamConfig = {
  configurable: { thread_id: 'preempt-seal-e2e' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

async function createSealRun(options: {
  runId: string;
  hook: HookCallback<'PreemptBoundary'>;
  responses: string[];
  stopHook?: HookCallback<'Stop'>;
  modelCallbacks?: FakeChatModel['callbacks'];
}): Promise<Run<t.IState>> {
  const registry = new HookRegistry();
  registry.register('PreemptBoundary', { hooks: [options.hook] });
  if (options.stopHook) {
    registry.register('Stop', { hooks: [options.stopHook] });
  }
  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o-mini',
        apiKey: 'test-key',
      },
      instructions: 'Answer plainly.',
    },
    hooks: registry,
    preemption: { shouldPreempt: () => true, maxSeals: 1 },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const model = new FakeChatModel({
    responses: options.responses,
  });
  if (options.modelCallbacks != null) {
    model.callbacks = options.modelCallbacks;
  }
  run.Graph.overrideModel = model;
  return run;
}

const aiContents = (messages: BaseMessage[]): string[] =>
  messages
    .filter((message) => message.getType() === 'ai')
    .map((message) =>
      typeof message.content === 'string'
        ? message.content
        : JSON.stringify(message.content)
    );

describe('cooperative seal (end-to-end via Run)', () => {
  jest.setTimeout(15000);

  it('surfaces an empty boundary as preempt_incomplete instead of a natural finish', async () => {
    const run = await createSealRun({
      runId: 'seal-empty-boundary',
      hook: async () => ({}),
      responses: [FULL_RESPONSE],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    /**
     * The answer really was cut short: the host asked to preempt, the seal
     * took the budget, and the drain had nothing to resume with. A terminal
     * consumer reading only completion events would persist a truncated
     * answer as finished — `getHaltReason()` is the channel that prevents
     * that (AgentSession emits `run.halted` off it).
     */
    expect(run.getHaltReason()).toBe('preempt_incomplete');
    expect(run.Graph?.preemptIncomplete).toBe(true);
    expect(run.Graph?.preemptEmptyBoundaries).toBe(1);

    const contents = aiContents(run.getRunMessages() ?? []);
    expect(contents).toHaveLength(1);
    expect(contents[0].length).toBeGreaterThan(0);
    expect(contents[0].length).toBeLessThan(FULL_RESPONSE.length);
    expect(FULL_RESPONSE.startsWith(contents[0])).toBe(true);
  });

  it('forwards a halting hook\'s own stopReason to Stop hooks and getHaltReason', async () => {
    let stopReasonSeen: string | undefined;
    const run = await createSealRun({
      runId: 'seal-halt-reason',
      hook: async () => ({
        preventContinuation: true,
        stopReason: 'host_policy_stop',
      }),
      responses: [FULL_RESPONSE],
      stopHook: async (input) => {
        stopReasonSeen = input.stopReason;
        return {};
      },
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    /**
     * The hook-supplied reason must win end to end: a persistence/audit Stop
     * hook records the actual cause, not the generic preempt_incomplete
     * label, and getHaltReason() reports the same string afterward. A
     * halting boundary that injected nothing also counts as an empty
     * boundary in the truncated-seal telemetry.
     */
    expect(stopReasonSeen).toBe('host_policy_stop');
    expect(run.getHaltReason()).toBe('host_policy_stop');
    expect(run.Graph?.preemptIncomplete).toBe(true);
    expect(run.Graph?.preemptEmptyBoundaries).toBe(1);
  });

  it('closes model-level callbacks for the sealed run, not just config-level ones', async () => {
    let starts = 0;
    let ends = 0;
    const run = await createSealRun({
      runId: 'seal-model-callbacks',
      hook: async () => ({
        injectedMessages: [
          { role: 'user' as const, content: 'Shorter.', source: 'steer' },
        ],
      }),
      responses: [FULL_RESPONSE, RESUMED_RESPONSE],
      /**
       * A handler supplied on the MODEL (clientOptions.callbacks) gets
       * handleChatModelStart from the real run, so the sealed turn's
       * synthetic close must reach it too — otherwise its span for the
       * sealed run never closes. Two runs (sealed + resumed): both must
       * balance.
       */
      modelCallbacks: [
        {
          handleChatModelStart: (): void => {
            starts += 1;
          },
          handleLLMEnd: (): void => {
            ends += 1;
          },
        },
      ],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    expect(run.Graph?.preemptSealCount).toBe(1);
    expect(starts).toBe(2);
    expect(ends).toBe(2);
  });

  it('resumes after an injecting boundary and completes without a halt reason', async () => {
    const run = await createSealRun({
      runId: 'seal-inject-resume',
      hook: async () => ({
        injectedMessages: [
          { role: 'user' as const, content: 'Make it shorter.', source: 'steer' },
        ],
      }),
      responses: [FULL_RESPONSE, RESUMED_RESPONSE],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    expect(run.getHaltReason()).toBeUndefined();
    expect(run.Graph?.preemptIncomplete).toBe(false);
    expect(run.Graph?.preemptSealCount).toBe(1);

    const messages = run.getRunMessages() ?? [];
    const steer = messages.find(
      (message) => message.additional_kwargs.source === 'steer'
    );
    expect(steer).toBeDefined();
    expect(steer?.content).toBe('Make it shorter.');

    /**
     * Two assistant turns: the sealed partial and the post-steer
     * continuation, which must have run to completion — the seal budget was
     * spent, so the second stream cannot seal again even though
     * `shouldPreempt` still answers true.
     */
    const contents = aiContents(messages);
    expect(contents).toHaveLength(2);
    expect(FULL_RESPONSE.startsWith(contents[0])).toBe(true);
    expect(contents[0].length).toBeLessThan(FULL_RESPONSE.length);
    expect(contents[1]).toBe(RESUMED_RESPONSE);
  });
});
