// src/graphs/__tests__/Graph.preemptSignal.test.ts
import type * as t from '@/types';
import { Providers } from '@/common';
import { HookRegistry } from '@/hooks/HookRegistry';
import { StandardGraph } from '../Graph';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

type DispatchResult = { messages: unknown[]; preventContinuation: boolean };

const dispatchBoundary = (
  graph: StandardGraph,
  agentId = 'agent'
): Promise<DispatchResult> =>
  (
    graph as unknown as {
      dispatchPreemptBoundary: (
        agentId: string,
        config: { configurable: { run_id: string } } | undefined
      ) => Promise<DispatchResult>;
    }
  ).dispatchPreemptBoundary(agentId, {
    configurable: { run_id: 'run_1' },
  });

/**
 * Registers a PreemptBoundary hook that aborts `controller` from inside its
 * own body, then reports whether the signal `executeHooks` handed it observed
 * that abort. Racing against a short timer keeps a broken composition from
 * stalling the suite for the 120-second boundary hook timeout.
 */
function registerAbortProbe(
  registry: HookRegistry,
  controller: AbortController
): { outcome: () => string | undefined } {
  let outcome: string | undefined;
  registry.register('PreemptBoundary', {
    hooks: [
      async (_input, signal): Promise<Record<string, never>> => {
        const aborted = new Promise<string>((resolve) => {
          if (signal.aborted) {
            resolve('aborted');
            return;
          }
          signal.addEventListener('abort', () => resolve('aborted'), {
            once: true,
          });
        });
        const timer = new Promise<string>((resolve) =>
          setTimeout(() => resolve('not-observed'), 200)
        );
        controller.abort();
        outcome = await Promise.race([aborted, timer]);
        return {};
      },
    ],
  });
  return { outcome: () => outcome };
}

describe('PreemptBoundary abort-signal composition', () => {
  it('observes a per-call caller abort even when a construction signal exists', async () => {
    const constructionController = new AbortController();
    const callerController = new AbortController();
    const graph = new StandardGraph({
      runId: 'run_1',
      signal: constructionController.signal,
      agents: [makeAgent('agent')],
    });
    const registry = new HookRegistry();
    const probe = registerAbortProbe(registry, callerController);
    graph.hookRegistry = registry;
    graph.callerSignal = callerController.signal;

    await dispatchBoundary(graph);

    expect(probe.outcome()).toBe('aborted');
    expect(constructionController.signal.aborted).toBe(false);
  });

  it('observes the caller abort when no construction signal exists (multi-agent shape)', async () => {
    const callerController = new AbortController();
    const graph = new StandardGraph({
      runId: 'run_1',
      agents: [makeAgent('agent')],
    });
    const registry = new HookRegistry();
    const probe = registerAbortProbe(registry, callerController);
    graph.hookRegistry = registry;
    graph.callerSignal = callerController.signal;

    await dispatchBoundary(graph);

    expect(probe.outcome()).toBe('aborted');
  });

  it('still observes the construction signal when no caller signal was supplied', async () => {
    const constructionController = new AbortController();
    const graph = new StandardGraph({
      runId: 'run_1',
      signal: constructionController.signal,
      agents: [makeAgent('agent')],
    });
    const registry = new HookRegistry();
    const probe = registerAbortProbe(registry, constructionController);
    graph.hookRegistry = registry;

    await dispatchBoundary(graph);

    expect(probe.outcome()).toBe('aborted');
  });

  it('drops the caller signal reference on clearHeavyState', () => {
    const graph = new StandardGraph({
      runId: 'run_1',
      agents: [makeAgent('agent')],
    });
    graph.callerSignal = new AbortController().signal;
    graph.clearHeavyState();
    expect(graph.callerSignal).toBeUndefined();
  });
});
