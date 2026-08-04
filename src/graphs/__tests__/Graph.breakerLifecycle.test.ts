import { describe, it, expect } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { StreamLimitExceededError } from '@/llm/streamLimits';
import { Providers } from '@/common';
import { StandardGraph } from '../Graph';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

const makeGraph = (): StandardGraph =>
  new StandardGraph({
    runId: 'run_1',
    agents: [makeAgent('agent')],
  });

const makeTrip = (): StreamLimitExceededError =>
  new StreamLimitExceededError({
    kind: 'tool_call_args',
    limit: 10,
    observed: 11,
    toolName: 'db_query',
  });

describe('run breaker lifecycle', () => {
  it('replaces the breaker at every run start, even when un-aborted', () => {
    const graph = makeGraph();
    const previous = graph.breakerAbort;
    expect(previous.signal.aborted).toBe(false);

    graph.resetValues();

    expect(graph.breakerAbort).not.toBe(previous);
    /** A straggler from the failed run trips the controller it captured;
     * the run starting now must not observe it. */
    previous.abort(makeTrip());
    expect(graph.breakerAbort.signal.aborted).toBe(false);
  });

  it('rejects a model node at entry when the breaker has already tripped', async () => {
    const graph = makeGraph();
    const trip = makeTrip();
    graph.breakerAbort.abort(trip);

    const node = graph.createCallModel('agent');
    await expect(
      node(
        { messages: [] } as unknown as t.AgentSubgraphState,
        {} as RunnableConfig
      )
    ).rejects.toBe(trip);
  });

  it('lets a model node proceed past entry when the breaker is live', async () => {
    const graph = makeGraph();
    const node = graph.createCallModel('agent');
    /** With a live breaker the entry guard must not fire; the node then
     * fails later, on the missing config, proving it got past the guard. */
    await expect(
      node(
        { messages: [] } as unknown as t.AgentSubgraphState,
        undefined
      )
    ).rejects.toThrow('No config provided');
  });
});
