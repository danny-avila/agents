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

  it('keeps straggler accounting through cleanup and one reset, then sweeps it', () => {
    const graph = makeGraph();
    const epoch = graph.breakerEpoch;
    graph.streamedToolCallArgTallies.set('gen:i:0', { bytes: 42, epoch });
    graph.streamDeltaEventCounts.set('gen', { count: 7, epoch });
    graph.streamLimitChargeCredits = new WeakMap();

    /** Cleanup runs while sibling attempts can still be unwinding on the
     * retained breaker; clearing here would hand a cancellation-ignoring
     * provider's late chunks a fresh budget. */
    graph.clearHeavyState();
    expect(graph.streamedToolCallArgTallies.get('gen:i:0')?.bytes).toBe(42);
    expect(graph.streamDeltaEventCounts.get('gen')?.count).toBe(7);
    expect(graph.streamLimitChargeCredits).toBeDefined();

    /** Producer loops of straggling attempts sit OUTSIDE the consumer-only
     * epoch gate, so their entries must survive the next run start too —
     * one grace reset keeps them on their original budgets. */
    graph.resetValues();
    expect(graph.streamedToolCallArgTallies.get('gen:i:0')?.bytes).toBe(42);
    expect(graph.streamDeltaEventCounts.get('gen')?.count).toBe(7);
    expect(graph.streamLimitChargeCredits).toBeUndefined();

    /** A second reset sweeps entries from older epochs. */
    graph.resetValues();
    expect(graph.streamedToolCallArgTallies.size).toBe(0);
    expect(graph.streamDeltaEventCounts.size).toBe(0);
  });

  it('holds a post-reset producer straggler to its original byte budget', async () => {
    const { enforceStreamLimitsForWireChunk } = await import(
      '@/llm/streamLimits'
    );
    const graph = makeGraph();
    graph.streamLimits = {
      maxToolCallArgBytes: 100,
      maxDeltaEventsPerTurn: 0,
      hasEnforceableToolCallArgLimit: true,
    };
    const metadata = {
      langgraph_checkpoint_ns: '',
      langgraph_node: 'agent',
      langgraph_step: 1,
    };
    const chunkOf = (
      args: string
    ): { tool_call_chunks: Array<Record<string, unknown>> } => ({
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args, index: 0 }],
    });

    enforceStreamLimitsForWireChunk({
      graph,
      metadata,
      chunk: chunkOf('x'.repeat(60)) as Parameters<
        typeof enforceStreamLimitsForWireChunk
      >[0]['chunk'],
    });

    /** The next run starts while this producer is still draining. Its next
     * chunk must land on the ORIGINAL 60-byte tally and trip, not open a
     * fresh allowance. */
    graph.resetValues();
    expect(() =>
      enforceStreamLimitsForWireChunk({
        graph,
        metadata,
        chunk: chunkOf('x'.repeat(60)) as Parameters<
          typeof enforceStreamLimitsForWireChunk
        >[0]['chunk'],
      })
    ).toThrow(StreamLimitExceededError);
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
