// src/graphs/__tests__/Graph.closeRunStep.test.ts
import type * as t from '@/types';
import { GraphEvents, StepTypes, Providers } from '@/common';
import { HandlerRegistry } from '@/events';
import { StandardGraph } from '../Graph';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

function createGraph(): {
  graph: StandardGraph;
  closed: t.RunStepClosedEvent[];
  } {
  const graph = new StandardGraph({
    runId: 'run_1',
    agents: [makeAgent('agent')],
  });
  const closed: t.RunStepClosedEvent[] = [];
  const registry = new HandlerRegistry();
  registry.register(GraphEvents.ON_RUN_STEP_CLOSED, {
    handle: (_event, data): void => {
      closed.push(data as t.RunStepClosedEvent);
    },
  });
  graph.handlerRegistry = registry;
  return { graph, closed };
}

function seedStep(
  graph: StandardGraph,
  id: string,
  type: StepTypes = StepTypes.MESSAGE_CREATION
): t.RunStep {
  const index = graph.contentData.length;
  const stepDetails: t.StepDetails =
    type === StepTypes.TOOL_CALLS
      ? { type: StepTypes.TOOL_CALLS, tool_calls: [] }
      : {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: `msg_${id}` },
      };
  const step: t.RunStep = {
    id,
    type,
    index,
    stepIndex: index,
    stepDetails,
    usage: null,
    created_at: 1_000,
    status: 'in_progress',
  };
  graph.contentData.push(step);
  graph.contentIndexMap.set(id, index);
  return step;
}

describe('StandardGraph.closeRunStep', () => {
  it('stamps the terminal status + timestamp and emits ON_RUN_STEP_CLOSED once', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a');

    const first = await graph.closeRunStep('step_a', 'completed', {
      at: 2_000,
    });
    expect(first).toBe(true);
    expect(step.status).toBe('completed');
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
    expect(closed[0]).toMatchObject({
      id: 'step_a',
      index: 0,
      type: StepTypes.MESSAGE_CREATION,
      status: 'completed',
      created_at: 1_000,
      closed_at: 2_000,
    });

    const second = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
    });
    expect(second).toBe(false);
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
  });

  it('keeps cancelled and failed immutable, even with restamp', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);

    await graph.closeRunStep('step_a', 'cancelled', { at: 2_000 });
    const restamped = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
      restamp: true,
    });
    expect(restamped).toBe(false);
    expect(step.status).toBe('cancelled');
    expect(step.cancelled_at).toBe(2_000);
    expect(step.completed_at).toBeUndefined();
    expect(closed).toHaveLength(1);
  });

  it('restamps a completed TOOL_CALLS step on a late completion', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);

    await graph.closeRunStep('step_a', 'completed', { at: 2_000 });
    const restamped = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
      restamp: true,
    });
    expect(restamped).toBe(true);
    expect(step.completed_at).toBe(3_000);
    expect(closed).toHaveLength(2);
    expect(closed[1].closed_at).toBe(3_000);
  });

  it('does not restamp completed MESSAGE_CREATION steps', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a');

    await graph.closeRunStep('step_a', 'completed', { at: 2_000 });
    const restamped = await graph.closeRunStep('step_a', 'completed', {
      at: 3_000,
      restamp: true,
    });
    expect(restamped).toBe(false);
    expect(step.completed_at).toBe(2_000);
    expect(closed).toHaveLength(1);
  });

  it('tolerates empty and unknown step ids', async () => {
    const { graph, closed } = createGraph();
    expect(await graph.closeRunStep('', 'completed')).toBe(false);
    expect(await graph.closeRunStep('step_missing', 'completed')).toBe(false);
    expect(closed).toHaveLength(0);
  });
});

describe('StandardGraph.recordStepCompletion', () => {
  it('closes a TOOL_CALLS step only after every registered call completes', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');
    graph.registerPendingToolCall('call_2', 'step_a');

    await graph.recordStepCompletion('step_a', 'call_1');
    expect(step.status).toBe('in_progress');
    expect(closed).toHaveLength(0);

    await graph.recordStepCompletion('step_a', 'call_2');
    expect(step.status).toBe('completed');
    expect(typeof step.completed_at).toBe('number');
    expect(closed).toHaveLength(1);
    expect(closed[0].status).toBe('completed');
  });

  it('closes steps without pending tracking on the first completion', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a');

    await graph.recordStepCompletion('step_a');
    expect(step.status).toBe('completed');
    expect(closed).toHaveLength(1);
  });

  it('absorbs duplicate completions without re-emitting', async () => {
    const { graph, closed } = createGraph();
    seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');

    await graph.recordStepCompletion('step_a', 'call_1');
    await graph.recordStepCompletion('step_a', 'call_1');
    expect(closed).toHaveLength(1);
  });

  it('restamps when a late-registered call completes after the step closed', async () => {
    const { graph, closed } = createGraph();
    const step = seedStep(graph, 'step_a', StepTypes.TOOL_CALLS);
    graph.registerPendingToolCall('call_1', 'step_a');

    await graph.recordStepCompletion('step_a', 'call_1');
    expect(closed).toHaveLength(1);
    const firstClosedAt = step.completed_at;

    graph.registerPendingToolCall('call_2', 'step_a');
    await graph.recordStepCompletion('step_a', 'call_2');
    expect(closed).toHaveLength(2);
    expect(step.completed_at).toBeGreaterThanOrEqual(firstClosedAt as number);
    expect(closed[1].status).toBe('completed');
  });
});

describe('StandardGraph.closeUnfinishedRunSteps', () => {
  it('closes only non-terminal steps with the sweep status', async () => {
    const { graph, closed } = createGraph();
    const done = seedStep(graph, 'step_done', StepTypes.TOOL_CALLS);
    const openMessage = seedStep(graph, 'step_msg');
    const openTool = seedStep(graph, 'step_tool', StepTypes.TOOL_CALLS);
    await graph.closeRunStep('step_done', 'completed', { at: 2_000 });
    closed.length = 0;

    await graph.closeUnfinishedRunSteps('cancelled', 5_000);

    expect(done.completed_at).toBe(2_000);
    expect(openMessage.status).toBe('cancelled');
    expect(openMessage.cancelled_at).toBe(5_000);
    expect(openTool.status).toBe('cancelled');
    expect(openTool.cancelled_at).toBe(5_000);
    expect(closed.map((event) => event.id)).toEqual(['step_msg', 'step_tool']);
    expect(closed.every((event) => event.status === 'cancelled')).toBe(true);
    expect(graph.pendingToolCallsByStep.size).toBe(0);
    expect(graph.openMessageStepByAgent.size).toBe(0);
  });

  it('stamps failed_at for failed sweeps', async () => {
    const { graph } = createGraph();
    const step = seedStep(graph, 'step_a');

    await graph.closeUnfinishedRunSteps('failed', 5_000);
    expect(step.status).toBe('failed');
    expect(step.failed_at).toBe(5_000);
    expect(step.cancelled_at).toBeUndefined();
    expect(step.completed_at).toBeUndefined();
  });
});
