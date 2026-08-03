/**
 * Regression tests for the streamed tool-argument circuit breaker.
 *
 * Incident: a model streamed a single malformed tool-call argument (a SQL
 * query) to 149,923 characters over 26 minutes, never completing it, until
 * the 64k output-token ceiling ended the run. Nothing bounded a single tool
 * call's streamed argument growth mid-flight.
 *
 * The guard lives on the SDK stream dispatch path: `ChatModelStreamHandler`
 * enforces `Graph.streamLimits` on every streamed chunk event and throws
 * `StreamLimitExceededError` out of the run's `streamEvents` loop, which
 * tears down the in-flight provider request (see the mid-flight halt notes
 * in `Run.processStream`).
 */
import { describe, it, expect, jest } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
  resolveStreamLimits,
} from '@/llm/streamLimits';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { HandlerRegistry } from '@/events';

function createGraph(overrides: Partial<StandardGraph> = {}): StandardGraph {
  const runSteps = new Map<string, t.RunStep>();
  const stepIdsByKey = new Map<string, string>();
  let stepCounter = 0;

  const graph = {
    config: {
      configurable: { user_id: 'user_1' },
      metadata: { run_id: 'run_1' },
    },
    eagerEventToolExecution: undefined,
    eagerEventToolExecutions: new Map(),
    eagerEventToolCallChunks: new Map(),
    eagerEventToolSuppressions: new Set<string>(),
    handlerRegistry: new HandlerRegistry(),
    hookRegistry: undefined,
    humanInTheLoop: undefined,
    toolOutputReferences: undefined,
    sessions: new Map(),
    toolCallStepIds: new Map(),
    messageIdsByStepKey: new Map(),
    messageStepHasToolCalls: new Map(),
    prelimMessageIdsByStepKey: new Map(),
    getAgentContext: jest.fn(
      (): Partial<AgentContext> => ({
        provider: Providers.OPENAI,
        reasoningKey: 'reasoning_content',
        toolDefinitions: [],
        graphTools: [],
        agentId: 'agent_1',
      })
    ),
    getStepKey: jest.fn(() => 'step-key'),
    getStepIdByKey: jest.fn((stepKey: string) => {
      const stepId = stepIdsByKey.get(stepKey);
      if (stepId == null) {
        throw new Error('no current step');
      }
      return stepId;
    }),
    getRunStep: jest.fn((stepId: string) => runSteps.get(stepId)),
    dispatchRunStep: jest.fn(async (stepKey: string, details: unknown) => {
      const id = `step_${++stepCounter}`;
      stepIdsByKey.set(stepKey, id);
      runSteps.set(id, {
        id,
        type: (details as { type: t.RunStep['type'] }).type,
        stepDetails: details as t.RunStep['stepDetails'],
      } as t.RunStep);
      return id;
    }),
    dispatchRunStepDelta: jest.fn(async () => undefined),
    ...overrides,
  };

  return graph as unknown as StandardGraph;
}

async function streamToolCallChunks(args: {
  handler: ChatModelStreamHandler;
  graph: StandardGraph;
  chunks: Array<Record<string, unknown>>;
}): Promise<void> {
  const { handler, graph, chunks } = args;
  for (const toolCallChunk of chunks) {
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [toolCallChunk],
        } as unknown as t.StreamChunk,
      },
      {},
      graph
    );
  }
}

describe('streamed tool-call argument circuit breaker', () => {
  it('aborts the stream once a single call exceeds the default 64 KiB', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({ streamLimits: resolveStreamLimits() });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [
        { id: 'call_1', name: 'query_database', args: '', index: 0 },
        ...Array.from({ length: 8 }, () => ({
          args: 'x'.repeat(8_192),
          index: 0,
        })),
      ],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'x', index: 0 }],
      })
    ).rejects.toMatchObject({
      name: 'StreamLimitExceededError',
      kind: 'tool_call_args',
      limit: DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
      observed: DEFAULT_MAX_TOOL_CALL_ARG_BYTES + 1,
      toolName: 'query_database',
    });

    const dispatchDelta = graph.dispatchRunStepDelta as jest.Mock;
    expect(dispatchDelta).toHaveBeenCalledTimes(9);
  });

  it('keeps parallel tool calls on independent budgets', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            { id: 'call_1', name: 'first', args: 'a'.repeat(60), index: 0 },
            { id: 'call_2', name: 'second', args: 'b'.repeat(60), index: 1 },
          ],
        } as unknown as t.StreamChunk,
      },
      {},
      graph
    );

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'b'.repeat(41), index: 1 }],
      })
    ).rejects.toThrow('(tool call: second)');
  });

  it('counts argument bytes, not characters', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 1_000 }),
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [
          { id: 'call_1', name: 'writer', args: '€'.repeat(300), index: 0 },
          { args: '€'.repeat(100), index: 0 },
        ],
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', observed: 1_200 });
  });

  it('streams unbounded when explicitly disabled', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 0 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [
        { id: 'call_1', name: 'query_database', args: '', index: 0 },
        ...Array.from({ length: 4 }, () => ({
          args: 'x'.repeat(65_536),
          index: 0,
        })),
      ],
    });

    const dispatchDelta = graph.dispatchRunStepDelta as jest.Mock;
    expect(dispatchDelta).toHaveBeenCalledTimes(5);
    expect(graph.streamedToolCallArgTallies).toBeUndefined();
  });

  it('creates the tool_calls run step before the breaker can trip', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'query_database', args: '{"q":', index: 0 }],
    });

    const dispatchStep = graph.dispatchRunStep as jest.Mock;
    const dispatchedTypes = dispatchStep.mock.calls.map(
      (call) => (call[1] as { type: string }).type
    );
    expect(dispatchedTypes).toContain(StepTypes.TOOL_CALLS);
  });
});

describe('per-turn delta event circuit breaker', () => {
  it('is opt-in and counts every streamed chunk event, including empty ones', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 3 }),
    });

    const emptyChunkEvent = (): Promise<void> =>
      handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        { chunk: { content: '' } as unknown as t.StreamChunk },
        {},
        graph
      );

    await emptyChunkEvent();
    await emptyChunkEvent();
    await emptyChunkEvent();
    await expect(emptyChunkEvent()).rejects.toMatchObject({
      name: 'StreamLimitExceededError',
      kind: 'delta_events',
      limit: 3,
      observed: 4,
    });
  });

  it('scopes the budget per generation turn', async () => {
    const handler = new ChatModelStreamHandler();
    let turn = 'turn-1';
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
      getStepKey: jest.fn(() => turn) as unknown as StandardGraph['getStepKey'],
    });

    const emptyChunkEvent = (): Promise<void> =>
      handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        { chunk: { content: '' } as unknown as t.StreamChunk },
        {},
        graph
      );

    await emptyChunkEvent();
    await emptyChunkEvent();
    turn = 'turn-2';
    await emptyChunkEvent();
    await emptyChunkEvent();
    await expect(emptyChunkEvent()).rejects.toMatchObject({
      kind: 'delta_events',
      observed: 3,
    });
  });

  it('never counts when left at the default (disabled)', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({ streamLimits: resolveStreamLimits() });

    for (let i = 0; i < 10; i++) {
      await handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        { chunk: { content: '' } as unknown as t.StreamChunk },
        {},
        graph
      );
    }
    expect(graph.streamDeltaEventCounts).toBeUndefined();
  });
});
