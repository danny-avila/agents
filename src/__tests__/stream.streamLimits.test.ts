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
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
  STREAM_LIMIT_REDISPATCH_KEY,
  enforceStreamLimitsForWireChunk,
  resolveGenerationKey,
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

const generation = (step: number): Record<string, unknown> => ({
  langgraph_checkpoint_ns: '',
  langgraph_node: 'agent',
  langgraph_step: step,
});

async function streamEvent(args: {
  handler: ChatModelStreamHandler;
  graph: StandardGraph;
  chunk: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}): Promise<void> {
  const { handler, graph, chunk, metadata } = args;
  await handler.handle(
    GraphEvents.CHAT_MODEL_STREAM,
    { chunk: chunk as unknown as t.StreamChunk },
    metadata ?? {},
    graph
  );
}

async function streamToolCallChunks(args: {
  handler: ChatModelStreamHandler;
  graph: StandardGraph;
  chunks: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
}): Promise<void> {
  const { handler, graph, chunks, metadata } = args;
  for (const toolCallChunk of chunks) {
    await streamEvent({
      handler,
      graph,
      metadata,
      chunk: { content: '', tool_call_chunks: [toolCallChunk] },
    });
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

    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_call_chunks: [
          { id: 'call_1', name: 'first', args: 'a'.repeat(60), index: 0 },
          { id: 'call_2', name: 'second', args: 'b'.repeat(60), index: 1 },
        ],
      },
    });

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

  it('enforces the cap before a complete arrival-sealed call can dispatch', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const oversized = JSON.stringify({ payload: 'x'.repeat(200) });

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_calls: [
            { id: 'call_1', name: 'side_effect', args: { payload: 'x'.repeat(200) } },
          ],
          tool_call_chunks: [
            { id: 'call_1', name: 'side_effect', args: oversized, index: 0 },
          ],
        },
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'side_effect' });

    const dispatchStep = graph.dispatchRunStep as jest.Mock;
    expect(dispatchStep).not.toHaveBeenCalled();
  });

  it('enforces the cap for chunks that carry no numeric index', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'no_index_tool', args: 'x'.repeat(80) }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ id: 'call_1', args: 'x'.repeat(21) }],
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'no_index_tool' });
  });

  it('does not double-count OpenAI Responses seal restatements', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
    });
    const fullArgs = 'x'.repeat(40);

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [
        { args: fullArgs.slice(0, 20), index: 0 },
        { args: fullArgs.slice(20), index: 0 },
      ],
    });
    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_call_chunks: [{ name: 'writer', args: fullArgs, index: 0 }],
        response_metadata: {
          [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
            OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
          [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 0 },
        },
      },
    });

    const key = `${resolveGenerationKey({})}:0`;
    expect(graph.streamedToolCallArgTallies.has(key)).toBe(false);
    expect(graph.streamedToolCallArgTallies.size).toBe(0);
  });

  it('keeps parallel anonymous arrival-sealed calls on separate budgets (Google)', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 64 }),
    });

    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_call_chunks: [
          { name: 'search_a', args: 'x'.repeat(40) },
          { name: 'search_b', args: 'y'.repeat(40) },
        ],
        response_metadata: {
          [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
        },
      },
    });
    expect(graph.streamedToolCallArgTallies.size).toBe(0);

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_call_chunks: [{ name: 'search_c', args: 'z'.repeat(70) }],
          response_metadata: {
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
          },
        },
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'search_c' });
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

describe('per-tool argument byte overrides', () => {
  it('raises the cap for the named tool without loosening others', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 100,
        maxToolCallArgBytesByTool: { create_file: 1_000 },
      }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [
        { id: 'call_1', name: 'create_file', args: 'x'.repeat(500), index: 0 },
      ],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [
          { id: 'call_2', name: 'query_database', args: 'y'.repeat(101), index: 1 },
        ],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 100,
      toolName: 'query_database',
    });
  });

  it('trips the named tool at its own lower cap', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 1_000,
        maxToolCallArgBytesByTool: { chatty_tool: 50 },
      }),
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [
          { id: 'call_1', name: 'chatty_tool', args: 'x'.repeat(51), index: 0 },
        ],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 50,
      toolName: 'chatty_tool',
    });
  });

  it('enforces a per-tool cap even when the global cap is disabled', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 0,
        maxToolCallArgBytesByTool: { create_file: 100 },
      }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [
        { id: 'call_1', name: 'unbounded_tool', args: 'x'.repeat(500), index: 0 },
      ],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [
          { id: 'call_2', name: 'create_file', args: 'y'.repeat(101), index: 1 },
        ],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 100,
      toolName: 'create_file',
    });
  });

  it('applies overrides to arrival-sealed complete calls', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 64,
        maxToolCallArgBytesByTool: { create_file: 1_000 },
      }),
    });

    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_call_chunks: [{ name: 'create_file', args: 'x'.repeat(500) }],
        response_metadata: {
          [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
        },
      },
    });

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_call_chunks: [{ name: 'search_c', args: 'z'.repeat(70) }],
          response_metadata: {
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
          },
        },
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'search_c' });
  });

  it('re-judges tallied bytes when the tool name arrives on a later empty chunk', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 1_000,
        maxToolCallArgBytesByTool: { capped_tool: 50 },
      }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ args: 'x'.repeat(60), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ id: 'call_1', name: 'capped_tool', args: '', index: 0 }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 50,
      observed: 60,
      toolName: 'capped_tool',
    });
  });

  it('honors an override for a tool named __proto__', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 100,
        maxToolCallArgBytesByTool: JSON.parse('{"__proto__": 1000}') as Record<
          string,
          number
        >,
      }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: '__proto__', args: 'x'.repeat(500), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'x'.repeat(501), index: 0 }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 1_000,
      toolName: '__proto__',
    });
  });

  it('keeps charging a call whose later deltas omit both id and index', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60) }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a'.repeat(60) }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 100,
      observed: 120,
      toolName: 'writer',
    });
  });

  it('does not charge argument bytes for marked OpenRouter redispatches', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    });
    await streamEvent({
      handler,
      graph,
      metadata: { [STREAM_LIMIT_REDISPATCH_KEY]: true },
      chunk: {
        content: '',
        tool_call_chunks: [{ args: 'a'.repeat(60), index: 0 }],
      },
    });
    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ args: 'a'.repeat(40), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a', index: 0 }],
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', observed: 101 });
  });

  it('judges client tool chunks arriving with a server-tool result before the early return', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    (graph.getAgentContext as jest.Mock).mockReturnValue({
      provider: Providers.ANTHROPIC,
      reasoningKey: 'reasoning_content',
      toolDefinitions: [],
      graphTools: [],
      agentId: 'agent_1',
    });
    await graph.dispatchRunStep('step-key', {
      type: StepTypes.TOOL_CALLS,
      tool_calls: [{ id: 'srvtoolu_1', name: 'web_search', args: {} }],
    });
    graph.toolCallStepIds.set('srvtoolu_1', 'step_1');

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: [
            { type: 'web_search_tool_result', tool_use_id: 'srvtoolu_1', content: [] },
          ],
          tool_call_chunks: [
            { id: 'call_big', name: 'side_effect', args: 'x'.repeat(200), index: 0 },
          ],
        },
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'side_effect' });
  });

  it('judges complete parsed tool calls that arrive without raw chunks', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 100,
        maxToolCallArgBytesByTool: { create_file: 1_000 },
      }),
    });

    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_calls: [
          { id: 'call_1', name: 'create_file', args: { content: 'x'.repeat(500) } },
        ],
      },
    });

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_calls: [
            { id: 'call_2', name: 'side_effect', args: { payload: 'x'.repeat(200) } },
          ],
        },
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 100,
      toolName: 'side_effect',
    });
  });

  it('charges a chunk once when the handler echo wins the race', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const wireChunk = {
      content: '',
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    };

    await streamEvent({ handler, graph, chunk: wireChunk });
    enforceStreamLimitsForWireChunk({
      graph,
      metadata: {},
      chunk: wireChunk as never,
    });
    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ args: 'a'.repeat(40), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a', index: 0 }],
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', observed: 101 });
  });

  it('charges wire chunks synchronously and skips the handler echo', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const wireChunk = {
      content: '',
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    };

    enforceStreamLimitsForWireChunk({
      graph,
      metadata: {},
      chunk: wireChunk as never,
    });
    await streamEvent({ handler, graph, chunk: wireChunk });
    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ args: 'a'.repeat(40), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a', index: 0 }],
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', observed: 101 });
  });

  it('keeps charging a mutable chunk object re-yielded across emissions', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const reused: Record<string, unknown> = {
      content: '',
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    };

    await streamEvent({ handler, graph, chunk: reused });
    reused.tool_call_chunks = [{ args: 'a'.repeat(40), index: 0 }];
    await streamEvent({ handler, graph, chunk: reused });
    reused.tool_call_chunks = [{ args: 'a', index: 0 }];

    await expect(streamEvent({ handler, graph, chunk: reused })).rejects.toMatchObject({
      kind: 'tool_call_args',
      observed: 101,
    });
  });

  it('pairs producer and echo charges per emission for reused chunk objects', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const reused: Record<string, unknown> = {
      content: '',
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    };

    enforceStreamLimitsForWireChunk({ graph, metadata: {}, chunk: reused as never });
    await streamEvent({ handler, graph, chunk: reused });
    reused.tool_call_chunks = [{ args: 'a'.repeat(40), index: 0 }];
    enforceStreamLimitsForWireChunk({ graph, metadata: {}, chunk: reused as never });
    await streamEvent({ handler, graph, chunk: reused });
    reused.tool_call_chunks = [{ args: 'a', index: 0 }];

    let thrown: unknown;
    try {
      enforceStreamLimitsForWireChunk({ graph, metadata: {}, chunk: reused as never });
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toMatchObject({ kind: 'tool_call_args', observed: 101 });
  });

  it('keeps parallel index-less calls on distinct budgets when one seals', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_a', name: 'a_tool', args: 'a'.repeat(60) }],
    });
    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_b', name: 'b_tool', args: 'b'.repeat(60) }],
    });
    await streamEvent({
      handler,
      graph,
      chunk: {
        content: '',
        tool_call_chunks: [{ id: 'call_a', args: '' }],
        response_metadata: {
          [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', id: 'call_a' },
        },
      },
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'b'.repeat(41) }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      observed: 101,
      toolName: 'b_tool',
    });
  });

  it('scopes charge credits per generation for shared chunk objects', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });
    const shared: Record<string, unknown> = {
      content: '',
      tool_call_chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    };

    await streamEvent({ handler, graph, chunk: shared, metadata: generation(1) });
    enforceStreamLimitsForWireChunk({
      graph,
      metadata: generation(2),
      chunk: shared as never,
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a'.repeat(41), index: 0 }],
        metadata: generation(2),
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', observed: 101 });
  });

  it('judges parsed calls even when a raw chunk representation is present', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_call_chunks: [{ id: 'call_1', name: 'side_effect', args: '', index: 0 }],
          tool_calls: [
            { id: 'call_1', name: 'side_effect', args: { payload: 'x'.repeat(200) } },
          ],
        },
      })
    ).rejects.toMatchObject({ kind: 'tool_call_args', toolName: 'side_effect' });
  });

  it('keeps one budget when a call drops its index but keeps its id', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ id: 'call_1', args: 'a'.repeat(41) }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      observed: 101,
      toolName: 'writer',
    });
  });

  it('keeps one budget when a call drops both identifiers on later deltas', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60), index: 0 }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'a'.repeat(41) }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      observed: 101,
      toolName: 'writer',
    });
  });

  it('adopts the existing tally when a later delta adds an index', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    });

    await streamToolCallChunks({
      handler,
      graph,
      chunks: [{ id: 'call_1', name: 'writer', args: 'a'.repeat(60) }],
    });

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ id: 'call_1', args: 'a'.repeat(41), index: 0 }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      observed: 101,
      toolName: 'writer',
    });
  });

  it('names anonymous raw chunks from parsed calls so overrides apply', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 100,
        maxToolCallArgBytesByTool: { create_file: 1_000 },
      }),
    });

    await expect(
      streamEvent({
        handler,
        graph,
        chunk: {
          content: '',
          tool_call_chunks: [{ id: 'call_1', args: 'x'.repeat(500), index: 0 }],
          tool_calls: [
            { id: 'call_1', name: 'create_file', args: { content: 'x'.repeat(500) } },
          ],
        },
      })
    ).resolves.toBeUndefined();

    await expect(
      streamToolCallChunks({
        handler,
        graph,
        chunks: [{ args: 'x'.repeat(501), index: 0 }],
      })
    ).rejects.toMatchObject({
      kind: 'tool_call_args',
      limit: 1_000,
      toolName: 'create_file',
    });
  });

  it('normalizes override entries like the global field', () => {
    const resolved = resolveStreamLimits({
      maxToolCallArgBytes: 100,
      maxToolCallArgBytesByTool: {
        create_file: 131_072.9,
        disabled_tool: -5,
        unlimited_tool: Infinity,
        invalid_tool: Number.NaN,
        '': 42,
      },
    });

    expect(resolved.maxToolCallArgBytesByTool).toEqual({
      create_file: 131_072,
      disabled_tool: 0,
      unlimited_tool: 0,
    });
  });
});

describe('per-generation delta event circuit breaker', () => {
  it('is opt-in and counts every streamed chunk event, including empty ones', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 3 }),
    });

    const emptyChunkEvent = (): Promise<void> =>
      streamEvent({ handler, graph, chunk: { content: '' }, metadata: generation(1) });

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

  it('scopes the budget per model generation, surviving step-key forks', async () => {
    const handler = new ChatModelStreamHandler();
    let stepKey = 'reasoning-key';
    const graph = createGraph({
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
      getStepKey: jest.fn(() => stepKey) as unknown as StandardGraph['getStepKey'],
    });

    const emptyChunkEvent = (step: number): Promise<void> =>
      streamEvent({ handler, graph, chunk: { content: '' }, metadata: generation(step) });

    await emptyChunkEvent(1);
    stepKey = 'post-reasoning-key';
    await emptyChunkEvent(1);
    await expect(emptyChunkEvent(1)).rejects.toMatchObject({
      kind: 'delta_events',
      observed: 3,
    });

    await emptyChunkEvent(2);
    expect(graph.streamDeltaEventCounts.get('|agent|2|')).toBe(1);
  });

  it('never counts when left at the default (disabled)', async () => {
    const handler = new ChatModelStreamHandler();
    const graph = createGraph({ streamLimits: resolveStreamLimits() });

    for (let i = 0; i < 10; i++) {
      await streamEvent({ handler, graph, chunk: { content: '' }, metadata: generation(1) });
    }
    expect(graph.streamDeltaEventCounts).toBeUndefined();
  });
});
