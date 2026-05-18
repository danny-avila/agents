import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import { Constants, GraphEvents, Providers, StepTypes } from '@/common';
import { HandlerRegistry } from '@/events';
import * as events from '@/utils/events';
import { ChatModelStreamHandler } from '@/stream';

function createGraph(overrides: Partial<StandardGraph> = {}): StandardGraph {
  const runSteps = new Map<string, t.RunStep>();
  const stepIdsByKey = new Map<string, string>();
  let stepCounter = 0;
  const handlerRegistry = new HandlerRegistry();
  handlerRegistry.register(GraphEvents.ON_TOOL_EXECUTE, {
    handle: async () => undefined,
  });

  const graph = {
    config: {
      configurable: { user_id: 'user_1' },
      metadata: { run_id: 'run_1' },
    },
    eagerEventToolExecution: { enabled: true },
    eagerEventToolExecutions: new Map(),
    eagerEventToolUsageCount: new Map(),
    eagerEventToolCallChunks: new Map(),
    handlerRegistry,
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
        toolDefinitions: [{ name: 'weather' }],
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
      if (
        (details as t.StepDetails).type === StepTypes.TOOL_CALLS &&
        Array.isArray((details as t.ToolCallsDetails).tool_calls)
      ) {
        for (const toolCall of (details as t.ToolCallsDetails).tool_calls ?? []) {
          if (toolCall.id != null && toolCall.id !== '') {
            graph.toolCallStepIds.set(toolCall.id, id);
          }
        }
      }
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

function chunkStateKey(stepKey: string, chunkKey: string | number): string {
  return `${stepKey}\u0000${String(chunkKey)}`;
}

describe('ChatModelStreamHandler eager event tool execution', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('prestarts a complete event-driven tool call from the stream', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(graph.eagerEventToolExecutions.get('call_weather')).toMatchObject({
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
    });
    expect(graph.toolCallStepIds.has('call_weather')).toBe(true);
  });

  it('records complete chunk-only tool calls after creating a tool step', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.toolCallStepIds.has('call_weather')).toBe(true);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"city":"NYC"}');
  });

  it('waits for final tool calls before prestarting streamed chunk calls', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    const handler = new ChatModelStreamHandler();

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: {},
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city"',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: ':"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: {},
            },
          ],
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"ticker":"CH"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('preserves repeated adjacent argument deltas', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_repeat',
            status: 'success',
            content: 'ok',
          },
        ]);
      }
    );

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: '{"word":"b',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'o',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'k"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"word":"book"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_repeat',
              name: 'weather',
              args: { word: 'book' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_repeat',
      name: 'weather',
      args: { word: 'book' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('does not prestart from cumulative streamed args before final tool calls', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_weather',
              name: 'weather',
              args: '{"ci',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"N',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: '{"city":"NYC","unit":"C"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"city":"NYC","unit":"C"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC', unit: 'C' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC', unit: 'C' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('ignores duplicate handling of the same stream chunk object', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const startChunk = {
      content: '',
      tool_call_chunks: [
        {
          id: 'call_weather',
          name: 'weather',
          args: '{"city"',
          index: 0,
        },
      ],
    } as unknown as t.StreamChunk;
    const finalChunk = {
      content: '',
      tool_call_chunks: [
        {
          args: ':"NYC"}',
          index: 0,
        },
      ],
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: startChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: startChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: finalChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: finalChunk },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"city":"NYC"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
  });

  it('prestarts a completed event tool before later parallel tool args finish', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      }
    );

    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_stock',
      name: 'stock',
      args: { ticker: 'CH' },
    });
  });

  it('scopes streamed chunk accumulation by step key', async () => {
    const graph = createGraph({
      getStepKey: jest.fn((metadata?: Record<string, unknown>) =>
        String(metadata?.langgraph_node ?? 'step-key')
      ),
    });
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((call) => ({
            toolCallId: call.id,
            status: 'success',
            content: `ok ${call.name}`,
          }))
        );
      }
    );

    const handler = new ChatModelStreamHandler();

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_agent_a',
              name: 'weather',
              args: '{"city":"N',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              id: 'call_agent_b',
              name: 'weather',
              args: '{"city":"S',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'F"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'YC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_a', 0))?.argsText
    ).toBe('{"city":"NYC"}');
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_b', 0))?.argsText
    ).toBe('{"city":"SF"}');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_b',
              name: 'weather',
              args: { city: 'SF' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_a',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_a' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_agent_b',
      name: 'weather',
      args: { city: 'SF' },
    });
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_agent_a',
      name: 'weather',
      args: { city: 'NYC' },
    });
  });

  it('does not prestart when batch-sensitive hooks are configured', async () => {
    const graph = createGraph({ hookRegistry: {} as StandardGraph['hookRegistry'] });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart local-engine direct coding tools', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: Constants.EXECUTE_CODE }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: { code: 'print(1)' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('continues eager turns after normal event-dispatch usage', async () => {
    const graph = createGraph();
    graph.eagerEventToolUsageCount.set('weather', 1);
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([
          {
            toolCallId: 'call_weather_2',
            status: 'success',
            content: 'sunny',
          },
        ]);
      }
    );

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather_2',
      name: 'weather',
      turn: 1,
    });
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
  });
});
