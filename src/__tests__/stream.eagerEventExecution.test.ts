import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  Providers,
  StepTypes,
} from '@/common';
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
  const eagerUsageCount = new Map<string, number>();

  const graph = {
    config: {
      configurable: { user_id: 'user_1' },
      metadata: { run_id: 'run_1' },
    },
    eagerEventToolExecution: { enabled: true },
    eagerEventToolExecutions: new Map(),
    eagerEventToolUsageCount: eagerUsageCount,
    getEagerEventToolUsageCount: jest.fn(() => eagerUsageCount),
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

const finalToolCallResponseMetadata = { finish_reason: 'tool_calls' };

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
          response_metadata: finalToolCallResponseMetadata,
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

  it('does not prestart parseable tool calls before a final tool-call signal', async () => {
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
          tool_calls: [
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'N' },
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(false);

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
          response_metadata: finalToolCallResponseMetadata,
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

  it('prestarts multiple complete event-driven tool calls as one batch', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'calendar' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
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
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.name} result`,
          }))
        );
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
            {
              id: 'call_calendar',
              name: 'calendar',
              args: { date: 'today' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toHaveLength(2);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
      expect.objectContaining({
        id: 'call_calendar',
        name: 'calendar',
        args: { date: 'today' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
    ]);
    const weatherExecution = graph.eagerEventToolExecutions.get('call_weather');
    const calendarExecution = graph.eagerEventToolExecutions.get('call_calendar');
    expect(weatherExecution).toMatchObject({
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
    });
    expect(calendarExecution).toMatchObject({
      toolCallId: 'call_calendar',
      toolName: 'calendar',
      args: { date: 'today' },
    });
    expect(weatherExecution?.promise).toBe(calendarExecution?.promise);
  });

  it('assigns same-tool eager turns in model emission order', async () => {
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
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.args.city} weather`,
          }))
        );
      }
    );

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls.map((call) => call.turn)).toEqual([
      0, 1,
    ]);
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
    expect(graph.eagerEventToolExecutions.get('call_weather_1')?.request.turn)
      .toBe(0);
    expect(graph.eagerEventToolExecutions.get('call_weather_2')?.request.turn)
      .toBe(1);
  });

  it('scopes eager turn reservation by agent', async () => {
    const usageByAgent = new Map<string, Map<string, number>>();
    const getUsageCount = (agentId?: string): Map<string, number> => {
      const key = agentId ?? 'default';
      let usage = usageByAgent.get(key);
      if (usage == null) {
        usage = new Map<string, number>();
        usageByAgent.set(key, usage);
      }
      return usage;
    };
    const graph = createGraph({
      getEagerEventToolUsageCount: jest.fn(getUsageCount),
      getAgentContext: jest.fn(
        (metadata?: Record<string, unknown>): AgentContext => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }],
          graphTools: [],
          agentId:
            metadata?.langgraph_node === 'agent_2' ? 'agent_2' : 'agent_1',
        }) as unknown as AgentContext
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
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: 'ok',
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
          tool_calls: [
            {
              id: 'call_agent_1_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_1' },
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_agent_2_weather',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_2' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls.map((call) => call.toolCalls[0].turn)).toEqual([
      0, 0,
    ]);
    expect(usageByAgent.get('agent_1')?.get('weather')).toBe(1);
    expect(usageByAgent.get('agent_2')?.get('weather')).toBe(1);
    expect(graph.eagerEventToolExecutions.get('call_agent_1_weather')?.request.turn)
      .toBe(0);
    expect(graph.eagerEventToolExecutions.get('call_agent_2_weather')?.request.turn)
      .toBe(0);
  });

  it('skips eager for the whole batch if any call is not request-plannable', async () => {
    const graph = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve([]);
      }
    );

    await new ChatModelStreamHandler().handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [
            {
              id: 'call_weather_bad',
              name: 'weather',
              args: '{"city":',
            },
            {
              id: 'call_weather_good',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent' },
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolUsageCount.size).toBe(0);
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

  it('prestarts final tool calls even when the final chunk also has tool-call chunks', async () => {
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
          tool_call_chunks: [
            {
              index: 0,
              id: 'call_weather',
              name: 'weather',
              args: '{"city":"NYC"}',
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
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
      turn: 0,
    });
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
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
          response_metadata: finalToolCallResponseMetadata,
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
          response_metadata: finalToolCallResponseMetadata,
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
          response_metadata: finalToolCallResponseMetadata,
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

  it('preserves identical incremental argument fragments', async () => {
    const graph = createGraph();
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

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('oo');
  });

  it('deduplicates repeated observed multi-character fragments', async () => {
    const graph = createGraph();
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
              args: '{"titl',
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
              args: '{"titl',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"titl');
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
          response_metadata: finalToolCallResponseMetadata,
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

  it('merges overlapping cumulative streamed args without duplicating suffixes', async () => {
    const graph = createGraph();
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
              args: '{"tit',
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
              args: '{"title":"alpha"',
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
              args: 'le":"alpha","city":"NYC"}',
              index: 0,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"title":"alpha","city":"NYC"}');
  });

  it('preserves repeated deltas from a reused stream chunk object', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableToolChunk: {
      id?: string;
      name?: string;
      args: string;
      index: number;
    } = {
      id: 'call_repeat',
      name: 'weather',
      args: '{"word":"b',
      index: 0,
    };
    const reusableChunk = {
      content: '',
      tool_call_chunks: [reusableToolChunk],
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.id = undefined;
    reusableToolChunk.name = undefined;
    reusableToolChunk.args = 'o';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.args = 'k"}';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"word":"book"}');
  });

  it('preserves repeated text deltas from a reused stream chunk object', async () => {
    const dispatchMessageDelta = jest.fn<StandardGraph['dispatchMessageDelta']>(
      async () => undefined
    );
    const graph = createGraph({
      dispatchMessageDelta,
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          currentTokenType: ContentTypes.TEXT,
          toolDefinitions: [],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableChunk = { content: 'ha' } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(dispatchMessageDelta).toHaveBeenCalledTimes(2);
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      1,
      expect.stringMatching(/^step_/),
      { content: [{ type: ContentTypes.TEXT, text: 'ha' }] },
      metadata
    );
    expect(dispatchMessageDelta).toHaveBeenNthCalledWith(
      2,
      expect.stringMatching(/^step_/),
      { content: [{ type: ContentTypes.TEXT, text: 'ha' }] },
      metadata
    );
  });

  it('processes a reused chunk object when its streamed payload changes', async () => {
    const graph = createGraph();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const reusableToolChunk = {
      id: 'call_weather',
      name: 'weather',
      args: '{"city"',
      index: 0,
    };
    const reusableChunk = {
      content: '',
      tool_call_chunks: [reusableToolChunk],
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    reusableToolChunk.args = ':"NYC"}';

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: reusableChunk },
      metadata,
      graph
    );

    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 0))
        ?.argsText
    ).toBe('{"city":"NYC"}');
  });

  it('does not share chunk object de-duplication across graphs', async () => {
    const graphA = createGraph();
    const graphB = createGraph();
    const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(
      async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        toolExecuteCalls.push(batch);
        batch.resolve(
          batch.toolCalls.map((request) => ({
            toolCallId: request.id,
            status: 'success' as const,
            content: `${request.id} result`,
          }))
        );
      }
    );

    const handler = new ChatModelStreamHandler();
    const sharedChunk = {
      content: '',
      tool_calls: [
        {
          id: 'call_weather',
          name: 'weather',
          args: { city: 'NYC' },
        },
      ],
      response_metadata: finalToolCallResponseMetadata,
    } as unknown as t.StreamChunk;

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: sharedChunk },
      { langgraph_node: 'agent' },
      graphA
    );
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      { chunk: sharedChunk },
      { langgraph_node: 'agent' },
      graphB
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(graphA.eagerEventToolExecutions.has('call_weather')).toBe(true);
    expect(graphB.eagerEventToolExecutions.has('call_weather')).toBe(true);
  });

  it('prestarts a completed streamed tool when a later tool call begins', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
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
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather',
        name: 'weather',
        args: { city: 'NYC' },
        stepId: expect.stringMatching(/^step_/),
        turn: 0,
      }),
    ]);
    expect(graph.eagerEventToolExecutions.has('call_weather')).toBe(true);
    expect(graph.eagerEventToolExecutions.has('call_stock')).toBe(false);
    expect(graph.eagerEventToolCallChunks.has(chunkStateKey('step-key', 0))).toBe(
      false
    );
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('step-key', 1))?.argsText
    ).toBe('{"ticker":"C');

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'H"}',
              index: 1,
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
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: { ticker: 'CH' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
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
      stepId: expect.stringMatching(/^step_/),
      turn: 0,
    });
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
  });

  it('does not seal a streamed tool when the same chunk also carries its own index', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'weather' }, { name: 'stock' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
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
              args: '{"city":"NYC"}',
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
              args: '',
              index: 0,
            },
            {
              id: 'call_stock',
              name: 'stock',
              args: '{"ticker":"C',
              index: 1,
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
              args: 'H"}',
              index: 1,
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
  });

  it('preserves same-tool turns across per-call streamed eager starts', async () => {
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
            content: `ok ${call.args.city}`,
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
          tool_call_chunks: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: '{"city":"NYC"}',
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
              id: 'call_weather_2',
              name: 'weather',
              args: '{"city":"B',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather_1',
      name: 'weather',
      args: { city: 'NYC' },
      turn: 0,
    });

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            {
              args: 'oston"}',
              index: 1,
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
          tool_calls: [
            {
              id: 'call_weather_1',
              name: 'weather',
              args: { city: 'NYC' },
            },
            {
              id: 'call_weather_2',
              name: 'weather',
              args: { city: 'Boston' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(2);
    expect(toolExecuteCalls[1].toolCalls[0]).toMatchObject({
      id: 'call_weather_2',
      name: 'weather',
      args: { city: 'Boston' },
      turn: 1,
    });
    expect(graph.eagerEventToolUsageCount.get('weather')).toBe(2);
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
          response_metadata: finalToolCallResponseMetadata,
        } as unknown as t.StreamChunk,
      },
      { langgraph_node: 'agent_b' },
      graph
    );

    expect(graph.eagerEventToolCallChunks.has(chunkStateKey('agent_b', 0))).toBe(
      false
    );
    expect(
      graph.eagerEventToolCallChunks.get(chunkStateKey('agent_a', 0))?.argsText
    ).toBe('{"city":"NYC"}');

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
          response_metadata: finalToolCallResponseMetadata,
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
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
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

  it('does not prestart streamed local-engine direct coding tools', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: Constants.EXECUTE_CODE }, { name: 'weather' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const dispatchSpy = jest.spyOn(events, 'safeDispatchCustomEvent');
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
              args: '{"city":"NYC"}',
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
              id: 'call_code',
              name: Constants.EXECUTE_CODE,
              args: '{"code":"print(1)"}',
              index: 1,
            },
          ],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(dispatchSpy).not.toHaveBeenCalledWith(
      GraphEvents.ON_TOOL_EXECUTE,
      expect.anything(),
      expect.anything()
    );
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart event tools in a mixed direct-tool batch', async () => {
    const graph = createGraph({
      toolExecution: {
        engine: 'local',
      } as StandardGraph['toolExecution'],
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [
            { name: Constants.EXECUTE_CODE },
            { name: 'weather' },
          ],
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
            {
              id: 'call_weather',
              name: 'weather',
              args: { city: 'NYC' },
            },
          ],
          response_metadata: finalToolCallResponseMetadata,
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
          response_metadata: finalToolCallResponseMetadata,
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
