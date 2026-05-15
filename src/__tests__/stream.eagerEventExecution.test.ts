import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { HandlerRegistry } from '@/events';
import * as events from '@/utils/events';
import { ChatModelStreamHandler } from '@/stream';

function createGraph(overrides: Partial<StandardGraph> = {}): StandardGraph {
  const runSteps = new Map<string, t.RunStep>();
  let currentStepId: string | undefined;
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
    getStepIdByKey: jest.fn(() => {
      if (currentStepId == null) {
        throw new Error('no current step');
      }
      return currentStepId;
    }),
    getRunStep: jest.fn((stepId: string) => runSteps.get(stepId)),
    dispatchRunStep: jest.fn(async (_stepKey: string, details: unknown) => {
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
      currentStepId = id;
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
});
