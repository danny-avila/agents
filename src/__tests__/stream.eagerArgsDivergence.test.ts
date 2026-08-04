/**
 * Regression tests for danny-avila/LibreChat#14371: eager tool execution
 * looped to the recursion limit for tools with large repetitive arguments.
 *
 * The eager prestart accumulator reconciles provider quirks with lossy
 * heuristics (repeat-fragment dedupe, overlap merge) that can swallow
 * legitimately repetitive payload fragments (SQL/code). The prestarted args
 * then diverged from the canonical LangChain `tool_call_chunks` accumulation
 * that materializes the final request, ToolNode's guard errored with "Tool
 * call changed after eager execution started", and the model's retry
 * re-prestarted and re-diverged — burning the whole recursion limit.
 *
 * Fixed two ways:
 * 1. Seal-time verification: prestart only when the heuristic accumulation is
 *    confirmed to match the canonical verbatim concatenation
 *    (`getStreamedReadyToolCalls` in src/stream.ts).
 * 2. Circuit breaker: if the "changed after eager execution" guard still
 *    fires, the tool name is suppressed from eager prestart for the rest of
 *    the run, so a retry executes normally and the loop is structurally
 *    impossible (ToolNode.takeMatchingEagerEventExecution +
 *    isEagerExecutionExcludedTool).
 */
import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  ToolMessage,
} from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  STREAM_LIMIT_EPOCH_KEY,
  StreamLimitExceededError,
} from '@/llm/streamLimits';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { ToolNode } from '@/tools/ToolNode';
import { HandlerRegistry } from '@/events';
import * as events from '@/utils/events';

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
    breakerAbort: new AbortController(),
    eagerEventToolExecution: { enabled: true },
    eagerEventToolExecutions: new Map(),
    eagerEventToolUsageCount: eagerUsageCount,
    getEagerEventToolUsageCount: jest.fn(() => eagerUsageCount),
    eagerEventToolCallChunks: new Map(),
    eagerEventToolSuppressions: new Set<string>(),
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
        provider: Providers.ANTHROPIC,
        reasoningKey: 'reasoning',
        toolDefinitions: [{ name: 'db_query' }, { name: 'stock' }],
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
        for (const toolCall of (details as t.ToolCallsDetails).tool_calls ??
          []) {
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

function createDummyTool(name: string): StructuredToolInterface {
  return tool(async () => 'direct should not run', {
    name,
    description: 'dummy',
    schema: z.object({ sql: z.string() }),
  }) as unknown as StructuredToolInterface;
}

function installToolExecuteResponder(): {
  toolExecuteCalls: t.ToolExecuteBatchRequest[];
  } {
  const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data): Promise<void> => {
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
    });
  return { toolExecuteCalls };
}

/**
 * A realistic Anthropic input_json_delta fragment sequence for a repetitive
 * payload: the model legitimately writes the same SQL statement three times
 * and the provider splits deltas on statement boundaries. The 2nd occurrence
 * collides with the overlap-merge heuristic (the accumulator ends with an
 * >=8-char prefix of it) and the 3rd with the repeat-fragment dedupe.
 */
const STATEMENT = 'INSERT INTO t VALUES (1);';
const REPETITIVE_FRAGMENTS = [
  `{"sql":"${STATEMENT}`,
  STATEMENT,
  STATEMENT,
  '"}',
];
const CANONICAL_SQL = `${STATEMENT}${STATEMENT}${STATEMENT}`;

function toToolCallChunks(
  callId: string,
  name: string,
  fragments: string[]
): Array<Record<string, unknown>> {
  return fragments.map((args, i) =>
    i === 0 ? { id: callId, name, args, index: 0 } : { args, index: 0 }
  );
}

async function streamChunks(args: {
  handler: ChatModelStreamHandler;
  graph: StandardGraph;
  metadata: Record<string, unknown>;
  toolCallChunks: Array<Record<string, unknown>>;
}): Promise<void> {
  const { handler, graph, metadata, toolCallChunks } = args;
  for (const toolCallChunk of toolCallChunks) {
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [toolCallChunk],
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
  }
}

/** Seal index 0 the way Anthropic does: the next tool-use block begins. */
async function streamNextToolIndex(args: {
  handler: ChatModelStreamHandler;
  graph: StandardGraph;
  metadata: Record<string, unknown>;
  callId: string;
}): Promise<void> {
  const { handler, graph, metadata, callId } = args;
  await handler.handle(
    GraphEvents.CHAT_MODEL_STREAM,
    {
      chunk: {
        content: '',
        tool_call_chunks: [
          { id: callId, name: 'stock', args: '{"ticker":"C', index: 1 },
        ],
      } as unknown as t.StreamChunk,
    },
    metadata,
    graph
  );
}

/**
 * The canonical accumulation LangChain performs in the model node: concat all
 * AIMessageChunks; the final message's tool_calls carry the args ToolNode
 * receives as the request.
 */
function canonicalToolCall(
  callId: string,
  name: string,
  fragments: string[]
): { id: string; name: string; args: Record<string, unknown> } {
  let accumulated: AIMessageChunk | undefined;
  for (const toolCallChunk of toToolCallChunks(callId, name, fragments)) {
    const chunk = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        { ...toolCallChunk, type: 'tool_call_chunk' },
      ] as AIMessageChunk['tool_call_chunks'],
    });
    accumulated = accumulated == null ? chunk : accumulated.concat(chunk);
  }
  const toolCall = accumulated?.tool_calls?.[0];
  if (toolCall?.id == null) {
    throw new Error('canonical accumulation produced no tool call');
  }
  return {
    id: toolCall.id,
    name: toolCall.name,
    args: toolCall.args as Record<string, unknown>,
  };
}

describe('eager args divergence (LibreChat#14371)', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('sanity: the canonical LangChain concat preserves every repeated fragment', () => {
    const canonical = canonicalToolCall(
      'call_1',
      'db_query',
      REPETITIVE_FRAGMENTS
    );
    expect(canonical.args).toEqual({ sql: CANONICAL_SQL });
  });

  it('does not prestart when overlap merge + repeat dedupe diverge from the canonical accumulation', async () => {
    const graph = createGraph();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await streamChunks({
      handler,
      graph,
      metadata,
      toolCallChunks: toToolCallChunks(
        'call_1',
        'db_query',
        REPETITIVE_FRAGMENTS
      ),
    });
    await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

    // Pre-fix: the seal prestarted `{"sql":"<one statement>"}` here — args
    // the model never asked for — and the run then looped on the "changed
    // after eager execution started" guard. Now the unconfirmed snapshot is
    // skipped and the call falls through to normal execution.
    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_1')).toBe(false);
  });

  it('does not prestart when the overlap heuristic alone swallows legitimate payload', async () => {
    // existing='{"code":"aaaaaaaaaa' ends with incoming.slice(0, 10), so the
    // merge drops 10 legitimate chars; canonical has 20 a's.
    const fragments = ['{"code":"aaaaaaaaaa', 'aaaaaaaaaab"}'];
    const graph = createGraph();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await streamChunks({
      handler,
      graph,
      metadata,
      toolCallChunks: toToolCallChunks('call_1', 'db_query', fragments),
    });
    await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_1')).toBe(false);
  });

  it('does not prestart when the repeat-fragment dedupe alone drops legitimate payload', async () => {
    // 'AB' fragments are too short for the overlap merge (< 8 chars), so only
    // isRepeatedObservedFragment fires — the 3rd fragment is dropped.
    const fragments = ['{"sql":"AB', 'AB', 'AB', '"}'];
    const graph = createGraph();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await streamChunks({
      handler,
      graph,
      metadata,
      toolCallChunks: toToolCallChunks('call_1', 'db_query', fragments),
    });
    await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_1')).toBe(false);
  });

  it('does not treat a pure-signal adapter seal as an args restatement (Bedrock)', async () => {
    // Bedrock's contentBlockStop seal chunk carries `args: ''` — a pure
    // signal, not a restatement. A repeated complete-JSON fragment leaves
    // the heuristic accumulator with lastArgsFragment === argsText while the
    // canonical concatenation differs; the seal must NOT bless that state as
    // authoritative (Codex P1 on #368).
    const bedrockMetadata = {
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
    };
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.BEDROCK,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'db_query' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    const fragment = '{"sql":"SELECT 1;"}';
    for (const toolCallChunk of [
      { id: 'call_1', name: 'db_query', args: fragment, index: 0 },
      { args: fragment, index: 0 },
    ]) {
      await handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        {
          chunk: {
            content: '',
            tool_call_chunks: [toolCallChunk],
            response_metadata: bedrockMetadata,
          } as unknown as t.StreamChunk,
        },
        metadata,
        graph
      );
    }
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [{ args: '', index: 0 }],
          response_metadata: {
            ...bedrockMetadata,
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: {
              kind: 'single',
              index: 0,
            },
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.has('call_1')).toBe(false);
  });

  it('still prestarts adapter-sealed calls whose seal chunk restates the args (OpenAI Responses contract)', async () => {
    const graph = createGraph({
      getAgentContext: jest.fn(
        (): Partial<AgentContext> => ({
          provider: Providers.OPENAI,
          reasoningKey: 'reasoning_content',
          toolDefinitions: [{ name: 'db_query' }],
          graphTools: [],
          agentId: 'agent_1',
        })
      ) as unknown as StandardGraph['getAgentContext'],
    });
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };
    const adapterMetadata = {
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
    };

    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            { id: 'call_1', name: 'db_query', args: '{"sql":"SELE', index: 0 },
          ],
          response_metadata: adapterMetadata,
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );
    // The `arguments.done` seal chunk restates the complete args.
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_call_chunks: [
            { id: 'call_1', args: '{"sql":"SELECT 1;"}', index: 0 },
          ],
          response_metadata: {
            ...adapterMetadata,
            [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: {
              kind: 'single',
              id: 'call_1',
              index: 0,
            },
          },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_1',
      name: 'db_query',
      args: { sql: 'SELECT 1;' },
    });
  });

  it('still prestarts sealed calls whose fragments accumulate cleanly', async () => {
    const fragments = ['{"sql":"SELECT ', '1;', '"}'];
    const graph = createGraph();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await streamChunks({
      handler,
      graph,
      metadata,
      toolCallChunks: toToolCallChunks('call_1', 'db_query', fragments),
    });
    await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_1',
      name: 'db_query',
      args: { sql: 'SELECT 1;' },
    });
  });

  it('does not prestart eager tools for stale-epoch events', async () => {
    const graph = createGraph({ breakerEpoch: 5 } as Partial<StandardGraph>);
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = {
      langgraph_node: 'agent',
      [STREAM_LIMIT_EPOCH_KEY]: 4,
    };

    /** A final tool-call chunk from a failed run handled after resetValues
     * advanced the epoch: dropped outright, so the dead run's call cannot
     * dispatch a host tool into the run now using the live controller. */
    await handler.handle(
      GraphEvents.CHAT_MODEL_STREAM,
      {
        chunk: {
          content: '',
          tool_calls: [{ id: 'call_1', name: 'db_query', args: { sql: 'SELECT 1;' } }],
          response_metadata: { finish_reason: 'tool_calls' },
        } as unknown as t.StreamChunk,
      },
      metadata,
      graph
    );

    expect(toolExecuteCalls).toHaveLength(0);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
  });

  it('does not prestart eager tools when the breaker trips during tool-call handling', async () => {
    const graph = createGraph();
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    const originalDispatch = graph.dispatchRunStep;
    graph.dispatchRunStep = (async (
      stepKey: string,
      details: unknown
    ): Promise<string> => {
      const stepId = await (
        originalDispatch as unknown as (
          stepKey: string,
          details: unknown
        ) => Promise<string>
      )(stepKey, details);
      graph.breakerAbort.abort(trip);
      return stepId;
    }) as typeof graph.dispatchRunStep;
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    /** The trip lands while handleToolCalls awaits the run-step dispatch —
     * after the queued-event guard already passed. The recheck before the
     * eager path must stop the prestart. */
    await expect(
      handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        {
          chunk: {
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'db_query', args: { sql: 'SELECT 1;' } },
            ],
            response_metadata: { finish_reason: 'tool_calls' },
          } as unknown as t.StreamChunk,
        },
        metadata,
        graph
      )
    ).rejects.toBe(trip);
    expect(toolExecuteCalls).toHaveLength(0);
  });

  it('sends a breaker-composed abort signal with eager prestart requests', async () => {
    const graph = createGraph();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const handler = new ChatModelStreamHandler();
    const metadata = { langgraph_node: 'agent' };

    await streamChunks({
      handler,
      graph,
      metadata,
      toolCallChunks: toToolCallChunks('call_1', 'db_query', [
        '{"sql":"SELECT 1;"}',
      ]),
    });
    await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

    expect(toolExecuteCalls).toHaveLength(1);
    const { signal } = toolExecuteCalls[0];
    expect(signal).toBeDefined();
    expect(signal?.aborted).toBe(false);

    graph.breakerAbort.abort(new Error('stream limit breach'));
    expect(signal?.aborted).toBe(true);
  });

  it('the retry path no longer loops: every round executes normally with canonical args', async () => {
    const metadata = { langgraph_node: 'agent' };
    const guardErrors: string[] = [];
    const normalExecutions: Array<Record<string, unknown>> = [];

    // Simulate the agent loop that previously burned the recursion limit:
    // each round the model streams the identical repetitive tool call and
    // ToolNode materializes the canonical request.
    for (let round = 0; round < 3; round += 1) {
      const graph = createGraph();
      const { toolExecuteCalls } = installToolExecuteResponder();
      const handler = new ChatModelStreamHandler();
      const callId = `call_round_${round}`;
      await streamChunks({
        handler,
        graph,
        metadata,
        toolCallChunks: toToolCallChunks(
          callId,
          'db_query',
          REPETITIVE_FRAGMENTS
        ),
      });
      await streamNextToolIndex({
        handler,
        graph,
        metadata,
        callId: `${callId}_next`,
      });

      const canonical = canonicalToolCall(
        callId,
        'db_query',
        REPETITIVE_FRAGMENTS
      );
      const toolNode = new ToolNode({
        tools: [createDummyTool('db_query')],
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: graph.eagerEventToolExecutions,
        eagerEventToolUsageCount: graph.getEagerEventToolUsageCount(),
        eagerEventToolSuppressions: graph.eagerEventToolSuppressions,
        toolCallStepIds: graph.toolCallStepIds,
      });
      const result = (await toolNode.invoke({
        messages: [new AIMessage({ content: '', tool_calls: [canonical] })],
      })) as { messages: ToolMessage[] };

      const toolMessage = result.messages.find(
        (message) => message.tool_call_id === callId
      );
      if (
        typeof toolMessage?.content === 'string' &&
        toolMessage.content.includes('changed after eager execution')
      ) {
        guardErrors.push(toolMessage.content);
      }
      const dispatched = toolExecuteCalls
        .flatMap((batch) => batch.toolCalls)
        .find((call) => call.id === callId);
      if (dispatched != null) {
        normalExecutions.push(dispatched.args);
      }
      jest.restoreAllMocks();
    }

    // Pre-fix: 3/3 rounds errored with the guard and nothing ever executed
    // with the args the model requested.
    expect(guardErrors).toHaveLength(0);
    expect(normalExecutions).toHaveLength(3);
    for (const args of normalExecutions) {
      expect(args).toEqual({ sql: CANONICAL_SQL });
    }
  });

  describe('circuit breaker', () => {
    it('suppresses eager prestart for a tool after the changed-args guard fires', async () => {
      jest.spyOn(console, 'warn').mockImplementation(() => undefined);
      const { toolExecuteCalls } = installToolExecuteResponder();
      const suppressions = new Set<string>();
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>();
      const request: t.ToolCallRequest = {
        id: 'call_1',
        name: 'db_query',
        args: { sql: 'diverged' },
        stepId: 'step_1',
        turn: 0,
      };
      eagerExecutions.set('call_1', {
        toolCallId: 'call_1',
        toolName: 'db_query',
        args: { sql: 'diverged' },
        request,
        promise: Promise.resolve({
          results: [
            { toolCallId: 'call_1', status: 'success', content: 'eager' },
          ],
        }),
      });

      const toolNode = new ToolNode({
        tools: [createDummyTool('db_query')],
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
        eagerEventToolSuppressions: suppressions,
        toolCallStepIds: new Map([['call_1', 'step_1']]),
      });
      const result = (await toolNode.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'db_query', args: { sql: CANONICAL_SQL } },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toContain(
        'changed after eager execution'
      );
      expect(suppressions.has('db_query')).toBe(true);
      expect(toolExecuteCalls).toHaveLength(0);
    });

    it('suppresses the eagerly executed name too when the identity mismatches', async () => {
      // If the stream prestarts name A but the final request materializes as
      // name B for the same call id, suppressing only B would let every
      // retry prestart A again — repeating A's side effects while the run
      // loops (Codex P2 on #368).
      jest.spyOn(console, 'warn').mockImplementation(() => undefined);
      installToolExecuteResponder();
      const suppressions = new Set<string>();
      const eagerExecutions = new Map<string, t.EagerEventToolExecution>();
      const request: t.ToolCallRequest = {
        id: 'call_1',
        name: 'tool_a',
        args: { sql: 'SELECT 1;' },
        stepId: 'step_1',
        turn: 0,
      };
      eagerExecutions.set('call_1', {
        toolCallId: 'call_1',
        toolName: 'tool_a',
        args: { sql: 'SELECT 1;' },
        request,
        promise: Promise.resolve({
          results: [
            { toolCallId: 'call_1', status: 'success', content: 'eager' },
          ],
        }),
      });

      const toolNode = new ToolNode({
        tools: [createDummyTool('tool_a'), createDummyTool('tool_b')],
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: eagerExecutions,
        eagerEventToolSuppressions: suppressions,
        toolCallStepIds: new Map([['call_1', 'step_1']]),
      });
      const result = (await toolNode.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'tool_b', args: { sql: 'SELECT 1;' } },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toContain(
        'changed after eager execution'
      );
      expect(suppressions.has('tool_b')).toBe(true);
      expect(suppressions.has('tool_a')).toBe(true);
    });

    it('stops prestarting a suppressed tool while siblings still prestart', async () => {
      const graph = createGraph();
      (graph.eagerEventToolSuppressions as Set<string>).add('db_query');
      const { toolExecuteCalls } = installToolExecuteResponder();
      const handler = new ChatModelStreamHandler();
      const metadata = { langgraph_node: 'agent' };

      // A clean, confirmable db_query stream: without the suppression this
      // would prestart (see "still prestarts sealed calls" above).
      await streamChunks({
        handler,
        graph,
        metadata,
        toolCallChunks: toToolCallChunks('call_1', 'db_query', [
          '{"sql":"SELECT 1;"}',
        ]),
      });
      await streamNextToolIndex({ handler, graph, metadata, callId: 'call_2' });

      expect(toolExecuteCalls).toHaveLength(0);
      expect(graph.eagerEventToolExecutions.has('call_1')).toBe(false);

      // The sibling tool (index 1) is not suppressed: sealing it via the
      // final tool-call signal still prestarts it.
      await handler.handle(
        GraphEvents.CHAT_MODEL_STREAM,
        {
          chunk: {
            content: '',
            tool_call_chunks: [{ args: 'H"}', index: 1 }],
            response_metadata: { finish_reason: 'tool_calls' },
          } as unknown as t.StreamChunk,
        },
        metadata,
        graph
      );

      expect(toolExecuteCalls).toHaveLength(1);
      expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
        id: 'call_2',
        name: 'stock',
        args: { ticker: 'CH' },
      });
    });

    it('lets the retry execute normally after suppression', async () => {
      const { toolExecuteCalls } = installToolExecuteResponder();
      const suppressions = new Set<string>(['db_query']);
      const toolNode = new ToolNode({
        tools: [createDummyTool('db_query')],
        eventDrivenMode: true,
        eagerEventToolExecution: { enabled: true },
        eagerEventToolExecutions: new Map(),
        eagerEventToolSuppressions: suppressions,
        toolCallStepIds: new Map([['call_retry', 'step_1']]),
      });

      const result = (await toolNode.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'call_retry',
                name: 'db_query',
                args: { sql: CANONICAL_SQL },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe('ok db_query');
      expect(toolExecuteCalls).toHaveLength(1);
      expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
        id: 'call_retry',
        name: 'db_query',
        args: { sql: CANONICAL_SQL },
      });
    });
  });
});
