import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
} from '@langchain/langgraph';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import {
  describe,
  it,
  expect,
  jest,
  afterEach,
  beforeEach,
} from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type {
  PreToolUseHookOutput,
  PostToolUseHookOutput,
  PostToolUseFailureHookOutput,
  PostToolBatchEntry,
  PostToolBatchHookInput,
  PostToolBatchHookOutput,
  RunStartHookOutput,
  UserPromptSubmitHookOutput,
} from '@/hooks';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { Providers as providers, GraphEvents } from '@/common';
import { ToolNode } from '../ToolNode';

/**
 * Schema-only tool stub. ToolNode in event-driven mode uses the schema
 * for binding/discovery but routes execution through the host via
 * `ON_TOOL_EXECUTE`, so the actual `func` here is never called.
 */
function createSchemaStub(name: string): StructuredToolInterface {
  return tool(async () => 'unused', {
    name,
    description: 'schema-only stub; host executes via ON_TOOL_EXECUTE',
    schema: z.object({ command: z.string() }),
  }) as unknown as StructuredToolInterface;
}

/**
 * Wires a fake host that responds to every `ON_TOOL_EXECUTE` event by
 * resolving the request promise with `mockResults`. Mirrors the pattern
 * used in `ToolNode.outputReferences.test.ts` so the event-driven path
 * actually returns ToolMessages without spinning up a real host.
 */
function mockEventDispatch(mockResults: t.ToolExecuteResult[]): void {
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data) => {
      if (event !== 'on_tool_execute') {
        return;
      }
      const request = data as Record<string, unknown>;
      if (typeof request.resolve === 'function') {
        (request.resolve as (r: t.ToolExecuteResult[]) => void)(mockResults);
      }
    });
}

type MessagesUpdate = { messages: BaseMessage[] };
type CompiledMessagesGraph = Runnable<unknown, { messages: BaseMessage[] }> & {
  invoke(input: unknown, config?: RunnableConfig): Promise<unknown>;
};

/** Factory for a minimal `agent → tools → END` graph wrapping the ToolNode. */
function buildHITLGraph(
  toolNode: ToolNode,
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }>
): CompiledMessagesGraph {
  let agentInvocations = 0;
  const builder = new StateGraph(MessagesAnnotation)
    .addNode('agent', (): MessagesUpdate => {
      agentInvocations += 1;
      /**
       * First entry → emit the AIMessage carrying tool_calls so the
       * ToolNode actually has work. After resume the agent re-enters
       * once more (a normal LangGraph loop), but at that point any
       * approved tool already has a ToolMessage in state, so we emit
       * an empty AIMessage to satisfy the loop and end the run.
       */
      if (agentInvocations === 1) {
        return {
          messages: [new AIMessage({ content: '', tool_calls: toolCalls })],
        };
      }
      return { messages: [new AIMessage({ content: 'done' })] };
    })
    .addNode('tools', toolNode)
    .addEdge(START, 'agent')
    .addEdge('agent', 'tools')
    .addEdge('tools', END);
  return builder.compile({
    checkpointer: new MemorySaver(),
  }) as unknown as CompiledMessagesGraph;
}

function makeHookRegistry(
  decision: 'allow' | 'deny' | 'ask',
  reason?: string
): HookRegistry {
  const registry = new HookRegistry();
  registry.register('PreToolUse', {
    hooks: [
      async (): Promise<PreToolUseHookOutput> => ({
        decision,
        ...(reason != null ? { reason } : {}),
      }),
    ],
  });
  return registry;
}

describe('ToolNode HITL — `ask` decision raises interrupt() when humanInTheLoop is enabled', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('raises a tool_approval interrupt with the pending tool call payload', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'should-not-run', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'review tool args'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-1' } };

    const result = await graph.invoke({ messages: [] }, config);

    expect(isInterrupted<t.HumanInterruptPayload>(result)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(result)) {
      throw new Error('expected interrupt');
    }
    const interrupts = result.__interrupt__;
    expect(interrupts).toHaveLength(1);
    const payload = interrupts[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.action_requests).toEqual([
      {
        tool_call_id: 'call_1',
        name: 'echo',
        arguments: { command: 'list /' },
        description: 'review tool args',
      },
    ]);
    expect(payload.review_configs).toEqual([
      {
        action_name: 'echo',
        tool_call_id: 'call_1',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
    ]);
  });

  it('resume with approve runs the tool through the host event path', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do-it' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-approve' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted(interrupted)).toBe(true);

    const resumed = (await graph.invoke(
      new Command({ resume: [{ type: 'approve' }] }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].content).toBe('host-result');
    expect(toolMessages[0].status).not.toBe('error');
  });

  it('resume with reject blocks the tool and emits an error ToolMessage', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'rm -rf /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-reject' } };

    await graph.invoke({ messages: [] }, config);

    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'reject', reason: 'destructive command' }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('destructive command');
  });

  it('resume with edit substitutes the tool input before invocation', async () => {
    const capturedRequests: t.ToolCallRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        capturedRequests.push(...request.toolCalls);
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-edit' } };

    await graph.invoke({ messages: [] }, config);

    await graph.invoke(
      new Command({
        resume: [{ type: 'edit', updatedInput: { command: 'patched' } }],
      }),
      config
    );

    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].args).toEqual({ command: 'patched' });
  });

  it('resume with respond emits the user-supplied text as a successful ToolMessage and skips host execution', async () => {
    const dispatchSpy = jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'search' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-respond' } };

    await graph.invoke({ messages: [] }, config);

    const dispatchCallsBefore = dispatchSpy.mock.calls.filter(
      ([event]) => event === 'on_tool_execute'
    ).length;

    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'respond', responseText: 'no relevant results' }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const dispatchCallsAfter = dispatchSpy.mock.calls.filter(
      ([event]) => event === 'on_tool_execute'
    ).length;

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].content).toBe('no relevant results');
    expect(toolMessages[0].status).not.toBe('error');
    expect(dispatchCallsAfter).toBe(dispatchCallsBefore);
  });

  it('advertises respond in review_configs.allowed_decisions', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = {
      configurable: { thread_id: 'thread-hitl-allowed-decisions' },
    };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.review_configs[0].allowed_decisions).toEqual([
      'approve',
      'reject',
      'edit',
      'respond',
    ]);
  });

  it('resume with a record keyed by tool_call_id is accepted', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do-it' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-map' } };

    await graph.invoke({ messages: [] }, config);

    const resumed = (await graph.invoke(
      new Command({ resume: { call_1: { type: 'approve' } } }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].content).toBe('host-result');
  });
});

describe('ToolNode HITL — opt-out (`humanInTheLoop: { enabled: false }`) is fail-closed', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('blocks the tool with a ToolMessage error and never raises an interrupt', async () => {
    mockEventDispatch([]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'HITL explicitly disabled'),
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-optout' } };

    const result = (await graph.invoke({ messages: [] }, config)) as {
      messages: BaseMessage[];
    };

    expect(isInterrupted(result)).toBe(false);
    const toolMessages = result.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'HITL explicitly disabled'
    );
  });

  it('blocks the tool when `humanInTheLoop` is omitted (default-off)', async () => {
    /**
     * Default is OFF until host UIs (notably LibreChat) ship the
     * approval-rendering affordances. With HITL omitted, an `ask`
     * decision must collapse into a synchronous block — same fail-
     * closed behavior as the explicit `{ enabled: false }` opt-out.
     * This test guards against accidentally re-enabling the default-on
     * path before the consumer ecosystem is ready.
     */
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'default-off-blocks'),
      // humanInTheLoop intentionally omitted — should default to disabled
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-default' } };

    const out = (await graph.invoke({ messages: [] }, config)) as {
      messages: BaseMessage[];
    };
    expect(isInterrupted<t.HumanInterruptPayload>(out)).toBe(false);
    const toolMessages = out.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('call_1');
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('default-off-blocks');
  });
});

describe('ToolNode HITL — multi-tool batches', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('bundles multiple ask decisions into a single interrupt and resolves per call', async () => {
    const capturedRequests: t.ToolCallRequest[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        capturedRequests.push(...request.toolCalls);
        request.resolve(
          request.toolCalls.map(
            (c): t.ToolExecuteResult => ({
              toolCallId: c.id,
              content: `ran:${c.name}`,
              status: 'success',
            })
          )
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo'), createSchemaStub('cat')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_1', 'step_call_1'],
        ['call_2', 'step_call_2'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'one' } },
      { id: 'call_2', name: 'cat', args: { command: 'two' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-batch' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(interrupted)).toBe(true);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    expect(payload.action_requests.map((r) => r.tool_call_id)).toEqual([
      'call_1',
      'call_2',
    ]);

    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'approve' }, { type: 'reject', reason: 'too risky' }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(2);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_1')!.content).toBe('ran:echo');
    expect(byId.get('call_1')!.status).not.toBe('error');
    expect(byId.get('call_2')!.status).toBe('error');
    expect(String(byId.get('call_2')!.content)).toContain('too risky');

    expect(capturedRequests).toHaveLength(1);
    expect(capturedRequests[0].id).toBe('call_1');
  });
});

describe('Run integration — HITL fallback checkpointer + resume', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('Run.create does NOT install a MemorySaver fallback by default (HITL is off until host UI ships)', async () => {
    /**
     * Default-off rationale: HITL ships the interrupt machinery but
     * stays opt-in until host UIs (notably LibreChat) can render and
     * resolve `tool_approval` interrupts. With HITL omitted, the SDK
     * must NOT silently install a checkpointer — that would suggest
     * the run can pause/resume when in fact the `ask` path will
     * fail-closed. Plan of record: flip the default to ON in a future
     * minor once the consumer ecosystem is ready.
     */
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-default-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      // humanInTheLoop intentionally omitted — default is OFF
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeUndefined();
  });

  it('Run.create installs a MemorySaver fallback when HITL is explicitly enabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-explicit-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);
    expect(run.Graph?.humanInTheLoop?.enabled).toBe(true);
  });

  it('Run.create preserves a host-supplied checkpointer when HITL is explicitly enabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const hostCheckpointer = new MemorySaver();
    const run = await Run.create<t.IState>({
      runId: 'hitl-host-checkpointer',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
        compileOptions: { checkpointer: hostCheckpointer },
      },
      humanInTheLoop: { enabled: true },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBe(hostCheckpointer);
  });

  it('re-exports langgraph HITL primitives from the SDK barrel for host use', async () => {
    const indexExports = await import('@/index');
    expect(indexExports.MemorySaver).toBe(MemorySaver);
    expect(indexExports.Command).toBe(Command);
    expect(indexExports.INTERRUPT).toBeDefined();
    expect(typeof indexExports.interrupt).toBe('function');
    expect(typeof indexExports.isInterrupted).toBe('function');
    expect(typeof indexExports.BaseCheckpointSaver).toBe('function');
  });

  it('Run.create does not attach a checkpointer when HITL is explicitly disabled', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');

    const run = await Run.create<t.IState>({
      runId: 'hitl-optout-run',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: false },
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeUndefined();
  });

  it('Run.resume() drives the host all the way through the resume command path', async () => {
    /** End-to-end on the Run wrapper: build a HITL graph that
     * interrupts on first invoke, then drive resume via the Run's
     * own `resume()` method (not raw graph.invoke + Command).
     * Validates the full Run.resume → processStream(Command) path. */
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'run-resume-direct',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'run-resume-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);
    expect(run.getInterrupt()).toBeDefined();
    expect(dispatchCount).toBe(0);

    /** This is the API contract under test: Run.resume() with a
     * decision array (not graph.invoke + Command). */
    await run.resume([{ type: 'approve' }], callerConfig);

    expect(dispatchCount).toBe(1);
    /** Resume completed naturally: interrupt cleared, no halt
     * reason carried over from the previous pass. */
    expect(run.getInterrupt()).toBeUndefined();
    expect(run.getHaltReason()).toBeUndefined();
  });

  it('Run.getHaltReason() reports prompt_denied when UserPromptSubmit denies the prompt', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'deny',
          reason: 'PII detected',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-deny-haltreason',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('please tell me their SSN')] },
      { configurable: { thread_id: 'prompt-deny-thread' }, version: 'v2' }
    );

    /** Hook denied the prompt — run returns undefined AND
     * `getHaltReason()` carries the reason so the host can
     * distinguish "blocked" from "natural empty completion". */
    expect(result).toBeUndefined();
    expect(run.getHaltReason()).toBe('PII detected');
  });

  it('Run.getHaltReason() falls back to canonical prompt_denied when deny carries no reason', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'deny',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-deny-canonical',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'prompt-deny-canonical-thread' },
        version: 'v2',
      }
    );

    /** Hook returned `deny` without a reason — host gets the
     * canonical 'prompt_denied' string so it can route on a stable
     * discriminator. */
    expect(run.getHaltReason()).toBe('prompt_denied');
  });

  it('Run.getHaltReason() reports prompt_requires_approval when UserPromptSubmit asks', async () => {
    const registry = new HookRegistry();
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => ({
          decision: 'ask',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'prompt-ask-haltreason',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('hello')] },
      { configurable: { thread_id: 'prompt-ask-thread' }, version: 'v2' }
    );

    /** Default reason when the hook didn't supply one — host can
     * route on the canonical string. */
    expect(run.getHaltReason()).toBe('prompt_requires_approval');
  });
});

describe('ToolNode HITL — additionalContext injection from hooks', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('injects PreToolUse + PostToolUse additionalContexts as a single HumanMessage', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'allow',
          additionalContext: 'pre-context: be careful',
        }),
      ],
    });
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          additionalContext: 'post-context: tool ran',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'ctx-thread-1' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain('pre-context: be careful');
    expect(String(injected!.content)).toContain('post-context: tool ran');
  });

  it('does not inject anything when no hook returns additionalContext', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({ decision: 'allow' }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'do' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'ctx-thread-2' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeUndefined();
  });
});

describe('ToolNode HITL — PostToolBatch hook', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('fires once per dispatch with all entries (success + error mix), in batch order', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          { toolCallId: 'call_1', content: 'ok', status: 'success' },
          {
            toolCallId: 'call_2',
            content: '',
            status: 'error',
            errorMessage: 'boom',
          },
        ]);
      });

    const registry = new HookRegistry();
    let captured: PostToolBatchEntry[] | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          captured = (input as PostToolBatchHookInput).entries;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo'), createSchemaStub('cat')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_1', 'step_1'],
        ['call_2', 'step_2'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
      { id: 'call_2', name: 'cat', args: { command: 'b' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-thread' } }
    );

    expect(captured).toBeDefined();
    expect(captured!).toHaveLength(2);
    expect(captured![0].toolUseId).toBe('call_1');
    expect(captured![0].status).toBe('success');
    expect(captured![0].toolOutput).toBe('ok');
    expect(captured![1].toolUseId).toBe('call_2');
    expect(captured![1].status).toBe('error');
    expect(captured![1].error).toContain('boom');
  });

  it('a PostToolBatch additionalContext gets injected as a HumanMessage', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          additionalContext: 'remember to format the response as JSON',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'a' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-ctx-thread' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain('format the response as JSON');
  });
});

describe('ToolNode HITL — per-hook allowedDecisions override', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('restricts the interrupt review_configs.allowed_decisions to the hook-supplied subset', async () => {
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const interrupted = await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'allowed-thread' } }
    );
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }
    expect(payload.review_configs[0].allowed_decisions).toEqual([
      'approve',
      'reject',
    ]);
  });
});

describe('Run — preventContinuation honored for pre-stream hooks', () => {
  beforeEach(() => {
    jest.restoreAllMocks();
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('returns undefined without invoking the graph when RunStart hook returns preventContinuation', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const registry = new HookRegistry();
    let runStartFired = false;
    registry.register('RunStart', {
      hooks: [
        async (): Promise<RunStartHookOutput> => {
          runStartFired = true;
          return {
            preventContinuation: true,
            stopReason: 'pre-flight policy halted run',
          };
        },
      ],
    });

    const run = await Run.create<t.IState>({
      runId: 'pc-runstart',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'pc-thread-1' },
        version: 'v2',
      }
    );

    expect(runStartFired).toBe(true);
    expect(result).toBeUndefined();
    /** Graph should not have been run — no messages added beyond the input. */
    expect(run.getInterrupt()).toBeUndefined();
  });

  it('returns undefined when UserPromptSubmit hook returns preventContinuation', async () => {
    const { Run } = await import('@/run');
    const { Providers } = await import('@/common');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const registry = new HookRegistry();
    let promptFired = false;
    registry.register('UserPromptSubmit', {
      hooks: [
        async (): Promise<UserPromptSubmitHookOutput> => {
          promptFired = true;
          return {
            preventContinuation: true,
            stopReason: 'rate limit reached',
          };
        },
      ],
    });

    const run = await Run.create<t.IState>({
      runId: 'pc-prompt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    const result = await run.processStream(
      { messages: [new HM('hello')] },
      {
        configurable: { thread_id: 'pc-thread-2' },
        version: 'v2',
      }
    );

    expect(promptFired).toBe(true);
    expect(result).toBeUndefined();
  });
});

describe('Mid-flight preventContinuation halts the run after the current step', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('PostToolBatch hook with preventContinuation breaks the stream loop and skips Stop', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ok', status: 'success' },
    ]);

    const registry = new HookRegistry();
    let stopFired = false;
    registry.register('PostToolBatch', {
      hooks: [
        async (): Promise<PostToolBatchHookOutput> => ({
          preventContinuation: true,
          stopReason: 'rate-limit policy halt',
        }),
      ],
    });
    registry.register('Stop', {
      hooks: [
        async (): Promise<Record<string, never>> => {
          stopFired = true;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('agent', () => ({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'echo', args: { command: 'x' } },
            ],
          }),
        ],
      }))
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'halt-mid-flight-1',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });
    /** Replace the SDK-built graph runnable with our handcrafted one so the
     * PostToolBatch hook fires under a real LangGraph stream. */
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'halt-thread-1' },
        version: 'v2',
      }
    );

    expect(run.getHaltReason()).toBe('rate-limit policy halt');
    expect(stopFired).toBe(false);
  });

  it('clears halt signal between processStream invocations', async () => {
    const registry = new HookRegistry();
    registry.register('RunStart', {
      hooks: [
        async (): Promise<RunStartHookOutput> => ({
          preventContinuation: true,
          stopReason: 'first run halted',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    const run = await Run.create<t.IState>({
      runId: 'halt-clear-1',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: false },
    });

    await run.processStream(
      { messages: [new HM('first')] },
      { configurable: { thread_id: 't-1' }, version: 'v2' }
    );
    /** RunStart preventContinuation is a pre-stream early return, but
     * `processStream` should still have cleared the registry signal
     * for this run id so a subsequent call starts fresh. */
    expect(registry.getHaltSignal('halt-clear-1')).toBeUndefined();
  });
});

describe('Async fire-and-forget hooks ignore decision/context fields', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('PreToolUse with `async: true` does not block the tool even when decision is `deny`', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ran', status: 'success' },
    ]);

    let bgFired = false;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          /** Side effect runs in background; agent doesn't wait. */
          void Promise.resolve().then(() => {
            bgFired = true;
          });
          return {
            async: true,
            decision: 'deny',
            reason: 'this should be ignored',
            additionalContext: 'this should also be ignored',
          };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'async-1' } }
    )) as { messages: BaseMessage[] };

    const toolMsg = result.messages.find(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMsg).toBeDefined();
    /** Tool ran (no Blocked: prefix) — async output's `decision: 'deny'` was
     * ignored as documented. */
    expect(toolMsg!.status).not.toBe('error');
    expect(toolMsg!.content).toBe('ran');
    /** Background work runs even though we ignored the output. */
    await new Promise((r) => setImmediate(r));
    expect(bgFired).toBe(true);
    /** No injected context message — `additionalContext` was also ignored. */
    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeUndefined();
  });

  it('PostToolUse with `async: true` does not halt the run even when preventContinuation is set', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'ran', status: 'success' },
    ]);

    const registry = new HookRegistry();
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          async: true,
          preventContinuation: true,
          stopReason: 'should not halt',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'async-2' } }
    );

    /** preventContinuation was on an async output → ignored → no halt
     * signal raised under any session id. The standalone graph here
     * runs with `runId = ''` (no `config.configurable.run_id` set),
     * so check that key explicitly. */
    expect(registry.getHaltSignal('')).toBeUndefined();
  });
});

describe('Codex review fixes', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('preserves session-scoped hooks across HITL interrupt so the policy still fires on resume', async () => {
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    let preCallCount = 0;
    /**
     * Register the policy hook against the runId via `registerSession`
     * (mirrors how a host scopes per-run policy without leaking it to
     * concurrent runs). The fix under test: this matcher MUST still be
     * present when `Run.resume()` re-runs the node so the policy
     * decision applies the second time too.
     */
    const runId = 'session-hook-preserve';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          preCallCount += 1;
          return { decision: 'ask', reason: 'session policy' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'session-thread-1' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);

    /** Interrupt fired; one hook invocation so far. Session matcher
     * MUST still be present — the regression was that finally cleared
     * it, leaving the resume to bypass the policy entirely. */
    expect(run.getInterrupt()).toBeDefined();
    expect(preCallCount).toBe(1);
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(true);
    expect(dispatchCalls).toBe(0);

    await run.resume([{ type: 'approve' }], callerConfig);

    /** Hook fired AGAIN on resume — policy was actually applied a
     * second time, not skipped. Tool then executed. */
    expect(preCallCount).toBe(2);
    expect(dispatchCalls).toBe(1);
    /** After natural completion, session matchers ARE cleared so the
     * next run on this registry starts clean. */
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('denied tool in a deny+ask batch dispatches ON_RUN_STEP_COMPLETED exactly once across interrupt + resume', async () => {
    const stepCompletedDispatches: string[] = [];
    /** Spy on the underlying custom event dispatcher to capture every
     * ON_RUN_STEP_COMPLETED event with its tool_call_id. Without the
     * blockEntry deferral, this would record `call_a` twice for one
     * logical denial (once before interrupt, once after resume
     * re-execution). */
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as {
            result?: { tool_call?: { id?: string } };
          };
          const id = payload.result?.tool_call?.id;
          if (id != null) {
            stepCompletedDispatches.push(id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (input.toolName === 'tool_a') {
            return { decision: 'deny', reason: 'policy:a' };
          }
          return { decision: 'ask', reason: 'policy:b-needs-review' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('tool_a'), createSchemaStub('tool_b')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_a', 'step_a'],
        ['call_b', 'step_b'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_a', name: 'tool_a', args: { command: 'a' } },
      { id: 'call_b', name: 'tool_b', args: { command: 'b' } },
    ]);
    const config = { configurable: { thread_id: 'dedup-thread' } };

    await graph.invoke({ messages: [] }, config);
    /** First pass: interrupt() threw, so the deferred denial side
     * effects were not flushed. Zero step-completed events for the
     * denied tool yet. */
    expect(stepCompletedDispatches.filter((id) => id === 'call_a')).toEqual([]);

    await graph.invoke(new Command({ resume: [{ type: 'approve' }] }), config);

    /** After resume: the denied tool dispatches exactly once (deferred
     * flush on the resume re-execution); the approved tool dispatches
     * once via the normal execution path. */
    expect(stepCompletedDispatches.filter((id) => id === 'call_a')).toEqual([
      'call_a',
    ]);
    expect(stepCompletedDispatches.filter((id) => id === 'call_b')).toEqual([
      'call_b',
    ]);
  });

  it('enforces allowedDecisions on resume — host-submitted decision outside the allowlist is rejected', async () => {
    const dispatchedToolNames: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedToolNames.push(c.name);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    /** Hook restricts to approve/reject only — edit/respond are
     * forbidden. Even if a buggy or hostile host UI submits an
     * `edit`, the SDK must fail closed instead of mutating the args
     * and running the tool. */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'allowed-enforce' } };

    await graph.invoke({ messages: [] }, config);

    /** Submit `edit` — outside the advertised allowlist. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'edit', updatedInput: { command: 'malicious' } }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    /** Tool was blocked; arg-mutation never reached the host. */
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'not in allowedDecisions'
    );
    expect(String(toolMessages[0].content)).toContain('approve');
    expect(String(toolMessages[0].content)).toContain('reject');
    expect(dispatchedToolNames).toEqual([]);
  });

  it('enforces allowedDecisions on resume — approved decision passes through when in the allowlist', async () => {
    const dispatchedArgs: Array<Record<string, unknown>> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedArgs.push(c.args);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          allowedDecisions: ['approve', 'reject'],
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'allowed-pass' } };

    await graph.invoke({ messages: [] }, config);

    /** Submit `approve` — explicitly in the allowlist. */
    await graph.invoke(new Command({ resume: [{ type: 'approve' }] }), config);

    expect(dispatchedArgs).toEqual([{ command: 'original' }]);
  });

  it('getInterrupt<T>() returns the captured payload typed as the host-asserted shape', async () => {
    /**
     * Custom graph node raises an interrupt with a payload shape the
     * SDK doesn't know about. `run.getInterrupt<MyCustomPayload>()`
     * returns the payload typed as the host's assertion — the SDK
     * doesn't validate, it just transports.
     */
    interface MyCustomPayload {
      type: 'custom_review';
      diff: string;
      reviewerHints: string[];
    }

    const langgraph = await import('@langchain/langgraph');

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('clarifier', () => {
        langgraph.interrupt({
          type: 'custom_review',
          diff: '+ added line',
          reviewerHints: ['check formatting'],
        } satisfies MyCustomPayload);
        return { messages: [] };
      })
      .addEdge(START, 'clarifier')
      .addEdge('clarifier', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'custom-interrupt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'custom-interrupt-thread' },
        version: 'v2',
      }
    );

    const interrupt = run.getInterrupt<MyCustomPayload>();
    expect(interrupt).toBeDefined();
    expect(interrupt!.payload.type).toBe('custom_review');
    expect(interrupt!.payload.diff).toBe('+ added line');
    expect(interrupt!.payload.reviewerHints).toEqual(['check formatting']);
  });

  it('isToolApprovalInterrupt / isAskUserQuestionInterrupt narrow safely from `unknown` (defensive)', async () => {
    const { isToolApprovalInterrupt, isAskUserQuestionInterrupt } =
      await import('@/types/hitl');

    /** The guards must accept arbitrary runtime values without throwing,
     * since hosts can pass anything from custom interrupts. */
    expect(isToolApprovalInterrupt(null as unknown)).toBe(false);
    expect(isToolApprovalInterrupt(undefined as unknown)).toBe(false);
    expect(isToolApprovalInterrupt('string' as unknown)).toBe(false);
    expect(isToolApprovalInterrupt(42 as unknown)).toBe(false);
    expect(isToolApprovalInterrupt({} as unknown)).toBe(false);
    expect(isToolApprovalInterrupt({ type: 'something_else' } as unknown)).toBe(
      false
    );
    expect(
      isToolApprovalInterrupt({
        type: 'tool_approval',
        action_requests: [],
        review_configs: [],
      } as unknown)
    ).toBe(true);

    expect(isAskUserQuestionInterrupt(null as unknown)).toBe(false);
    expect(
      isAskUserQuestionInterrupt({ type: 'tool_approval' } as unknown)
    ).toBe(false);
    expect(
      isAskUserQuestionInterrupt({
        type: 'ask_user_question',
        question: { question: 'why' },
      } as unknown)
    ).toBe(true);
  });

  it('hook returning ask + updatedInput rewrites args BEFORE the interrupt and BEFORE host execution', async () => {
    const dispatchedArgs: Array<Record<string, unknown>> = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedArgs.push(c.args);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'ran',
            status: 'success' as const,
          }))
        );
      });

    /**
     * Hook returns BOTH a sanitization rewrite AND `ask`. Real-world
     * pattern: one matcher redacts secrets in the args, another
     * matcher requires human approval. Both signals must apply.
     */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review redacted args',
          updatedInput: { command: 'redacted-command' },
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original-secret' } },
    ]);
    const config = { configurable: { thread_id: 'ask-with-update' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }
    /** The interrupt payload surfaces the REWRITTEN args to the
     * reviewer, not the original. Without the fix, the reviewer
     * would see the secret. */
    expect(payload.action_requests[0].arguments).toEqual({
      command: 'redacted-command',
    });

    await graph.invoke(new Command({ resume: [{ type: 'approve' }] }), config);

    /** And the host execution dispatches the rewritten args, not
     * the original. Without the fix, the policy redaction would be
     * silently dropped after approval. */
    expect(dispatchedArgs).toEqual([{ command: 'redacted-command' }]);
  });

  it('captures interrupt even when payload is null (custom node calling interrupt(null))', async () => {
    const langgraph = await import('@langchain/langgraph');

    let stopFired = false;
    const registry = new HookRegistry();
    registry.register('Stop', {
      hooks: [
        async (): Promise<Record<string, never>> => {
          stopFired = true;
          return {};
        },
      ],
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('pauser', () => {
        /** Custom node pauses without payload — valid use case (the
         * pause itself is the signal; no metadata needed). */
        langgraph.interrupt(null);
        return { messages: [] };
      })
      .addEdge(START, 'pauser')
      .addEdge('pauser', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'null-payload-interrupt',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'null-payload-thread' },
        version: 'v2',
      }
    );

    /** Run was paused, NOT completed — getInterrupt returns a result
     * (with the null payload preserved) and the Stop hook does not
     * fire. Without the fix, both inversions held. */
    const interrupt = run.getInterrupt<unknown>();
    expect(interrupt).toBeDefined();
    expect(interrupt!.payload).toBeNull();
    expect(stopFired).toBe(false);
  });

  it('halt signal raised by run A does not bleed into a concurrent run B sharing the same registry', async () => {
    /**
     * One registry, two runs. RunStart hook for run A raises
     * preventContinuation; run B has no halt signal. Without
     * per-session scoping, run B's stream-loop poll would see A's
     * signal and silently terminate. With scoping, each run reads
     * only its own halt entry.
     */
    const registry = new HookRegistry();
    let runStartFires = 0;
    registry.register('RunStart', {
      hooks: [
        async (input): Promise<RunStartHookOutput> => {
          runStartFires += 1;
          /** Halt only run A, not run B. */
          if (input.runId === 'run-a') {
            return {
              preventContinuation: true,
              stopReason: 'A halted',
            };
          }
          return {};
        },
      ],
    });

    const { Run } = await import('@/run');
    const { HumanMessage: HM } = await import('@langchain/core/messages');

    /** No-op graph so we never hit the real model. */
    const makeNoopGraph = (): t.CompiledStateWorkflow => {
      const builder = new StateGraph(MessagesAnnotation)
        .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
        .addEdge(START, 'noop')
        .addEdge('noop', END);
      return builder.compile() as unknown as t.CompiledStateWorkflow;
    };

    const makeRun = async (
      runId: string
    ): Promise<Awaited<ReturnType<typeof Run.create<t.IState>>>> => {
      const r = await Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'noop',
              maxContextTokens: 8000,
            },
          ],
        },
        hooks: registry,
        humanInTheLoop: { enabled: false },
      });
      r.graphRunnable = makeNoopGraph();
      return r;
    };

    const runA = await makeRun('run-a');
    const runB = await makeRun('run-b');

    /** Run A — its preventContinuation lands in the per-session halt
     * map under key `'run-a'` and triggers a pre-stream early
     * return. Note that the early-return path also clears its own
     * halt signal in the same step, so run B can never observe it
     * even momentarily. */
    await runA.processStream(
      { messages: [new HM('a')] },
      { configurable: { thread_id: 'thread-a' }, version: 'v2' }
    );
    expect(runA.getHaltReason()).toBe('A halted');

    /** Run B's signal must be undefined — A's halt is scoped to A's
     * session id, and was cleared in A's pre-stream finally path. */
    expect(registry.getHaltSignal('run-b')).toBeUndefined();
    expect(registry.getHaltSignal('run-a')).toBeUndefined();

    /** Run B — RunStart returns no halt, so processStream proceeds
     * past the pre-stream gate, executes the no-op graph, and
     * completes without halt. */
    runStartFires = 0;
    await runB.processStream(
      { messages: [new HM('b')] },
      { configurable: { thread_id: 'thread-b' }, version: 'v2' }
    );
    expect(runStartFires).toBe(1);
    expect(runB.getHaltReason()).toBeUndefined();
  });

  it('review_configs entries carry tool_call_id so duplicate-tool batches map unambiguously', async () => {
    mockEventDispatch([]);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    /** Same tool name called twice in one batch — by-position
     * mapping breaks down for hosts that reorder; tool_call_id
     * lets the UI map review_configs → action_requests directly. */
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_first', 'step_first'],
        ['call_second', 'step_second'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_first', name: 'echo', args: { command: 'a' } },
      { id: 'call_second', name: 'echo', args: { command: 'b' } },
    ]);
    const config = { configurable: { thread_id: 'duplicate-tool' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval');
    }

    /** Each review_config carries its own tool_call_id matching the
     * action_request at the same index. UI can build a Map keyed by
     * tool_call_id rather than relying on positional order. */
    expect(payload.review_configs).toEqual([
      {
        action_name: 'echo',
        tool_call_id: 'call_first',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
      {
        action_name: 'echo',
        tool_call_id: 'call_second',
        allowed_decisions: ['approve', 'reject', 'edit', 'respond'],
      },
    ]);
    /** And the action_requests carry the same ids — pairing is
     * always derivable from id even when names collide. */
    expect(payload.action_requests.map((r) => r.tool_call_id)).toEqual([
      'call_first',
      'call_second',
    ]);
  });

  it('malformed edit decision (missing updatedInput) is blocked, not approved with garbage args', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-malformed' } };

    await graph.invoke({ messages: [] }, config);

    /** `{ type: 'edit' }` with no updatedInput — same trust-boundary
     * issue as malformed respond. Must fail closed, NOT pass undefined
     * into applyInputOverride and approve a tool with garbage args. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'edit' } as unknown as t.ToolApprovalDecision],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing object updatedInput'
    );
    expect(String(toolMessages[0].content)).toContain('<missing>');
    expect(dispatchCount).toBe(0);
  });

  it('malformed edit decision (non-object updatedInput) is blocked', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-nonobject' } };

    await graph.invoke({ messages: [] }, config);

    /** `updatedInput: 'string'` — wire deserializer didn't enforce
     * object shape; SDK must reject. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [
          {
            type: 'edit',
            updatedInput: 'not-an-object' as unknown as Record<string, unknown>,
          },
        ],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing object updatedInput'
    );
    expect(String(toolMessages[0].content)).toContain('string');
    expect(dispatchCount).toBe(0);
  });

  it('malformed edit decision (array updatedInput) is blocked — arrays are objects but not plain records', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'original' } },
    ]);
    const config = { configurable: { thread_id: 'edit-array' } };

    await graph.invoke({ messages: [] }, config);

    const resumed = (await graph.invoke(
      new Command({
        resume: [
          {
            type: 'edit',
            updatedInput: [1, 2, 3] as unknown as Record<string, unknown>,
          },
        ],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain('array');
  });

  it('malformed respond decision (missing responseText) is blocked, not crashed', async () => {
    let dispatchCount = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCount += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-malformed' } };

    await graph.invoke({ messages: [] }, config);

    /** Submit a `respond` decision with NO responseText — wire shape
     * the SDK can't honor. Must fail closed (blockEntry path), NOT
     * crash truncateToolResultContent on `undefined.length`. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'respond' } as unknown as t.ToolApprovalDecision],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing string responseText'
    );
    expect(String(toolMessages[0].content)).toContain('<missing>');
    /** Tool was never dispatched — fail-closed worked. */
    expect(dispatchCount).toBe(0);
  });

  it('malformed respond decision (non-string responseText) is blocked, not crashed', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-nonstring' } };

    await graph.invoke({ messages: [] }, config);

    /** `responseText: 42` — wire deserializer didn't enforce string;
     * SDK must reject without crashing. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [
          {
            type: 'respond',
            responseText: 42 as unknown as string,
          },
        ],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'missing string responseText'
    );
    expect(String(toolMessages[0].content)).toContain('number');
  });

  it('respond decision truncates oversized text the same way real tool output is truncated', async () => {
    mockEventDispatch([]);

    /** Build a ToolNode with a tiny `maxToolResultChars` so the
     * truncation kicks in for a 200-char response. Without the fix,
     * the full string would land in the ToolMessage and PostToolBatch
     * entry — bypassing the model context budget. */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [async (): Promise<PreToolUseHookOutput> => ({ decision: 'ask' })],
    });
    let captured: PostToolBatchEntry | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          captured = (input as PostToolBatchHookInput).entries[0];
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
      maxToolResultChars: 50,
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'x' } },
    ]);
    const config = { configurable: { thread_id: 'respond-truncate' } };

    await graph.invoke({ messages: [] }, config);

    /** 200-char response — well over the 50-char cap. */
    const oversized = 'A'.repeat(200);
    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'respond', responseText: oversized }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    /** The ToolMessage content is truncated; not the raw 200 chars. */
    const content = String(toolMessages[0].content);
    expect(content.length).toBeLessThan(oversized.length);
    /** And the PostToolBatch entry sees the SAME truncated value
     * — batch hooks observe what the model will actually see. */
    expect(captured).toBeDefined();
    expect(typeof captured!.toolOutput).toBe('string');
    expect(captured!.toolOutput).toBe(content);
  });

  it('hook returning both ask + preventContinuation halts cleanly and clears session hooks', async () => {
    mockEventDispatch([]);

    const registry = new HookRegistry();
    /** Session-scoped policy hook returns BOTH `ask` (which would
     * raise an interrupt) AND `preventContinuation: true` (which
     * raises a halt signal). The halt wins — no resume is expected,
     * sessions must clear. */
    const runId = 'ask-and-halt';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          preventContinuation: true,
          stopReason: 'policy halted ask',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'ask-and-halt-thread' },
        version: 'v2',
      }
    );

    /** Both signals landed: interrupt was captured AND halt fired. */
    expect(run.getInterrupt()).toBeDefined();
    expect(run.getHaltReason()).toBe('policy halted ask');
    /** Session hooks MUST be cleared — no resume is expected on a
     * halted run, even one that also captured an interrupt. */
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('preserves Graph sidecars across HITL interrupt + resume so tool completions keep their step ids', async () => {
    /**
     * Regression test for the cleanup-vs-resume bug: previously
     * `processStream` always called `Graph.clearHeavyState()` in its
     * `finally` block AND `Graph.resetValues()` on entry, even when
     * pausing on a HITL interrupt. That wiped `toolCallStepIds`,
     * `_toolOutputRegistry`, and `sessions` between pause and resume,
     * so the resumed `ToolNode` could no longer find the original
     * step id and dispatched `ON_RUN_STEP_COMPLETED` with an empty id
     * — the host's stream consumer would then drop the result.
     *
     * The fix is two gated cleanups:
     *   - `clearHeavyState` skipped when `_interrupt != null && _haltedReason == null && !streamThrew`
     *   - `resetValues` skipped when entering processStream via `Command` (resume)
     *
     * To exercise the SDK Graph's actual sidecar state (not a private
     * test ToolNode), this test wires the custom ToolNode to share
     * the SDK Graph's `toolCallStepIds` Map by reference. After the
     * interrupt fires AND after the resume completes, the
     * pre-populated entry must still be present.
     */
    const dispatchedStepIds: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as { result?: { id?: string } };
          if (payload.result?.id != null) {
            dispatchedStepIds.push(payload.result.id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: 'host-result',
            status: 'success' as const,
          }))
        );
      });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'sidecar-preserve',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
    });

    /** Wire the test ToolNode to share the SDK Graph's
     * `toolCallStepIds` Map by reference — this is how the real
     * StandardGraph builds its inner ToolNode at Graph.ts:587. */
    const toolNode = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'a',
      toolCallStepIds: run.Graph!.toolCallStepIds,
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    /** The agent node simulates `attemptInvoke`'s sidecar-population
     * step: in a real run, the model invocation creates a run step
     * and writes its id into `toolCallStepIds` before tools dispatch.
     * Doing it here means the entry lands AFTER `processStream`'s
     * `resetValues` (which fires once on entry) and BEFORE the
     * ToolNode's hook + interrupt — exactly mirroring the production
     * timing the cleanup gate has to preserve. */
    const builder = new StateGraph(MessagesAnnotation)
      .addNode('agent', (): MessagesUpdate => {
        run.Graph!.toolCallStepIds.set('call_1', 'step_real_id');
        return {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        };
      })
      .addNode('tools', toolNode)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'sidecar-thread' },
      version: 'v2' as const,
    };

    await run.processStream({ messages: [] }, callerConfig);

    /** After interrupt: sidecar entry MUST still be present. Without
     * the fix, `clearHeavyState` in the `finally` block would have
     * wiped this map. */
    expect(run.getInterrupt()).toBeDefined();
    expect(run.Graph!.toolCallStepIds.has('call_1')).toBe(true);
    expect(run.Graph!.toolCallStepIds.get('call_1')).toBe('step_real_id');

    /** Resume: without the resetValues gate, this would also wipe
     * the map at the START of the second processStream invocation. */
    await run.resume([{ type: 'approve' }], callerConfig);

    /** After resume completes naturally: dispatch fired with the real
     * step id (not an empty string from a wiped map). Without either
     * fix, `dispatchedStepIds` would contain `''`. */
    expect(dispatchedStepIds).toContain('step_real_id');
    expect(dispatchedStepIds).not.toContain('');
    /** And clearHeavyState DID fire on the natural-completion side
     * — sidecar map is now empty after the resume settled. */
    expect(run.Graph!.toolCallStepIds.size).toBe(0);
  });

  it('clears Graph sidecars on natural completion when no interrupt was raised', async () => {
    /** Negative case: when no interrupt fires, `clearHeavyState`
     * MUST run as before. This pins the gate so a future change
     * doesn't accidentally preserve sidecars on natural completion
     * (memory leak across runs). */
    mockEventDispatch([]);

    const { Run } = await import('@/run');
    const run = await Run.create<t.IState>({
      runId: 'sidecar-clear-natural',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      humanInTheLoop: { enabled: false },
    });

    /** No-op graph — runs to completion without an interrupt. */
    const builder = new StateGraph(MessagesAnnotation)
      .addNode('noop', (): MessagesUpdate => ({ messages: [] }))
      .addEdge(START, 'noop')
      .addEdge('noop', END);
    const graph = builder.compile();
    run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    /** Stash an entry so we can verify clearHeavyState wiped it. */
    run.Graph!.toolCallStepIds.set('stale_call', 'stale_step');

    await run.processStream(
      { messages: [] },
      {
        configurable: { thread_id: 'sidecar-clear-thread' },
        version: 'v2',
      }
    );

    /** No interrupt → clearHeavyState ran → sidecar wiped. */
    expect(run.getInterrupt()).toBeUndefined();
    expect(run.Graph!.toolCallStepIds.size).toBe(0);
  });

  it('clears session hooks when the stream throws AFTER an interrupt is captured (stale interrupt)', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async () => {
        return;
      });

    const registry = new HookRegistry();
    const runId = 'stream-error-after-interrupt';
    registry.registerSession(runId, 'PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'session policy',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode(
        'agent',
        (): MessagesUpdate => ({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'call_1', name: 'echo', args: { command: 'x' } },
              ],
            }),
          ],
        })
      )
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const { Run } = await import('@/run');
    /**
     * Holder for forward-referencing the run inside the sentinel
     * handler closure. The handler is constructed before `Run.create`
     * runs (it's passed into `customHandlers`) but needs to read
     * `run.getInterrupt()` at firing time.
     */
    const holder: {
      run: Awaited<ReturnType<typeof Run.create<t.IState>>> | undefined;
    } = { run: undefined };

    /**
     * Handler keyed to a chain-stream event that throws ONLY after the
     * interrupt has been captured. The stream loop captures the
     * interrupt on the chunk that carries `__interrupt__`, then
     * dispatches to handlers in the same iteration — so the throw
     * exits the loop with `_interrupt != null`. Without the
     * `streamThrew` guard, the `finally` block would preserve session
     * hooks on this stale interrupt.
     */
    const sentinelHandler = {
      handle: async (): Promise<void> => {
        if (holder.run?.getInterrupt() != null) {
          throw new Error('post-interrupt handler failure');
        }
      },
    };

    holder.run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'a',
            provider: providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
      },
      hooks: registry,
      humanInTheLoop: { enabled: true },
      customHandlers: {
        [GraphEvents.CHAIN_STREAM]: sentinelHandler,
        [GraphEvents.CHAIN_END]: sentinelHandler,
      },
    });
    holder.run.graphRunnable = graph as unknown as t.CompiledStateWorkflow;

    const callerConfig = {
      configurable: { thread_id: 'stale-interrupt-thread' },
      version: 'v2' as const,
    };

    await expect(
      holder.run.processStream({ messages: [] }, callerConfig)
    ).rejects.toThrow('post-interrupt handler failure');

    /** Interrupt WAS captured on the run instance, but because the
     * stream subsequently threw, session hooks must be cleared so the
     * next run on this registry isn't poisoned by stale state. */
    expect(holder.run.getInterrupt()).toBeDefined();
    expect(registry.hasHookFor('PreToolUse', runId)).toBe(false);
  });

  it('mixed deny/ask/allow batch: deny short-circuits, allow runs immediately, ask interrupts; resume completes the asked tool', async () => {
    const dispatchedToolNames: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        for (const c of request.toolCalls) {
          dispatchedToolNames.push(c.name);
        }
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    /**
     * Per-tool policy hook: tool_a denied, tool_b asks, tool_c allowed.
     * The hook is registered without a pattern so it fires once per
     * tool call and dispatches by tool name.
     */
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (input.toolName === 'tool_a') {
            return { decision: 'deny', reason: 'policy:a' };
          }
          if (input.toolName === 'tool_b') {
            return { decision: 'ask', reason: 'policy:b-needs-review' };
          }
          return { decision: 'allow' };
        },
      ],
    });
    /**
     * Listen on PostToolBatch to verify the batch entry shape after
     * resume reflects the final outcomes (deny + run + run), not
     * stale state from the first pass.
     */
    const batchSnapshots: PostToolBatchEntry[][] = [];
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchSnapshots.push(
            (input as PostToolBatchHookInput).entries.map((e) => ({ ...e }))
          );
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [
        createSchemaStub('tool_a'),
        createSchemaStub('tool_b'),
        createSchemaStub('tool_c'),
      ],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_a', 'step_a'],
        ['call_b', 'step_b'],
        ['call_c', 'step_c'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_a', name: 'tool_a', args: { command: 'a' } },
      { id: 'call_b', name: 'tool_b', args: { command: 'b' } },
      { id: 'call_c', name: 'tool_c', args: { command: 'c' } },
    ]);
    const config = { configurable: { thread_id: 'mixed-thread' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    if (!isInterrupted<t.HumanInterruptPayload>(interrupted)) {
      throw new Error('expected interrupt');
    }
    const payload = interrupted.__interrupt__[0].value!;
    if (payload.type !== 'tool_approval') {
      throw new Error('expected tool_approval payload');
    }
    /** Only tool_b appears in the interrupt — deny short-circuited
     * locally, allow was queued for dispatch but never reached it
     * because `interrupt()` threw inside the same node first. LangGraph
     * rolls back the entire node's effects on throw, so no host event
     * fires for any tool until after resume. This is the safe
     * semantic: partial execution while a human is being asked would
     * leak side effects ahead of approval. */
    expect(payload.action_requests).toHaveLength(1);
    expect(payload.action_requests[0].tool_call_id).toBe('call_b');
    expect(dispatchedToolNames).toEqual([]);

    const resumed = (await graph.invoke(
      new Command({ resume: [{ type: 'approve' }] }),
      config
    )) as { messages: BaseMessage[] };

    /**
     * After resume, all three tools have ToolMessages: tool_a blocked
     * (deny), tool_b ran (host approved), tool_c ran (allow). The
     * ToolNode re-executed from scratch, so both tool_b and tool_c
     * dispatch in this pass.
     */
    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(3);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_a')!.status).toBe('error');
    expect(String(byId.get('call_a')!.content)).toContain('policy:a');
    expect(byId.get('call_b')!.status).not.toBe('error');
    expect(byId.get('call_b')!.content).toBe('ran:tool_b');
    expect(byId.get('call_c')!.status).not.toBe('error');
    expect(byId.get('call_c')!.content).toBe('ran:tool_c');
    /** Both approved tools dispatched on resume; tool_a (deny) never did. */
    expect(new Set(dispatchedToolNames)).toEqual(new Set(['tool_b', 'tool_c']));
    expect(dispatchedToolNames).not.toContain('tool_a');

    /**
     * PostToolBatch is dispatched at the bottom of `dispatchToolEvents`,
     * after tool execution. On the FIRST pass `interrupt()` throws
     * before reaching that line, so PostToolBatch does NOT fire for
     * the interrupted pass. Only the resume pass yields a snapshot —
     * carrying all three entries with their final outcomes (tool_a
     * blocked by deny, tool_b approved + ran, tool_c approved + ran).
     */
    expect(batchSnapshots).toHaveLength(1);
    const finalSnapshot = batchSnapshots[0];
    /**
     * Order assertion: entries must match the original toolCalls
     * sequence (`call_a`, `call_b`, `call_c`) regardless of when each
     * outcome was recorded — `call_a` was denied synchronously in the
     * hook loop, `call_b` was approved through the resume branch,
     * `call_c` was approved+executed via the host event path. Hooks
     * correlating outcomes by position (per the API doc) depend on
     * this stability.
     */
    expect(finalSnapshot.map((e) => e.toolUseId)).toEqual([
      'call_a',
      'call_b',
      'call_c',
    ]);
    const byCallId = new Map(finalSnapshot.map((e) => [e.toolUseId, e]));
    expect(byCallId.size).toBe(3);
    expect(byCallId.get('call_a')!.status).toBe('error');
    expect(byCallId.get('call_a')!.error).toContain('policy:a');
    expect(byCallId.get('call_b')!.status).toBe('success');
    expect(byCallId.get('call_b')!.toolOutput).toBe('ran:tool_b');
    expect(byCallId.get('call_c')!.status).toBe('success');
    expect(byCallId.get('call_c')!.toolOutput).toBe('ran:tool_c');
  });

  it('mixed respond + reject in the same resume: dispatches once each, batch entries in toolCalls order', async () => {
    const stepCompletedDispatches: string[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const payload = data as {
            result?: { tool_call?: { id?: string } };
          };
          const id = payload.result?.tool_call?.id;
          if (id != null) {
            stepCompletedDispatches.push(id);
          }
          return;
        }
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const registry = new HookRegistry();
    /** Both tools `ask`; the resume picks `respond` for one and
     * `reject` for the other. Exercises the timing interaction
     * between respond's immediate dispatch and reject's deferred
     * flush in the same resume pass. */
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'review',
        }),
      ],
    });
    const batchSnapshots: PostToolBatchEntry[][] = [];
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchSnapshots.push(
            (input as PostToolBatchHookInput).entries.map((e) => ({ ...e }))
          );
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [
        createSchemaStub('respond_tool'),
        createSchemaStub('reject_tool'),
      ],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([
        ['call_respond', 'step_respond'],
        ['call_reject', 'step_reject'],
      ]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_respond', name: 'respond_tool', args: { command: 'r' } },
      { id: 'call_reject', name: 'reject_tool', args: { command: 'j' } },
    ]);
    const config = { configurable: { thread_id: 'mixed-respond-reject' } };

    await graph.invoke({ messages: [] }, config);
    /** First pass: interrupt fires before either dispatch path runs. */
    expect(stepCompletedDispatches).toEqual([]);

    const resumed = (await graph.invoke(
      new Command({
        resume: [
          { type: 'respond', responseText: 'fake answer' },
          { type: 'reject', reason: 'no thanks' },
        ],
      }),
      config
    )) as { messages: BaseMessage[] };

    /** Each tool dispatched ON_RUN_STEP_COMPLETED exactly once on
     * resume — respond via its immediate path, reject via the
     * deferred flush. */
    expect(
      stepCompletedDispatches.filter((id) => id === 'call_respond')
    ).toEqual(['call_respond']);
    expect(
      stepCompletedDispatches.filter((id) => id === 'call_reject')
    ).toEqual(['call_reject']);

    /** PostToolBatch fires once on the resume pass, with entries in
     * the original toolCalls order (respond first, reject second)
     * regardless of which dispatch path landed first into the Map. */
    expect(batchSnapshots).toHaveLength(1);
    expect(batchSnapshots[0].map((e) => e.toolUseId)).toEqual([
      'call_respond',
      'call_reject',
    ]);
    expect(batchSnapshots[0][0].status).toBe('success');
    expect(batchSnapshots[0][0].toolOutput).toBe('fake answer');
    expect(batchSnapshots[0][1].status).toBe('error');
    expect(String(batchSnapshots[0][1].error)).toContain('no thanks');

    /** ToolMessage state matches: success with response text, error with reason. */
    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(2);
    const byId = new Map(toolMessages.map((m) => [m.tool_call_id, m]));
    expect(byId.get('call_respond')!.status).not.toBe('error');
    expect(byId.get('call_respond')!.content).toBe('fake answer');
    expect(byId.get('call_reject')!.status).toBe('error');
    expect(String(byId.get('call_reject')!.content)).toContain('no thanks');
  });

  it('PostToolBatch entries preserve toolCalls order even when first call is denied and second is approved', async () => {
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve(
          request.toolCalls.map((c) => ({
            toolCallId: c.id,
            content: `ran:${c.name}`,
            status: 'success' as const,
          }))
        );
      });

    /**
     * Two different orderings to verify the asserted order really
     * tracks the input — not just incidental ordering from one path
     * landing first.
     */
    const cases: Array<{
      thread: string;
      input: Array<{ id: string; name: string; args: Record<string, unknown> }>;
      expected: string[];
    }> = [
      {
        thread: 'order-deny-first',
        input: [
          { id: 'call_first', name: 'denied_tool', args: { command: 'a' } },
          { id: 'call_second', name: 'allowed_tool', args: { command: 'b' } },
        ],
        expected: ['call_first', 'call_second'],
      },
      {
        thread: 'order-approve-first',
        input: [
          { id: 'call_first', name: 'allowed_tool', args: { command: 'a' } },
          { id: 'call_second', name: 'denied_tool', args: { command: 'b' } },
        ],
        expected: ['call_first', 'call_second'],
      },
    ];

    for (const { thread, input, expected } of cases) {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (hookInput): Promise<PreToolUseHookOutput> => {
            if (hookInput.toolName === 'denied_tool') {
              return { decision: 'deny', reason: 'no' };
            }
            return { decision: 'allow' };
          },
        ],
      });
      const captured: PostToolBatchEntry[] = [];
      registry.register('PostToolBatch', {
        hooks: [
          async (i): Promise<PostToolBatchHookOutput> => {
            captured.push(...(i as PostToolBatchHookInput).entries);
            return {};
          },
        ],
      });

      const node = new ToolNode({
        tools: [
          createSchemaStub('denied_tool'),
          createSchemaStub('allowed_tool'),
        ],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map(input.map((c) => [c.id, `step_${c.id}`])),
        hookRegistry: registry,
        humanInTheLoop: { enabled: false },
      });

      const graph = buildHITLGraph(node, input);
      await graph.invoke(
        { messages: [] },
        { configurable: { thread_id: thread } }
      );

      expect(captured.map((e) => e.toolUseId)).toEqual(expected);
    }
  });

  it('fails closed when the host resume payload carries an unknown decision type', async () => {
    /** Spy MUST be reachable inside Promise.resolve handlers — must not run after mock is restored. */
    let dispatchCalls = 0;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        dispatchCalls += 1;
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([]);
      });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask'),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'sensitive' } },
    ]);
    const config = { configurable: { thread_id: 'unknown-decision' } };

    await graph.invoke({ messages: [] }, config);

    /** Host sends a typo'd / malformed decision. Must NOT silently approve. */
    const resumed = (await graph.invoke(
      new Command({
        resume: [{ type: 'aproved' as 'approve' }],
      }),
      config
    )) as { messages: BaseMessage[] };

    const toolMessages = resumed.messages.filter(
      (m): m is ToolMessage => m._getType() === 'tool'
    );
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].status).toBe('error');
    expect(String(toolMessages[0].content)).toContain(
      'Unknown approval decision type'
    );
    /** Tool was never dispatched — fail-closed worked. */
    expect(dispatchCalls).toBe(0);
  });

  it('PostToolBatch entry sees the PostToolUse-rewritten output, not the original', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'raw-secret-1234', status: 'success' },
    ]);

    const registry = new HookRegistry();
    /** PostToolUse redacts the output before the model sees it. */
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          updatedOutput: 'raw-secret-[REDACTED]',
        }),
      ],
    });
    let batchEntries: PostToolBatchEntry[] | undefined;
    registry.register('PostToolBatch', {
      hooks: [
        async (input): Promise<PostToolBatchHookOutput> => {
          batchEntries = (input as PostToolBatchHookInput).entries;
          return {};
        },
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'fetch' } },
    ]);
    await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'batch-rewrite' } }
    );

    expect(batchEntries).toBeDefined();
    expect(batchEntries).toHaveLength(1);
    /** Batch hook sees the redacted value, not the raw secret. */
    expect(batchEntries![0].toolOutput).toBe('raw-secret-[REDACTED]');
    expect(batchEntries![0].toolOutput).not.toContain('raw-secret-1234');
  });

  it('PostToolUseFailure additionalContext is injected for the next model turn', async () => {
    /** Force the host event dispatch to return an error so the failure path runs. */
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data) => {
        if (event !== 'on_tool_execute') {
          return;
        }
        const request = data as {
          toolCalls: t.ToolCallRequest[];
          resolve: (r: t.ToolExecuteResult[]) => void;
        };
        request.resolve([
          {
            toolCallId: 'call_1',
            content: '',
            status: 'error',
            errorMessage: 'network timeout',
          },
        ]);
      });

    const registry = new HookRegistry();
    registry.register('PostToolUseFailure', {
      hooks: [
        async (): Promise<PostToolUseFailureHookOutput> => ({
          additionalContext:
            'Tool failed — suggest the user retry with a smaller batch size',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      hookRegistry: registry,
      humanInTheLoop: { enabled: false },
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'fetch' } },
    ]);
    const result = (await graph.invoke(
      { messages: [] },
      { configurable: { thread_id: 'failure-ctx' } }
    )) as { messages: BaseMessage[] };

    const injected = result.messages.find(
      (m) =>
        m._getType() === 'human' &&
        (m as { additional_kwargs?: { source?: string } }).additional_kwargs
          ?.source === 'hook'
    );
    expect(injected).toBeDefined();
    expect(String(injected!.content)).toContain(
      'suggest the user retry with a smaller batch size'
    );
  });
});

describe('AskUserQuestion — interrupt + resume', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('a node calling askUserQuestion() raises an ask_user_question interrupt and resumes with the answer', async () => {
    const { askUserQuestion } = await import('@/hitl');

    let resumedAnswer: string | undefined;

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('clarifier', () => {
        const resolution = askUserQuestion({
          question: 'Which environment?',
          options: [
            { label: 'Staging', value: 'staging' },
            { label: 'Production', value: 'production' },
          ],
        });
        resumedAnswer = resolution.answer;
        return { messages: [] };
      })
      .addEdge(START, 'clarifier')
      .addEdge('clarifier', END);
    const graph = builder.compile({ checkpointer: new MemorySaver() });

    const config = { configurable: { thread_id: 'ask-q-thread' } };

    const interrupted = (await graph.invoke({ messages: [] }, config)) as {
      __interrupt__?: Array<{ value?: t.HumanInterruptPayload }>;
    };
    expect(interrupted.__interrupt__).toBeDefined();
    const payload = interrupted.__interrupt__![0].value!;
    if (payload.type !== 'ask_user_question') {
      throw new Error('expected ask_user_question');
    }
    expect(payload.question.question).toBe('Which environment?');
    expect(payload.question.options).toHaveLength(2);

    const resolution: t.AskUserQuestionResolution = { answer: 'production' };
    await graph.invoke(new Command({ resume: resolution }), config);

    expect(resumedAnswer).toBe('production');
  });

  it('isAskUserQuestionInterrupt narrows the payload union correctly', async () => {
    const { isAskUserQuestionInterrupt, isToolApprovalInterrupt } =
      await import('@/types/hitl');

    const askPayload: t.HumanInterruptPayload = {
      type: 'ask_user_question',
      question: { question: 'why?' },
    };
    const approvalPayload: t.HumanInterruptPayload = {
      type: 'tool_approval',
      action_requests: [],
      review_configs: [],
    };

    expect(isAskUserQuestionInterrupt(askPayload)).toBe(true);
    expect(isAskUserQuestionInterrupt(approvalPayload)).toBe(false);
    expect(isToolApprovalInterrupt(approvalPayload)).toBe(true);
    expect(isToolApprovalInterrupt(askPayload)).toBe(false);
  });
});
