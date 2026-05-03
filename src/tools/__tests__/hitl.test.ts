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
  PostToolBatchEntry,
  PostToolBatchHookInput,
  PostToolBatchHookOutput,
  RunStartHookOutput,
  UserPromptSubmitHookOutput,
} from '@/hooks';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { Providers as providers } from '@/common';
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

  it('raises an interrupt when `humanInTheLoop` is omitted (default-on)', async () => {
    mockEventDispatch([
      { toolCallId: 'call_1', content: 'host-result', status: 'success' },
    ]);
    const node = new ToolNode({
      tools: [createSchemaStub('echo')],
      eventDrivenMode: true,
      agentId: 'agent-x',
      toolCallStepIds: new Map([['call_1', 'step_call_1']]),
      hookRegistry: makeHookRegistry('ask', 'default-on'),
      // humanInTheLoop intentionally omitted — should default to enabled
    });

    const graph = buildHITLGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'list /' } },
    ]);
    const config = { configurable: { thread_id: 'thread-hitl-default' } };

    const interrupted = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(interrupted)).toBe(true);
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

  it('Run.create installs a MemorySaver fallback by default (HITL on, no checkpointer provided)', async () => {
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
      // humanInTheLoop intentionally omitted — default is on
    });

    expect(run.Graph?.compileOptions?.checkpointer).toBeDefined();
    expect(run.Graph?.compileOptions?.checkpointer).toBeInstanceOf(MemorySaver);
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

  it('Run.create preserves a host-supplied checkpointer (default-on HITL keeps it intact)', async () => {
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
     * `processStream` should still have cleared the registry signal in
     * its `finally` block so a subsequent call starts fresh. */
    expect(registry.getHaltSignal()).toBeUndefined();
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

    /** preventContinuation was on an async output → ignored → no halt signal. */
    expect(registry.getHaltSignal()).toBeUndefined();
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
