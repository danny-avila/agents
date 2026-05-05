import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import {
  END,
  START,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
  Command,
} from '@langchain/langgraph';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { PreToolUseHookOutput } from '@/hooks';
import type * as t from '@/types';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

/**
 * Pins the resume-scope behaviour for direct-path interrupts. The
 * existing JSDoc on `HumanInTheLoopConfig` warned that mixed
 * direct+event batches re-execute the direct half on resume because
 * LangGraph rolls back the entire ToolNode on `interrupt()`. After
 * lifting HITL into the direct path, the same rollback applies — but
 * for direct-only batches too, since `interrupt()` always rewinds to
 * the start of the suspending node.
 *
 * This test makes that concrete: a direct tool whose call is gated
 * by a PreToolUse 'ask' hook fires its execute callback exactly N
 * times, where N == number of resume passes that lead to an
 * 'approve'. Side-effect-bearing tools should be designed
 * idempotent regardless of whether they're direct or event-dispatched.
 */

function aiCall(
  callId: string,
  name: string,
  args: Record<string, unknown>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id: callId, name, args }],
  });
}

type MessagesUpdate = { messages: BaseMessage[] };
type CompiledMessagesGraph = Runnable<unknown, { messages: BaseMessage[] }> & {
  invoke(input: unknown, config?: RunnableConfig): Promise<unknown>;
};

function buildGraph(
  toolNode: ToolNode,
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }>
): CompiledMessagesGraph {
  let agentInvocations = 0;
  const builder = new StateGraph(MessagesAnnotation)
    .addNode('agent', (): MessagesUpdate => {
      agentInvocations += 1;
      if (agentInvocations === 1) {
        return { messages: [aiCall(toolCalls[0].id, toolCalls[0].name, toolCalls[0].args)] };
      }
      return { messages: [] };
    })
    .addNode('tools', toolNode)
    .addEdge(START, 'agent')
    .addEdge('agent', 'tools')
    .addEdge('tools', END);
  return builder.compile({
    checkpointer: new MemorySaver(),
  }) as unknown as CompiledMessagesGraph;
}

describe('direct-path HITL: resume scope', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('re-executes the direct tool body on resume when interrupt() fires from the direct path', async () => {
    const sideEffect = jest.fn(() => 'EXECUTED');
    const directTool = tool(
      async () => sideEffect(),
      {
        name: 'echo',
        description: 'direct tool that records every body invocation',
        schema: z.object({ command: z.string().optional() }).passthrough(),
      }
    ) as unknown as StructuredToolInterface;

    const registry = new HookRegistry();
    let hookInvocations = 0;
    // Realistic shape: ask the FIRST time the hook sees a tool call,
    // allow on subsequent invocations. A real policy hook would key
    // off persistent state (an "approved paths" set, a session
    // approval token, etc.); we just count.
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => {
          hookInvocations += 1;
          if (hookInvocations === 1) {
            return { decision: 'ask', reason: 'first-time-ask' };
          }
          return { decision: 'allow' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [directTool],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
      humanInTheLoop: { enabled: true },
    });

    const graph = buildGraph(node, [
      { id: 'call_1', name: 'echo', args: { command: 'go' } },
    ]);
    const config = { configurable: { thread_id: 'thread-resume-1' } };

    const first = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);

    // Body should NOT have run yet — the hook intercepted before
    // the tool executed.
    expect(sideEffect).not.toHaveBeenCalled();
    // Hook fires once per attempt; the first interrupt is attempt #1.
    expect(hookInvocations).toBe(1);

    // Resume with approve. LangGraph re-enters the ToolNode body,
    // PreToolUse fires again (the "idempotency" caveat — see the
    // HumanInTheLoopConfig JSDoc).
    const second = await graph.invoke(
      { resume: [{ tool_call_id: 'call_1', type: 'approve' }] },
      config
    );

    // PreToolUse fired a SECOND time on the resume re-entry.
    expect(hookInvocations).toBe(2);
    // Body executed exactly once — only on the resume pass, after
    // the hook returned 'allow'. The interrupted first pass never
    // reached the body. This pins the resume scope: LangGraph
    // restarts the ToolNode at the top, but the body itself only
    // runs once because the first pass interrupted before the
    // body, not after.
    expect(sideEffect).toHaveBeenCalledTimes(1);

    // Result should carry the executed output.
    const messages = (second as { messages: ToolMessage[] }).messages;
    const toolMsg = messages.find((m) => m instanceof ToolMessage) as ToolMessage;
    expect(String(toolMsg.content)).toBe('EXECUTED');
  });

  it('re-runs sibling tools that already executed in the same batch when a later tool interrupts', async () => {
    // Two direct tools in the same batch. Tool A is a no-op the
    // hook always allows; tool B asks the first time. On resume,
    // LangGraph rewinds the entire ToolNode — meaning tool A's
    // body runs twice (once per pass). This pins the side-effect
    // caveat: tools called BEFORE an interrupting sibling MUST be
    // idempotent regardless of whether they're direct or
    // event-dispatched.
    const aSideEffect = jest.fn(() => 'A-OK');
    const bSideEffect = jest.fn(() => 'B-OK');
    const a = tool(async () => aSideEffect(), {
      name: 'tool_a',
      description: 'allowed direct tool',
      schema: z.object({}).passthrough(),
    }) as unknown as StructuredToolInterface;
    const b = tool(async () => bSideEffect(), {
      name: 'tool_b',
      description: 'asks first time, allows after',
      schema: z.object({}).passthrough(),
    }) as unknown as StructuredToolInterface;

    let bHookInvocations = 0;
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async ({ toolName }): Promise<PreToolUseHookOutput> => {
          if (toolName === 'tool_b') {
            bHookInvocations += 1;
            if (bHookInvocations === 1) {
              return { decision: 'ask', reason: 'b-first-ask' };
            }
          }
          return { decision: 'allow' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [a, b],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['tool_a', 'tool_b']),
      humanInTheLoop: { enabled: true },
    });

    const builder = new StateGraph(MessagesAnnotation)
      .addNode('agent', (): MessagesUpdate => ({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'a1', name: 'tool_a', args: {} },
              { id: 'b1', name: 'tool_b', args: {} },
            ],
          }),
        ],
      }))
      .addNode('tools', node)
      .addEdge(START, 'agent')
      .addEdge('agent', 'tools')
      .addEdge('tools', END);
    const graph = builder.compile({
      checkpointer: new MemorySaver(),
    }) as unknown as CompiledMessagesGraph;

    const config = { configurable: { thread_id: 'thread-mixed-batch' } };
    const first = await graph.invoke({ messages: [] }, config);
    expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);

    // First pass: A ran (allowed), B asked.
    expect(aSideEffect).toHaveBeenCalledTimes(1);
    expect(bSideEffect).not.toHaveBeenCalled();
    expect(bHookInvocations).toBe(1);

    await graph.invoke(
      { resume: [{ tool_call_id: 'b1', type: 'approve' }] },
      config
    );

    // Resume: LangGraph rewinds the ToolNode and re-enters from the
    // start. A's body runs AGAIN. B's body runs once now that the
    // hook allowed.
    expect(aSideEffect).toHaveBeenCalledTimes(2);
    expect(bSideEffect).toHaveBeenCalledTimes(1);
    expect(bHookInvocations).toBe(2);
  });

  describe('edit decision (Codex P1 #16)', () => {
    it('applies decision.updatedInput (the documented field) to the executed tool args', async () => {
      const receivedArgs: Array<Record<string, unknown>> = [];
      const directTool = tool(
        async (input) => {
          receivedArgs.push(input as Record<string, unknown>);
          return JSON.stringify(input);
        },
        {
          name: 'echo',
          description: 'records the args it actually executed with',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      // Pattern: hook ALWAYS asks. interrupt() throws on the first
      // pass (sends the host the askEntry payload) and RETURNS the
      // resume value on the second pass — the resume value is what
      // actually carries the approve/edit/reject decision.
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (): Promise<PreToolUseHookOutput> => ({
            decision: 'ask',
            allowedDecisions: ['approve', 'edit'],
          }),
        ],
      });

      const node = new ToolNode({
        tools: [directTool],
        eventDrivenMode: true,
        hookRegistry: registry,
        directToolNames: new Set(['echo']),
        humanInTheLoop: { enabled: true },
      });

      const graph = buildGraph(node, [
        { id: 'call_1', name: 'echo', args: { command: 'original' } },
      ]);
      const config = { configurable: { thread_id: 'thread-edit-1' } };

      const first = await graph.invoke({ messages: [] }, config);
      expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
      // Body did not run yet.
      expect(receivedArgs).toEqual([]);

      const second = await graph.invoke(
        new Command({
          resume: [
            { type: 'edit', updatedInput: { command: 'edited-by-host' } },
          ],
        }),
        config
      );

      // The whole point of the fix: the edited input flows through.
      // Pre-fix the direct path read `decision.args` (wrong field) so
      // updatedInput was silently dropped and the tool ran with
      // `{ command: 'original' }`.
      expect(receivedArgs).toHaveLength(1);
      expect(receivedArgs[0]).toEqual({ command: 'edited-by-host' });

      const messages = (second as { messages: ToolMessage[] }).messages;
      const toolMsg = messages.find(
        (m) => m instanceof ToolMessage
      ) as ToolMessage;
      expect(String(toolMsg.content)).toContain('edited-by-host');
    });

    it('fails closed when updatedInput is missing or wrong-shaped', async () => {
      const directTool = tool(
        async () => 'should-not-execute',
        {
          name: 'echo',
          description: 'must not execute on malformed edit',
          schema: z.object({ command: z.string().optional() }).passthrough(),
        }
      ) as unknown as StructuredToolInterface;

      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (): Promise<PreToolUseHookOutput> => ({
            decision: 'ask',
            allowedDecisions: ['approve', 'edit'],
          }),
        ],
      });

      const node = new ToolNode({
        tools: [directTool],
        eventDrivenMode: true,
        hookRegistry: registry,
        directToolNames: new Set(['echo']),
        humanInTheLoop: { enabled: true },
      });

      const graph = buildGraph(node, [
        { id: 'call_1', name: 'echo', args: { command: 'original' } },
      ]);
      const config = { configurable: { thread_id: 'thread-edit-2' } };

      await graph.invoke({ messages: [] }, config);

      // Send `{ type: 'edit' }` with no updatedInput at all (simulates
      // a host that misnamed the field, e.g. used `args` like the old
      // bug expected). Must fail closed instead of executing.
      const second = await graph.invoke(
        new Command({
          resume: [
            {
              type: 'edit',
              args: { command: 'this-field-name-is-wrong' },
            } as unknown as t.ToolApprovalDecision,
          ],
        }),
        config
      );

      const messages = (second as { messages: ToolMessage[] }).messages;
      const toolMsg = messages.find(
        (m) => m instanceof ToolMessage
      ) as ToolMessage;
      expect(toolMsg.status).toBe('error');
      expect(String(toolMsg.content)).toContain(
        'Decision "edit" missing object updatedInput'
      );
      expect(String(toolMsg.content)).not.toContain('should-not-execute');
    });
  });

  describe('usage counter stability across resume (Codex P2 #30)', () => {
    it('turn stays the same across an interrupt + resume — does not double-increment', async () => {
      // Pre-fix the P2 #27 turn-race fix incremented before the
      // hook fired, and never rolled back on `ask`. LangGraph
      // re-runs ToolNode from the start on resume, so a single
      // call that asks once before approval got turn=1 instead of
      // turn=0. Now turns are cached per call.id and re-used on
      // re-entry.
      const observed: number[] = [];
      const directTool = tool(
        async (_, config) => {
          const tc = (config as { toolCall?: { turn?: number } } | undefined)
            ?.toolCall;
          if (typeof tc?.turn === 'number') observed.push(tc.turn);
          return 'EXECUTED';
        },
        {
          name: 'echo',
          description: 'records the turn it ran under',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      // Hook ALWAYS asks. The resume value is what unblocks.
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async (): Promise<PreToolUseHookOutput> => ({
            decision: 'ask',
            allowedDecisions: ['approve'],
          }),
        ],
      });

      const node = new ToolNode({
        tools: [directTool],
        eventDrivenMode: true,
        hookRegistry: registry,
        directToolNames: new Set(['echo']),
        humanInTheLoop: { enabled: true },
      });

      const graph = buildGraph(node, [
        { id: 'call_1', name: 'echo', args: { command: 'go' } },
      ]);
      const config = { configurable: { thread_id: 'thread-turn-stable' } };

      const first = await graph.invoke({ messages: [] }, config);
      expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);

      await graph.invoke(
        new Command({ resume: [{ type: 'approve' }] }),
        config
      );

      // Body ran once. The turn it observed must be 0 (the slot
      // assigned on the FIRST entry, reused on resume), not 1
      // (which is what pre-fix produced because the second entry
      // re-incremented).
      expect(observed).toEqual([0]);
    });
  });
});
