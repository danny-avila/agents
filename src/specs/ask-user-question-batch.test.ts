import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import {
  END,
  START,
  Command,
  StateGraph,
  MemorySaver,
  isInterrupted,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { askUserQuestion } from '@/hitl';
import { ToolNode } from '@/tools/ToolNode';

/**
 * Pins the `interruptingToolNames` guard for the `ask_user_question`
 * shape: a normal tool whose BODY raises a LangGraph `interrupt()`
 * mid-execution (via {@link askUserQuestion}) to collect a human answer.
 *
 * When such a tool shares a tool-call batch with a NON-idempotent
 * sibling (send_email, billing), LangGraph's resume contract — re-run
 * the whole interrupted node from the top — would otherwise execute the
 * sibling once on the first pass and AGAIN on resume, duplicating the
 * side effect. Declaring the interrupter in `interruptingToolNames`
 * schedules it ahead of its siblings, so the batch unwinds before any
 * sibling runs and the sibling executes exactly once (on resume).
 *
 * Runs entirely in-memory (StateGraph + MemorySaver) — no LLM, no Run
 * machinery — so it exercises the ToolNode batch-execution change
 * directly. Companion dist probe: `node -e` against dist/cjs/main.cjs.
 */

type MessagesUpdate = { messages: BaseMessage[] };
type CompiledMessagesGraph = Runnable<unknown, { messages: BaseMessage[] }> & {
  invoke(input: unknown, config?: RunnableConfig): Promise<unknown>;
};

/** Tool whose body suspends the run to ask the user (no side effect). */
function makeAskTool(): StructuredToolInterface {
  return tool(
    async () => {
      const { answer } = askUserQuestion({ question: 'Proceed?' });
      return `answered: ${answer}`;
    },
    {
      name: 'ask_user_question',
      description: 'suspends to collect a human answer',
      schema: z.object({}).passthrough(),
    }
  ) as unknown as StructuredToolInterface;
}

/** Non-idempotent sibling — records every body invocation. */
function makeSideEffectTool(sideEffect: () => string): StructuredToolInterface {
  return tool(async () => sideEffect(), {
    name: 'side_effect',
    description: 'non-idempotent side effect (e.g. send_email)',
    schema: z.object({}).passthrough(),
  }) as unknown as StructuredToolInterface;
}

function buildGraph(
  node: ToolNode,
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }>
): CompiledMessagesGraph {
  const builder = new StateGraph(MessagesAnnotation)
    .addNode(
      'agent',
      (): MessagesUpdate => ({
        messages: [new AIMessage({ content: '', tool_calls: toolCalls })],
      })
    )
    .addNode('tools', node)
    .addEdge(START, 'agent')
    .addEdge('agent', 'tools')
    .addEdge('tools', END);
  return builder.compile({
    checkpointer: new MemorySaver(),
  }) as unknown as CompiledMessagesGraph;
}

const ASK = { id: 'ask1', name: 'ask_user_question', args: {} };
const SIDE_EFFECT = { id: 'se1', name: 'side_effect', args: {} };

describe('ask_user_question batched with a sibling tool', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('direct sibling — guarded by interruptingToolNames', () => {
    // The exposed shape: both tools run in-process, so absent the guard
    // they share one Promise.all and the sibling double-executes.
    for (const order of [
      { label: '[ask, side_effect]', calls: [ASK, SIDE_EFFECT] },
      { label: '[side_effect, ask]', calls: [SIDE_EFFECT, ASK] },
    ]) {
      it(`runs the sibling exactly once across pause→resume — order ${order.label}`, async () => {
        const sideEffect = jest.fn(() => 'EXECUTED');
        const node = new ToolNode({
          tools: [makeAskTool(), makeSideEffectTool(sideEffect)],
          eventDrivenMode: true,
          directToolNames: new Set(['ask_user_question', 'side_effect']),
          interruptingToolNames: new Set(['ask_user_question']),
        });
        const graph = buildGraph(node, order.calls);
        const config = {
          configurable: { thread_id: `guarded-${order.label}` },
        };

        const first = await graph.invoke({ messages: [] }, config);
        expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
        // Guard: the interrupter is scheduled first, so the sibling has
        // NOT run when the batch unwinds on the first pass.
        expect(sideEffect).not.toHaveBeenCalled();

        const second = await graph.invoke(
          new Command({ resume: { answer: 'yes' } }),
          config
        );

        // Sibling ran exactly once, on the resume pass.
        expect(sideEffect).toHaveBeenCalledTimes(1);

        const messages = (second as { messages: ToolMessage[] }).messages;
        const seMsg = messages.find(
          (m) => m instanceof ToolMessage && m.name === 'side_effect'
        ) as ToolMessage;
        expect(String(seMsg.content)).toBe('EXECUTED');
        const askMsg = messages.find(
          (m) => m instanceof ToolMessage && m.name === 'ask_user_question'
        ) as ToolMessage;
        expect(String(askMsg.content)).toBe('answered: yes');
      });
    }
  });

  describe('baseline (no guard) — documents the defect the guard closes', () => {
    it('double-executes an unguarded direct sibling on resume', async () => {
      const sideEffect = jest.fn(() => 'EXECUTED');
      const node = new ToolNode({
        tools: [makeAskTool(), makeSideEffectTool(sideEffect)],
        eventDrivenMode: true,
        directToolNames: new Set(['ask_user_question', 'side_effect']),
        // interruptingToolNames intentionally omitted.
      });
      const graph = buildGraph(node, [ASK, SIDE_EFFECT]);
      const config = { configurable: { thread_id: 'baseline-unguarded' } };

      const first = await graph.invoke({ messages: [] }, config);
      expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
      // Unguarded: the sibling shared the interrupter's Promise.all and
      // already ran its side effect before the pause.
      expect(sideEffect).toHaveBeenCalledTimes(1);

      await graph.invoke(new Command({ resume: { answer: 'yes' } }), config);
      // LangGraph re-runs the batch → the side effect fires a SECOND time.
      expect(sideEffect).toHaveBeenCalledTimes(2);
    });
  });

  describe('event-dispatched sibling — already safe without the guard', () => {
    it('never runs the event sibling on the first pass', async () => {
      const sideEffect = jest.fn(() => 'EXECUTED');
      // Host executes event-dispatched tools; record side_effect there.
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const req = data as {
            toolCalls: Array<{ id: string; name: string }>;
            resolve: (results: t.ToolExecuteResult[]) => void;
          };
          const results = req.toolCalls.map((tc) => {
            if (tc.name === 'side_effect') {
              sideEffect();
            }
            return {
              toolCallId: tc.id,
              content: 'EXECUTED',
              status: 'success' as const,
            };
          });
          req.resolve(results);
          return true;
        });

      const node = new ToolNode({
        tools: [
          makeAskTool(),
          // schema-only stub; the host "executes" side_effect via event.
          tool(async () => 'unused', {
            name: 'side_effect',
            description: 'event tool',
            schema: z.object({}).passthrough(),
          }) as unknown as StructuredToolInterface,
        ],
        eventDrivenMode: true,
        // side_effect is NOT direct → dispatched as an event.
        directToolNames: new Set(['ask_user_question']),
        interruptingToolNames: new Set(['ask_user_question']),
      });
      const graph = buildGraph(node, [SIDE_EFFECT, ASK]);
      const config = { configurable: { thread_id: 'event-sibling' } };

      const first = await graph.invoke({ messages: [] }, config);
      expect(isInterrupted<t.HumanInterruptPayload>(first)).toBe(true);
      // The direct interrupter unwinds the batch before event tools are
      // dispatched, so the host never executed side_effect.
      expect(sideEffect).not.toHaveBeenCalled();

      await graph.invoke(new Command({ resume: { answer: 'yes' } }), config);
      expect(sideEffect).toHaveBeenCalledTimes(1);
    });
  });

  describe('interruptingToolNames only reorders; never forces direct (Codex #294)', () => {
    // A self-spawned child scrubs inherited `graphTools` but keeps the
    // event `toolDefinition`, so a name like `ask_user_question` exists
    // only as a schema-only stub. Propagating `interruptingToolNames`
    // must NOT force that name onto the direct path — invoking the stub
    // throws "should not be invoked directly". It must stay dispatched.
    it('dispatches an interrupting name that has no in-process implementation', async () => {
      const directlyInvoked = jest.fn();
      // Mirrors createSchemaOnlyTool: throws if invoked in-process.
      const stub = tool(
        async () => {
          directlyInvoked();
          throw new Error(
            'Tool "ask_user_question" should not be invoked directly in event-driven mode.'
          );
        },
        {
          name: 'ask_user_question',
          description: 'schema-only event stub',
          schema: z.object({}).passthrough(),
        }
      ) as unknown as StructuredToolInterface;

      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const req = data as {
            toolCalls: Array<{ id: string; name: string }>;
            resolve: (results: t.ToolExecuteResult[]) => void;
          };
          req.resolve(
            req.toolCalls.map((tc) => ({
              toolCallId: tc.id,
              content: 'HOST-HANDLED',
              status: 'success' as const,
            }))
          );
          return true;
        });

      const node = new ToolNode({
        tools: [stub],
        eventDrivenMode: true,
        // NOT in directToolNames → it is an event tool, even though it is
        // named in interruptingToolNames.
        interruptingToolNames: new Set(['ask_user_question']),
      });
      const graph = buildGraph(node, [ASK]);
      const config = { configurable: { thread_id: 'stub-stays-event' } };

      const result = await graph.invoke({ messages: [] }, config);

      // Dispatched to the host, NOT invoked in-process.
      expect(directlyInvoked).not.toHaveBeenCalled();
      const msg = (result as { messages: ToolMessage[] }).messages.find(
        (m) => m instanceof ToolMessage
      ) as ToolMessage;
      expect(msg.status).toBe('success');
      expect(String(msg.content)).toBe('HOST-HANDLED');
    });
  });
});
