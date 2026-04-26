import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import {
  ToolOutputReferenceRegistry,
  TOOL_OUTPUT_REF_KEY,
} from '../toolOutputReferences';

/**
 * Captures the `command` arg each time the tool is invoked and returns
 * a configurable string output. The tool shape matches a typical bash
 * executor: single required string arg, string response.
 */
function createEchoTool(options: {
  capturedArgs: string[];
  outputs: string[];
  name?: string;
}): StructuredToolInterface {
  const { capturedArgs, outputs, name = 'echo' } = options;
  let callCount = 0;
  return tool(
    async (input) => {
      const args = input as { command: string };
      capturedArgs.push(args.command);
      const output = outputs[callCount] ?? outputs[outputs.length - 1];
      callCount++;
      return output;
    },
    {
      name,
      description: 'Echo test tool',
      schema: z.object({ command: z.string() }),
    }
  ) as unknown as StructuredToolInterface;
}

function aiMsgWithCalls(
  calls: Array<{ id: string; name: string; command: string }>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: calls.map((c) => ({
      id: c.id,
      name: c.name,
      args: { command: c.command },
    })),
  });
}

async function invokeBatch(
  toolNode: ToolNode,
  calls: Array<{ id: string; name: string; command: string }>,
  runId: string = 'test-run'
): Promise<ToolMessage[]> {
  const aiMsg = aiMsgWithCalls(calls);
  const result = (await toolNode.invoke(
    { messages: [aiMsg] },
    { configurable: { run_id: runId } }
  )) as ToolMessage[] | { messages: ToolMessage[] };
  return Array.isArray(result) ? result : result.messages;
}

describe('ToolNode tool output references', () => {
  describe('disabled (default)', () => {
    it('does not annotate outputs or register anything when disabled', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['plain-output'],
      });
      const node = new ToolNode({ tools: [t1] });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'hello' },
      ]);

      expect(msg.content).toBe('plain-output');
      expect(node._unsafeGetToolOutputRegistry()).toBeUndefined();
    });

    it('does not substitute placeholders when disabled', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({ capturedArgs, outputs: ['X'] });
      const node = new ToolNode({ tools: [t1] });

      await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'raw {{tool0turn0}}' },
      ]);

      expect(capturedArgs).toEqual(['raw {{tool0turn0}}']);
    });
  });

  describe('enabled', () => {
    it('annotates string outputs with a [ref: …] prefix line', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['hello world'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      expect(msg.content).toBe('[ref: tool0turn0]\nhello world');
    });

    it('injects _ref into JSON-object string outputs', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['{"a":1,"b":"x"}'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      const parsed = JSON.parse(msg.content as string);
      expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
      expect(parsed.a).toBe(1);
    });

    it('uses the [ref: …] prefix for JSON array outputs', async () => {
      const t1 = createEchoTool({ capturedArgs: [], outputs: ['[1,2,3]'] });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'echo', command: 'run' },
      ]);

      expect(msg.content).toBe('[ref: tool0turn0]\n[1,2,3]');
    });

    it('registers the un-annotated output for piping into later calls', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['raw-payload', 'second-call'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'first' }]);
      await invokeBatch(node, [
        {
          id: 'c2',
          name: 'echo',
          command: 'echo {{tool0turn0}}',
        },
      ]);

      expect(capturedArgs).toEqual(['first', 'echo raw-payload']);
    });

    it('increments the turn counter per ToolNode batch', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['one', 'two', 'three'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [m0] = await invokeBatch(node, [
        { id: 'b1c1', name: 'echo', command: 'a' },
      ]);
      const [m1] = await invokeBatch(node, [
        { id: 'b2c1', name: 'echo', command: 'b' },
      ]);
      const [m2] = await invokeBatch(node, [
        { id: 'b3c1', name: 'echo', command: '{{tool0turn1}}' },
      ]);

      expect(m0.content).toContain('[ref: tool0turn0]');
      expect(m1.content).toContain('[ref: tool0turn1]');
      expect(m2.content).toContain('[ref: tool0turn2]');
      expect(capturedArgs[2]).toBe('two');
    });

    it('uses array index within a batch for the tool<idx> segment', async () => {
      const capturedA: string[] = [];
      const capturedB: string[] = [];
      const tA = createEchoTool({
        capturedArgs: capturedA,
        outputs: ['A-out'],
        name: 'alpha',
      });
      const tB = createEchoTool({
        capturedArgs: capturedB,
        outputs: ['B-out'],
        name: 'beta',
      });
      const node = new ToolNode({
        tools: [tA, tB],
        toolOutputReferences: { enabled: true },
      });

      const messages = await invokeBatch(node, [
        { id: 'c1', name: 'alpha', command: 'a' },
        { id: 'c2', name: 'beta', command: 'b' },
      ]);

      expect(messages[0].content).toContain('[ref: tool0turn0]');
      expect(messages[1].content).toContain('[ref: tool1turn0]');
    });

    it('reports unresolved placeholders after the output', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({ capturedArgs, outputs: ['done'] });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        {
          id: 'c1',
          name: 'echo',
          command: 'see {{tool9turn9}}',
        },
      ]);

      expect(capturedArgs[0]).toBe('see {{tool9turn9}}');
      expect(msg.content).toContain('[unresolved refs: tool9turn9]');
    });

    it('stores the raw untruncated output in the registry, independent of the LLM-visible truncation', async () => {
      const raw = 'X'.repeat(8_000);
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: [raw, 'second'],
      });
      const node = new ToolNode({
        tools: [t1],
        maxToolResultChars: 200,
        toolOutputReferences: { enabled: true },
      });

      const [first] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'echo', command: 'first' }],
        'raw-preservation'
      );

      expect((first.content as string).length).toBeLessThan(raw.length);
      expect(first.content).toContain('truncated');

      await invokeBatch(
        node,
        [{ id: 'c2', name: 'echo', command: 'echo {{tool0turn0}}' }],
        'raw-preservation'
      );

      expect(capturedArgs[1]).toBe(`echo ${raw}`);
      expect(
        node
          ._unsafeGetToolOutputRegistry()!
          .get('raw-preservation', 'tool0turn0')
      ).toBe(raw);
    });

    it('uses each batch\'s own turn when ToolNode is invoked concurrently within a run', async () => {
      const gates: Record<string, () => void> = {};
      const slowTool = tool(
        async (input) => {
          const args = input as { command: string };
          await new Promise<void>((resolve) => {
            gates[args.command] = resolve;
          });
          return `output-${args.command}`;
        },
        {
          name: 'slow',
          description: 'awaits a per-command gate',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [slowTool],
        toolOutputReferences: { enabled: true },
      });

      // Two batches of the SAME run, started concurrently — batch A
      // captures turn 0 in its sync prefix, batch B captures turn 1.
      // If the turn were read from shared state after the awaits, the
      // reads would race and both batches would see the latest value.
      const first = node.invoke(
        {
          messages: [aiMsgWithCalls([{ id: 'a', name: 'slow', command: 'A' }])],
        },
        { configurable: { run_id: 'concurrent-run' } }
      );
      const second = node.invoke(
        {
          messages: [aiMsgWithCalls([{ id: 'b', name: 'slow', command: 'B' }])],
        },
        { configurable: { run_id: 'concurrent-run' } }
      );

      await new Promise<void>((resolve) => {
        const check = (): void => {
          if (
            Object.prototype.hasOwnProperty.call(gates, 'A') &&
            Object.prototype.hasOwnProperty.call(gates, 'B')
          ) {
            resolve();
          } else {
            setTimeout(check, 5);
          }
        };
        check();
      });
      // Release B first (the later-scheduled batch), then A. Under the
      // old code this would bake turn=1 into BOTH results because
      // `currentTurn` was overwritten during B's sync prefix.
      gates.B();
      gates.A();

      const [resA, resB] = (await Promise.all([first, second])) as Array<{
        messages: ToolMessage[];
      }>;

      expect(resA.messages[0].content).toContain('[ref: tool0turn0]');
      expect(resA.messages[0].content).toContain('output-A');
      expect(resB.messages[0].content).toContain('[ref: tool0turn1]');
      expect(resB.messages[0].content).toContain('output-B');

      const registry = node._unsafeGetToolOutputRegistry()!;
      expect(registry.get('concurrent-run', 'tool0turn0')).toBe('output-A');
      expect(registry.get('concurrent-run', 'tool0turn1')).toBe('output-B');
    });

    it('clips registered outputs to maxOutputSize', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['{"payload":"' + 'y'.repeat(200) + '"}'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true, maxOutputSize: 40 },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'x' }]);

      const registry = node._unsafeGetToolOutputRegistry();
      expect(registry).toBeDefined();
      expect(
        registry!.get('test-run', 'tool0turn0')!.length
      ).toBeLessThanOrEqual(40);
    });

    it('honors maxTotalSize via FIFO eviction across batches', async () => {
      const t1 = createEchoTool({
        capturedArgs: [],
        outputs: ['aaaaa', 'bbbbb', 'ccccc'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: {
          enabled: true,
          maxOutputSize: 10,
          maxTotalSize: 10,
        },
      });

      await invokeBatch(node, [{ id: 'c1', name: 'echo', command: 'x' }]);
      await invokeBatch(node, [{ id: 'c2', name: 'echo', command: 'x' }]);
      await invokeBatch(node, [{ id: 'c3', name: 'echo', command: 'x' }]);

      const registry = node._unsafeGetToolOutputRegistry()!;
      expect(registry.get('test-run', 'tool0turn0')).toBeUndefined();
      expect(registry.get('test-run', 'tool0turn1')).toBe('bbbbb');
      expect(registry.get('test-run', 'tool0turn2')).toBe('ccccc');
    });

    it('does not register error outputs', async () => {
      const boom = tool(
        async () => {
          throw new Error('nope');
        },
        {
          name: 'boom',
          description: 'always errors',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [boom],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'boom', command: 'x' },
      ]);

      expect((msg.content as string).startsWith('[ref:')).toBe(false);
      expect(
        node._unsafeGetToolOutputRegistry()!.get('test-run', 'tool0turn0')
      ).toBeUndefined();
    });

    it('surfaces unresolved refs on thrown-error ToolMessages', async () => {
      const boom = tool(
        async () => {
          throw new Error('nope');
        },
        {
          name: 'boom',
          description: 'always errors',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [boom],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'boom', command: 'see {{tool9turn9}}' },
      ]);

      expect(msg.content).toContain('Error: nope');
      expect(msg.content).toContain('[unresolved refs: tool9turn9]');
    });

    it('surfaces unresolved refs on tool-returned error ToolMessages', async () => {
      const errReturn = tool(
        async () =>
          new ToolMessage({
            status: 'error',
            content: 'handled failure',
            name: 'errReturn',
            tool_call_id: 'c1',
          }),
        {
          name: 'errReturn',
          description: 'returns error ToolMessage',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [errReturn],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(node, [
        { id: 'c1', name: 'errReturn', command: 'see {{tool9turn9}}' },
      ]);

      expect(msg.content).toContain('handled failure');
      expect(msg.content).toContain('[unresolved refs: tool9turn9]');
    });

    it('isolates state between overlapping runs on the same ToolNode', async () => {
      const sharedRegistry = new ToolOutputReferenceRegistry();
      const capturedArgs: string[] = [];
      const tl = createEchoTool({
        capturedArgs,
        outputs: ['out-A', 'out-B', 'resolved-in-B'],
      });

      const node = new ToolNode({
        tools: [tl],
        toolOutputRegistry: sharedRegistry,
      });

      // Run A records `tool0turn0` → 'out-A' in its bucket.
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'a1', name: 'echo', command: 'a' }]),
          ],
        },
        { configurable: { run_id: 'run-A' } }
      );
      expect(sharedRegistry.get('run-A', 'tool0turn0')).toBe('out-A');

      // Run B records `tool0turn0` → 'out-B' in its own bucket.
      // Under the old global-reset design, starting run B would have
      // wiped run A's registered output; with partitioning, A's
      // bucket survives untouched.
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'b1', name: 'echo', command: 'b' }]),
          ],
        },
        { configurable: { run_id: 'run-B' } }
      );
      expect(sharedRegistry.get('run-A', 'tool0turn0')).toBe('out-A');
      expect(sharedRegistry.get('run-B', 'tool0turn0')).toBe('out-B');

      // Run B's next batch resolves `{{tool0turn0}}` against its own
      // partition (out-B), not run A's partition (out-A).
      await node.invoke(
        {
          messages: [
            aiMsgWithCalls([
              { id: 'b2', name: 'echo', command: 'see {{tool0turn0}}' },
            ]),
          ],
        },
        { configurable: { run_id: 'run-B' } }
      );
      expect(capturedArgs[2]).toBe('see out-B');
    });

    it('gives concurrent anonymous invocations independent scopes', async () => {
      const gates: Record<string, () => void> = {};
      const slowTool = tool(
        async (input) => {
          const args = input as { command: string };
          await new Promise<void>((resolve) => {
            gates[args.command] = resolve;
          });
          return `out-${args.command}`;
        },
        {
          name: 'slow',
          description: 'awaits a per-command gate',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [slowTool],
        toolOutputReferences: { enabled: true },
      });

      // Two invocations without `run_id`, started concurrently. Before
      // the unique-anon-scope fix, the second invocation's sync prefix
      // would have deleted the shared anonymous bucket that the first
      // invocation's tool was about to register into.
      const first = node.invoke({
        messages: [aiMsgWithCalls([{ id: 'a1', name: 'slow', command: 'A' }])],
      });
      const second = node.invoke({
        messages: [aiMsgWithCalls([{ id: 'b1', name: 'slow', command: 'B' }])],
      });

      await new Promise<void>((resolve) => {
        const check = (): void => {
          if (
            Object.prototype.hasOwnProperty.call(gates, 'A') &&
            Object.prototype.hasOwnProperty.call(gates, 'B')
          ) {
            resolve();
          } else {
            setTimeout(check, 5);
          }
        };
        check();
      });
      gates.B();
      gates.A();

      const [resA, resB] = (await Promise.all([first, second])) as Array<{
        messages: ToolMessage[];
      }>;

      // Each invocation produces its own annotated output — neither's
      // registered tool0turn0 was clobbered by the other's sync-prefix
      // reset.
      expect(resA.messages[0].content).toContain('[ref: tool0turn0]');
      expect(resA.messages[0].content).toContain('out-A');
      expect(resB.messages[0].content).toContain('[ref: tool0turn0]');
      expect(resB.messages[0].content).toContain('out-B');
    });

    it('clears state on every batch when run_id is absent (anonymous caller)', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['first-anonymous', 'second-anonymous'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      await node.invoke({
        messages: [aiMsgWithCalls([{ id: 'a1', name: 'echo', command: 'a' }])],
      });
      const result = (await node.invoke({
        messages: [
          aiMsgWithCalls([
            { id: 'a2', name: 'echo', command: 'echo {{tool0turn0}}' },
          ]),
        ],
      })) as { messages: ToolMessage[] };

      expect(capturedArgs[1]).toBe('echo {{tool0turn0}}');
      expect(result.messages[0].content).toContain(
        '[unresolved refs: tool0turn0]'
      );
    });

    it('lets two ToolNodes sharing a registry resolve each other\'s refs', async () => {
      const sharedRegistry = new ToolOutputReferenceRegistry();
      const capturedA: string[] = [];
      const capturedB: string[] = [];
      const toolA = createEchoTool({
        capturedArgs: capturedA,
        outputs: ['agent-A-output'],
        name: 'alpha',
      });
      const toolB = createEchoTool({
        capturedArgs: capturedB,
        outputs: ['agent-B-output'],
        name: 'beta',
      });

      // Two independent ToolNodes (simulating one per agent in a
      // multi-agent graph) sharing one registry instance.
      const nodeA = new ToolNode({
        tools: [toolA],
        toolOutputRegistry: sharedRegistry,
      });
      const nodeB = new ToolNode({
        tools: [toolB],
        toolOutputRegistry: sharedRegistry,
      });

      await nodeA.invoke(
        {
          messages: [
            aiMsgWithCalls([{ id: 'a1', name: 'alpha', command: 'first' }]),
          ],
        },
        { configurable: { run_id: 'shared-run' } }
      );

      await nodeB.invoke(
        {
          messages: [
            aiMsgWithCalls([
              { id: 'b1', name: 'beta', command: 'see {{tool0turn0}}' },
            ]),
          ],
        },
        { configurable: { run_id: 'shared-run' } }
      );

      // nodeB resolved nodeA's tool0turn0 placeholder (cross-node),
      // and its own output landed under the *next* turn (1), not 0.
      expect(capturedB[0]).toBe('see agent-A-output');
      expect(sharedRegistry.get('shared-run', 'tool0turn0')).toBe(
        'agent-A-output'
      );
      expect(sharedRegistry.get('shared-run', 'tool0turn1')).toBe(
        'agent-B-output'
      );
    });

    it('emits resolved args in ON_RUN_STEP_COMPLETED, not the template', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['STORED', 'second'],
      });
      const stepCompletedArgs: string[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event === 'on_run_step_completed') {
            const step = data as {
              result: { tool_call: { args: string } };
            };
            stepCompletedArgs.push(step.result.tool_call.args);
          }
        });

      const node = new ToolNode({
        tools: [t1],
        toolCallStepIds: new Map([
          ['a1', 'step_a1'],
          ['a2', 'step_a2'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      await invokeBatch(
        node,
        [{ id: 'a1', name: 'echo', command: 'first' }],
        'resolved-args'
      );
      await invokeBatch(
        node,
        [
          {
            id: 'a2',
            name: 'echo',
            command: 'echo {{tool0turn0}}',
          },
        ],
        'resolved-args'
      );

      // Second step-completed event should reflect the post-
      // substitution command, not the `{{…}}` template.
      expect(stepCompletedArgs).toHaveLength(2);
      expect(JSON.parse(stepCompletedArgs[1]).command).toBe('echo STORED');
    });

    it('prepends unresolved-refs warning to non-string ToolMessage content', async () => {
      const complexTool = tool(
        async () =>
          new ToolMessage({
            status: 'success',
            content: [
              { type: 'text', text: 'data' },
              { type: 'image_url', image_url: { url: 'data:...' } },
            ],
            name: 'complex',
            tool_call_id: 'c1',
          }),
        {
          name: 'complex',
          description: 'returns multi-part content',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;

      const node = new ToolNode({
        tools: [complexTool],
        toolOutputReferences: { enabled: true },
      });

      const [msg] = await invokeBatch(
        node,
        [{ id: 'c1', name: 'complex', command: 'see {{tool9turn9}}' }],
        'non-string'
      );

      expect(Array.isArray(msg.content)).toBe(true);
      const blocks = msg.content as Array<{ type: string; text?: string }>;
      expect(blocks[0].type).toBe('text');
      expect(blocks[0].text).toContain('[unresolved refs: tool9turn9]');
      // Original blocks follow the warning.
      expect(blocks[1].type).toBe('text');
      expect(blocks[1].text).toBe('data');
      expect(blocks[2].type).toBe('image_url');
    });

    it('resets the registry and turn counter when the runId changes', async () => {
      const capturedArgs: string[] = [];
      const t1 = createEchoTool({
        capturedArgs,
        outputs: ['from-run-A', 'from-run-B'],
      });
      const node = new ToolNode({
        tools: [t1],
        toolOutputReferences: { enabled: true },
      });

      const aiMsgA = aiMsgWithCalls([
        { id: 'a1', name: 'echo', command: 'first' },
      ]);
      await node.invoke(
        { messages: [aiMsgA] },
        { configurable: { run_id: 'run-A' } }
      );

      const aiMsgB = aiMsgWithCalls([
        {
          id: 'b1',
          name: 'echo',
          command: 'echo {{tool0turn0}}',
        },
      ]);
      const resultB = (await node.invoke(
        { messages: [aiMsgB] },
        { configurable: { run_id: 'run-B' } }
      )) as { messages: ToolMessage[] };

      expect(capturedArgs[1]).toBe('echo {{tool0turn0}}');
      expect(resultB.messages[0].content).toContain('[ref: tool0turn0]');
      expect(resultB.messages[0].content).toContain(
        '[unresolved refs: tool0turn0]'
      );
    });
  });

  describe('event-driven dispatch path', () => {
    afterEach(() => {
      jest.restoreAllMocks();
    });

    function mockEventDispatch(mockResults: t.ToolExecuteResult[]): void {
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const request = data as Record<string, unknown>;
          if (typeof request.resolve === 'function') {
            (request.resolve as (r: t.ToolExecuteResult[]) => void)(
              mockResults
            );
          }
        });
    }

    function createSchemaStub(name: string): StructuredToolInterface {
      return tool(async () => 'unused', {
        name,
        description: 'schema-only stub; host executes via ON_TOOL_EXECUTE',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;
    }

    it('annotates the output the host returns', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'host-output', status: 'success' },
      ]);

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'ec1', name: 'echo', args: { command: 'run' } }],
      });
      const result = (await node.invoke(
        { messages: [aiMsg] },
        { configurable: { run_id: 'run-host' } }
      )) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe('[ref: tool0turn0]\nhost-output');
      expect(
        node._unsafeGetToolOutputRegistry()!.get('run-host', 'tool0turn0')
      ).toBe('host-output');
    });

    it('substitutes `{{…}}` in the request sent to the host', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([
          ['ec1', 'step_ec1'],
          ['ec2', 'step_ec2'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'FIRST', status: 'success' },
      ]);
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [{ id: 'ec1', name: 'echo', args: { command: 'a' } }],
            }),
          ],
        },
        { configurable: { run_id: 'run-subst' } }
      );

      jest.restoreAllMocks();
      const capturedRequests: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            capturedRequests.push(req);
          }
          batch.resolve([
            { toolCallId: 'ec2', content: 'SECOND', status: 'success' },
          ]);
        });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'ec2',
                  name: 'echo',
                  args: { command: 'see {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-subst' } }
      );

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].args).toEqual({ command: 'see FIRST' });
    });

    it('surfaces unresolved refs on host-returned error results', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        {
          toolCallId: 'ec1',
          content: '',
          status: 'error',
          errorMessage: 'host failure',
        },
      ]);
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'ec1',
                name: 'echo',
                args: { command: 'see {{tool9turn9}}' },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toContain('Error: host failure');
      expect(result.messages[0].content).toContain(
        '[unresolved refs: tool9turn9]'
      );
    });

    it('reports unresolved refs even when the host succeeds', async () => {
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'done', status: 'success' },
      ]);
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'ec1',
                name: 'echo',
                args: { command: 'see {{tool9turn9}}' },
              },
            ],
          }),
        ],
      })) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toContain(
        '[unresolved refs: tool9turn9]'
      );
    });

    it('registers the post-hook output when PostToolUse replaces it', async () => {
      const hooks = new HookRegistry();
      hooks.register('PostToolUse', {
        hooks: [
          async (): Promise<{ updatedOutput: string }> => ({
            updatedOutput: 'hooked-output',
          }),
        ],
      });
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([['ec1', 'step_ec1']]),
        toolOutputReferences: { enabled: true },
        hookRegistry: hooks,
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'raw-output', status: 'success' },
      ]);
      const result = (await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'ec1', name: 'echo', args: { command: 'run' } },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-posthook' } }
      )) as { messages: ToolMessage[] };

      expect(result.messages[0].content).toBe(
        '[ref: tool0turn0]\nhooked-output'
      );
      expect(
        node._unsafeGetToolOutputRegistry()!.get('run-posthook', 'tool0turn0')
      ).toBe('hooked-output');
    });

    it('aborts event dispatch when a direct tool throws with handleToolErrors=false', async () => {
      const directBoom = tool(
        async () => {
          throw new Error('direct branch failed');
        },
        {
          name: 'directBoom',
          description: 'direct tool that throws',
          schema: z.object({ command: z.string() }),
        }
      ) as unknown as StructuredToolInterface;
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      let hostCalled = false;
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event === 'on_tool_execute') {
            hostCalled = true;
            (data as t.ToolExecuteBatchRequest).resolve([]);
          }
        });

      const node = new ToolNode({
        tools: [directBoom, eventStub],
        eventDrivenMode: true,
        handleToolErrors: false,
        agentId: 'agent-failfast',
        directToolNames: new Set(['directBoom']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
        ]),
        toolOutputReferences: { enabled: true },
      });

      await expect(
        node.invoke(
          {
            messages: [
              new AIMessage({
                content: '',
                tool_calls: [
                  { id: 'd1', name: 'directBoom', args: { command: 'x' } },
                  { id: 'e1', name: 'eventTool', args: { command: 'y' } },
                ],
              }),
            ],
          },
          { configurable: { run_id: 'failfast-run' } }
        )
      ).rejects.toThrow('direct branch failed');

      expect(hostCalled).toBe(false);
    });

    it('isolates PreToolUse-injected refs from same-turn direct outputs in the mixed path', async () => {
      // PreToolUse hook rewrites the event call's args to include
      // `{{tool0turn0}}`. In the mixed direct+event path that
      // placeholder must NOT resolve to the same-turn direct
      // output that just registered — it should be reported as
      // unresolved (matching cross-batch resolution semantics).
      const directCapturedArgs: string[] = [];
      const directTool = createEchoTool({
        capturedArgs: directCapturedArgs,
        outputs: ['direct-same-turn'],
        name: 'directTool',
      });
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        pattern: 'eventTool',
        hooks: [
          async (): Promise<{ updatedInput: { command: string } }> => ({
            updatedInput: { command: 'see {{tool0turn0}}' },
          }),
        ],
      });

      const hostCapturedArgs: Record<string, unknown>[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            hostCapturedArgs.push(req.args);
          }
          batch.resolve(
            batch.toolCalls.map((req) => ({
              toolCallId: req.id,
              content: 'event-out',
              status: 'success' as const,
            }))
          );
        });

      const node = new ToolNode({
        tools: [directTool, eventStub],
        eventDrivenMode: true,
        agentId: 'agent-snap',
        directToolNames: new Set(['directTool']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
        ]),
        hookRegistry: hooks,
        toolOutputReferences: { enabled: true },
      });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd1',
                  name: 'directTool',
                  args: { command: 'first' },
                },
                {
                  id: 'e1',
                  name: 'eventTool',
                  args: { command: 'orig' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'snap-run' } }
      );

      // Hook injected `{{tool0turn0}}`. The direct tool registered
      // `tool0turn0` in the same batch, but the snapshot was taken
      // pre-direct so the placeholder must remain unresolved.
      expect(hostCapturedArgs).toHaveLength(1);
      expect(hostCapturedArgs[0]).toEqual({
        command: 'see {{tool0turn0}}',
      });
    });

    it('keeps same-turn refs isolated in the mixed direct+event path', async () => {
      // Build a ToolNode with both a direct tool (via directToolNames)
      // and an event-driven schema stub. Share one registry across
      // both batches so refs only cross batch boundaries.
      const sharedRegistry = new ToolOutputReferenceRegistry();

      const directCapturedArgs: string[] = [];
      const directTool = createEchoTool({
        capturedArgs: directCapturedArgs,
        outputs: ['direct-A-output', 'direct-B-output'],
        name: 'directTool',
      });
      const eventStub = tool(async () => 'unused', {
        name: 'eventTool',
        description: 'schema-only stub',
        schema: z.object({ command: z.string() }),
      }) as unknown as StructuredToolInterface;

      const hostCapturedArgs: Record<string, unknown>[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            hostCapturedArgs.push(req.args);
          }
          batch.resolve(
            batch.toolCalls.map((req) => ({
              toolCallId: req.id,
              content: `event-${(req.args as { command: string }).command}`,
              status: 'success' as const,
            }))
          );
        });

      const node = new ToolNode({
        tools: [directTool, eventStub],
        eventDrivenMode: true,
        agentId: 'agent-mixed',
        directToolNames: new Set(['directTool']),
        toolCallStepIds: new Map([
          ['d1', 'step_d1'],
          ['e1', 'step_e1'],
          ['d2', 'step_d2'],
          ['e2', 'step_e2'],
        ]),
        toolOutputRegistry: sharedRegistry,
      });

      // Batch 1: mixed direct (index 0) + event (index 1). The event
      // call attempts `{{tool0turn0}}` — which points at the direct
      // call running *in the same batch*. Correct behavior: the
      // placeholder stays unresolved (cross-batch only), and the
      // event args received by the host must carry the literal
      // template string plus the LLM-visible `[unresolved refs:…]`
      // trailer.
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd1',
                  name: 'directTool',
                  args: { command: 'first' },
                },
                {
                  id: 'e1',
                  name: 'eventTool',
                  args: { command: 'echo {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'mixed-run' } }
      );

      expect(hostCapturedArgs).toHaveLength(1);
      expect(hostCapturedArgs[0]).toEqual({
        command: 'echo {{tool0turn0}}',
      });

      // Batch 2: ref across the boundary now resolves — direct's
      // registered output from batch 1 (tool0turn0) is available.
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'd2',
                  name: 'directTool',
                  args: { command: 'second' },
                },
                {
                  id: 'e2',
                  name: 'eventTool',
                  args: { command: 'echo {{tool0turn0}}' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'mixed-run' } }
      );

      expect(hostCapturedArgs[1]).toEqual({
        command: 'echo direct-A-output',
      });
    });

    it('re-resolves placeholders when PreToolUse rewrites args', async () => {
      const hooks = new HookRegistry();
      hooks.register('PreToolUse', {
        hooks: [
          async (): Promise<{ updatedInput: { command: string } }> => ({
            updatedInput: { command: 'rewritten {{tool0turn0}}' },
          }),
        ],
      });
      const node = new ToolNode({
        tools: [createSchemaStub('echo')],
        eventDrivenMode: true,
        agentId: 'agent-x',
        toolCallStepIds: new Map([
          ['ec1', 'step_ec1'],
          ['ec2', 'step_ec2'],
        ]),
        toolOutputReferences: { enabled: true },
        hookRegistry: hooks,
      });

      mockEventDispatch([
        { toolCallId: 'ec1', content: 'STORED', status: 'success' },
      ]);
      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'ec1', name: 'echo', args: { command: 'first' } },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-hookresolve' } }
      );

      jest.restoreAllMocks();
      const capturedRequests: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data) => {
          if (event !== 'on_tool_execute') {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          for (const req of batch.toolCalls) {
            capturedRequests.push(req);
          }
          batch.resolve([
            { toolCallId: 'ec2', content: 'done', status: 'success' },
          ]);
        });

      await node.invoke(
        {
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                {
                  id: 'ec2',
                  name: 'echo',
                  args: { command: 'input-without-placeholder' },
                },
              ],
            }),
          ],
        },
        { configurable: { run_id: 'run-hookresolve' } }
      );

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].args).toEqual({
        command: 'rewritten STORED',
      });
    });
  });
});
