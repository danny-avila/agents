import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import { TOOL_OUTPUT_REF_KEY } from '../toolOutputReferences';

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
  calls: Array<{ id: string; name: string; command: string }>
): Promise<ToolMessage[]> {
  const aiMsg = aiMsgWithCalls(calls);
  const result = (await toolNode.invoke({ messages: [aiMsg] })) as
    | ToolMessage[]
    | { messages: ToolMessage[] };
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
      expect(registry!.get('tool0turn0')!.length).toBeLessThanOrEqual(40);
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
      expect(registry.get('tool0turn0')).toBeUndefined();
      expect(registry.get('tool0turn1')).toBe('bbbbb');
      expect(registry.get('tool0turn2')).toBe('ccccc');
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
        node._unsafeGetToolOutputRegistry()!.get('tool0turn0')
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
      const result = (await node.invoke({ messages: [aiMsg] })) as {
        messages: ToolMessage[];
      };

      expect(result.messages[0].content).toBe('[ref: tool0turn0]\nhost-output');
      expect(node._unsafeGetToolOutputRegistry()!.get('tool0turn0')).toBe(
        'host-output'
      );
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
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'ec1', name: 'echo', args: { command: 'a' } }],
          }),
        ],
      });

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

      await node.invoke({
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
      });

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
      expect(node._unsafeGetToolOutputRegistry()!.get('tool0turn0')).toBe(
        'hooked-output'
      );
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
