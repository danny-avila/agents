import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
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
      expect(node.getToolOutputRegistry()).toBeUndefined();
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

      const registry = node.getToolOutputRegistry();
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

      const registry = node.getToolOutputRegistry()!;
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
      expect(node.getToolOutputRegistry()!.get('tool0turn0')).toBeUndefined();
    });
  });
});
