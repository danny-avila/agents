import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect } from '@jest/globals';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolRuntime } from '@langchain/core/tools';
import { ToolNode } from '../ToolNode';

type CapturedRuntime = {
  state: unknown;
  toolCallId: string;
  toolCallIdFromConfig: string | undefined;
};

/**
 * Tool that records the langgraph runtime it received as its second
 * argument. Proves our forked ToolNode forwards `runtime.state` (the
 * deprecation-free replacement for `getCurrentTaskInput()`).
 */
function createRuntimeProbeTool(captured: CapturedRuntime[]) {
  return tool(
    async (_input, runtime: ToolRuntime) => {
      captured.push({
        state: runtime.state,
        toolCallId: runtime.toolCallId,
        toolCallIdFromConfig: runtime.toolCall?.id,
      });
      return 'ok';
    },
    {
      name: 'probe_runtime',
      description: 'Records the runtime forwarded by the ToolNode',
      schema: z.object({ value: z.string() }),
    }
  );
}

function aiMessageWithProbeCall(callId: string): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [
      {
        id: callId,
        name: 'probe_runtime',
        args: { value: 'hello' },
      },
    ],
  });
}

describe('ToolNode runtime.state forwarding (langgraph 1.4)', () => {
  it('forwards the message-state input as runtime.state', async () => {
    const captured: CapturedRuntime[] = [];
    const toolNode = new ToolNode({
      tools: [createRuntimeProbeTool(captured)],
    });

    const messages: BaseMessage[] = [
      new HumanMessage('start'),
      aiMessageWithProbeCall('call_1'),
    ];
    const input = { messages };

    await toolNode.invoke(input);

    expect(captured).toHaveLength(1);
    /* state is the exact run input object the ToolNode was invoked with */
    expect(captured[0].state).toBe(input);
    expect((captured[0].state as { messages: BaseMessage[] }).messages).toBe(
      messages
    );
    /* toolCallId and the langchain-extracted toolCall both resolve */
    expect(captured[0].toolCallId).toBe('call_1');
    expect(captured[0].toolCallIdFromConfig).toBe('call_1');
  });

  it('forwards the array-shaped input as runtime.state', async () => {
    const captured: CapturedRuntime[] = [];
    const toolNode = new ToolNode({
      tools: [createRuntimeProbeTool(captured)],
    });

    const input: BaseMessage[] = [
      new HumanMessage('start'),
      aiMessageWithProbeCall('call_2'),
    ];

    await toolNode.invoke(input);

    expect(captured).toHaveLength(1);
    expect(captured[0].state).toBe(input);
    expect(captured[0].toolCallId).toBe('call_2');
  });

  it('forwards the de-enveloped state for Send-shaped input', async () => {
    const captured: CapturedRuntime[] = [];
    const toolNode = new ToolNode({
      tools: [createRuntimeProbeTool(captured)],
    });

    const sidecar = { tenant: 'acme' };
    await toolNode.invoke({
      lg_tool_call: {
        id: 'call_3',
        name: 'probe_runtime',
        args: { value: 'hi' },
        type: 'tool_call',
      },
      ...sidecar,
    });

    expect(captured).toHaveLength(1);
    /* The `lg_tool_call` envelope key is stripped, the rest is state */
    expect(captured[0].state).toEqual(sidecar);
    expect(
      (captured[0].state as Record<string, unknown>).lg_tool_call
    ).toBeUndefined();
    expect(captured[0].toolCallId).toBe('call_3');
  });
});
