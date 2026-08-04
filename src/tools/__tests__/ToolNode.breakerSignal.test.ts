import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { GraphEvents } from '@/common';
import { ToolNode } from '../ToolNode';

function createSignalCaptureTool(name: string): {
  tool: StructuredToolInterface;
  observed: () => AbortSignal | undefined;
} {
  let signal: AbortSignal | undefined;
  const captureTool = tool(
    async (_input, config) => {
      signal = (config as RunnableConfig | undefined)?.signal;
      return 'captured';
    },
    {
      name,
      description: 'captures the abort signal its runtime receives',
      schema: z.object({}),
    }
  ) as unknown as StructuredToolInterface;
  return { tool: captureTool, observed: () => signal };
}

function createToolCallMessage(id: string, name: string): AIMessage {
  return new AIMessage({ content: '', tool_calls: [{ id, name, args: {} }] });
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
          status: 'success' as const,
          content: `ok ${call.name}`,
        }))
      );
    });
  return { toolExecuteCalls };
}

describe('ToolNode breaker signal composition', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('exposes the run breaker to direct tool runtimes', async () => {
    const breaker = new AbortController();
    const { tool: capture, observed } = createSignalCaptureTool('capture');
    const node = new ToolNode({
      tools: [capture],
      getBreakerSignal: () => breaker.signal,
    });

    const result = (await node.invoke({
      messages: [createToolCallMessage('call_1', 'capture')],
    })) as { messages: ToolMessage[] };

    expect(result.messages[0].content).toBe('captured');
    const signal = observed();
    expect(signal).toBeDefined();
    expect(signal?.aborted).toBe(false);

    breaker.abort(new Error('stream limit breach'));
    expect(signal?.aborted).toBe(true);
  });

  it('composes the breaker with the caller signal instead of replacing it', async () => {
    const breaker = new AbortController();
    const caller = new AbortController();
    const { tool: capture, observed } = createSignalCaptureTool('capture');
    const node = new ToolNode({
      tools: [capture],
      getBreakerSignal: () => breaker.signal,
    });

    await node.invoke(
      { messages: [createToolCallMessage('call_1', 'capture')] },
      { signal: caller.signal }
    );

    const signal = observed();
    expect(signal).toBeDefined();
    expect(signal?.aborted).toBe(false);

    caller.abort(new Error('caller cancelled'));
    expect(signal?.aborted).toBe(true);
  });

  it('leaves the caller signal untouched when no breaker accessor is set', async () => {
    const caller = new AbortController();
    const { tool: capture, observed } = createSignalCaptureTool('capture');
    const node = new ToolNode({ tools: [capture] });

    await node.invoke(
      { messages: [createToolCallMessage('call_1', 'capture')] },
      { signal: caller.signal }
    );

    expect(observed()).toBe(caller.signal);
  });

  it('sends a breaker-composed signal on ON_TOOL_EXECUTE batch requests', async () => {
    const breaker = new AbortController();
    const { toolExecuteCalls } = installToolExecuteResponder();
    const node = new ToolNode({
      tools: [],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_1', 'step_1']]),
      getBreakerSignal: () => breaker.signal,
    });

    await node.invoke({
      messages: [createToolCallMessage('call_1', 'remote_tool')],
    });

    expect(toolExecuteCalls).toHaveLength(1);
    const { signal } = toolExecuteCalls[0];
    expect(signal).toBeDefined();
    expect(signal?.aborted).toBe(false);

    breaker.abort(new Error('stream limit breach'));
    expect(signal?.aborted).toBe(true);
  });
});
