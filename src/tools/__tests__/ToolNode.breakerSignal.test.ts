import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { StreamLimitExceededError } from '@/llm/streamLimits';
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

/** A minimal tool whose invoke ignores the abort signal entirely — unlike
 * langchain `tool()` Runnables, which race the signal and reject with its
 * reason the moment a trip fires mid-execution. */
function createSignalBlindTool(
  name: string,
  fn: () => Promise<string>
): StructuredToolInterface {
  return {
    name,
    description: `signal-blind ${name}`,
    invoke: fn,
  } as unknown as StructuredToolInterface;
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

  it('rejects the batch at entry when the breaker has already tripped', async () => {
    const breaker = new AbortController();
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    breaker.abort(trip);
    let toolRan = false;
    const sideEffect = tool(
      async () => {
        toolRan = true;
        return 'ran';
      },
      {
        name: 'side_effect',
        description: 'must never run on a failed run',
        schema: z.object({}),
      }
    ) as unknown as StructuredToolInterface;
    const node = new ToolNode({
      tools: [sideEffect],
      getBreakerSignal: () => breaker.signal,
    });

    await expect(
      node.invoke({
        messages: [createToolCallMessage('call_1', 'side_effect')],
      })
    ).rejects.toBe(trip);
    expect(toolRan).toBe(false);
  });

  it('stops regular siblings when an interrupting tool trips the breaker', async () => {
    const breaker = new AbortController();
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    let sideEffectRan = false;
    /** Plain tool objects, NOT langchain `tool()`: Runnable invokes race
     * the composed signal and reject with its reason the instant the trip
     * fires, which would mask the stage boundary this test targets. The
     * exposure is exactly tools that ignore cancellation. */
    const tripper = createSignalBlindTool('ask_question', async () => {
      breaker.abort(trip);
      return 'completed normally across the trip';
    });
    const sideEffect = createSignalBlindTool('send_email', async () => {
      sideEffectRan = true;
      return 'sent';
    });
    const node = new ToolNode({
      tools: [tripper, sideEffect],
      interruptingToolNames: new Set(['ask_question']),
      getBreakerSignal: () => breaker.signal,
    });

    await expect(
      node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'ask_question', args: {} },
              { id: 'call_2', name: 'send_email', args: {} },
            ],
          }),
        ],
      })
    ).rejects.toBe(trip);
    expect(sideEffectRan).toBe(false);
  });

  it('stops event dispatch when a direct tool trips the breaker mid-batch', async () => {
    const breaker = new AbortController();
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });
    const { toolExecuteCalls } = installToolExecuteResponder();
    const tripper = createSignalBlindTool('direct_tripper', async () => {
      breaker.abort(trip);
      return 'completed normally across the trip';
    });
    const node = new ToolNode({
      tools: [tripper],
      eventDrivenMode: true,
      directToolNames: new Set(['direct_tripper']),
      toolCallStepIds: new Map([
        ['call_1', 'step_1'],
        ['call_2', 'step_2'],
      ]),
      getBreakerSignal: () => breaker.signal,
    });

    await expect(
      node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'call_1', name: 'direct_tripper', args: {} },
              { id: 'call_2', name: 'remote_tool', args: {} },
            ],
          }),
        ],
      })
    ).rejects.toBe(trip);
    expect(toolExecuteCalls).toHaveLength(0);
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
