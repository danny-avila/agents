import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import * as events from '@/utils/events';
import { GraphEvents } from '@/common';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

function createDummyTool(name: string): StructuredToolInterface {
  return tool(async () => 'direct should not run', {
    name,
    description: 'dummy',
    schema: z.object({}).passthrough(),
  }) as unknown as StructuredToolInterface;
}

function createAIMessageWithToolCalls(
  toolCalls: Array<{ id: string; name: string; args: Record<string, unknown> }>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: toolCalls,
  });
}

type CompletionEvent = {
  result: {
    id: string;
    tool_call: { id: string; output: string; args?: string };
  };
};

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe('ToolNode per-call onResult completion emission', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('emits a completion as the host reports each result, before the batch resolves', async () => {
    const timeline: string[] = [];
    const completions: CompletionEvent[] = [];

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          const completion = data as CompletionEvent;
          completions.push(completion);
          timeline.push(`completed:${completion.result.tool_call.id}`);
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        expect(batch.onResult).toBeDefined();

        // Host finishes the fast call first and reports it immediately.
        timeline.push('onResult:call_weather');
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'success',
          content: 'sunny',
        });

        await flushAsync();
        timeline.push('resolving-batch');
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
          {
            toolCallId: 'call_stock',
            status: 'success',
            content: '42',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather'), createDummyTool('stock')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([
        ['call_weather', 'step_weather'],
        ['call_stock', 'step_stock'],
      ]),
    });

    const result = (await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
          { id: 'call_stock', name: 'stock', args: { ticker: 'CH' } },
        ]),
      ],
    })) as { messages: ToolMessage[] };

    // The fast call's completion was emitted before the batch resolved.
    expect(timeline.indexOf('completed:call_weather')).toBeGreaterThan(
      timeline.indexOf('onResult:call_weather')
    );
    expect(timeline.indexOf('completed:call_weather')).toBeLessThan(
      timeline.indexOf('resolving-batch')
    );

    // Exactly one completion per call: early for weather, batch for stock.
    const byId = completions.map((c) => c.result.tool_call.id);
    expect(byId.filter((id) => id === 'call_weather')).toHaveLength(1);
    expect(byId.filter((id) => id === 'call_stock')).toHaveLength(1);
    expect(
      completions.find((c) => c.result.tool_call.id === 'call_weather')?.result
        .tool_call.output
    ).toBe('sunny');

    expect(result.messages).toHaveLength(2);
    expect(result.messages.map((m) => m.content)).toEqual(['sunny', '42']);
  });

  it('serializes bigint output before early and batch completion paths', async () => {
    const completions: CompletionEvent[] = [];
    const structuredOutput = [{ rowsRead: BigInt(42), status: 'complete' }];

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as CompletionEvent);
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.onResult?.({
          toolCallId: 'call_query',
          status: 'success',
          content: structuredOutput,
        });
        await flushAsync();
        batch.resolve([
          {
            toolCallId: 'call_query',
            status: 'success',
            content: structuredOutput,
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('query')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_query', 'step_query']]),
    });
    const result = (await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_query', name: 'query', args: {} },
        ]),
      ],
    })) as { messages: ToolMessage[] };

    const serialized = '[{"rowsRead":"42","status":"complete"}]';
    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call.output).toBe(serialized);
    expect(result.messages[0].content).toBe(serialized);
  });

  it('omits native computer screenshots from completion events', async () => {
    const completions: CompletionEvent[] = [];
    const screenshot = `data:image/png;base64,${'A'.repeat(2_000)}`;
    const computerOutput = new ToolMessage({
      content: screenshot,
      tool_call_id: 'call_computer',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const computer = createDummyTool('computer_use');
    (
      computer as unknown as {
        invoke: () => Promise<ToolMessage>;
      }
    ).invoke = async () => computerOutput;
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as CompletionEvent);
        }
      });
    const toolNode = new ToolNode({
      tools: [computer],
      eventDrivenMode: true,
      directToolNames: new Set(['computer_use']),
      toolCallStepIds: new Map([['call_computer', 'step_computer']]),
      maxToolResultChars: 80,
    });

    const result = (await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_computer', name: 'computer_use', args: {} },
        ]),
      ],
    })) as { messages: ToolMessage[] };

    expect(result.messages[0]).toBe(computerOutput);
    expect(result.messages[0].content).toBe(screenshot);
    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call.output).toContain(
      'Computer screenshot omitted'
    );
    expect(completions[0].result.tool_call.output.length).toBeLessThanOrEqual(
      80
    );
  });

  it('bounds cyclic tool args without losing the completion event', async () => {
    const completions: CompletionEvent[] = [];
    const cyclicArgs: Record<string, unknown> = { city: 'NYC' };
    cyclicArgs.self = cyclicArgs;

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as CompletionEvent);
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'success',
          content: 'sunny',
        });
        await flushAsync();
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: cyclicArgs },
        ]),
      ],
    });

    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call.args).toContain('[Circular]');
    expect(completions[0].result.tool_call.output).toBe('sunny');
  });

  it('ignores duplicate and unknown onResult reports', async () => {
    const completions: CompletionEvent[] = [];

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as CompletionEvent);
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'success',
          content: 'sunny',
        });
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'success',
          content: 'sunny again',
        });
        batch.onResult?.({
          toolCallId: 'call_unknown',
          status: 'success',
          content: 'never requested',
        });
        await flushAsync();
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    const byId = completions.map((c) => c.result.tool_call.id);
    expect(byId.filter((id) => id === 'call_weather')).toHaveLength(1);
    expect(byId.filter((id) => id === 'call_unknown')).toHaveLength(0);
  });

  it('does not offer onResult when result-altering hooks are configured', async () => {
    let observedBatch: t.ToolExecuteBatchRequest | undefined;

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        observedBatch = batch;
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const hookRegistry = new HookRegistry();
    hookRegistry.register('PostToolUse', {
      hooks: [async () => ({})],
    });
    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      hookRegistry,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    expect(observedBatch).toBeDefined();
    expect(observedBatch?.onResult).toBeUndefined();
  });

  it('offers onResult when the registry only has observation hooks (PostToolBatch)', async () => {
    let observedBatch: t.ToolExecuteBatchRequest | undefined;

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        observedBatch = batch;
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const hookRegistry = new HookRegistry();
    hookRegistry.register('PostToolBatch', {
      hooks: [async () => ({})],
    });
    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      hookRegistry,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    expect(observedBatch).toBeDefined();
    expect(observedBatch?.onResult).toBeDefined();
  });

  it('does not offer onResult when human-in-the-loop is enabled', async () => {
    let observedBatch: t.ToolExecuteBatchRequest | undefined;

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        observedBatch = batch;
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      humanInTheLoop: { enabled: true },
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    expect(observedBatch).toBeDefined();
    expect(observedBatch?.onResult).toBeUndefined();
  });

  it('falls back to batch emission when the early dispatch is not delivered', async () => {
    const completionAttempts: CompletionEvent[] = [];
    let failNextCompletion = true;

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<boolean | void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completionAttempts.push(data as CompletionEvent);
          if (failNextCompletion) {
            failNextCompletion = false;
            return false;
          }
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'success',
          content: 'sunny',
        });
        await flushAsync();
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'sunny',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    // First attempt (early) was rejected by the dispatcher; the batch path
    // re-emitted it.
    expect(completionAttempts).toHaveLength(2);
    expect(completionAttempts[0].result.tool_call.id).toBe('call_weather');
    expect(completionAttempts[1].result.tool_call.id).toBe('call_weather');
  });

  it('emits error-status results with the standard error formatting', async () => {
    const completions: CompletionEvent[] = [];

    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
          completions.push(data as CompletionEvent);
          return;
        }
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        batch.onResult?.({
          toolCallId: 'call_weather',
          status: 'error',
          content: '',
          errorMessage: 'city not found',
        });
        await flushAsync();
        batch.resolve([
          {
            toolCallId: 'call_weather',
            status: 'error',
            content: '',
            errorMessage: 'city not found',
          },
        ]);
      });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          { id: 'call_weather', name: 'weather', args: { city: 'NYC' } },
        ]),
      ],
    });

    expect(completions).toHaveLength(1);
    expect(completions[0].result.tool_call.output).toBe(
      'Error: city not found\n Please fix your mistakes.'
    );
  });
});
