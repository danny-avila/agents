import { GraphEvents } from '@/common';
import {
  buildResponse,
  createResponseTracker,
  createResponsesEventHandlers,
  emitResponseCompleted,
} from '@/responses';
import type * as t from '@/types';

describe('Responses-compatible adapters', () => {
  it('streams semantic response events through a generic writer', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_1', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );
    await emitResponseCompleted({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_1', model: 'agent', createdAt: 1 },
      tracker,
    });

    expect(writes.join('')).toContain('response.created');
    expect(writes.join('')).toContain('response.output_text.delta');
    expect(writes.join('')).toContain('response.completed');
    expect(writes.at(-1)).toBe('data: [DONE]\n\n');
  });

  it('emits response.created before output item events once', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_created', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );
    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: ' again' }] },
      } satisfies t.MessageDeltaEvent
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map((data) => JSON.parse(data.slice(6)) as { type: string });
    expect(events[0].type).toBe('response.created');
    expect(
      events.filter((event) => event.type === 'response.created')
    ).toHaveLength(1);
    expect(
      events.findIndex((event) => event.type === 'response.created')
    ).toBeLessThan(
      events.findIndex((event) => event.type === 'response.output_item.added')
    );
  });

  it('emits output item completion events before response completion', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_done', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );
    await handlers[GraphEvents.ON_REASONING_DELTA].handle(
      GraphEvents.ON_REASONING_DELTA,
      {
        id: 'reasoning',
        delta: { content: [{ type: 'text', text: 'thinking' }] },
      } as t.ReasoningDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, id: 'call_1', name: 'search' }],
        },
      } as t.RunStepDeltaEvent
    );

    await emitResponseCompleted({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_done', model: 'agent', createdAt: 1 },
      tracker,
    });

    const events = writes
      .filter(
        (data) => data.startsWith('data: ') && data !== 'data: [DONE]\n\n'
      )
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            output_index?: number;
          }
      );
    const completedIndex = events.findIndex(
      (event) => event.type === 'response.completed'
    );
    const doneEvents = events.filter(
      (event) => event.type === 'response.output_item.done'
    );
    expect(doneEvents.map((event) => event.output_index)).toEqual([0, 1, 2]);
    expect(
      doneEvents.every(
        (event) =>
          events.indexOf(event) >= 0 && events.indexOf(event) < completedIndex
      )
    ).toBe(true);
    expect(events.map((event) => event.type)).toEqual([
      'response.created',
      'response.output_item.added',
      'response.output_text.delta',
      'response.output_item.added',
      'response.reasoning_text.delta',
      'response.output_item.added',
      'response.output_text.done',
      'response.output_item.done',
      'response.reasoning_text.done',
      'response.output_item.done',
      'response.function_call_arguments.done',
      'response.output_item.done',
      'response.completed',
    ]);
  });

  it('builds completed response usage from tracker state', () => {
    const tracker = createResponseTracker();
    tracker.usage.inputTokens = 2;
    tracker.usage.outputTokens = 7;

    expect(
      buildResponse(
        { responseId: 'resp_2', model: 'agent', createdAt: 1 },
        tracker,
        'completed'
      ).usage
    ).toEqual({
      input_tokens: 2,
      output_tokens: 7,
      total_tokens: 9,
    });
  });

  it('streams reasoning text with official Responses event names', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_reasoning', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.ON_REASONING_DELTA].handle(
      GraphEvents.ON_REASONING_DELTA,
      {
        id: 'reasoning',
        delta: { content: [{ type: 'text', text: 'thinking' }] },
      } as t.ReasoningDeltaEvent
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            delta?: string;
          }
      );
    expect(events.map((event) => event.type)).toEqual([
      'response.created',
      'response.output_item.added',
      'response.reasoning_text.delta',
    ]);
    expect(events[2]).toMatchObject({
      delta: 'thinking',
    });
  });

  it('tracks partial usage metadata without NaN totals', async () => {
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: jest.fn() },
      context: { responseId: 'resp_usage', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.CHAT_MODEL_END].handle(
      GraphEvents.CHAT_MODEL_END,
      {
        output: { usage_metadata: { input_tokens: 2 } },
      } as t.ModelEndData
    );
    await handlers[GraphEvents.CHAT_MODEL_END].handle(
      GraphEvents.CHAT_MODEL_END,
      {
        output: { usage_metadata: { output_tokens: 4 } },
      } as t.ModelEndData
    );

    expect(
      buildResponse(
        { responseId: 'resp_usage', model: 'agent', createdAt: 1 },
        tracker,
        'completed'
      ).usage
    ).toEqual({
      input_tokens: 2,
      output_tokens: 4,
      total_tokens: 6,
    });
  });

  it('streams function call items and argument events', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { responseId: 'resp_tools', model: 'agent', createdAt: 1 },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              name: 'search',
              args: '{"query"',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              args: ':"sessions"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_COMPLETED].handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: 'step_1',
          index: 7,
          type: 'tool_call',
          tool_call: {
            id: 'call_1',
            name: 'search',
            args: '{"query":"sessions"}',
            output: 'ok',
            progress: 1,
          },
        },
      }
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            call_id?: string;
            name?: string;
            arguments?: string;
          }
      );
    expect(events.map((event) => event.type)).toEqual([
      'response.created',
      'response.output_item.added',
      'response.function_call_arguments.delta',
      'response.function_call_arguments.delta',
      'response.function_call_arguments.done',
      'response.output_item.done',
    ]);
    expect(
      events.find(
        (event) => event.type === 'response.function_call_arguments.done'
      )
    ).toMatchObject({
      call_id: 'call_1',
      name: 'search',
      arguments: '{"query":"sessions"}',
    });
    expect(tracker.items[0]).toMatchObject({
      type: 'function_call',
      call_id: 'call_1',
      name: 'search',
      arguments: '{"query":"sessions"}',
      status: 'completed',
    });
  });

  it('emits function call arguments done before closing unfinished tool items', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        responseId: 'resp_unfinished_tool',
        model: 'agent',
        createdAt: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              name: 'search',
              args: '{"query":"sessions"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );

    await emitResponseCompleted({
      writer: { write: (data) => void writes.push(data) },
      context: {
        responseId: 'resp_unfinished_tool',
        model: 'agent',
        createdAt: 1,
      },
      tracker,
    });

    const events = writes
      .filter(
        (data) => data.startsWith('data: ') && data !== 'data: [DONE]\n\n'
      )
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            call_id?: string;
            arguments?: string;
          }
      );
    const argumentsDoneIndex = events.findIndex(
      (event) => event.type === 'response.function_call_arguments.done'
    );
    const itemDoneIndex = events.findIndex(
      (event) => event.type === 'response.output_item.done'
    );

    expect(argumentsDoneIndex).toBeGreaterThan(-1);
    expect(itemDoneIndex).toBeGreaterThan(argumentsDoneIndex);
    expect(events[argumentsDoneIndex]).toMatchObject({
      call_id: 'call_1',
      arguments: '{"query":"sessions"}',
    });
    expect(tracker.items[0]).toMatchObject({
      type: 'function_call',
      status: 'completed',
    });
  });

  it('keeps tool-call items stable when ids arrive and indexes differ', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        responseId: 'resp_tools_late_id',
        model: 'agent',
        createdAt: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, args: '{"query"' }],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            {
              index: 0,
              id: 'call_1',
              name: 'search',
              args: ':"sessions"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_COMPLETED].handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: 'step_1',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_1',
            name: 'search',
            args: '{"query":"sessions"}',
            output: 'ok',
            progress: 1,
          },
        },
      }
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            call_id?: string;
            name?: string;
            arguments?: string;
          }
      );
    expect(
      events.filter((event) => event.type === 'response.output_item.added')
    ).toHaveLength(1);
    expect(
      events.find(
        (event) => event.type === 'response.function_call_arguments.done'
      )
    ).toMatchObject({
      call_id: 'call_1',
      name: 'search',
      arguments: '{"query":"sessions"}',
    });
    expect(tracker.items).toHaveLength(1);
    expect(tracker.items[0]).toMatchObject({
      type: 'function_call',
      call_id: 'call_1',
      name: 'search',
      arguments: '{"query":"sessions"}',
      status: 'completed',
    });
  });

  it('completes id-less tool calls using the tool-call position', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        responseId: 'resp_tools_no_id',
        model: 'agent',
        createdAt: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, name: 'search', args: '{"query"' }],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_COMPLETED].handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: 'step_1',
          index: 7,
          type: 'tool_call',
          tool_call: {
            name: 'search',
            args: '{"query":"sessions"}',
            output: 'ok',
            progress: 1,
          },
        },
      }
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
            call_id?: string;
            arguments?: string;
          }
      );
    expect(
      events.filter((event) => event.type === 'response.output_item.added')
    ).toHaveLength(1);
    expect(
      events.find(
        (event) => event.type === 'response.function_call_arguments.done'
      )
    ).toMatchObject({
      call_id: 'step_1:0',
      arguments: '{"query":"sessions"}',
    });
    expect(tracker.items).toHaveLength(1);
    expect(tracker.items[0]).toMatchObject({
      type: 'function_call',
      call_id: 'step_1:0',
      name: 'search',
      arguments: '{"query":"sessions"}',
      status: 'completed',
    });
  });

  it('keeps id-less single-chunk tool calls distinct by explicit index', async () => {
    const writes: string[] = [];
    const tracker = createResponseTracker();
    const handlers = createResponsesEventHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        responseId: 'resp_tools_indexed',
        model: 'agent',
        createdAt: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [{ index: 0, name: 'search', args: '{"query":"first"}' }],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_1',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            { index: 1, name: 'search', args: '{"query":"second"}' },
          ],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_COMPLETED].handle(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: 'step_1',
          type: 'tool_call',
          tool_call: {
            index: 1,
            name: 'search',
            args: '{"query":"second"}',
            output: 'ok',
            progress: 1,
          },
        },
      }
    );

    const events = writes
      .filter((data) => data.startsWith('data: '))
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            type: string;
          }
      );
    expect(
      events.filter((event) => event.type === 'response.output_item.added')
    ).toHaveLength(2);
    expect(tracker.items).toHaveLength(2);
    expect(tracker.items[0]).toMatchObject({
      call_id: 'step_1:0',
      arguments: '{"query":"first"}',
      status: 'in_progress',
    });
    expect(tracker.items[1]).toMatchObject({
      call_id: 'step_1:1',
      arguments: '{"query":"second"}',
      status: 'completed',
    });
  });
});
