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

    expect(writes.join('')).toContain('response.output_text.delta');
    expect(writes.join('')).toContain('response.completed');
    expect(writes.at(-1)).toBe('data: [DONE]\n\n');
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
      .map((data) => JSON.parse(data.slice(6)) as { type: string });
    expect(events.map((event) => event.type)).toEqual([
      'response.output_item.added',
      'response.function_call_arguments.delta',
      'response.function_call_arguments.delta',
      'response.function_call_arguments.done',
      'response.output_item.done',
    ]);
    expect(tracker.items[0]).toMatchObject({
      type: 'function_call',
      call_id: 'call_1',
      name: 'search',
      arguments: '{"query":"sessions"}',
      status: 'completed',
    });
  });
});
