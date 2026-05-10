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
});
