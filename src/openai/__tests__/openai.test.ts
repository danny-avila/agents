import { GraphEvents } from '@/common';
import {
  createChatCompletionChunk,
  createOpenAIHandlers,
  createOpenAIStreamTracker,
  sendOpenAIFinalChunk,
} from '@/openai';
import type * as t from '@/types';

describe('OpenAI-compatible adapters', () => {
  it('creates chunks and streams message deltas as SSE data', async () => {
    const writes: string[] = [];
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_1', model: 'agent', created: 1 },
      tracker: createOpenAIStreamTracker(),
    });

    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );

    expect(writes.join('')).toContain('"content":"hello"');
  });

  it('sends a final usage chunk and done marker', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    tracker.usage.promptTokens = 3;
    tracker.usage.completionTokens = 5;

    await sendOpenAIFinalChunk({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_2', model: 'agent', created: 1 },
      tracker,
    });

    expect(writes.join('')).toContain('"total_tokens":8');
    expect(writes.at(-1)).toBe('data: [DONE]\n\n');
  });

  it('tracks partial usage metadata without NaN totals', async () => {
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: jest.fn() },
      context: { requestId: 'chatcmpl_usage', model: 'agent', created: 1 },
      tracker,
    });

    await handlers[GraphEvents.CHAT_MODEL_END].handle(
      GraphEvents.CHAT_MODEL_END,
      {
        output: { usage_metadata: { input_tokens: 3 } },
      } as t.ModelEndData
    );
    await handlers[GraphEvents.CHAT_MODEL_END].handle(
      GraphEvents.CHAT_MODEL_END,
      {
        output: { usage_metadata: { output_tokens: 5 } },
      } as t.ModelEndData
    );

    expect(tracker.usage.promptTokens).toBe(3);
    expect(tracker.usage.completionTokens).toBe(5);
  });

  it('builds a chat completion chunk without transport dependencies', () => {
    expect(
      createChatCompletionChunk(
        { requestId: 'chatcmpl_3', model: 'agent', created: 1 },
        { content: 'x' }
      )
    ).toEqual({
      id: 'chatcmpl_3',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'agent',
      choices: [{ index: 0, delta: { content: 'x' }, finish_reason: null }],
    });
  });
});
