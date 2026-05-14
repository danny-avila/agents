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

    expect(writes).toHaveLength(2);
    expect(writes[0]).toContain('"role":"assistant"');
    expect(writes[1]).toContain('"content":"hello"');
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

    expect(writes).toHaveLength(4);
    expect(writes[0]).toContain('"role":"assistant"');
    expect(writes[1]).toContain('"finish_reason":"stop"');
    expect(writes[1]).not.toContain('"usage"');
    expect(writes[2]).toContain('"choices":[]');
    expect(writes[2]).toContain('"total_tokens":8');
    expect(writes[3]).toBe('data: [DONE]\n\n');
  });

  it('uses tool_calls finish reason after streaming tool deltas', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_tools', model: 'agent', created: 1 },
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
    await sendOpenAIFinalChunk({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_tools', model: 'agent', created: 1 },
      tracker,
    });

    expect(writes[0]).toContain('"role":"assistant"');
    expect(writes[1]).toContain('"tool_calls"');
    expect(writes.at(-3)).toContain('"finish_reason":"tool_calls"');
    expect(writes.at(-2)).toContain('"choices":[]');
  });

  it('uses stop finish reason when assistant text follows tool calls', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_tools_done', model: 'agent', created: 1 },
      tracker,
    });

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
    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'msg',
        delta: { content: [{ type: 'text', text: 'done' }] },
      } satisfies t.MessageDeltaEvent
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
    await sendOpenAIFinalChunk({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_tools_done', model: 'agent', created: 1 },
      tracker,
    });

    expect(writes.at(-3)).toContain('"finish_reason":"stop"');
    expect(writes.at(-2)).toContain('"choices":[]');
  });

  it('scopes tool-call argument state by run step', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: { requestId: 'chatcmpl_step_tools', model: 'agent', created: 1 },
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
              args: '{"query":"first"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );
    await handlers[GraphEvents.ON_RUN_STEP_DELTA].handle(
      GraphEvents.ON_RUN_STEP_DELTA,
      {
        id: 'step_2',
        delta: {
          type: 'tool_calls',
          tool_calls: [
            {
              index: 0,
              id: 'call_2',
              name: 'search',
              args: '{"query":"second"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent
    );

    const toolCallDeltas = writes
      .map(
        (data) =>
          JSON.parse(data.slice(6)) as {
            choices: Array<{
              delta: {
                tool_calls?: Array<{
                  id?: string;
                  function?: { name?: string; arguments?: string };
                }>;
              };
            }>;
          }
      )
      .flatMap((chunk) => chunk.choices[0].delta.tool_calls ?? []);

    expect(toolCallDeltas).toHaveLength(2);
    expect(toolCallDeltas[1]).toMatchObject({
      id: 'call_2',
      function: { name: 'search', arguments: '{"query":"second"}' },
    });
    expect(tracker.toolCalls.get(0)?.function.arguments).toBe(
      '{"query":"second"}'
    );
  });

  it('streams completed tool-call run steps without deltas', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        requestId: 'chatcmpl_complete_tools',
        model: 'agent',
        created: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.ON_RUN_STEP].handle(GraphEvents.ON_RUN_STEP, {
      id: 'step_complete',
      index: 2,
      type: 'tool_calls',
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [
          {
            id: 'call_complete',
            type: 'function',
            function: {
              name: 'search',
              arguments: { query: 'sessions' },
            },
          },
        ],
      },
    } as t.RunStep);
    await sendOpenAIFinalChunk({
      writer: { write: (data) => void writes.push(data) },
      context: {
        requestId: 'chatcmpl_complete_tools',
        model: 'agent',
        created: 1,
      },
      tracker,
    });

    expect(writes[0]).toContain('"role":"assistant"');
    expect(writes[1]).toContain('"tool_calls"');
    expect(writes[1]).toContain('"id":"call_complete"');
    expect(writes[1]).toContain('"name":"search"');
    expect(writes[1]).toContain('"{\\"query\\":\\"sessions\\"}"');
    expect(writes.at(-3)).toContain('"finish_reason":"tool_calls"');
    expect(writes.at(-2)).toContain('"choices":[]');
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

  it('includes reasoning token usage in the final chunk', async () => {
    const writes: string[] = [];
    const tracker = createOpenAIStreamTracker();
    const handlers = createOpenAIHandlers({
      writer: { write: (data) => void writes.push(data) },
      context: {
        requestId: 'chatcmpl_reasoning_usage',
        model: 'agent',
        created: 1,
      },
      tracker,
    });

    await handlers[GraphEvents.CHAT_MODEL_END].handle(
      GraphEvents.CHAT_MODEL_END,
      {
        output: {
          usage_metadata: {
            input_tokens: 3,
            output_tokens: 5,
            output_token_details: { reasoning: 2 },
          },
        },
      } as t.ModelEndData
    );
    await sendOpenAIFinalChunk({
      writer: { write: (data) => void writes.push(data) },
      context: {
        requestId: 'chatcmpl_reasoning_usage',
        model: 'agent',
        created: 1,
      },
      tracker,
    });

    expect(writes.at(-2)).toContain('"choices":[]');
    expect(writes.at(-2)).toContain(
      '"completion_tokens_details":{"reasoning_tokens":2}'
    );
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
