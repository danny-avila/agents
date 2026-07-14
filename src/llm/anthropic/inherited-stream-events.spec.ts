// Inherited from @langchain/anthropic@1.5.1
//   tests/chat_models_stream_events.test.ts
//
// Exercises the native content-block streaming protocol (`ChatModelStreamEvent`)
// that `BaseChatModel.streamEvents()` / `_streamChatModelEvents()` expose as typed
// sub-streams (`.text` / `.reasoning` / `.toolCalls` / `.usage` / `.output`).
//
// Applicability to our fork:
//   Our `CustomAnthropic` (imported here as `ChatAnthropic` from `@/llm/anthropic`)
//   extends upstream `ChatAnthropicMessages` (1.5.1). It overrides both the legacy
//   `_streamResponseChunks` path and `_streamChatModelEvents`, whose request setup
//   mirrors upstream while using the local cumulative-usage converter. Mocking
//   `createStreamWithRetry` (as upstream's `MockStreamChatAnthropic` does) drives
//   the real native conversion path without making API calls.
//
// Adaptation:
//   - vitest -> jest (`@jest/globals`; `vi.*` -> `jest.*`).
//   - Import the class from `@/llm/anthropic` (our fork), not `@langchain/anthropic`.
//   - Feed canned SSE events by overriding `createStreamWithRetry` (mirrors
//     `MockStreamingAnthropic` in llm.spec.ts), keeping these UNIT tests with a
//     mocked transport — no live API call.
//   - Upstream's final `streaming events` describe used vitest custom matchers
//     (`toHaveStreamText` / `toHaveStreamReasoning` / `toHaveStreamToolCalls`), which
//     do not exist in jest. They are re-expressed against the public `ChatModelStream`
//     sub-streams those matchers wrap.

import { test, expect, describe } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import type { BaseChatModelCallOptions } from '@langchain/core/language_models/chat_models';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type {
  AnthropicMessageStreamEvent,
  AnthropicRequestOptions,
  AnthropicStreamingMessageCreateParams,
} from './types';
import type { CustomAnthropicCallOptions } from './index';
import { CustomAnthropic as ChatAnthropic } from './index';

// ─── Mock model ─────────────────────────────────────────────────
//
// Subclass that overrides `createStreamWithRetry` to yield a canned async
// iterable of native Anthropic SSE events, capturing the constructed request
// so we can assert on top-level request fields.

class MockStreamChatAnthropic extends ChatAnthropic {
  private mockEvents: unknown[];

  capturedRequest: AnthropicStreamingMessageCreateParams | undefined;

  constructor(mockEvents: unknown[]) {
    super({ apiKey: 'fake-key', model: 'claude-sonnet-4-20250514' });
    this.mockEvents = mockEvents;
  }

  // Inherit the base method's own Stream identity instead of importing
  // @anthropic-ai/sdk/streaming here: the SDK ships dual .d.ts/.d.mts types,
  // and an independent import can resolve to the other format's identity
  // depending on compile order (flaky TS2416 in sharded CI).
  protected override async createStreamWithRetry(
    request: AnthropicStreamingMessageCreateParams,
    _options?: AnthropicRequestOptions
  ): ReturnType<ChatAnthropic['createStreamWithRetry']> {
    this.capturedRequest = request;
    const events = this.mockEvents;
    return {
      controller: { abort: (): void => {} },
      async *[Symbol.asyncIterator](): AsyncGenerator<AnthropicMessageStreamEvent> {
        for (const event of events) {
          yield event as AnthropicMessageStreamEvent;
        }
      },
    } as unknown as Awaited<ReturnType<ChatAnthropic['createStreamWithRetry']>>;
  }
}

type StreamEventsModel = {
  _streamChatModelEvents: (
    messages: unknown[],
    options: BaseChatModelCallOptions
  ) => AsyncGenerator<ChatModelStreamEvent>;
};

function streamEvents(
  model: ChatAnthropic,
  options: BaseChatModelCallOptions = {} as BaseChatModelCallOptions
): AsyncGenerator<ChatModelStreamEvent> {
  return (model as unknown as StreamEventsModel)._streamChatModelEvents(
    [],
    options
  );
}

async function collectEvents(
  model: ChatAnthropic,
  options?: BaseChatModelCallOptions
): Promise<ChatModelStreamEvent[]> {
  const events: ChatModelStreamEvent[] = [];
  for await (const event of streamEvents(model, options)) {
    events.push(event);
  }
  return events;
}

// ─── Fixtures (native Anthropic SSE events) ─────────────────────

function textOnlyEvents(): unknown[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_01ABC',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 25, output_tokens: 0 },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'Hello' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: ' world' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 2 },
    },
    { type: 'message_stop' },
  ];
}

function thinkingPlusTextEvents(): unknown[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_02DEF',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 50, output_tokens: 0 },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'thinking', thinking: '' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'thinking_delta', thinking: 'Let me' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'thinking_delta', thinking: ' reason...' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'signature_delta', signature: 'sig_abc' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'content_block_start',
      index: 1,
      content_block: { type: 'text', text: '' },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'text_delta', text: 'The answer is 42.' },
    },
    { type: 'content_block_stop', index: 1 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 20 },
    },
    { type: 'message_stop' },
  ];
}

function toolCallEvents(): unknown[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_03GHI',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 100, output_tokens: 0 },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'Let me search.' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'content_block_start',
      index: 1,
      content_block: {
        type: 'tool_use',
        id: 'toolu_01ABC',
        name: 'web_search',
      },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: '{"query"' },
    },
    {
      type: 'content_block_delta',
      index: 1,
      delta: { type: 'input_json_delta', partial_json: ':"weather"}' },
    },
    { type: 'content_block_stop', index: 1 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'tool_use', stop_sequence: null },
      usage: { output_tokens: 15 },
    },
    { type: 'message_stop' },
  ];
}

function cacheUsageEvents(): unknown[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_04JKL',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 100,
          output_tokens: 0,
          cache_creation_input_tokens: 500,
          cache_read_input_tokens: 200,
        },
      },
    },
    {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    },
    {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'Cached response' },
    },
    { type: 'content_block_stop', index: 0 },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: 3 },
    },
    { type: 'message_stop' },
  ];
}

function lateUsageEvents(): unknown[] {
  return [
    {
      type: 'message_start',
      message: {
        id: 'msg_05MNO',
        type: 'message',
        role: 'assistant',
        content: [],
        model: 'claude-sonnet-4-20250514',
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
        },
      },
    },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: {
        input_tokens: 100,
        output_tokens: 20,
        cache_creation_input_tokens: 500,
        cache_read_input_tokens: 200,
      },
    },
    {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: {
        input_tokens: 100,
        output_tokens: 42,
        cache_creation_input_tokens: 500,
        cache_read_input_tokens: 200,
      },
    },
    { type: 'message_stop' },
  ];
}

// ─── Tests ───────────────────────────────────────────────────────

describe('CustomAnthropic._streamChatModelEvents (inherited native)', () => {
  describe('text-only streaming', () => {
    test('passes invocation cache_control as a top-level request field', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const callOptions = {
        cache_control: { type: 'ephemeral' as const, ttl: '1h' as const },
      };
      for await (const _event of (
        model as unknown as StreamEventsModel
      )._streamChatModelEvents(
        [new HumanMessage('hello')],
        callOptions as unknown as BaseChatModelCallOptions
      )) {
        // Consume stream to trigger request construction.
      }

      expect(model.capturedRequest).toBeDefined();
      expect(model.capturedRequest?.cache_control).toEqual(
        callOptions.cache_control
      );
      expect(JSON.stringify(model.capturedRequest?.messages)).not.toContain(
        '"cache_control"'
      );
    });

    test('emits correct lifecycle events', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const eventNames = (
        await collectEvents(model, {
          streamUsage: true,
        } as BaseChatModelCallOptions)
      ).map((e) => e.event);

      expect(eventNames).toContain('message-start');
      expect(eventNames).toContain('content-block-start');
      expect(eventNames).toContain('content-block-delta');
      expect(eventNames).toContain('content-block-finish');
      expect(eventNames).toContain('message-finish');
    });

    test('message-start carries id and usage', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const start = events.find((e) => e.event === 'message-start');
      expect(start).toBeDefined();
      expect((start as { id?: string }).id).toBe('msg_01ABC');
      expect(
        (start as { usage?: { input_tokens: number } }).usage?.input_tokens
      ).toBe(25);
    });

    test('text deltas accumulate correctly', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const deltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' && 'index' in e && e.index === 0
      );
      expect(deltas.length).toBe(2);

      const d1 = deltas[0] as { delta: { type: string; text?: string } };
      expect(d1.delta.type).toBe('text-delta');
      expect(d1.delta.text).toBe('Hello');

      const d2 = deltas[1] as { delta: { type: string; text?: string } };
      expect(d2.delta.type).toBe('text-delta');
      expect(d2.delta.text).toBe(' world');
    });

    test('content-block-finish carries finalized text', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const events = await collectEvents(model);

      expect(
        events.find((e) => e.event === 'content-block-finish' && e.index === 0)
      ).toMatchObject({
        content: { type: 'text', text: 'Hello world' },
      });
    });

    test('message-finish carries stop reason', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const events = await collectEvents(model);

      const finish = events.find((e) => e.event === 'message-finish') as {
        reason: string;
      };
      expect(finish.reason).toBe('stop');
    });
  });

  describe('thinking + text streaming', () => {
    test('reasoning block accumulates correctly', async () => {
      const model = new MockStreamChatAnthropic(thinkingPlusTextEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const reasoningDeltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' &&
          'index' in e &&
          e.index === 0 &&
          'delta' in e &&
          (e.delta as { type: string }).type === 'reasoning-delta'
      );
      expect(reasoningDeltas.length).toBe(2);

      const rd1 = reasoningDeltas[0] as {
        delta: { type: string; reasoning: string };
      };
      expect(rd1.delta.reasoning).toBe('Let me');

      const rd2 = reasoningDeltas[1] as {
        delta: { type: string; reasoning: string };
      };
      expect(rd2.delta.reasoning).toBe(' reason...');

      expect(
        events.find((e) => e.event === 'content-block-finish' && e.index === 0)
      ).toMatchObject({
        content: { type: 'reasoning', reasoning: 'Let me reason...' },
      });
    });

    test('text block follows reasoning with correct index', async () => {
      const model = new MockStreamChatAnthropic(thinkingPlusTextEvents());
      const events = await collectEvents(model);

      expect(
        events.find((e) => e.event === 'content-block-finish' && e.index === 1)
      ).toMatchObject({
        content: { type: 'text', text: 'The answer is 42.' },
      });
    });

    test('signature delta is handled as non_standard', async () => {
      const model = new MockStreamChatAnthropic(thinkingPlusTextEvents());
      const events = await collectEvents(model);

      const sigDelta = events.find(
        (e) =>
          e.event === 'content-block-delta' &&
          'delta' in e &&
          (e.delta as { fields?: { signature?: string } }).fields?.signature ===
            'sig_abc'
      ) as { delta: { type: string; fields?: { signature?: string } } };
      expect(sigDelta).toBeDefined();
      expect(sigDelta.delta.type).toBe('block-delta');
      expect(sigDelta.delta.fields?.signature).toBe('sig_abc');
    });
  });

  describe('tool call streaming', () => {
    test('tool call args accumulate correctly', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const events = await collectEvents(model);

      expect(
        events.find((e) => e.event === 'content-block-start' && e.index === 1)
      ).toMatchObject({
        content: {
          type: 'tool_call_chunk',
          name: 'web_search',
          id: 'toolu_01ABC',
        },
      });

      const toolDeltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' && 'index' in e && e.index === 1
      );
      expect(toolDeltas.length).toBe(2);

      const td1 = toolDeltas[0] as {
        delta: { type: string; fields: { type: string; args?: string } };
      };
      expect(td1.delta.type).toBe('block-delta');
      expect(td1.delta.fields.type).toBe('tool_call_chunk');
      expect(td1.delta.fields.args).toBe('{"query"');

      const td2 = toolDeltas[1] as {
        delta: { type: string; fields: { type: string; args?: string } };
      };
      expect(td2.delta.type).toBe('block-delta');
      expect(td2.delta.fields.type).toBe('tool_call_chunk');
      expect(td2.delta.fields.args).toBe('{"query":"weather"}');
    });

    test('tool call finish has parsed args', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const events = await collectEvents(model);

      expect(
        events.find((e) => e.event === 'content-block-finish' && e.index === 1)
      ).toMatchObject({
        content: {
          type: 'tool_call',
          name: 'web_search',
          id: 'toolu_01ABC',
          args: { query: 'weather' },
        },
      });
    });

    test('message-finish has tool_use reason', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const events = await collectEvents(model);

      const finish = events.find((e) => e.event === 'message-finish') as {
        reason: string;
      };
      expect(finish.reason).toBe('tool_use');
    });
  });

  describe('usage streaming', () => {
    test('usage snapshot with cache details', async () => {
      const model = new MockStreamChatAnthropic(cacheUsageEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const start = events.find((e) => e.event === 'message-start') as {
        usage: {
          input_tokens: number;
          input_token_details: {
            cache_creation: number;
            cache_read: number;
          };
        };
      };
      // 100 + 500 + 200 = 800
      expect(start.usage.input_tokens).toBe(800);
      expect(start.usage.input_token_details.cache_creation).toBe(500);
      expect(start.usage.input_token_details.cache_read).toBe(200);
    });

    test('usage event emitted on message_delta', async () => {
      const model = new MockStreamChatAnthropic(cacheUsageEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const usageEvents = events.filter((e) => e.event === 'usage');
      expect(usageEvents.length).toBeGreaterThanOrEqual(1);

      const lastUsage = usageEvents[usageEvents.length - 1] as {
        usage: { output_tokens: number };
      };
      expect(lastUsage.usage.output_tokens).toBe(3);
    });

    test('message-finish carries final usage', async () => {
      const model = new MockStreamChatAnthropic(cacheUsageEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const finish = events.find((e) => e.event === 'message-finish') as {
        usage: { input_tokens: number; output_tokens: number };
      };
      expect(finish.usage.input_tokens).toBe(800);
      expect(finish.usage.output_tokens).toBe(3);
    });

    test('usage events preserve cumulative message_delta totals', async () => {
      const model = new MockStreamChatAnthropic(lateUsageEvents());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const usageEvents = events.filter((event) => event.event === 'usage');
      const lastUsage = usageEvents[usageEvents.length - 1] as {
        usage: {
          input_tokens: number;
          output_tokens: number;
          total_tokens: number;
          input_token_details: {
            cache_creation: number;
            cache_read: number;
          };
        };
      };
      const expectedUsage = {
        input_tokens: 800,
        output_tokens: 42,
        total_tokens: 842,
        input_token_details: {
          cache_creation: 500,
          cache_read: 200,
        },
      };
      expect(lastUsage.usage).toEqual(expectedUsage);
      expect(
        events.find((event) => event.event === 'message-finish')
      ).toMatchObject({ usage: expectedUsage });
    });

    test('no usage events when streamUsage is false', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      model.streamUsage = false;
      const events = await collectEvents(model, {
        streamUsage: false,
      } as BaseChatModelCallOptions);

      const usageEvents = events.filter((e) => e.event === 'usage');
      expect(usageEvents.length).toBe(0);

      const start = events.find((e) => e.event === 'message-start') as {
        usage?: unknown;
      };
      expect(start.usage).toBeUndefined();
    });
  });

  describe('provider passthrough', () => {
    test('message_start metadata is forwarded as provider event', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const events = await collectEvents(model);

      const providerEvents = events.filter((e) => e.event === 'provider');
      const metaEvent = providerEvents.find(
        (e) => (e as { name: string }).name === 'message_start'
      ) as { provider: string; payload: { model: string; id: string } };
      expect(metaEvent).toBeDefined();
      expect(metaEvent.provider).toBe('anthropic');
      expect(metaEvent.payload.model).toBe('claude-sonnet-4-20250514');
      expect(metaEvent.payload.id).toBe('msg_01ABC');
    });

    test('unknown events are forwarded as provider events', async () => {
      const eventsWithPing = [
        ...textOnlyEvents().slice(0, -1), // everything except message_stop
        { type: 'ping' },
        textOnlyEvents().slice(-1)[0], // message_stop
      ];
      const model = new MockStreamChatAnthropic(eventsWithPing);
      const events = await collectEvents(model);

      const pingEvent = events.find(
        (e) => e.event === 'provider' && (e as { name: string }).name === 'ping'
      );
      expect(pingEvent).toBeDefined();
    });
  });

  describe('integration with ChatModelStream', () => {
    test('text sub-stream works end-to-end', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const stream = new ChatModelStream(streamEvents(model));
      const text = await stream.text;
      expect(text).toBe('Hello world');
    });

    test('toolCalls sub-stream works end-to-end', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const stream = new ChatModelStream(streamEvents(model));
      const calls = await stream.toolCalls;
      expect(calls.length).toBe(1);
      expect(calls[0]!.name).toBe('web_search');
      expect(calls[0]!.args).toEqual({ query: 'weather' });
    });

    test('reasoning sub-stream works end-to-end', async () => {
      const model = new MockStreamChatAnthropic(thinkingPlusTextEvents());
      const stream = new ChatModelStream(streamEvents(model));
      const reasoning = await stream.reasoning;
      expect(reasoning).toBe('Let me reason...');
    });

    test('output assembles correct AIMessage', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const stream = new ChatModelStream(streamEvents(model));
      const message = await stream.output;

      expect(message.id).toBe('msg_03GHI');
      expect(message._getType()).toBe('ai');

      const content = message.content as Array<{
        type: string;
        text?: string;
        name?: string;
        args?: unknown;
      }>;

      expect(content.length).toBe(2);
      expect(content[0]!.type).toBe('text');
      expect(content[0]!.text).toBe('Let me search.');
      expect(content[1]!.type).toBe('tool_call');
      expect(content[1]!.name).toBe('web_search');
      expect(content[1]!.args).toEqual({ query: 'weather' });

      expect(message.tool_calls?.length).toBe(1);
      expect(message.tool_calls?.[0]?.name).toBe('web_search');
    });

    test('usage sub-stream works end-to-end', async () => {
      const model = new MockStreamChatAnthropic(cacheUsageEvents());
      const stream = new ChatModelStream(
        streamEvents(model, { streamUsage: true } as BaseChatModelCallOptions)
      );
      const usage = await stream.usage;
      expect(usage?.input_tokens).toBe(800);
      expect(usage?.output_tokens).toBe(3);
    });

    test('await stream returns AIMessage directly', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const message = await new ChatModelStream(streamEvents(model));
      expect(message._getType()).toBe('ai');
      expect(message.id).toBe('msg_01ABC');
    });

    test('sequential sub-stream consumption', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const stream = new ChatModelStream(streamEvents(model));

      const text = await stream.text;
      expect(text).toBe('Let me search.');

      const tools = await stream.toolCalls;
      expect(tools.length).toBe(1);
      expect(tools[0]!.name).toBe('web_search');
    });

    test('parallel sub-stream consumption from events', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const events = await collectEvents(model);

      async function* replay(): AsyncGenerator<ChatModelStreamEvent> {
        for (const e of events) {
          yield e;
        }
      }
      const stream = new ChatModelStream(replay());

      const [text, tools] = await Promise.all([stream.text, stream.toolCalls]);

      expect(text).toBe('Let me search.');
      expect(tools.length).toBe(1);
      expect(tools[0]!.name).toBe('web_search');
    });
  });

  // Re-expressed from upstream's `streaming events` describe, which used vitest
  // custom matchers (`toHaveStreamText` / `toHaveStreamReasoning` /
  // `toHaveStreamToolCalls`) that do not exist in jest. The matchers wrap the
  // `ChatModelStream` sub-streams asserted here directly. `model.streamEvents(...)`
  // is the inherited public entry; we drive it via the internal generator to keep
  // the mocked transport in play.
  describe('streaming events (sub-stream assertions)', () => {
    test('streams text', async () => {
      const model = new MockStreamChatAnthropic(textOnlyEvents());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.text).toBe('Hello world');
    });

    test('streams reasoning', async () => {
      const model = new MockStreamChatAnthropic(thinkingPlusTextEvents());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.reasoning).toBe('Let me reason...');
    });

    test('streams tool calls', async () => {
      const model = new MockStreamChatAnthropic(toolCallEvents());
      const stream = new ChatModelStream(streamEvents(model));
      const calls = await stream.toolCalls;
      expect(calls).toEqual([
        expect.objectContaining({
          name: 'web_search',
          args: { query: 'weather' },
        }),
      ]);
    });
  });

  // Dropped (inherited): live — upstream had no live-API cases in this file; every
  // case is unit-testable with the mocked transport above.
});
