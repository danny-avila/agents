// Inherited stream-events specs for the LibreChat Vertex fork, ported from two
// upstream @langchain/google-common@2.2.0 suites:
//
//   1. utils/tests/stream_events.test.ts  (the original prompt references it as
//      google-common_stream_events.test.ts) — unit tests for the pure converter
//      `convertGoogleGeminiStream`, which turns Gemini-style stream responses
//      into LangChain `ChatModelStreamEvent`s. Our fork's `ChatVertexAI` does
//      not re-implement this converter; it is re-exported from
//      `@langchain/google-common` and exercised verbatim here as a pure unit.
//
//   2. chat_models/tests/chat_models_stream_events.test.ts (the prompt references
//      it as google-common_chat_models_stream_events.test.ts) — tests
//      `ChatGoogle.streamEvents()`. Upstream drove a `TestChatGoogle` helper whose
//      `authOptions.resultFile` replayed recorded mock JSON and asserted via
//      vitest custom matchers (`toHaveStreamText` / `toHaveStreamReasoning` /
//      `toHaveStreamToolCalls`). Those fixture files and matchers are not
//      available in this repo, so each case is routed through OUR fork:
//      `new ChatVertexAI({...})` (which extends ChatGoogle and inherits the
//      native `_streamChatModelEvents` path) with the google transport mocked at
//      the `streamedConnection.request` boundary — it returns `{ data: <json
//      stream> }`, the exact shape `_streamChatModelEvents` consumes. The matchers
//      are re-expressed against the public `ChatModelStream` sub-streams
//      (`.text` / `.reasoning` / `.toolCalls`) that those matchers wrap.
//
// All cases are deterministic and run against a mocked transport — no live
// Vertex access. vitest -> jest, `zod/v3` -> n/a (no zod here).

import { describe, it, test, expect } from '@jest/globals';
import { convertGoogleGeminiStream } from '@langchain/google-common';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import type { BaseChatModelCallOptions } from '@langchain/core/language_models/chat_models';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import { ChatVertexAI } from '@/llm/vertexai';

type GeminiChunk = Record<string, unknown>;

// --- Converter unit-test harness (upstream source 1) -----------------------

async function* asAsyncIterable(
  chunks: GeminiChunk[]
): AsyncGenerator<GeminiChunk> {
  for (const chunk of chunks) {
    yield chunk;
  }
}

async function collectEvents(
  chunks: GeminiChunk[]
): Promise<ChatModelStreamEvent[]> {
  const out: ChatModelStreamEvent[] = [];
  for await (const event of convertGoogleGeminiStream(
    asAsyncIterable(chunks)
  )) {
    out.push(event);
  }
  return out;
}

// --- Class streamEvents harness (upstream source 2, routed through fork) ----

interface MockJsonStream {
  readonly streamDone: boolean;
  nextChunk(): Promise<GeminiChunk | null>;
}

function makeJsonStream(chunks: GeminiChunk[]): MockJsonStream {
  let i = 0;
  return {
    get streamDone(): boolean {
      return i >= chunks.length;
    },
    async nextChunk(): Promise<GeminiChunk | null> {
      if (i >= chunks.length) return null;
      const chunk = chunks[i];
      i += 1;
      return chunk;
    },
  };
}

interface StreamedConnectionModel {
  streamedConnection: {
    request: () => Promise<{ data: MockJsonStream }>;
  };
  _streamChatModelEvents: (
    messages: unknown[],
    options: BaseChatModelCallOptions
  ) => AsyncGenerator<ChatModelStreamEvent>;
}

function createStreamModel(chunks: GeminiChunk[]): ChatVertexAI {
  const model = new ChatVertexAI({
    model: 'gemini-2.0-flash',
    authOptions: {
      credentials: { client_email: 'test@example.com', private_key: 'test' },
      projectId: 'test-project',
    },
  } as ConstructorParameters<typeof ChatVertexAI>[0]);
  (model as unknown as StreamedConnectionModel).streamedConnection.request =
    async () => ({ data: makeJsonStream(chunks) });
  return model;
}

function streamEvents(
  model: ChatVertexAI,
  options: BaseChatModelCallOptions = {} as BaseChatModelCallOptions
): AsyncGenerator<ChatModelStreamEvent> {
  return (model as unknown as StreamedConnectionModel)._streamChatModelEvents(
    [],
    options
  );
}

// ---------------------------------------------------------------------------
// Source 1: convertGoogleGeminiStream (pure converter, re-exported from
// @langchain/google-common and inherited unchanged by the fork)
// ---------------------------------------------------------------------------

describe('convertGoogleGeminiStream (inherited from @langchain/google-common)', () => {
  test('text-only streaming', async () => {
    const events = await collectEvents([
      { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
      { candidates: [{ content: { parts: [{ text: ' world' }] } }] },
    ]);

    const textDeltas = events.filter(
      (e) =>
        e.event === 'content-block-delta' &&
        (e as { delta: { type: string } }).delta.type === 'text-delta'
    );
    expect(textDeltas).toHaveLength(2);

    expect(
      events.find((e) => e.event === 'content-block-finish')
    ).toMatchObject({
      content: { text: 'Hello world' },
    });
  });

  test('maps Gemini finish reasons', async () => {
    const lengthEvents = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Hello' }] },
            finishReason: 'MAX_TOKENS',
          },
        ],
      },
    ]);
    const lengthFinish = lengthEvents.find((e) => e.event === 'message-finish');
    expect(lengthFinish).toMatchObject({ reason: 'length' });

    const filterEvents = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Hello' }] },
            finishReason: 'SAFETY',
          },
        ],
      },
    ]);
    const filterFinish = filterEvents.find((e) => e.event === 'message-finish');
    expect(filterFinish).toMatchObject({ reason: 'content_filter' });
  });

  test('thinking parts map to reasoning', async () => {
    const events = await collectEvents([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Let me think', thought: true }] },
          },
        ],
      },
    ]);

    expect(
      events.find(
        (e) =>
          e.event === 'content-block-finish' &&
          (e as { content: { type: string } }).content.type === 'reasoning'
      )
    ).toMatchObject({
      content: { reasoning: 'Let me think' },
    });
  });

  test('usage snapshots', async () => {
    const events = await collectEvents([
      {
        usageMetadata: {
          promptTokenCount: 10,
          candidatesTokenCount: 4,
          totalTokenCount: 14,
        },
        candidates: [{ content: { parts: [{ text: 'Hi' }] } }],
      },
    ]);

    expect(events.filter((e) => e.event === 'usage').length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Source 2: ChatGoogle.streamEvents — inherited native streamEvents path on
// the fork (ChatVertexAI does not override _streamChatModelEvents). Re-expressed
// against the public ChatModelStream sub-streams that the upstream vitest
// matchers wrapped.
// ---------------------------------------------------------------------------

describe('ChatVertexAI.streamEvents (inherited native path)', () => {
  test('streams text', async () => {
    const model = createStreamModel([
      { candidates: [{ content: { parts: [{ text: 'Hello' }] } }] },
      {
        candidates: [
          { content: { parts: [{ text: ' world' }] }, finishReason: 'STOP' },
        ],
      },
    ]);
    const stream = new ChatModelStream(streamEvents(model));
    expect(await stream.text).toBe('Hello world');
  });

  test('streams reasoning', async () => {
    const model = createStreamModel([
      {
        candidates: [
          {
            content: { parts: [{ text: 'Let me reason...', thought: true }] },
            finishReason: 'STOP',
          },
        ],
      },
    ]);
    const stream = new ChatModelStream(streamEvents(model));
    expect(await stream.reasoning).toBe('Let me reason...');
  });

  test('streams tool calls', async () => {
    const model = createStreamModel([
      {
        candidates: [
          {
            content: {
              parts: [
                {
                  functionCall: {
                    name: 'web_search',
                    args: { query: 'weather' },
                  },
                },
              ],
            },
            finishReason: 'STOP',
          },
        ],
      },
    ]);
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
