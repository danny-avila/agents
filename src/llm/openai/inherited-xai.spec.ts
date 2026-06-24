// Inherited xAI specs for the LibreChat OpenAI fork, ported from two
// upstream-derived suites (each adapted to the fork's actual surface):
//
//   1. completions — Inherited from @langchain/xai@1.4.3
//      chat_models/tests/completions.test.ts.
//      xAI 1.4.x splits the model into chat_models/{completions,responses} +
//      converters (like openai 1.5.x). Our fork's `ChatXAI` (from `@/llm/openai`)
//      extends the upstream top-level `ChatXAI`, which IS the completions class
//      (`ChatXAI extends ChatOpenAICompletions`), so the completions-path behavior
//      tested upstream lives directly on the instance (no `.completions`
//      sub-delegate, unlike the fork's `ChatOpenAI`). The live-search /
//      invocationParams / `_hasBuiltInTools` / `_getEffectiveSearchParameters`
//      surface is inherited verbatim and exercised here against the fork.
//
//   2. stream events — Inherited from @langchain/xai@1.4.3
//      chat_models/tests/chat_models_stream_events.test.ts.
//      Upstream extends `ChatXAI` (completions) and drives `streamEvents` with
//      vitest custom matchers (`toHaveStreamText` / `toHaveStreamReasoning` /
//      `toHaveStreamToolCalls`) that are unavailable in jest. The fork inherits
//      `_streamChatModelEvents` from langchain-core, so we keep these as a parity
//      check: construct a `MockStreamChatXAI` overriding `completionWithRetry`
//      (mirroring the upstream mock + custom-chat-models.smoke.test.ts), wrap
//      `_streamChatModelEvents` in a `ChatModelStream`, and assert on the same
//      promise-backed sub-streams (`.text` / `.reasoning` / `.toolCalls`) that
//      the vitest matchers wrap.
//
// Import notes (per task rules):
//   - The model class is imported from `@/llm/openai` (our fork), NOT
//     `@langchain/xai`.
//   - `isXAIBuiltInTool` and the `ChatXAICompletionsInvocationParams` type are
//     re-exported by `@langchain/xai`'s main entry and imported from there
//     (the fork does not re-wrap them).
//   - The live-search tool constants/types (`XAI_LIVE_SEARCH_TOOL_NAME`,
//     `XAI_LIVE_SEARCH_TOOL_TYPE`, `XAILiveSearchTool`, `XAISearchParameters`)
//     are NOT exported from `@langchain/xai` (only `.` is in package `exports`,
//     and they live behind unexposed subpaths). They are reproduced locally from
//     `dist/tools/live_search.d.ts` + `dist/live_search.d.ts` to keep the
//     fixtures faithful without reaching into deep import paths.
//
// Where the fork intentionally diverges from upstream we assert OUR behavior and
// note the diff inline. UNIT only (mocked transport); the upstream live cases are
// dropped with reasons below.
//
// Dropped (inherited): the upstream suite has no `it.skip`/live integration
// cases — every upstream case is deterministic and ported here. No xAI Responses
// API cases existed in either source, so none needed a probe/drop.

import { isXAIBuiltInTool } from '@langchain/xai';
import { describe, test, expect, beforeEach } from '@jest/globals';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import {
  openAIReasoningTextChunks,
  openAITextOnlyChunks,
  openAIToolCallChunks,
} from '@langchain/core/testing';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { ChatXAICompletionsInvocationParams } from '@langchain/xai';
import { ChatXAI } from '@/llm/openai';

const TEST_MODEL = process.env.XAI_TEST_MODEL ?? 'grok-3-fast';

// Local copies of the (unexported) live-search tool literals, verbatim from
// @langchain/xai@1.4.3 dist/tools/live_search.d.ts.
const XAI_LIVE_SEARCH_TOOL_NAME = 'live_search' as const;
const XAI_LIVE_SEARCH_TOOL_TYPE = 'live_search_deprecated_20251215' as const;

type XAILiveSearchTool = {
  name: typeof XAI_LIVE_SEARCH_TOOL_NAME;
  type: typeof XAI_LIVE_SEARCH_TOOL_TYPE;
};

// Subset of the (unexported) XAISearchParameters shape used by these fixtures.
type XAISearchParameters = NonNullable<
  ChatXAICompletionsInvocationParams['search_parameters']
>;

type ParsedCallOptions = ChatXAI['ParsedCallOptions'];

// `clientConfig` and the protected search helpers are reached the way upstream
// reaches them (an internal cast); typed structurally to avoid `any`.
type ChatXAIInternals = {
  clientConfig: { baseURL?: string };
  searchParameters?: XAISearchParameters;
  _getEffectiveSearchParameters(
    options?: ParsedCallOptions
  ): XAISearchParameters | undefined;
  _hasBuiltInTools(tools?: unknown[]): boolean;
};

const internals = (model: ChatXAI): ChatXAIInternals =>
  model as unknown as ChatXAIInternals;

beforeEach(() => {
  process.env.XAI_API_KEY = 'foo';
});

describe('inherited xAI completions (chat_models/tests/completions.test.ts)', () => {
  describe('baseURL configuration', () => {
    test('should use default baseURL when not specified', () => {
      const model = new ChatXAI();
      expect(internals(model).clientConfig.baseURL).toBe('https://api.x.ai/v1');
    });

    test('should use custom baseURL when provided', () => {
      const model = new ChatXAI({
        baseURL: 'https://custom.api.example.com/v1',
      });
      expect(internals(model).clientConfig.baseURL).toBe(
        'https://custom.api.example.com/v1'
      );
    });
  });

  // Fork divergence: the serialized `lc_id` reflects the LibreChat subclass name
  // (`LibreChatXAI` via `lc_name()`), not upstream's `ChatXAI`. Everything else
  // (the `model` kwarg, secret stripping) is identical, so we assert OUR id.
  describe('Serialization', () => {
    test('serializes model + apiKey (lc_id = LibreChatXAI)', () => {
      delete process.env.XAI_API_KEY;
      const model = new ChatXAI({ model: TEST_MODEL, apiKey: 'bar' });
      expect(JSON.stringify(model)).toEqual(
        `{"lc":1,"type":"constructor","id":["langchain","chat_models","xai","LibreChatXAI"],"kwargs":{"model":"${TEST_MODEL}"}}`
      );
    });

    test('serializes with no params (default model grok-3-fast)', () => {
      const model = new ChatXAI();
      expect(JSON.stringify(model)).toEqual(
        '{"lc":1,"type":"constructor","id":["langchain","chat_models","xai","LibreChatXAI"],"kwargs":{"model":"grok-3-fast"}}'
      );
    });

    test('serializes with model shorthand', () => {
      // The string-shorthand constructor is inherited from the upstream base but
      // is narrowed away by the fork's field type, so cast to reach that path.
      const model = new ChatXAI(
        TEST_MODEL as unknown as ConstructorParameters<typeof ChatXAI>[0]
      );
      expect(JSON.stringify(model)).toEqual(
        `{"lc":1,"type":"constructor","id":["langchain","chat_models","xai","LibreChatXAI"],"kwargs":{"model":"${TEST_MODEL}"}}`
      );
    });
  });

  describe('Server Tool Calling', () => {
    describe('isXAIBuiltInTool', () => {
      test('should identify live_search as a built-in tool', () => {
        const liveSearchTool: XAILiveSearchTool = {
          name: XAI_LIVE_SEARCH_TOOL_NAME,
          type: XAI_LIVE_SEARCH_TOOL_TYPE,
        };
        expect(isXAIBuiltInTool(liveSearchTool)).toBe(true);
      });

      test('should not identify function tools as built-in', () => {
        const functionTool = {
          type: 'function',
          function: {
            name: 'get_weather',
            description: 'Get the weather',
            parameters: { type: 'object', properties: {} },
          },
        };
        expect(isXAIBuiltInTool(functionTool as never)).toBe(false);
      });

      test('should not identify invalid objects as built-in', () => {
        expect(isXAIBuiltInTool(null as never)).toBe(false);
        expect(isXAIBuiltInTool(undefined as never)).toBe(false);
        expect(isXAIBuiltInTool({} as never)).toBe(false);
        expect(isXAIBuiltInTool({ type: 'other' } as never)).toBe(false);
      });
    });

    describe('ChatXAI with searchParameters', () => {
      test('should store searchParameters from constructor', () => {
        const searchParams: XAISearchParameters = {
          mode: 'auto',
          max_search_results: 5,
        };
        const model = new ChatXAI({ searchParameters: searchParams });
        expect(internals(model).searchParameters).toEqual(searchParams);
      });

      test('should have undefined searchParameters by default', () => {
        const model = new ChatXAI();
        expect(internals(model).searchParameters).toBeUndefined();
      });

      test('should merge search parameters correctly', () => {
        const model = new ChatXAI({
          searchParameters: { mode: 'auto', max_search_results: 5 },
        });

        const effectiveParams = internals(model)._getEffectiveSearchParameters({
          searchParameters: {
            max_search_results: 10,
            from_date: '2024-01-01',
          },
        } as ParsedCallOptions);

        expect(effectiveParams).toEqual({
          mode: 'auto',
          max_search_results: 10,
          from_date: '2024-01-01',
        });
      });
    });

    describe('invocationParams with server tools', () => {
      test('should add search_parameters when live_search tool is bound', () => {
        const model = new ChatXAI();

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({
            tools: [
              {
                type: XAI_LIVE_SEARCH_TOOL_TYPE,
                name: XAI_LIVE_SEARCH_TOOL_NAME,
              },
            ],
          } as unknown as ParsedCallOptions);

        expect(params.search_parameters).toBeDefined();
        expect(params.search_parameters?.mode).toBe('auto');
      });

      test('should add search_parameters from call options', () => {
        const model = new ChatXAI();

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({
            searchParameters: {
              mode: 'on',
              max_search_results: 10,
              from_date: '2024-01-01',
            },
          } as unknown as ParsedCallOptions);

        expect(params.search_parameters).toEqual({
          mode: 'on',
          max_search_results: 10,
          from_date: '2024-01-01',
        });
      });

      test('should include sources in search_parameters when provided', () => {
        const model = new ChatXAI();

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({
            searchParameters: {
              mode: 'on',
              sources: [
                { type: 'web', allowed_websites: ['x.ai'] },
                { type: 'news', excluded_websites: ['bbc.co.uk'] },
                { type: 'x', included_x_handles: ['xai'] },
                { type: 'rss', links: ['https://example.com/feed.rss'] },
              ],
            },
          } as unknown as ParsedCallOptions);

        expect(params.search_parameters).toBeDefined();
        expect(params.search_parameters?.sources).toEqual([
          { type: 'web', allowed_websites: ['x.ai'] },
          { type: 'news', excluded_websites: ['bbc.co.uk'] },
          { type: 'x', included_x_handles: ['xai'] },
          { type: 'rss', links: ['https://example.com/feed.rss'] },
        ]);
      });

      test('should omit sources field when none are configured', () => {
        const model = new ChatXAI();

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({
            searchParameters: { mode: 'auto' },
          } as unknown as ParsedCallOptions);

        expect(params.search_parameters).toEqual({ mode: 'auto' });
        expect(
          Object.prototype.hasOwnProperty.call(
            params.search_parameters as NonNullable<
              ChatXAICompletionsInvocationParams['search_parameters']
            >,
            'sources'
          )
        ).toBe(false);
      });

      test('should merge instance and call option search parameters', () => {
        const model = new ChatXAI({
          searchParameters: {
            mode: 'auto',
            max_search_results: 5,
            return_citations: true,
          },
        });

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({
            searchParameters: { max_search_results: 10 },
          } as unknown as ParsedCallOptions);

        expect(params.search_parameters).toEqual({
          mode: 'auto',
          max_search_results: 10,
          return_citations: true,
        });
      });

      test('should not add search_parameters when no search config is present', () => {
        const model = new ChatXAI();

        const params: ChatXAICompletionsInvocationParams =
          model.invocationParams({} as ParsedCallOptions);

        expect(params.search_parameters).toBeUndefined();
      });
    });

    describe('_hasBuiltInTools', () => {
      test('should return true when live_search tool is present', () => {
        const model = new ChatXAI();
        const result = internals(model)._hasBuiltInTools([
          { type: XAI_LIVE_SEARCH_TOOL_TYPE, name: XAI_LIVE_SEARCH_TOOL_NAME },
          { type: 'function', function: { name: 'test', parameters: {} } },
        ]);
        expect(result).toBe(true);
      });

      test('should return false when no built-in tools are present', () => {
        const model = new ChatXAI();
        const result = internals(model)._hasBuiltInTools([
          { type: 'function', function: { name: 'test', parameters: {} } },
        ]);
        expect(result).toBe(false);
      });

      test('should return false for undefined or empty tools', () => {
        const model = new ChatXAI();
        expect(internals(model)._hasBuiltInTools(undefined)).toBe(false);
        expect(internals(model)._hasBuiltInTools([])).toBe(false);
      });
    });
  });
});

// Parity check for the inherited native `streamEvents` protocol. Upstream's
// MockStreamChatXAI overrides `completionWithRetry` to feed raw OpenAI chunks;
// our fork's `ChatXAI` exposes the same method directly (it extends the
// completions class), so the mock transplants verbatim.
type RawChunks = ReturnType<typeof openAITextOnlyChunks>;

class MockStreamChatXAI extends ChatXAI {
  private readonly mockChunks: RawChunks;

  constructor(chunks: RawChunks) {
    super({ apiKey: 'fake-key', model: 'grok-3', streaming: true });
    this.mockChunks = chunks;
  }

  override async completionWithRetry(): Promise<never> {
    const chunks = this.mockChunks;
    return {
      async *[Symbol.asyncIterator]() {
        for (const chunk of chunks) {
          yield chunk;
        }
      },
    } as never;
  }
}

function streamEvents(model: ChatXAI): AsyncGenerator<ChatModelStreamEvent> {
  return (
    model as unknown as {
      _streamChatModelEvents(
        messages: unknown[],
        options: Record<string, unknown>
      ): AsyncGenerator<ChatModelStreamEvent>;
    }
  )._streamChatModelEvents([], {});
}

describe('inherited xAI streamEvents (chat_models/tests/chat_models_stream_events.test.ts)', () => {
  // Re-expressed from upstream's vitest matchers
  // (`toHaveStreamText` / `toHaveStreamReasoning` / `toHaveStreamToolCalls`),
  // which wrap exactly the `ChatModelStream` sub-streams asserted here.
  test('streams text', async () => {
    const stream = new ChatModelStream(
      streamEvents(new MockStreamChatXAI(openAITextOnlyChunks()))
    );
    expect(await stream.text).toBe('Hello world');
  });

  test('streams reasoning', async () => {
    const stream = new ChatModelStream(
      streamEvents(new MockStreamChatXAI(openAIReasoningTextChunks()))
    );
    expect(await stream.reasoning).toBe('Let me reason...');
  });

  test('streams tool calls', async () => {
    const stream = new ChatModelStream(
      streamEvents(new MockStreamChatXAI(openAIToolCallChunks()))
    );
    const calls = await stream.toolCalls;
    expect(calls).toEqual([
      expect.objectContaining({
        name: 'web_search',
        args: { query: 'weather' },
      }),
    ]);
  });
});
