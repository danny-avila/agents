// Inherited DeepSeek specs for the LibreChat `@langchain/agents` fork, ported
// from three upstream @langchain/deepseek@1.1.3 suites and adapted to the
// fork's actual surface (`ChatDeepSeek` from `@/llm/openai`):
//
//   1. reasoning — Inherited from @langchain/deepseek@1.1.3
//      src/tests/chat_models_reasoning.test.ts.
//      Upstream drives `model.stream("hi")` through a `configuration.fetch`
//      SSE mock. Our fork heavily overrides the legacy streaming path
//      (`_streamResponseChunks` -> `_streamResponseChunksWithReasoning`) to
//      split `<think>` fallback tags into `additional_kwargs.reasoning_content`.
//      We exercise that override directly by overriding `completionWithRetry`
//      (the `deepseek.test.ts` harness pattern) and feeding OpenAI-shaped
//      chunks. The fork's `<think>` parser reproduces upstream's observable
//      content/reasoning split exactly for every deterministic case below.
//
//   2. stream events — Inherited from @langchain/deepseek@1.1.3
//      src/tests/chat_models_stream_events.test.ts.
//      Our `ChatDeepSeek` does NOT override `_streamChatModelEvents`; it
//      inherits the native `ChatModelStreamEvent` protocol from upstream's
//      ChatOpenAICompletions, a separate code path from the legacy
//      `_streamResponseChunks` override. These are kept as a parity check.
//      Upstream's vitest custom matchers (`toHaveStreamText` /
//      `toHaveStreamReasoning` / `toHaveStreamToolCalls`) do not exist in jest;
//      they are re-expressed against the public `ChatModelStream` sub-streams
//      (`.text` / `.reasoning` / `.toolCalls`) that those matchers wrap.
//
//   3. standard unit tests — Inherited from @langchain/deepseek@1.1.3
//      src/tests/chat_models.standard.test.ts.
//      Upstream runs `ChatModelUnitTests` from `@langchain/standard-tests/vitest`.
//      That package is vitest-only and is not installed in this repo, so the
//      harness class itself is dropped. Its deterministic, fork-relevant
//      assertions (api-key precedence, tool-calling, structured output, model
//      field) are re-expressed inline against the fork.

import { describe, it, test, expect } from '@jest/globals';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import {
  openAIReasoningTextChunks,
  openAITextOnlyChunks,
  openAIToolCallChunks,
} from '@langchain/core/testing';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { OpenAIClient } from '@langchain/openai';
import { ChatDeepSeek } from '@/llm/openai';

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;

function createContentChunk(content: string): RawChunk {
  return {
    id: 'chatcmpl-123',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'deepseek-chat',
    choices: [
      {
        index: 0,
        delta: { content },
        finish_reason: null,
        logprobs: null,
      },
    ],
  };
}

function createStopChunk(): RawChunk {
  return {
    id: 'chatcmpl-123',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'deepseek-chat',
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
  };
}

/**
 * Mirrors the `deepseek.test.ts` / smoke-test transport mock: override
 * `completionWithRetry` so the legacy streaming path consumes fixed chunks.
 */
class MockStreamChatDeepSeek extends ChatDeepSeek {
  constructor(private readonly chunks: RawChunk[]) {
    super({ apiKey: 'fake-key', model: 'deepseek-chat', streaming: true });
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAIClient.RequestOptions
  ): Promise<AsyncIterable<RawChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAIClient.RequestOptions
  ): Promise<OpenAIClient.Chat.Completions.ChatCompletion>;
  async completionWithRetry(): Promise<
    AsyncIterable<RawChunk> | OpenAIClient.Chat.Completions.ChatCompletion
  > {
    const chunks = this.chunks;
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<RawChunk> {
        for (const chunk of chunks) {
          yield chunk;
        }
      },
    };
  }
}

type AggregatedStream = { content: string; reasoning: string };

async function streamThinkContents(
  contents: string[]
): Promise<AggregatedStream> {
  const chunks = [...contents.map(createContentChunk), createStopChunk()];
  const model = new MockStreamChatDeepSeek(chunks);

  const collected: AIMessageChunk[] = [];
  for await (const chunk of await model.stream([new HumanMessage('hi')])) {
    collected.push(chunk);
  }

  const content = collected
    .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
    .join('');
  const reasoning = collected
    .map((chunk) => (chunk.additional_kwargs.reasoning_content as string) ?? '')
    .join('');
  return { content, reasoning };
}

function toDeepSeekChunks(
  chunks: ReturnType<typeof openAITextOnlyChunks>
): RawChunk[] {
  return chunks.map((chunk) => ({
    ...chunk,
    id: chunk.id ?? 'chatcmpl-test',
    object: 'chat.completion.chunk',
    created: 0,
    model: chunk.model ?? 'deepseek-chat',
    service_tier: null,
    choices: (chunk.choices ?? []).map((choice) => ({
      ...choice,
      delta: choice.delta ?? {},
    })) as RawChunk['choices'],
  })) as RawChunk[];
}

type StreamEventsModel = {
  _streamChatModelEvents: (
    messages: unknown[],
    options: Record<string, unknown>
  ) => AsyncGenerator<ChatModelStreamEvent>;
};

function streamEvents(
  model: ChatDeepSeek
): AsyncGenerator<ChatModelStreamEvent> {
  return (model as unknown as StreamEventsModel)._streamChatModelEvents([], {});
}

describe('ChatDeepSeek (inherited @langchain/deepseek@1.1.3 reasoning)', () => {
  // Dropped (inherited): `new ChatDeepSeek("deepseek-chat", {...})` shorthand —
  // the fork constructor only accepts a single fields object
  // (ConstructorParameters<typeof OriginalChatDeepSeek>[0]); the positional
  // (model, fields) overload does not type-check. Model-field init is covered
  // in the standard-unit-tests describe below.

  test('separates <think> tags into reasoning_content', async () => {
    const { content, reasoning } = await streamThinkContents([
      '<think>',
      'thinking process...',
      '</think>',
      'Hello world',
    ]);

    // Parity with upstream: tags stripped, thought routed to reasoning.
    expect(content).toBe('Hello world');
    expect(reasoning).toBe('thinking process...');
  });

  test('handles multiple think blocks and content before/after', async () => {
    const { content, reasoning } = await streamThinkContents([
      'Start ',
      '<think>',
      'Reason 1',
      '</think>',
      ' Middle ',
      '<think>',
      'Reason 2',
      '</think>',
      ' End',
    ]);

    // Parity with upstream.
    expect(content).toBe('Start  Middle  End');
    expect(reasoning).toBe('Reason 1Reason 2');
  });

  test('handles unclosed think tags (flush at end)', async () => {
    const { reasoning } = await streamThinkContents([
      'Start ',
      '<think>',
      'Unclosed thought',
    ]);

    // Parity with upstream: trailing open block flushes as reasoning.
    expect(reasoning).toBe('Unclosed thought');
  });

  test('handles split tags across chunks', async () => {
    const { reasoning } = await streamThinkContents([
      '<th',
      'ink>Thought',
      '</th',
      'ink>',
    ]);

    // Parity with upstream: partial-tag buffering reassembles boundaries.
    expect(reasoning).toBe('Thought');
  });

  test('handles empty think blocks', async () => {
    const { content, reasoning } = await streamThinkContents([
      'Before ',
      '<think>',
      '</think>',
      ' After',
    ]);

    // Parity with upstream.
    expect(content).toBe('Before  After');
    expect(reasoning).toBe('');
  });

  test('handles nested think tags (inner treated as reasoning text)', async () => {
    const { content, reasoning } = await streamThinkContents([
      '<think>',
      'Outer ',
      '<think>',
      'Inner',
      '</think>',
      ' Content',
    ]);

    // Parity with upstream: first </think> closes the outer block; the inner
    // <think> is kept verbatim inside the reasoning text.
    expect(content).toBe(' Content');
    expect(reasoning).toBe('Outer <think>Inner');
  });

  test('handles malformed tags gracefully', async () => {
    const { content } = await streamThinkContents([
      '</think>',
      'Text ',
      '<think',
      ' more',
    ]);

    // Parity with upstream's loose assertion: orphan/incomplete tags are
    // surfaced as content. The fork emits the full
    // "</think>Text <think more" verbatim.
    expect(content).toContain('Text');
  });
});

describe('ChatDeepSeek.streamEvents (inherited @langchain/deepseek@1.1.3, native path)', () => {
  // The fork inherits `_streamChatModelEvents` from upstream's completions
  // model (no override), so the native ChatModelStreamEvent protocol behaves
  // identically — kept as a parity check.

  test('streams text', async () => {
    const model = new MockStreamChatDeepSeek(
      toDeepSeekChunks(openAITextOnlyChunks())
    );
    const stream = new ChatModelStream(streamEvents(model));
    expect(await stream.text).toBe('Hello world');
  });

  test('streams reasoning', async () => {
    const model = new MockStreamChatDeepSeek(
      toDeepSeekChunks(openAIReasoningTextChunks())
    );
    const stream = new ChatModelStream(streamEvents(model));
    expect(await stream.reasoning).toBe('Let me reason...');
  });

  test('streams tool calls', async () => {
    const model = new MockStreamChatDeepSeek(
      toDeepSeekChunks(openAIToolCallChunks())
    );
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

describe('ChatDeepSeek (inherited @langchain/deepseek@1.1.3 standard unit tests)', () => {
  // Re-expressed from upstream's `ChatModelUnitTests` standard suite, which
  // imports `@langchain/standard-tests/vitest` (vitest-only, not installed).
  // The harness class is dropped; its deterministic assertions are inlined.

  it('initializes the model field from constructor args', () => {
    process.env.DEEPSEEK_API_KEY = 'test';
    const model = new ChatDeepSeek({ model: 'deepseek-chat', apiKey: 'test' });
    expect(model.model).toBe('deepseek-chat');
  });

  it('reads the api key from the apiKey field when the env var is unset', () => {
    process.env.DEEPSEEK_API_KEY = '';
    const model = new ChatDeepSeek({
      apiKey: 'arg-key',
      model: 'deepseek-chat',
    });
    expect(model.apiKey).toBe('arg-key');
    process.env.DEEPSEEK_API_KEY = 'test';
  });

  it('throws when no api key is provided via field or env', () => {
    process.env.DEEPSEEK_API_KEY = '';
    expect(() => new ChatDeepSeek({ model: 'deepseek-chat' })).toThrow(
      /API key not found/i
    );
    process.env.DEEPSEEK_API_KEY = 'test';
  });

  it('supports tool calling (bindTools) and structured output', () => {
    process.env.DEEPSEEK_API_KEY = 'test';
    const model = new ChatDeepSeek({ model: 'deepseek-chat', apiKey: 'test' });
    expect(typeof model.bindTools).toBe('function');
    expect(typeof model.withStructuredOutput).toBe('function');
    expect(model.bindTools([])).toBeDefined();
  });
});

// Dropped (inherited): live `.invoke()` / `.stream()`-to-API cases — none of
// the three source suites contained network-backed cases; all upstream
// scenarios were deterministic mocked transport.
