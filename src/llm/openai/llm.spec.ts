// Consolidated inherited specs for the LibreChat OpenAI fork, merged from three
// upstream-derived suites (each adapted to the fork's actual surface):
//
//   1. converters — Inherited from @langchain/openai@1.5.3
//      converters/tests/completions.test.ts.
//      Our fork does NOT export the standalone converter functions
//      (convertCompletionsMessageToBaseMessage, convertCompletionsDeltaToBaseMessageChunk,
//      convertMessagesToCompletionsMessageParams, completionsApiContentBlockConverter,
//      convertStandardContentBlockToCompletionsContentPart). The upstream fixtures are kept,
//      but each case is routed through our fork's actual surface:
//        - convertMessagesToCompletionsMessageParams -> _convertMessagesToOpenAIParams (@/llm/openai/utils)
//        - convertCompletionsMessageToBaseMessage / ...DeltaToBaseMessageChunk -> the protected
//          delegate methods on a ChatOpenAI instance's `.completions` (wrapping upstream + attachLibreChat*)
//        - completionsApiContentBlockConverter is module-private; it is exercised indirectly via
//          _convertMessagesToOpenAIParams, which runs data content blocks through it.
//      Where our fork intentionally diverges from upstream, we assert OUR behavior and note the diff inline.
//
//   2. completions — Inherited from @langchain/openai@1.5.3
//      chat_models/tests/completions.test.ts.
//      Our ChatOpenAI wraps the upstream completions class via a `completions`
//      delegate; the completions-path behavior tested upstream lives on that
//      delegate, so we reach it the way the smoke-test harness does.
//
//   3. stream events — Inherited from @langchain/openai@1.5.3
//      chat_models/tests/chat_models_stream_events.test.ts.
//      Upstream extends ChatOpenAICompletions directly and calls
//      `model._streamChatModelEvents(...)`. Our fork's `ChatOpenAI` (from `@/llm/openai`)
//      extends upstream's top-level `ChatOpenAI`, which inherits
//      `_streamChatModelEvents` and delegates it to its `completions` sub-model. The
//      native `ChatModelStreamEvent` protocol (`convertOpenAICompletionsStream`) is
//      therefore fully supported by the fork: it is a separate code path from the
//      legacy `_streamResponseChunks` override and from
//      `_convertCompletionsDeltaToBaseMessageChunk`.
//
//      Adaptation: construct the top-level fork `ChatOpenAI`, feed raw chunks by
//      overriding `completions.completionWithRetry` (mirroring
//      custom-chat-models.smoke.test.ts), and drive the inherited
//      `_streamChatModelEvents` / `streamEvents`.
//
//      The upstream `streamEvents` describe used vitest custom matchers
//      (`toHaveStreamText` / `toHaveStreamToolCalls` / `toHaveStreamReasoning`),
//      which do not exist in jest. They are re-expressed here against the public
//      `ChatModelStream` sub-streams (`.text` / `.toolCalls` / `.reasoning`) that
//      those matchers wrap.

import { z } from 'zod';
import { describe, it, test, expect, jest } from '@jest/globals';
import { toJsonSchema } from '@langchain/core/utils/json_schema';
import { ChatModelStream } from '@langchain/core/language_models/stream';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { BaseChatModelCallOptions } from '@langchain/core/language_models/chat_models';
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, BaseMessageChunk } from '@langchain/core/messages';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { OpenAIClient } from '@langchain/openai';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { AzureChatOpenAI, ChatOpenAI } from '@/llm/openai';

type CompletionsDelegate = {
  _convertCompletionsMessageToBaseMessage(
    message: Record<string, unknown>,
    rawResponse: Record<string, unknown>
  ): BaseMessage;
  _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: Record<string, unknown>,
    defaultRole?: string
  ): BaseMessageChunk;
};

type CompletionsStreamDelegate = {
  completionWithRetry: (request: unknown) => Promise<AsyncIterable<unknown>>;
  _streamResponseChunks: (
    messages: BaseMessage[],
    options: Record<string, unknown>,
    runManager?: CallbackManagerForLLMRun
  ) => AsyncGenerator<ChatGenerationChunk>;
};

type InvocationParamsDelegate = {
  invocationParams: (
    options: Record<string, unknown>,
    extra?: { streaming?: boolean }
  ) => { tools?: { function: { strict?: boolean } }[] };
};

type ResponsesInvocationParamsDelegate = {
  invocationParams: (options: Record<string, unknown>) => {
    include?: string[];
  };
};

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;

type StreamEventsModel = {
  completions: {
    streamUsage: boolean;
    completionWithRetry: (request: unknown) => Promise<AsyncIterable<RawChunk>>;
  };
  _streamChatModelEvents: (
    messages: unknown[],
    options: BaseChatModelCallOptions
  ) => AsyncGenerator<ChatModelStreamEvent>;
};

function completionsDelegateOf(model: ChatOpenAI): CompletionsDelegate {
  return (model as unknown as { completions: CompletionsDelegate }).completions;
}

const completionsOf = <T>(model: ChatOpenAI): T =>
  (model as unknown as { completions: T }).completions;

const responsesOf = <T>(model: ChatOpenAI | AzureChatOpenAI): T =>
  (model as unknown as { responses: T }).responses;

function newOpenAI(): ChatOpenAI {
  return new ChatOpenAI({ model: 'gpt-4o-mini', apiKey: 'test-key' });
}

describe('Responses hosted web search sources', () => {
  const models = [
    new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' }),
    new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiInstanceName: 'test-instance',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
    }),
  ];

  test.each(models)('requests sources for hosted web search', (model) => {
    const params = responsesOf<ResponsesInvocationParamsDelegate>(
      model
    ).invocationParams({ tools: [{ type: 'web_search' }] });

    expect(params.include).toContain('web_search_call.action.sources');
  });

  test('does not request web sources without a hosted web search tool', () => {
    const params = responsesOf<ResponsesInvocationParamsDelegate>(
      models[0]
    ).invocationParams({});

    expect(params.include).toBeUndefined();
  });
});

function createStreamModel(mockChunks: RawChunk[]): ChatOpenAI {
  const model = new ChatOpenAI({ model: 'gpt-4o-mini', apiKey: 'test-key' });
  (model as unknown as StreamEventsModel).completions.completionWithRetry =
    async (): Promise<AsyncIterable<RawChunk>> => ({
      async *[Symbol.asyncIterator](): AsyncGenerator<RawChunk> {
        for (const chunk of mockChunks) {
          yield chunk;
        }
      },
    });
  return model;
}

function streamEvents(
  model: ChatOpenAI,
  options: BaseChatModelCallOptions = {} as BaseChatModelCallOptions
): AsyncGenerator<ChatModelStreamEvent> {
  return (model as unknown as StreamEventsModel)._streamChatModelEvents(
    [],
    options
  );
}

async function collectEvents(
  model: ChatOpenAI,
  options?: BaseChatModelCallOptions
): Promise<ChatModelStreamEvent[]> {
  const events: ChatModelStreamEvent[] = [];
  for await (const event of streamEvents(model, options)) {
    events.push(event);
  }
  return events;
}

function textOnlyChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-abc',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant', content: 'Hello' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-abc',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { content: ' world' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-abc',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: 'fp_abc',
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    },
  ] as unknown as RawChunk[];
}

function reasoningPlusTextChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-reason',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-5.4',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            content: '',
            reasoning_content: 'Let me',
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-reason',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-5.4',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { reasoning_content: ' reason...' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-reason',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-5.4',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { content: 'The answer is 42.' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-reason',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-5.4',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    },
  ] as unknown as RawChunk[];
}

function toolCallChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-tools',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant', content: 'Let me search.' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-tools',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: 'call_abc',
                type: 'function',
                function: { name: 'web_search', arguments: '{"query"' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-tools',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                function: { arguments: ':"weather"}' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-tools',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'tool_calls',
          logprobs: null,
        },
      ],
    },
  ] as unknown as RawChunk[];
}

function invalidToolCallChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-bad',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            tool_calls: [
              {
                index: 0,
                id: 'call_bad',
                type: 'function',
                function: { name: 'broken', arguments: 'not-json' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-bad',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'tool_calls',
          logprobs: null,
        },
      ],
    },
  ] as unknown as RawChunk[];
}

function usageChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-usage',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: { role: 'assistant', content: 'Cached response' },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-usage',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-usage',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [],
      usage: {
        prompt_tokens: 100,
        completion_tokens: 3,
        total_tokens: 103,
        prompt_tokens_details: { cached_tokens: 50, audio_tokens: null },
        completion_tokens_details: {
          reasoning_tokens: 2,
          audio_tokens: null,
        },
      },
    },
  ] as unknown as RawChunk[];
}

function parallelToolCallChunks(): RawChunk[] {
  return [
    {
      id: 'chatcmpl-parallel',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            tool_calls: [
              {
                index: 0,
                id: 'call_1',
                type: 'function',
                function: { name: 'tool_a', arguments: '{}' },
              },
              {
                index: 1,
                id: 'call_2',
                type: 'function',
                function: { name: 'tool_b', arguments: '{}' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    },
    {
      id: 'chatcmpl-parallel',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'gpt-4o-mini',
      service_tier: null,
      system_fingerprint: null,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'tool_calls',
          logprobs: null,
        },
      ],
    },
  ] as unknown as RawChunk[];
}

describe('convertCompletionsMessageToBaseMessage (fork delegate)', () => {
  it('preserves assistant reasoning_content in additional_kwargs', () => {
    const mockMessage = {
      role: 'assistant' as const,
      content: '2',
      reasoning_content: 'The user asked 1+1.',
    };
    const mockRawResponse = {
      id: 'chatcmpl-reasoning',
      model: 'gpt-5.4',
      choices: [{ index: 0, message: mockMessage }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };

    const result = completionsDelegateOf(
      newOpenAI()
    )._convertCompletionsMessageToBaseMessage(
      mockMessage,
      mockRawResponse
    ) as AIMessage;

    expect(result.additional_kwargs.reasoning_content).toBe(
      'The user asked 1+1.'
    );
  });

  it('preserves delta reasoning_content in streaming chunks', () => {
    const delta = {
      role: 'assistant' as const,
      content: '',
      reasoning_content: 'The user',
    };
    const rawResponse = {
      id: 'chatcmpl-reasoning-stream',
      choices: [{ index: 0, delta, finish_reason: null }],
      usage: { total_tokens: 0, total_characters: 0 },
    };

    const result = completionsDelegateOf(
      newOpenAI()
    )._convertCompletionsDeltaToBaseMessageChunk(
      delta,
      rawResponse
    ) as AIMessageChunk;

    expect(result.additional_kwargs.reasoning_content).toBe('The user');
  });

  describe('OpenRouter image response handling', () => {
    it('Should correctly parse OpenRouter-style image responses', () => {
      const mockMessage = {
        role: 'assistant' as const,
        content: 'Here is your image of a cute cat:',
      };
      const mockRawResponse = {
        id: 'chatcmpl-12345',
        object: 'chat.completion',
        created: 1234567890,
        model: 'google/gemini-2.5-flash-image-preview',
        choices: [
          {
            index: 0,
            message: {
              ...mockMessage,
              images: [
                {
                  type: 'image_url',
                  image_url: {
                    url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
                  },
                },
              ],
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      const result = completionsDelegateOf(
        newOpenAI()
      )._convertCompletionsMessageToBaseMessage(mockMessage, mockRawResponse);

      expect(result.constructor.name).toBe('AIMessage');
      expect(result.content).toEqual([
        { type: 'text', text: 'Here is your image of a cute cat:' },
        {
          type: 'image',
          url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
        },
      ]);
    });

    it('Should handle OpenRouter responses with multiple images', () => {
      const mockMessage = {
        role: 'assistant' as const,
        content: 'Here are multiple images:',
      };
      const mockRawResponse = {
        id: 'chatcmpl-12345',
        object: 'chat.completion',
        created: 1234567890,
        model: 'google/gemini-2.5-flash-image-preview',
        choices: [
          {
            index: 0,
            message: {
              ...mockMessage,
              images: [
                {
                  type: 'image_url',
                  image_url: { url: 'data:image/png;base64,image1' },
                },
                {
                  type: 'image_url',
                  image_url: { url: 'data:image/png;base64,image2' },
                },
              ],
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };

      const result = completionsDelegateOf(
        newOpenAI()
      )._convertCompletionsMessageToBaseMessage(mockMessage, mockRawResponse);

      expect(result.content).toEqual([
        { type: 'text', text: 'Here are multiple images:' },
        { type: 'image', url: 'data:image/png;base64,image1' },
        { type: 'image', url: 'data:image/png;base64,image2' },
      ]);
    });
  });
});

// Dropped (inherited): the `convertStandardContentBlockToCompletionsContentPart`
// describe block tested an upstream-only standalone function (input shape
// { type, data, mimeType }) that our fork does not expose. Our fork's equivalent is the
// module-private `completionsApiContentBlockConverter` (StandardContentBlockConverter,
// source_type/mime_type shape), exercised below through _convertMessagesToOpenAIParams.

describe('completionsApiContentBlockConverter via _convertMessagesToOpenAIParams', () => {
  const partsOf = (block: Record<string, unknown>): unknown[] => {
    const [param] = _convertMessagesToOpenAIParams([
      new HumanMessage({
        content: [block as never],
      }),
    ]);
    return param.content as unknown[];
  };

  it('converts image block with base64 data to image_url data URL', () => {
    expect(
      partsOf({
        type: 'image',
        source_type: 'base64',
        mime_type: 'image/png',
        data: 'iVBORw0KGgoAAAANSUhEUgAAAAE',
      })
    ).toEqual([
      {
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE' },
      },
    ]);
  });

  it('converts image block with url to image_url', () => {
    expect(
      partsOf({
        type: 'image',
        source_type: 'url',
        url: 'https://example.com/cat.png',
      })
    ).toEqual([
      { type: 'image_url', image_url: { url: 'https://example.com/cat.png' } },
    ]);
  });

  it('omits filename for a base64 file block missing filename metadata', () => {
    // FORK DIFFERS: upstream substitutes a "LC_AUTOGENERATED" placeholder filename
    // and console.warns. Our fork has no such placeholder/warning and simply omits
    // the filename key when no filename/name/title metadata is present.
    expect(
      partsOf({
        type: 'file',
        source_type: 'base64',
        mime_type: 'application/pdf',
        data: 'iVBORw0KGgoAAAANSUhEUgAAAAE',
      })
    ).toEqual([
      {
        type: 'file',
        file: {
          file_data: 'data:application/pdf;base64,iVBORw0KGgoAAAANSUhEUgAAAAE',
        },
      },
    ]);
  });

  it('converts a base64 file block to an openai file payload when a filename is provided', () => {
    expect(
      partsOf({
        type: 'file',
        source_type: 'base64',
        mime_type: 'application/pdf',
        data: 'iVBORw0KGgoAAAANSUhEUgAAAAE',
        metadata: { filename: 'cat.pdf' },
      })
    ).toEqual([
      {
        type: 'file',
        file: {
          file_data: 'data:application/pdf;base64,iVBORw0KGgoAAAANSUhEUgAAAAE',
          filename: 'cat.pdf',
        },
      },
    ]);
  });

  it('converts a url data-url file block with filename from metadata.name', () => {
    const dataUrl = 'data:application/pdf;base64,AAABBB';
    expect(
      partsOf({
        type: 'file',
        source_type: 'url',
        url: dataUrl,
        metadata: { name: 'report.pdf' },
      })
    ).toEqual([
      { type: 'file', file: { file_data: dataUrl, filename: 'report.pdf' } },
    ]);
  });

  it('returns file_id for id source_type', () => {
    expect(
      partsOf({
        type: 'file',
        source_type: 'id',
        id: 'file_123',
      })
    ).toEqual([{ type: 'file', file: { file_id: 'file_123' } }]);
  });

  it('throws when a file url is not a data URL', () => {
    expect(() =>
      partsOf({
        type: 'file',
        source_type: 'url',
        url: 'https://example.com/file.pdf',
        metadata: { filename: 'file.pdf' },
      })
    ).toThrow(
      'URL file blocks with source_type url must be formatted as a data URL for ChatOpenAI'
    );
  });
});

describe('convertMessagesToCompletionsMessageParams (_convertMessagesToOpenAIParams)', () => {
  it('wipes string content to empty string when tool_calls are present', () => {
    // FORK DIFFERS: upstream@1.5.3 preserves string content alongside tool_calls.
    // Our fork sets content to '' for AI messages carrying tool_calls (unless an
    // Anthropic thinking block is present).
    const message = new AIMessage({
      content:
        "I'll check the status of item 730 for identifier X1110 to find out why it's not active.",
      tool_calls: [
        {
          id: 'call_zGKlzVl2Ee3Lyob4AsyqfGXb',
          name: 'getStatus',
          args: { identifier: 'X1110', itemId: '730' },
        },
      ],
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call_zGKlzVl2Ee3Lyob4AsyqfGXb',
          type: 'function',
          function: {
            name: 'getStatus',
            arguments: '{"identifier":"X1110","itemId":"730"}',
          },
        },
      ],
    });
  });

  it('handles AIMessage with empty content and tool_calls', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'call_123', name: 'someFunction', args: { key: 'value' } },
      ],
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call_123',
          type: 'function',
          function: { name: 'someFunction', arguments: '{"key":"value"}' },
        },
      ],
    });
  });

  it('emits empty-string content for output_version v1 assistant messages', () => {
    // FORK DIFFERS: upstream@1.5.3 emits content: [] for v1 array content with tool_calls.
    // Our fork emits content: '' (the tool_call content array is wiped, not normalized to []).
    const message = new AIMessage({
      content: [
        {
          type: 'tool_call',
          id: 'call_123',
          name: 'someFunction',
          args: { key: 'value' },
        } as never,
      ],
      tool_calls: [
        { id: 'call_123', name: 'someFunction', args: { key: 'value' } },
      ],
      response_metadata: { output_version: 'v1' },
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call_123',
          type: 'function',
          function: { name: 'someFunction', arguments: '{"key":"value"}' },
        },
      ],
    });
  });

  it('wipes content to empty string when function_call is in additional_kwargs', () => {
    // FORK DIFFERS: upstream@1.5.3 preserves the string content next to function_call.
    // Our fork clears content to '' whenever additional_kwargs.function_call is present.
    const message = new AIMessage({
      content: 'Let me call a function for you.',
      additional_kwargs: {
        function_call: { name: 'myFunction', arguments: '{"arg":"value"}' },
      },
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: 'assistant',
      content: '',
      function_call: { name: 'myFunction', arguments: '{"arg":"value"}' },
    });
  });
});

describe('Anthropic cross-provider compatibility', () => {
  it('wipes content (incl. tool_use/text) to empty string when tool_calls present', () => {
    // FORK DIFFERS: upstream@1.5.3 drops the tool_use block but keeps the text block,
    // yielding content: [{ type: 'text', ... }]. Our fork wipes the whole array to ''
    // because there is no Anthropic thinking block to trigger pass-through.
    const message = new AIMessage({
      content: [
        { type: 'text', text: 'I will search for that.' },
        {
          type: 'tool_use',
          id: 'toolu_abc123',
          name: 'get_weather',
          input: { location: 'SF' },
        } as never,
      ],
      tool_calls: [
        { id: 'toolu_abc123', name: 'get_weather', args: { location: 'SF' } },
      ],
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'toolu_abc123',
          type: 'function',
          function: { name: 'get_weather', arguments: '{"location":"SF"}' },
        },
      ],
    });
  });

  it('passes the full content array through when a thinking block is present', () => {
    // FORK DIFFERS: upstream@1.5.3 drops tool_use while keeping thinking + text.
    // Our fork passes the entire content array through unchanged (including the
    // tool_use block) once any Anthropic thinking block is detected.
    const message = new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking: 'I need to consider...',
          signature: 'sig123',
        } as never,
        { type: 'text', text: 'Here is my answer.' },
        {
          type: 'tool_use',
          id: 'toolu_1',
          name: 'search',
          input: { q: 'langchain' },
        } as never,
      ],
      tool_calls: [{ id: 'toolu_1', name: 'search', args: { q: 'langchain' } }],
    });

    const result = _convertMessagesToOpenAIParams([message]);

    expect(result).toHaveLength(1);
    const contentArr = result[0].content as Array<{ type: string }>;
    expect(contentArr.some((c) => c.type === 'thinking')).toBe(true);
    expect(contentArr.some((c) => c.type === 'text')).toBe(true);
    // FORK DIFFERS: tool_use is NOT dropped under thinking-block pass-through.
    expect(contentArr.some((c) => c.type === 'tool_use')).toBe(true);
    expect((result[0] as { tool_calls?: unknown[] }).tool_calls).toHaveLength(
      1
    );
  });
});

describe('ChatOpenAICompletions constructor', () => {
  // Adapted: the string-model shorthand `new ChatOpenAICompletions("model", {...})`
  // is a ChatOpenAICompletions-only constructor overload; our ChatOpenAI wrapper
  // only accepts the object form, so we assert the same `model`/`temperature`
  // outcome through it.
  it('applies model and temperature from constructor fields', () => {
    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0.1,
    });
    expect(model.model).toBe('gpt-4o-mini');
    expect(model.temperature).toBe(0.1);
  });
});

describe('ChatOpenAICompletions streaming usage_metadata callback', () => {
  it('should call handleLLMNewToken for the usage chunk', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-4o-mini',
      apiKey: 'test-key',
      streaming: true,
      streamUsage: true,
    });
    const completions = completionsOf<CompletionsStreamDelegate>(model);

    // Mock completionWithRetry to return a fake async iterable
    // that simulates: one content chunk, then a usage-only chunk
    const fakeStream = (async function* () {
      // Content chunk
      yield {
        choices: [
          {
            index: 0,
            delta: { role: 'assistant' as const, content: 'Hello' },
            finish_reason: null,
            logprobs: null,
          },
        ],
        usage: null,
        system_fingerprint: null,
        model: 'gpt-4o-mini',
        service_tier: null,
      };
      // Final chunk with finish_reason
      yield {
        choices: [
          {
            index: 0,
            delta: { content: '' },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage: null,
        system_fingerprint: 'fp_abc123',
        model: 'gpt-4o-mini',
        service_tier: null,
      };
      // Usage-only chunk (no choices)
      yield {
        choices: [],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
          prompt_tokens_details: null,
          completion_tokens_details: null,
        },
        system_fingerprint: null,
        model: 'gpt-4o-mini',
        service_tier: null,
      };
    })();

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<unknown>
    > => fakeStream;

    // Create a mock runManager
    const handleLLMNewToken = jest.fn();
    const runManager = {
      handleLLMNewToken,
    } as unknown as CallbackManagerForLLMRun;

    const chunks: ChatGenerationChunk[] = [];
    for await (const chunk of completions._streamResponseChunks(
      [new HumanMessage('test')],
      {},
      runManager
    )) {
      chunks.push(chunk);
    }

    // Should have 3 chunks: content, finish, and usage
    expect(chunks.length).toBe(3);

    // The last chunk should have usage_metadata
    const usageChunk = chunks[chunks.length - 1];
    const usageMessage = usageChunk.message as AIMessageChunk;
    expect(usageMessage.usage_metadata).toBeDefined();
    expect(usageMessage.usage_metadata?.input_tokens).toBe(10);
    expect(usageMessage.usage_metadata?.output_tokens).toBe(5);
    expect(usageMessage.usage_metadata?.total_tokens).toBe(15);

    // handleLLMNewToken should have been called for EVERY chunk,
    // including the usage chunk (this is the bug fix)
    expect(handleLLMNewToken).toHaveBeenCalledTimes(3);

    // Verify the last call includes the usage chunk
    const lastCall = handleLLMNewToken.mock.calls[2];
    const lastCallFields = lastCall[5] as {
      chunk: { message: AIMessageChunk };
    };
    expect(lastCallFields.chunk.message.usage_metadata).toBeDefined();
    expect(lastCallFields.chunk.message.usage_metadata?.input_tokens).toBe(10);
  });
});

describe('ChatOpenAICompletions reasoning_content compatibility', () => {
  it('should preserve reasoning_content on streamed assistant chunks', async () => {
    const model = new ChatOpenAI({
      model: 'gpt-5.4',
      apiKey: 'test-key',
      streaming: true,
    });
    const completions = completionsOf<CompletionsStreamDelegate>(model);

    const fakeStream = (async function* () {
      yield {
        choices: [
          {
            index: 0,
            delta: {
              role: 'assistant' as const,
              content: '',
              reasoning_content: 'The user',
            },
            finish_reason: null,
            logprobs: null,
          },
        ],
        usage: null,
        system_fingerprint: null,
        model: 'gpt-5.4',
        service_tier: null,
      };
      yield {
        choices: [
          {
            index: 0,
            delta: { content: '' },
            finish_reason: 'stop',
            logprobs: null,
          },
        ],
        usage: null,
        system_fingerprint: null,
        model: 'gpt-5.4',
        service_tier: null,
      };
    })();

    completions.completionWithRetry = async (): Promise<
      AsyncIterable<unknown>
    > => fakeStream;

    const chunks: ChatGenerationChunk[] = [];
    for await (const chunk of completions._streamResponseChunks(
      [new HumanMessage('1+1=?')],
      {}
    )) {
      chunks.push(chunk);
    }

    const firstChunk = chunks[0].message as AIMessageChunk;
    expect(firstChunk.additional_kwargs.reasoning_content).toBe('The user');
  });
});

describe('ChatOpenAICompletions strict tools for structured output', () => {
  const weatherTool = {
    type: 'function' as const,
    function: {
      name: 'get_current_weather',
      description: 'Get the current weather in a location',
      parameters: toJsonSchema(z.object({ location: z.string() })),
    },
  };
  const jsonSchemaResponseFormat = {
    type: 'json_schema' as const,
    json_schema: {
      name: 'answer',
      schema: toJsonSchema(z.object({ answer: z.string() })),
    },
  };

  /** Return the per-tool `strict` flag invocationParams produces for `options`. */
  function toolStrict(
    options: Record<string, unknown>,
    extra?: { streaming?: boolean }
  ): boolean | undefined {
    const model = new ChatOpenAI({
      model: 'gpt-4',
      apiKey: 'test-key',
    });
    const completions = completionsOf<InvocationParamsDelegate>(model);
    const params = completions.invocationParams(
      { tools: [weatherTool], ...options },
      extra
    );
    return params.tools?.[0]?.function?.strict;
  }

  it('defaults strict to true when a json_schema response_format is requested', () => {
    expect(toolStrict({ response_format: jsonSchemaResponseFormat })).toBe(
      true
    );
  });

  it('respects an explicit strict:false even with a json_schema response_format', () => {
    expect(
      toolStrict({ response_format: jsonSchemaResponseFormat, strict: false })
    ).toBe(false);
  });

  it('does not set strict when no response_format is requested', () => {
    expect(toolStrict({})).toBeUndefined();
  });

  it('does not set strict for a streaming json_schema request (create() path)', () => {
    // Streaming goes through create(), not .parse(), so strict isn't required.
    expect(
      toolStrict(
        { response_format: jsonSchemaResponseFormat },
        { streaming: true }
      )
    ).toBeUndefined();
  });

  it('does not set strict for a json_object response_format (JSON mode)', () => {
    expect(
      toolStrict({ response_format: { type: 'json_object' } })
    ).toBeUndefined();
  });
});

describe('ChatOpenAI._streamChatModelEvents (native, fork)', () => {
  describe('text-only streaming', () => {
    test('emits correct lifecycle events', async () => {
      const model = createStreamModel(textOnlyChunks());
      const eventNames = (await collectEvents(model)).map((e) => e.event);

      expect(eventNames).toContain('message-start');
      expect(eventNames).toContain('content-block-start');
      expect(eventNames).toContain('content-block-delta');
      expect(eventNames).toContain('content-block-finish');
      expect(eventNames).toContain('message-finish');
    });

    test('message-start carries id', async () => {
      const model = createStreamModel(textOnlyChunks());
      const events = await collectEvents(model);

      const start = events.find((e) => e.event === 'message-start');
      expect(start).toBeDefined();
      expect((start as { id?: string }).id).toBe('chatcmpl-abc');
    });

    test('text deltas are incremental', async () => {
      const model = createStreamModel(textOnlyChunks());
      const events = await collectEvents(model);

      const textDeltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' &&
          'delta' in e &&
          (e.delta as { type: string }).type === 'text-delta'
      );
      expect(textDeltas.length).toBe(2);
      expect((textDeltas[0] as { delta: { text: string } }).delta.text).toBe(
        'Hello'
      );
      expect((textDeltas[1] as { delta: { text: string } }).delta.text).toBe(
        ' world'
      );
    });

    test('content-block-finish carries finalized text', async () => {
      const model = createStreamModel(textOnlyChunks());
      const events = await collectEvents(model);

      expect(
        events.find((e) => e.event === 'content-block-finish')
      ).toMatchObject({
        content: { type: 'text', text: 'Hello world' },
      });
    });

    test('message-finish carries stop reason', async () => {
      const model = createStreamModel(textOnlyChunks());
      const events = await collectEvents(model);

      const finish = events.find((e) => e.event === 'message-finish') as {
        reason: string;
      };
      expect(finish.reason).toBe('stop');
    });
  });

  describe('reasoning + text streaming', () => {
    test('reasoning block accumulates correctly', async () => {
      const model = createStreamModel(reasoningPlusTextChunks());
      const events = await collectEvents(model);

      const reasoningDeltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' &&
          'delta' in e &&
          (e.delta as { type: string }).type === 'reasoning-delta'
      );
      expect(reasoningDeltas.length).toBe(2);
      expect(
        (reasoningDeltas[0] as { delta: { reasoning: string } }).delta.reasoning
      ).toBe('Let me');
      expect(
        (reasoningDeltas[1] as { delta: { reasoning: string } }).delta.reasoning
      ).toBe(' reason...');

      expect(
        events.find(
          (e) =>
            e.event === 'content-block-finish' && e.content.type === 'reasoning'
        )
      ).toMatchObject({
        content: { reasoning: 'Let me reason...' },
      });
    });

    test('text block uses separate index from reasoning', async () => {
      const model = createStreamModel(reasoningPlusTextChunks());
      const events = await collectEvents(model);

      expect(
        events.find(
          (e) => e.event === 'content-block-finish' && e.content.type === 'text'
        )
      ).toMatchObject({
        content: { text: 'The answer is 42.' },
      });
    });
  });

  describe('tool call streaming', () => {
    test('tool call args accumulate correctly', async () => {
      const model = createStreamModel(toolCallChunks());
      const events = await collectEvents(model);

      const toolDeltas = events.filter(
        (e) =>
          e.event === 'content-block-delta' &&
          'delta' in e &&
          (e.delta as { type: string }).type === 'block-delta'
      );
      const toolArgDeltas = toolDeltas.filter(
        (e) =>
          (e as unknown as { delta: { fields?: { args?: string } } }).delta
            .fields?.args != null
      );
      expect(toolArgDeltas.length).toBe(2);
      expect(
        (toolArgDeltas[0] as unknown as { delta: { fields: { args: string } } })
          .delta.fields.args
      ).toBe('{"query"');
      expect(
        (toolArgDeltas[1] as unknown as { delta: { fields: { args: string } } })
          .delta.fields.args
      ).toBe('{"query":"weather"}');
    });

    test('tool call finish has parsed args', async () => {
      const model = createStreamModel(toolCallChunks());
      const events = await collectEvents(model);

      expect(
        events.find(
          (e) =>
            e.event === 'content-block-finish' && e.content.type === 'tool_call'
        )
      ).toMatchObject({
        content: {
          name: 'web_search',
          id: 'call_abc',
          args: { query: 'weather' },
        },
      });
    });

    test('invalid tool call JSON becomes invalid_tool_call', async () => {
      const model = createStreamModel(invalidToolCallChunks());
      const events = await collectEvents(model);

      expect(
        events.find(
          (e) =>
            e.event === 'content-block-finish' &&
            e.content.type === 'invalid_tool_call'
        )
      ).toMatchObject({
        content: {
          name: 'broken',
          error: expect.stringContaining('JSON'),
        },
      });
    });

    test('message-finish has tool_use reason', async () => {
      const model = createStreamModel(toolCallChunks());
      const events = await collectEvents(model);

      const finish = events.find((e) => e.event === 'message-finish') as {
        reason: string;
      };
      expect(finish.reason).toBe('tool_use');
    });

    test('parallel tool calls get distinct block indices', async () => {
      const model = createStreamModel(parallelToolCallChunks());
      const events = await collectEvents(model);

      const toolStarts = events.filter(
        (e) =>
          e.event === 'content-block-start' &&
          'content' in e &&
          (e.content as { type: string }).type === 'tool_call_chunk'
      );
      expect(toolStarts.length).toBe(2);
      const indices = toolStarts.map((e) => (e as { index: number }).index);
      expect(new Set(indices).size).toBe(2);
    });
  });

  describe('usage streaming', () => {
    test('usage snapshot with cache details', async () => {
      const model = createStreamModel(usageChunks());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const usageEvents = events.filter((e) => e.event === 'usage');
      expect(usageEvents.length).toBe(1);

      const usage = (
        usageEvents[0] as {
          usage: {
            input_tokens: number;
            input_token_details: { cache_read: number };
            output_token_details: { reasoning: number };
          };
        }
      ).usage;
      expect(usage.input_tokens).toBe(100);
      expect(usage.input_token_details.cache_read).toBe(50);
      expect(usage.output_token_details.reasoning).toBe(2);
    });

    test('message-finish carries final usage', async () => {
      const model = createStreamModel(usageChunks());
      const events = await collectEvents(model, {
        streamUsage: true,
      } as BaseChatModelCallOptions);

      const finish = events.find((e) => e.event === 'message-finish') as {
        usage: { total_tokens: number };
      };
      expect(finish.usage.total_tokens).toBe(103);
    });

    test('no usage events when streamUsage is false', async () => {
      const model = createStreamModel(usageChunks());
      // Upstream extends the completions class, so `model.streamUsage` is the
      // completions delegate's field. Our fork's top-level model delegates to
      // its `completions` sub-model, whose `streamUsage` is what gates emission.
      (model as unknown as StreamEventsModel).completions.streamUsage = false;
      const events = await collectEvents(model, {
        streamUsage: false,
      } as BaseChatModelCallOptions);

      expect(events.filter((e) => e.event === 'usage').length).toBe(0);
      const finish = events.find((e) => e.event === 'message-finish') as {
        usage?: unknown;
      };
      expect(finish.usage).toBeUndefined();
    });
  });

  describe('provider passthrough', () => {
    test('stream metadata is forwarded as provider event', async () => {
      const model = createStreamModel(textOnlyChunks());
      const events = await collectEvents(model);

      const meta = events.find(
        (e) =>
          e.event === 'provider' &&
          (e as { name: string }).name === 'stream_metadata'
      ) as { provider: string; payload: { model: string } };
      expect(meta.provider).toBe('openai');
      expect(meta.payload.model).toBe('gpt-4o-mini');
    });
  });

  describe('integration with ChatModelStream', () => {
    test('text sub-stream works end-to-end', async () => {
      const model = createStreamModel(textOnlyChunks());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.text).toBe('Hello world');
    });

    test('toolCalls sub-stream works end-to-end', async () => {
      const model = createStreamModel(toolCallChunks());
      const stream = new ChatModelStream(streamEvents(model));
      const calls = await stream.toolCalls;
      expect(calls.length).toBe(1);
      expect(calls[0]!.name).toBe('web_search');
      expect(calls[0]!.args).toEqual({ query: 'weather' });
    });

    test('reasoning sub-stream works end-to-end', async () => {
      const model = createStreamModel(reasoningPlusTextChunks());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.reasoning).toBe('Let me reason...');
    });

    test('usage sub-stream works end-to-end', async () => {
      const model = createStreamModel(usageChunks());
      const stream = new ChatModelStream(
        streamEvents(model, { streamUsage: true } as BaseChatModelCallOptions)
      );
      expect(await stream.usage).toMatchObject({
        input_tokens: 100,
        output_tokens: 3,
        total_tokens: 103,
      });
    });

    test('output assembles correct AIMessage', async () => {
      const model = createStreamModel(toolCallChunks());
      const stream = new ChatModelStream(streamEvents(model));
      const message = await stream.output;

      expect(message.id).toBe('chatcmpl-tools');
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
    });

    test('await stream returns AIMessage directly', async () => {
      const model = createStreamModel(textOnlyChunks());
      const message = await new ChatModelStream(streamEvents(model));
      expect(message._getType()).toBe('ai');
      expect(message.id).toBe('chatcmpl-abc');
    });
  });

  // Re-expressed from upstream's `streaming events` describe, which used vitest
  // custom matchers (`toHaveStreamText` / `toHaveStreamToolCalls` /
  // `toHaveStreamReasoning`) unavailable in jest. The matchers wrap the
  // `ChatModelStream` sub-streams asserted here directly.
  describe('streaming events (sub-stream assertions)', () => {
    test('streams text', async () => {
      const model = createStreamModel(textOnlyChunks());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.text).toBe('Hello world');
    });

    test('streams tool calls', async () => {
      const model = createStreamModel(toolCallChunks());
      const stream = new ChatModelStream(streamEvents(model));
      const calls = await stream.toolCalls;
      expect(calls).toEqual([
        expect.objectContaining({
          name: 'web_search',
          args: { query: 'weather' },
        }),
      ]);
    });

    test('streams reasoning', async () => {
      const model = createStreamModel(reasoningPlusTextChunks());
      const stream = new ChatModelStream(streamEvents(model));
      expect(await stream.reasoning).toBe('Let me reason...');
    });
  });
});
