import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseMessage } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';

import { ChatDeepSeek } from './index';

type DeepSeekRequest =
  | OpenAIClient.Chat.ChatCompletionCreateParamsStreaming
  | OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming;
type OpenAIChatCompletion = OpenAIClient.Chat.Completions.ChatCompletion;
type OpenAIChatCompletionChunk =
  OpenAIClient.Chat.Completions.ChatCompletionChunk;
type ReasoningAssistantMessageParam =
  OpenAIClient.Chat.Completions.ChatCompletionAssistantMessageParam & {
    reasoning_content?: string;
  };

class CapturingChatDeepSeek extends ChatDeepSeek {
  readonly requests: DeepSeekRequest[] = [];

  constructor(
    fields: ConstructorParameters<typeof ChatDeepSeek>[0],
    private readonly streamChunks = createCompletionStreamChunks(),
    private readonly completion = createCompletion()
  ) {
    super(fields);
  }

  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsStreaming,
    requestOptions?: OpenAIClient.RequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk>>;
  async completionWithRetry(
    request: OpenAIClient.Chat.ChatCompletionCreateParamsNonStreaming,
    requestOptions?: OpenAIClient.RequestOptions
  ): Promise<OpenAIChatCompletion>;
  async completionWithRetry(
    request: DeepSeekRequest,
    _requestOptions?: OpenAIClient.RequestOptions
  ): Promise<AsyncIterable<OpenAIChatCompletionChunk> | OpenAIChatCompletion> {
    this.requests.push(request);

    if (request.stream === true) {
      return createCompletionStream(this.streamChunks);
    }

    return this.completion;
  }

  streamChunksWithSignal(
    signal: AbortSignal
  ): AsyncGenerator<ChatGenerationChunk> {
    return this._streamResponseChunks([new HumanMessage('hi')], {
      signal,
    } as this['ParsedCallOptions']);
  }
}

function createToolContextMessages(): BaseMessage[] {
  return [
    new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'call_1',
          name: 'web_search',
          args: { query: 'trending news today' },
          type: 'tool_call',
        },
      ],
      additional_kwargs: {
        reasoning_content: 'Need current news from the web.',
      },
    }),
    new ToolMessage({
      content: 'Search results',
      tool_call_id: 'call_1',
    }),
  ];
}

function createCompletionStreamChunks(): OpenAIChatCompletionChunk[] {
  return [
    createContentChunk('ok'),
    {
      id: 'chatcmpl-deepseek-test',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'deepseek-v4-pro',
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: 'stop',
          logprobs: null,
        },
      ],
    },
  ];
}

function createContentChunk(content: string): OpenAIChatCompletionChunk {
  return {
    id: 'chatcmpl-deepseek-test',
    object: 'chat.completion.chunk',
    created: 0,
    model: 'deepseek-v4-pro',
    choices: [
      {
        index: 0,
        delta: {
          role: 'assistant',
          content,
        },
        finish_reason: null,
        logprobs: null,
      },
    ],
  };
}

async function* createCompletionStream(
  chunks: OpenAIChatCompletionChunk[]
): AsyncGenerator<OpenAIChatCompletionChunk> {
  for (const chunk of chunks) {
    yield chunk;
  }
}

function createCompletion(
  usage: OpenAIClient.Completions.CompletionUsage = {
    prompt_tokens: 1,
    completion_tokens: 1,
    total_tokens: 2,
  }
): OpenAIChatCompletion {
  return {
    id: 'chatcmpl-deepseek-test',
    object: 'chat.completion',
    created: 0,
    model: 'deepseek-v4-pro',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: 'ok',
          refusal: null,
        },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage,
  };
}

function getReasoningAssistantMessage(
  request: DeepSeekRequest
): ReasoningAssistantMessageParam {
  return request.messages[0] as ReasoningAssistantMessageParam;
}

async function drainStream(stream: AsyncIterable<unknown>): Promise<void> {
  for await (const chunk of stream) {
    void chunk;
  }
}

describe('ChatDeepSeek', () => {
  it('passes reasoning_content back on same-run streaming tool continuations', async () => {
    const model = new CapturingChatDeepSeek({
      apiKey: 'test-key',
      model: 'deepseek-v4-pro',
      streaming: true,
    });
    const chunks = [];

    for await (const chunk of await model.stream(createToolContextMessages())) {
      chunks.push(chunk);
    }

    expect(chunks).toHaveLength(2);
    expect(model.requests).toHaveLength(1);
    expect(getReasoningAssistantMessage(model.requests[0])).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'Need current news from the web.',
      })
    );
  });

  it('passes reasoning_content back on same-run non-streaming tool continuations', async () => {
    const model = new CapturingChatDeepSeek({
      apiKey: 'test-key',
      model: 'deepseek-v4-pro',
      streaming: false,
    });

    await model.invoke(createToolContextMessages());

    expect(model.requests).toHaveLength(1);
    expect(getReasoningAssistantMessage(model.requests[0])).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'Need current news from the web.',
      })
    );
  });

  it('keeps raw think fallback content out of streamed assistant content', async () => {
    const model = new CapturingChatDeepSeek(
      {
        apiKey: 'test-key',
        model: 'deepseek-v4-pro',
        streaming: true,
      },
      [
        createContentChunk('prefix <thi'),
        createContentChunk('nk>hidden'),
        createContentChunk('</think>visible'),
      ]
    );
    const chunks = [];
    const callbackTokens: string[] = [];

    const stream = await model.stream([new HumanMessage('hi')], {
      callbacks: [
        {
          handleLLMNewToken(token: string): void {
            callbackTokens.push(token);
          },
        },
      ],
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const streamedText = chunks
      .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
      .join('');
    const hasHiddenReasoning = chunks.some(
      (chunk) => chunk.additional_kwargs.reasoning_content === 'hidden'
    );

    expect(streamedText).toBe('prefix visible');
    expect(callbackTokens.join('')).toBe('prefix visible');
    expect(callbackTokens.join('')).not.toContain('hidden');
    expect(callbackTokens.join('')).not.toContain('think');
    expect(hasHiddenReasoning).toBe(true);
  });

  it('keeps multiple raw think fallback blocks hidden from content and callbacks', async () => {
    const model = new CapturingChatDeepSeek(
      {
        apiKey: 'test-key',
        model: 'deepseek-v4-pro',
        streaming: true,
      },
      [
        createContentChunk(
          'before<think>hidden one</think>visible<think>hidden two</think>done'
        ),
      ]
    );
    const chunks = [];
    const callbackTokens: string[] = [];

    const stream = await model.stream([new HumanMessage('hi')], {
      callbacks: [
        {
          handleLLMNewToken(token: string): void {
            callbackTokens.push(token);
          },
        },
      ],
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const streamedText = chunks
      .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
      .join('');
    const reasoningContent = chunks
      .map((chunk) => chunk.additional_kwargs.reasoning_content)
      .filter((content): content is string => typeof content === 'string');

    expect(streamedText).toBe('beforevisibledone');
    expect(callbackTokens.join('')).toBe('beforevisibledone');
    expect(reasoningContent).toEqual(['hidden one', 'hidden two']);
  });

  it('emits trailing unfinished raw think fallback as reasoning content', async () => {
    const model = new CapturingChatDeepSeek(
      {
        apiKey: 'test-key',
        model: 'deepseek-v4-pro',
        streaming: true,
      },
      [createContentChunk('<think>truncated')]
    );
    const chunks = [];
    const callbackTokens: string[] = [];

    const stream = await model.stream([new HumanMessage('hi')], {
      callbacks: [
        {
          handleLLMNewToken(token: string): void {
            callbackTokens.push(token);
          },
        },
      ],
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    const streamedText = chunks
      .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
      .join('');
    const reasoningContent = chunks
      .map((chunk) => chunk.additional_kwargs.reasoning_content)
      .filter((content): content is string => typeof content === 'string');

    expect(streamedText).toBe('');
    expect(callbackTokens.join('')).toBe('');
    expect(reasoningContent).toEqual(['truncated']);
  });

  it('preserves detailed usage metadata in non-streaming responses', async () => {
    const model = new CapturingChatDeepSeek(
      {
        apiKey: 'test-key',
        model: 'deepseek-v4-pro',
        streaming: false,
      },
      createCompletionStreamChunks(),
      createCompletion({
        prompt_tokens: 11,
        completion_tokens: 7,
        total_tokens: 18,
        prompt_tokens_details: {
          audio_tokens: 2,
          cached_tokens: 3,
        },
        completion_tokens_details: {
          audio_tokens: 4,
          reasoning_tokens: 5,
        },
      })
    );

    const response = await model.invoke([new HumanMessage('hi')]);

    expect(response.usage_metadata).toEqual({
      input_tokens: 11,
      output_tokens: 7,
      total_tokens: 18,
      input_token_details: {
        audio: 2,
        cache_read: 3,
      },
      output_token_details: {
        audio: 4,
        reasoning: 5,
      },
    });
  });

  it('does not serialize non-streaming requests when aborted before generation', async () => {
    const controller = new AbortController();
    const model = new CapturingChatDeepSeek({
      apiKey: 'test-key',
      model: 'deepseek-v4-pro',
      streaming: false,
    });

    controller.abort();

    await expect(
      model.invoke([new HumanMessage('hi')], {
        signal: controller.signal,
      })
    ).rejects.toThrow();
    expect(model.requests).toHaveLength(0);
  });

  it('throws AbortError when a DeepSeek stream is canceled', async () => {
    const controller = new AbortController();
    const model = new CapturingChatDeepSeek({
      apiKey: 'test-key',
      model: 'deepseek-v4-pro',
      streaming: true,
    });

    controller.abort();

    await expect(
      drainStream(model.streamChunksWithSignal(controller.signal))
    ).rejects.toThrow('AbortError');
  });
});
