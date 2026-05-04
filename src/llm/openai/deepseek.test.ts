import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
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
    private readonly streamChunks = createCompletionStreamChunks()
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

    return createCompletion();
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

function createCompletion(): OpenAIChatCompletion {
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
    usage: {
      prompt_tokens: 1,
      completion_tokens: 1,
      total_tokens: 2,
    },
  };
}

function getReasoningAssistantMessage(
  request: DeepSeekRequest
): ReasoningAssistantMessageParam {
  return request.messages[0] as ReasoningAssistantMessageParam;
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
});
