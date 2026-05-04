import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { OpenAIChatInput, OpenAIClient } from '@langchain/openai';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import type { CustomAnthropicInput } from '@/llm/anthropic';
import type {
  ChatAnthropicToolType,
  AnthropicMCPServerURLDefinition,
} from '@/llm/anthropic/types';
import {
  ChatXAI,
  ChatOpenAI,
  ChatDeepSeek,
  ChatMoonshot,
  AzureChatOpenAI,
  CustomOpenAIClient,
  CustomAzureOpenAIClient,
} from '@/llm/openai';
import { CustomChatGoogleGenerativeAI } from '@/llm/google';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { ChatOpenRouter } from '@/llm/openrouter';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatVertexAI } from '@/llm/vertexai';

type OpenAIRequestOptions = Parameters<ChatOpenAI['_getClientOptions']>[0];
type OpenAIRequestOptionsWithBaseURL = ReturnType<
  ChatXAI['_getClientOptions']
> & {
  baseURL?: string;
};
type OpenAIResponsesDelegate = {
  client?: unknown;
  _getClientOptions: (options?: OpenAIRequestOptions) => OpenAIRequestOptions;
};
type AnthropicCallOptions = Parameters<
  CustomAnthropic['invocationParams']
>[0] & {
  outputConfig?: CustomAnthropicInput['outputConfig'] & {
    task_budget?: { type: 'token_budget'; value: number };
  };
  inferenceGeo?: CustomAnthropicInput['inferenceGeo'];
  betas?: CustomAnthropicInput['betas'];
  container?: string;
  mcp_servers?: AnthropicMCPServerURLDefinition[];
  tools?: ChatAnthropicToolType[];
};
type AzureReasoningModel = AzureChatOpenAI & {
  reasoning?: { effort: 'low' | 'high' };
};
type OpenRouterFields = Partial<
  ChatOpenRouterCallOptions & Pick<OpenAIChatInput, 'model' | 'apiKey'>
>;
type CompletionDelegate = {
  completionWithRetry: (request: { messages?: unknown }) => Promise<unknown>;
};
type CompletionBackedModel = {
  completions: CompletionDelegate;
};
type StreamingCompletionRequest = {
  messages?: unknown;
  stream?: boolean;
};
type StreamingCompletionDelegate = {
  completionWithRetry: (
    request: StreamingCompletionRequest
  ) => Promise<
    AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
  >;
};
type StreamingCompletionBackedModel = {
  completions: StreamingCompletionDelegate;
};
type OpenAIStreamEvent = {
  event: string;
  data?: unknown;
};
type OpenAIStreamItem =
  | OpenAIClient.Chat.Completions.ChatCompletionChunk
  | OpenAIStreamEvent;
type MockableCompletionCreate = (
  request: unknown,
  options?: unknown
) => Promise<
  AsyncIterable<OpenAIStreamItem> | OpenAIClient.Chat.Completions.ChatCompletion
>;
type MockableCompletionClient = {
  chat: {
    completions: {
      create: MockableCompletionCreate;
    };
  };
};
type MockableCompletionDelegate = OpenAIResponsesDelegate & {
  client?: MockableCompletionClient;
};
type OpenRouterReasoningStreamDelta =
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice.Delta & {
    reasoning_details?: Array<
      | {
          type: 'reasoning.text';
          text?: string;
          format?: string;
          index?: number;
        }
      | {
          type: 'reasoning.encrypted';
          id?: string;
          data?: string;
          format?: string;
          index?: number;
        }
    >;
  };
type OpenRouterReasoningStreamChoice = Omit<
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice,
  'delta'
> & {
  delta: OpenRouterReasoningStreamDelta;
};
type OpenAIStreamModel = ChatOpenAI | AzureChatOpenAI;

const baseAzureFields = {
  azureOpenAIApiKey: 'test-azure-key',
  azureOpenAIApiVersion: '2024-10-21',
  azureOpenAIApiInstanceName: 'test-instance',
  azureOpenAIApiDeploymentName: 'test-deployment',
};

const baseBedrockFields = {
  region: 'us-east-1',
  credentials: {
    accessKeyId: 'test-access-key',
    secretAccessKey: 'test-secret-key',
  },
};

const createOpenAIStreamChunk = (
  content: string,
  finishReason: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['finish_reason'] = null
): OpenAIClient.Chat.Completions.ChatCompletionChunk => ({
  id: 'chatcmpl-hermes-test',
  object: 'chat.completion.chunk',
  created: 0,
  model: 'hermes-agent',
  choices: [
    {
      index: 0,
      delta: { content },
      finish_reason: finishReason,
    },
  ],
});

async function* createOpenAIStreamWithCustomEvents(): AsyncGenerator<OpenAIStreamItem> {
  yield createOpenAIStreamChunk('Hello ');
  yield {
    event: 'hermes.tool.progress',
    data: {
      tool: 'execute_code',
      toolCallId: 'call_1',
      status: 'running',
    },
  };
  yield {
    event: 'hermes.tool.progress',
    data: null,
  };
  yield {
    event: 'message',
    data: createOpenAIStreamChunk('world', 'stop'),
  };
}

function mockCompletionStream(
  model: OpenAIStreamModel
): MockableCompletionCreate {
  const completions = (
    model as unknown as { completions: MockableCompletionDelegate }
  ).completions;
  completions._getClientOptions(undefined);
  const client = completions.client;
  if (client == null) {
    throw new Error('Expected OpenAI completions client');
  }

  const createMock = jest.fn(async () =>
    createOpenAIStreamWithCustomEvents()
  ) as MockableCompletionCreate;
  client.chat.completions.create = createMock;
  return createMock;
}

function mockCompletion(
  model: ChatOpenAI,
  response: OpenAIClient.Chat.Completions.ChatCompletion
): MockableCompletionCreate {
  const completions = (
    model as unknown as { completions: MockableCompletionDelegate }
  ).completions;
  completions._getClientOptions(undefined);
  const client = completions.client;
  if (client == null) {
    throw new Error('Expected OpenAI completions client');
  }

  const createMock = jest.fn(async () => response) as MockableCompletionCreate;
  client.chat.completions.create = createMock;
  return createMock;
}

async function expectCustomSSEEventsSkipped(
  model: OpenAIStreamModel
): Promise<void> {
  const createMock = mockCompletionStream(model);
  const chunks: AIMessageChunk[] = [];
  const stream = await model.stream([new HumanMessage('use a tool')]);
  for await (const chunk of stream) {
    chunks.push(chunk);
  }

  const text = chunks
    .map((chunk) => (typeof chunk.content === 'string' ? chunk.content : ''))
    .join('');
  expect(chunks).toHaveLength(2);
  expect(text).toBe('Hello world');
  expect(createMock).toHaveBeenCalledWith(
    expect.objectContaining({ stream: true }),
    expect.any(Object)
  );
}

describe('custom chat model class smoke tests', () => {
  it('keeps the custom OpenAI client, stream delay, and reasoning precedence', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      reasoning: { effort: 'low' },
      _lc_stream_delay: 3,
    });

    const requestOptions = model._getClientOptions({
      headers: { 'x-smoke': 'openai' },
    } as OpenAIRequestOptions);

    expect(ChatOpenAI.lc_name()).toBe('LibreChatOpenAI');
    expect(model._lc_stream_delay).toBe(3);
    expect(model.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(requestOptions.headers).toEqual(
      expect.objectContaining({ 'x-smoke': 'openai' })
    );
    expect(model.getReasoningParams({ reasoning: { effort: 'high' } })).toEqual(
      { effort: 'high' }
    );
    const params = model.invocationParams({ reasoningEffort: 'medium' }) as {
      reasoning?: unknown;
      reasoning_effort?: unknown;
    };
    expect(params.reasoning ?? { effort: params.reasoning_effort }).toEqual({
      effort: 'low',
    });
  });

  it('keeps the custom OpenAI client on the Responses API path', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5',
      apiKey: 'test-key',
      useResponsesApi: true,
      maxTokens: 32,
      reasoning: { effort: 'low' },
    });

    const params = model.invocationParams({
      reasoningEffort: 'high',
    }) as {
      max_output_tokens?: number;
      max_completion_tokens?: number;
      reasoning?: unknown;
    };
    const responses = (
      model as unknown as { responses: OpenAIResponsesDelegate }
    ).responses;
    const requestOptions = responses._getClientOptions({
      headers: { 'x-smoke': 'responses' },
    } as OpenAIRequestOptions);

    expect(params.max_output_tokens).toBe(32);
    expect(params.max_completion_tokens).toBeUndefined();
    expect(params.reasoning).toEqual({ effort: 'low' });
    expect(responses.client).toBeInstanceOf(CustomOpenAIClient);
    const responsesClient = responses.client as CustomOpenAIClient;
    responsesClient.abortHandler = (): void => undefined;
    expect(model.exposedClient).toBe(responsesClient);
    expect(requestOptions?.headers).toEqual(
      expect.objectContaining({ 'x-smoke': 'responses' })
    );
  });

  it('keeps Azure client customization and gates reasoning to reasoning models', () => {
    const model = new AzureChatOpenAI({
      ...baseAzureFields,
      _lc_stream_delay: 4,
    }) as AzureReasoningModel;
    model.model = 'gpt-5';
    model.reasoning = { effort: 'low' };

    const requestOptions = model._getClientOptions({
      headers: { 'x-smoke': 'azure' },
    });

    expect(AzureChatOpenAI.lc_name()).toBe('LibreChatAzureOpenAI');
    expect(model._lc_stream_delay).toBe(4);
    expect(model.exposedClient).toBeInstanceOf(CustomAzureOpenAIClient);
    const azureResponses = (
      model as unknown as { responses: OpenAIResponsesDelegate }
    ).responses;
    azureResponses._getClientOptions(undefined);
    expect(azureResponses.client).toBeInstanceOf(CustomAzureOpenAIClient);
    const azureResponsesClient =
      azureResponses.client as CustomAzureOpenAIClient;
    azureResponsesClient.abortHandler = (): void => undefined;
    expect(model.exposedClient).toBe(azureResponsesClient);
    expect(requestOptions.headers).toEqual(
      expect.objectContaining({
        'api-key': 'test-azure-key',
        'x-smoke': 'azure',
      })
    );
    expect(requestOptions.query).toEqual(
      expect.objectContaining({ 'api-version': '2024-10-21' })
    );
    expect(model.getReasoningParams()).toEqual({ effort: 'low' });

    const nonReasoningModel = new AzureChatOpenAI({
      ...baseAzureFields,
    }) as AzureReasoningModel;
    nonReasoningModel.model = 'gpt-4o';
    nonReasoningModel.reasoning = { effort: 'low' };
    expect(nonReasoningModel.getReasoningParams()).toBeUndefined();
  });

  it('keeps DeepSeek, Moonshot, and xAI on LibreChat wrapper semantics', () => {
    const deepSeek = new ChatDeepSeek({
      model: 'deepseek-chat',
      apiKey: 'test-key',
      _lc_stream_delay: 5,
    });
    deepSeek._getClientOptions();

    const moonshot = new ChatMoonshot({
      model: 'moonshot-v1-8k',
      apiKey: 'test-key',
      _lc_stream_delay: 6,
    });

    const xai = new ChatXAI({
      model: 'grok-3-fast',
      apiKey: 'test-key',
      configuration: { baseURL: 'https://xai.test/v1' },
      _lc_stream_delay: 7,
    });
    const xaiRequestOptions =
      xai._getClientOptions() as OpenAIRequestOptionsWithBaseURL;

    expect(ChatDeepSeek.lc_name()).toBe('LibreChatDeepSeek');
    expect(deepSeek._lc_stream_delay).toBe(5);
    expect(deepSeek.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(ChatMoonshot.lc_name()).toBe('LibreChatMoonshot');
    expect(moonshot._lc_stream_delay).toBe(6);
    expect(ChatXAI.lc_name()).toBe('LibreChatXAI');
    expect(xai._lc_stream_delay).toBe(7);
    expect(xai.exposedClient).toBeInstanceOf(CustomOpenAIClient);
    expect(xaiRequestOptions.baseURL).toBe('https://xai.test/v1');
  });

  it('skips custom OpenAI-compatible SSE events during OpenAI streaming', async () => {
    await expectCustomSSEEventsSkipped(
      new ChatOpenAI({
        model: 'hermes-agent',
        apiKey: 'test-key',
        streaming: true,
      })
    );
  });

  it('skips custom OpenAI-compatible SSE events during Azure streaming', async () => {
    await expectCustomSSEEventsSkipped(
      new AzureChatOpenAI({
        ...baseAzureFields,
      })
    );
  });

  it('passes non-streaming OpenAI completions through unchanged', async () => {
    const model = new ChatOpenAI({
      model: 'hermes-agent',
      apiKey: 'test-key',
    });
    const createMock = mockCompletion(model, {
      id: 'chatcmpl-nonstream-test',
      object: 'chat.completion',
      created: 0,
      model: 'hermes-agent',
      choices: [
        {
          index: 0,
          finish_reason: 'stop',
          logprobs: null,
          message: {
            role: 'assistant',
            content: 'plain response',
            refusal: null,
          },
        },
      ],
    });

    const response = await model.invoke([new HumanMessage('no stream')]);

    expect(response.content).toBe('plain response');
    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({ stream: false }),
      expect.any(Object)
    );
  });

  it('keeps Moonshot reasoning content in completion requests', async () => {
    const moonshot = new ChatMoonshot({
      model: 'moonshot-v1-8k',
      apiKey: 'test-key',
      streaming: false,
    });
    const completions = (moonshot as unknown as CompletionBackedModel)
      .completions;
    let requestMessages: unknown;

    completions.completionWithRetry = async (request): Promise<unknown> => {
      requestMessages = request.messages;
      return {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        created: 0,
        model: 'moonshot-v1-8k',
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            message: {
              role: 'assistant',
              content: 'ok',
            },
          },
        ],
      };
    };

    await moonshot.invoke([
      new AIMessage({
        content: '',
        additional_kwargs: { reasoning_content: 'kept-thinking' },
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: { q: 'test' },
            type: 'tool_call',
          },
        ],
      }),
    ]);

    expect(requestMessages).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: '',
        reasoning_content: 'kept-thinking',
        tool_calls: expect.any(Array),
      }),
    ]);
  });

  it('keeps OpenRouter reasoning isolated from OpenAI reasoning_effort', () => {
    const fields: OpenRouterFields = {
      model: 'openrouter/test-model',
      apiKey: 'test-key',
      reasoning: { effort: 'xhigh', max_tokens: 2048 },
    };
    const model = new ChatOpenRouter(fields);

    const params = model.invocationParams();

    expect(ChatOpenRouter.lc_name()).toBe('LibreChatOpenRouter');
    expect(params.reasoning).toEqual({ effort: 'xhigh', max_tokens: 2048 });
    expect(params.reasoning_effort).toBeUndefined();

    const callParams = model.invocationParams({
      reasoning: { effort: 'low', exclude: true },
    } as ChatOpenRouterCallOptions);
    expect(callParams.reasoning).toEqual({
      effort: 'low',
      max_tokens: 2048,
      exclude: true,
    });

    const legacyModel = new ChatOpenRouter({
      model: 'openrouter/test-model',
      apiKey: 'test-key',
      include_reasoning: true,
    });
    expect(legacyModel.invocationParams().reasoning).toEqual({
      enabled: true,
    });
  });

  it('keeps OpenRouter streaming reasoning details stable', async () => {
    const model = new ChatOpenRouter({
      model: 'anthropic/claude-sonnet-test',
      apiKey: 'test-key',
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    let requestMessages: unknown;
    const createChunk = (
      choice: OpenRouterReasoningStreamChoice
    ): OpenAIClient.Chat.Completions.ChatCompletionChunk => ({
      id: 'chatcmpl-openrouter-test',
      object: 'chat.completion.chunk',
      created: 0,
      model: 'anthropic/claude-sonnet-test',
      choices: [choice],
    });

    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield createChunk({
        index: 0,
        delta: {
          role: 'assistant',
          content: '',
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'Think ',
              format: 'text',
              index: 0,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: {
          content: 'answer',
          reasoning_details: [
            { type: 'reasoning.text', text: 'hard', index: 0 },
            {
              type: 'reasoning.encrypted',
              id: 'sig_1',
              data: 'encrypted',
              format: 'anthropic',
              index: 1,
            },
          ],
        },
        finish_reason: null,
      });
      yield createChunk({
        index: 0,
        delta: { content: '' },
        finish_reason: 'stop',
      });
    }

    completions.completionWithRetry = async (
      request
    ): Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    > => {
      requestMessages = request.messages;
      return streamChunks();
    };

    const chunks: AIMessageChunk[] = [];
    const stream = await model.stream([
      new AIMessage({
        content: '',
        additional_kwargs: {
          reasoning_details: [
            {
              type: 'reasoning.text',
              text: 'previous thought',
              index: 0,
            },
            {
              type: 'reasoning.encrypted',
              id: 'prev_sig',
              data: 'previous encrypted',
              index: 1,
            },
          ],
        },
        tool_calls: [
          {
            id: 'call_1',
            name: 'lookup',
            args: { q: 'test' },
            type: 'tool_call',
          },
        ],
      }),
    ]);
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(requestMessages).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: expect.any(Array),
        content: [
          expect.objectContaining({
            type: 'thinking',
            thinking: 'previous thought',
          }),
          expect.objectContaining({
            type: 'redacted_thinking',
            data: 'previous encrypted',
            id: 'prev_sig',
          }),
        ],
      }),
    ]);
    expect(chunks).toHaveLength(3);
    expect(chunks[0].additional_kwargs.reasoning).toBe('Think ');
    expect(chunks[0].additional_kwargs.reasoning_details).toBeUndefined();
    expect(chunks[1].additional_kwargs.reasoning).toBe('hard');
    expect(chunks[1].additional_kwargs.reasoning_details).toBeUndefined();
    expect(chunks[2].additional_kwargs.reasoning_details).toEqual([
      {
        type: 'reasoning.text',
        text: 'Think hard',
        format: 'text',
        index: 0,
      },
      {
        type: 'reasoning.encrypted',
        id: 'sig_1',
        data: 'encrypted',
        format: 'anthropic',
        index: 1,
      },
    ]);
  });

  it('keeps Anthropic output, residency, compaction, and stream-delay options', () => {
    const contextManagement = {
      edits: [
        {
          type: 'compact_20260112' as const,
          trigger: { type: 'input_tokens' as const, value: 50000 },
        },
      ],
    };
    const model = new CustomAnthropic({
      model: 'claude-sonnet-4-5-20250929',
      apiKey: 'test-key',
      maxTokens: 4096,
      outputConfig: { effort: 'medium' },
      inferenceGeo: 'us',
      contextManagement,
      _lc_stream_delay: 8,
    });

    const params = model.invocationParams({
      outputConfig: { effort: 'low' },
      inferenceGeo: 'eu',
    } as AnthropicCallOptions);

    expect(CustomAnthropic.lc_name()).toBe('LibreChatAnthropic');
    expect(model._lc_stream_delay).toBe(8);
    expect(params.output_config).toEqual({ effort: 'low' });
    expect(params.inference_geo).toBe('eu');
    expect(params.context_management).toEqual(contextManagement);
  });

  it('keeps Anthropic beta, MCP, and container request wiring current', () => {
    const contextManagement = {
      edits: [
        {
          type: 'compact_20260112' as const,
          trigger: { type: 'input_tokens' as const, value: 50000 },
        },
      ],
    };
    const mcpServers: AnthropicMCPServerURLDefinition[] = [
      {
        type: 'url',
        url: 'https://example.com/mcp',
        name: 'docs',
      },
    ];
    const model = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      contextManagement,
      betas: ['model-beta'],
    });

    const params = model.invocationParams({
      outputConfig: {
        effort: 'low',
        task_budget: { type: 'token_budget', value: 1024 },
      },
      betas: ['request-beta', 'model-beta'],
      container: 'container_123',
      mcp_servers: mcpServers,
      tools: [
        {
          type: 'tool_search_tool_bm25_20251119',
          name: 'search',
        } as ChatAnthropicToolType,
      ],
    } as AnthropicCallOptions);

    expect(params.betas).toEqual([
      'model-beta',
      'request-beta',
      'advanced-tool-use-2025-11-20',
      'compact-2026-01-12',
      'task-budgets-2026-03-13',
    ]);
    expect(params.container).toBe('container_123');
    expect(params.mcp_servers).toBe(mcpServers);
    expect(params.temperature).toBeUndefined();
    expect(params.top_k).toBeUndefined();
    expect(params.top_p).toBeUndefined();
  });

  it('matches Anthropic Opus 4.7 sampling compatibility checks', () => {
    const thinkingModel = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      thinking: { type: 'enabled', budget_tokens: 1024 },
    });
    const topKModel = new CustomAnthropic({
      model: 'claude-opus-4-7-test',
      apiKey: 'test-key',
      maxTokens: 4096,
      topK: 5,
    });

    expect(() => thinkingModel.invocationParams()).toThrow(
      'thinking.type="enabled" is not supported for claude-opus-4-7'
    );
    expect(() => topKModel.invocationParams()).toThrow(
      'topK is not supported for claude-opus-4-7'
    );
  });

  it('keeps Bedrock Converse application profiles and service tier passthroughs', () => {
    const applicationInferenceProfile =
      'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
    const model = new CustomChatBedrockConverse({
      ...baseBedrockFields,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      applicationInferenceProfile,
      serviceTier: 'priority',
    });

    expect(CustomChatBedrockConverse.lc_name()).toBe(
      'LibreChatBedrockConverse'
    );
    expect(model.applicationInferenceProfile).toBe(applicationInferenceProfile);
    expect(model.invocationParams({}).serviceTier).toEqual({
      type: 'priority',
    });
    expect(model.invocationParams({ serviceTier: 'flex' }).serviceTier).toEqual(
      { type: 'flex' }
    );
  });

  it('keeps Google and Vertex thinking configuration wiring offline', () => {
    const thinkingConfig = {
      thinkingLevel: 'HIGH' as const,
      includeThoughts: true,
    };
    const google = new CustomChatGoogleGenerativeAI({
      model: 'models/gemini-3-pro-preview',
      apiKey: 'test-key',
      thinkingConfig,
    });
    const vertex = new ChatVertexAI({
      model: 'gemini-3-pro-preview',
      location: 'global',
      thinkingBudget: -1,
      thinkingConfig,
    });

    expect(CustomChatGoogleGenerativeAI.lc_name()).toBe(
      'LibreChatGoogleGenerativeAI'
    );
    expect(google.model).toBe('gemini-3-pro-preview');
    expect(google._isMultimodalModel).toBe(true);
    expect(google.thinkingConfig).toEqual(thinkingConfig);
    expect(ChatVertexAI.lc_name()).toBe('LibreChatVertexAI');
    expect(vertex.dynamicThinkingBudget).toBe(true);
    expect(vertex.thinkingConfig).toEqual(thinkingConfig);
    expect(vertex.invocationParams({}).maxReasoningTokens).toBe(-1);
  });

  it('uppercases custom OpenAI fetch methods before dispatch', async () => {
    let method: string | undefined;
    const client = new CustomOpenAIClient({
      apiKey: 'test-key',
      fetch: async (_url, init): Promise<Response> => {
        method = init?.method;
        return new Response('{}', { status: 200 });
      },
    });

    const response = await client.fetchWithTimeout(
      'https://example.test/v1/chat/completions',
      { method: 'patch' },
      1000,
      new AbortController()
    );

    expect(response.status).toBe(200);
    expect(method).toBe('PATCH');
    expect(client.abortHandler).toBeDefined();
  });
});
