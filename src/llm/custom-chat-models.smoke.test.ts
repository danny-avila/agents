import type { OpenAIChatInput } from '@langchain/openai';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import type { CustomAnthropicInput } from '@/llm/anthropic';
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
type AnthropicCallOptions = Parameters<
  CustomAnthropic['invocationParams']
>[0] & {
  outputConfig?: CustomAnthropicInput['outputConfig'];
  inferenceGeo?: CustomAnthropicInput['inferenceGeo'];
};
type AzureReasoningModel = AzureChatOpenAI & {
  reasoning?: { effort: 'low' | 'high' };
};
type OpenRouterFields = Partial<
  ChatOpenRouterCallOptions & Pick<OpenAIChatInput, 'model' | 'apiKey'>
>;

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
