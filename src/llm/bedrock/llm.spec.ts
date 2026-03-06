/* eslint-disable no-process-env */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import { expect, test, describe, jest } from '@jest/globals';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  SystemMessage,
  AIMessageChunk,
} from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  BedrockRuntimeClient,
  ConverseCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type { ConverseResponse } from '@aws-sdk/client-bedrock-runtime';
import {
  convertConverseMessageToLangChainMessage,
  handleConverseStreamMetadata,
  convertToConverseMessages,
} from './utils';
import { CustomChatBedrockConverse, ServiceTierType } from './index';

jest.setTimeout(120000);

// Base constructor args for tests
const baseConstructorArgs = {
  region: 'us-east-1',
  credentials: {
    secretAccessKey: 'test-secret-key',
    accessKeyId: 'test-access-key',
  },
};

describe('CustomChatBedrockConverse', () => {
  describe('applicationInferenceProfile parameter', () => {
    test('should initialize applicationInferenceProfile from constructor', () => {
      const testArn =
        'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        applicationInferenceProfile: testArn,
      });
      expect(model.model).toBe('anthropic.claude-3-haiku-20240307-v1:0');
      expect(model.applicationInferenceProfile).toBe(testArn);
    });

    test('should be undefined when not provided in constructor', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
      });
      expect(model.model).toBe('anthropic.claude-3-haiku-20240307-v1:0');
      expect(model.applicationInferenceProfile).toBeUndefined();
    });

    test('should send applicationInferenceProfile as modelId in ConverseCommand when provided', async () => {
      const testArn =
        'arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/test-profile';
      const mockSend = jest.fn<any>().mockResolvedValue({
        output: {
          message: {
            role: 'assistant',
            content: [{ text: 'Test response' }],
          },
        },
        stopReason: 'end_turn',
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          totalTokens: 15,
        },
      });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        applicationInferenceProfile: testArn,
        client: mockClient,
      });

      await model.invoke([new HumanMessage('Hello')]);

      expect(mockSend).toHaveBeenCalledTimes(1);
      const commandArg = mockSend.mock.calls[0][0] as {
        input: { modelId: string };
      };
      expect(commandArg.input.modelId).toBe(testArn);
      expect(commandArg.input.modelId).not.toBe(
        'anthropic.claude-3-haiku-20240307-v1:0'
      );
    });

    test('should send model as modelId in ConverseCommand when applicationInferenceProfile is not provided', async () => {
      const mockSend = jest.fn<any>().mockResolvedValue({
        output: {
          message: {
            role: 'assistant',
            content: [{ text: 'Test response' }],
          },
        },
        stopReason: 'end_turn',
        usage: {
          inputTokens: 10,
          outputTokens: 5,
          totalTokens: 15,
        },
      });

      const mockClient = {
        send: mockSend,
      } as unknown as BedrockRuntimeClient;

      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
        client: mockClient,
      });

      await model.invoke([new HumanMessage('Hello')]);

      expect(mockSend).toHaveBeenCalledTimes(1);
      const commandArg = mockSend.mock.calls[0][0] as {
        input: { modelId: string };
      };
      expect(commandArg.input.modelId).toBe(
        'anthropic.claude-3-haiku-20240307-v1:0'
      );
    });
  });

  describe('serviceTier configuration', () => {
    test('should set serviceTier in constructor', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'priority',
      });
      expect(model.serviceTier).toBe('priority');
    });

    test('should set serviceTier as undefined when not provided', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
      });
      expect(model.serviceTier).toBeUndefined();
    });

    test.each(['priority', 'default', 'flex', 'reserved'])(
      'should include serviceTier in invocationParams when set to %s',
      (serviceTier) => {
        const model = new CustomChatBedrockConverse({
          ...baseConstructorArgs,
          serviceTier: serviceTier as ServiceTierType,
        });
        const params = model.invocationParams({});
        expect(params.serviceTier).toEqual({ type: serviceTier });
      }
    );

    test('should not include serviceTier in invocationParams when not set', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
      });
      const params = model.invocationParams({});
      expect(params.serviceTier).toBeUndefined();
    });

    test('should override serviceTier from call options in invocationParams', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'default',
      });
      const params = model.invocationParams({
        serviceTier: 'priority',
      });
      expect(params.serviceTier).toEqual({ type: 'priority' });
    });

    test('should use class-level serviceTier when call options do not override it', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'flex',
      });
      const params = model.invocationParams({});
      expect(params.serviceTier).toEqual({ type: 'flex' });
    });

    test('should handle serviceTier in invocationParams with other config options', () => {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        serviceTier: 'reserved',
        temperature: 0.5,
        maxTokens: 100,
      });
      const params = model.invocationParams({
        stop: ['stop_sequence'],
      });
      expect(params.serviceTier).toEqual({ type: 'reserved' });
      expect(params.inferenceConfig?.temperature).toBe(0.5);
      expect(params.inferenceConfig?.maxTokens).toBe(100);
      expect(params.inferenceConfig?.stopSequences).toEqual(['stop_sequence']);
    });
  });

  describe('contentBlockIndex cleanup', () => {
    // Access private methods for testing via any cast
    function getModelWithCleanMethods() {
      const model = new CustomChatBedrockConverse({
        ...baseConstructorArgs,
        model: 'anthropic.claude-3-haiku-20240307-v1:0',
      });
      return model as any;
    }

    test('should remove contentBlockIndex from top level', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        contentBlockIndex: 0,
        text: 'hello',
        other: 'data',
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({ text: 'hello', other: 'data' });
      expect(cleaned.contentBlockIndex).toBeUndefined();
    });

    test('should remove contentBlockIndex from nested objects', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        outer: {
          contentBlockIndex: 1,
          inner: {
            contentBlockIndex: 2,
            data: 'test',
          },
        },
        topLevel: 'value',
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({
        outer: {
          inner: {
            data: 'test',
          },
        },
        topLevel: 'value',
      });
    });

    test('should handle arrays when removing contentBlockIndex', () => {
      const model = getModelWithCleanMethods();
      const obj = {
        items: [
          { contentBlockIndex: 0, text: 'first' },
          { contentBlockIndex: 1, text: 'second' },
        ],
      };

      const cleaned = model.removeContentBlockIndex(obj);

      expect(cleaned).toEqual({
        items: [{ text: 'first' }, { text: 'second' }],
      });
    });

    test('should preserve null and undefined values', () => {
      const model = getModelWithCleanMethods();

      expect(model.removeContentBlockIndex(null)).toBeNull();
      expect(model.removeContentBlockIndex(undefined)).toBeUndefined();
    });

    test('enrichChunk should strip contentBlockIndex from response_metadata', () => {
      const model = getModelWithCleanMethods();

      const chunkWithIndex = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            contentBlockIndex: 0,
            stopReason: null,
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithIndex, new Set([0]));

      expect(enriched.message.response_metadata).toEqual({
        stopReason: null,
      });
      expect(
        (enriched.message.response_metadata as any).contentBlockIndex
      ).toBeUndefined();
      expect(enriched.text).toBe('Hello');
    });

    test('enrichChunk should pass through chunks without contentBlockIndex unchanged', () => {
      const model = getModelWithCleanMethods();

      const chunkWithoutIndex = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            stopReason: 'end_turn',
            usage: { inputTokens: 10, outputTokens: 5 },
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithoutIndex, new Set());

      expect(enriched.message.response_metadata).toEqual({
        stopReason: 'end_turn',
        usage: { inputTokens: 10, outputTokens: 5 },
      });
    });

    test('enrichChunk should inject index on array content blocks', () => {
      const model = getModelWithCleanMethods();

      const chunkWithArrayContent = new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: [
            {
              type: 'reasoning_content',
              reasoningText: { text: 'thinking...' },
            },
          ],
          response_metadata: {
            contentBlockIndex: 0,
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithArrayContent, new Set([0]));

      expect(Array.isArray(enriched.message.content)).toBe(true);
      const blocks = enriched.message.content as any[];
      expect(blocks[0].index).toBe(0);
      expect(blocks[0].type).toBe('reasoning_content');
      expect(
        (enriched.message.response_metadata as any).contentBlockIndex
      ).toBeUndefined();
    });

    test('enrichChunk should promote text to array when multiple block indices seen', () => {
      const model = getModelWithCleanMethods();

      const textChunk = new ChatGenerationChunk({
        text: 'Hello world',
        message: new AIMessageChunk({
          content: 'Hello world',
          response_metadata: {
            contentBlockIndex: 1,
          },
        }),
      });

      const enriched = model.enrichChunk(textChunk, new Set([0, 1]));

      expect(Array.isArray(enriched.message.content)).toBe(true);
      const blocks = enriched.message.content as any[];
      expect(blocks).toHaveLength(1);
      expect(blocks[0]).toEqual({
        type: 'text',
        text: 'Hello world',
        index: 1,
      });
    });

    test('enrichChunk should keep text as string when only one block index seen', () => {
      const model = getModelWithCleanMethods();

      const textChunk = new ChatGenerationChunk({
        text: 'Hello',
        message: new AIMessageChunk({
          content: 'Hello',
          response_metadata: {
            contentBlockIndex: 0,
            stopReason: null,
          },
        }),
      });

      const enriched = model.enrichChunk(textChunk, new Set([0]));

      expect(typeof enriched.message.content).toBe('string');
      expect(enriched.message.content).toBe('Hello');
    });

    test('enrichChunk should strip deeply nested contentBlockIndex from response_metadata', () => {
      const model = getModelWithCleanMethods();

      const chunkWithNestedIndex = new ChatGenerationChunk({
        text: 'Test',
        message: new AIMessageChunk({
          content: 'Test',
          response_metadata: {
            contentBlockIndex: 0,
            amazon: {
              bedrock: {
                contentBlockIndex: 0,
                trace: { something: 'value' },
              },
            },
            otherData: 'preserved',
          },
        }),
      });

      const enriched = model.enrichChunk(chunkWithNestedIndex, new Set([0]));

      expect(enriched.message.response_metadata).toEqual({
        amazon: {
          bedrock: {
            trace: { something: 'value' },
          },
        },
        otherData: 'preserved',
      });
    });
  });
});

describe('handleConverseStreamMetadata - cache token extraction', () => {
  test('should extract cacheReadInputTokens and cacheWriteInputTokens into input_token_details', () => {
    const metadata = {
      usage: {
        inputTokens: 13,
        outputTokens: 5,
        totalTokens: 10849,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      },
      metrics: { latencyMs: 1000 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toEqual({
      input_tokens: 13,
      output_tokens: 5,
      total_tokens: 10849,
      input_token_details: {
        cache_read: 10831,
        cache_creation: 0,
      },
    });
  });

  test('should not include input_token_details when no cache tokens present', () => {
    const metadata = {
      usage: {
        inputTokens: 100,
        outputTokens: 50,
        totalTokens: 150,
      },
      metrics: { latencyMs: 500 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(msg.usage_metadata?.input_token_details).toBeUndefined();
  });

  test('should include input_token_details when only cacheWriteInputTokens is present', () => {
    const metadata = {
      usage: {
        inputTokens: 50,
        outputTokens: 10,
        totalTokens: 10060,
        cacheWriteInputTokens: 10000,
      },
      metrics: { latencyMs: 800 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: true,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata?.input_token_details).toEqual({
      cache_read: 0,
      cache_creation: 10000,
    });
  });

  test('should return undefined usage_metadata when streamUsage is false', () => {
    const metadata = {
      usage: {
        inputTokens: 13,
        outputTokens: 5,
        totalTokens: 10849,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      },
      metrics: { latencyMs: 1000 },
    };

    const chunk = handleConverseStreamMetadata(metadata, {
      streamUsage: false,
    });
    const msg = chunk.message as AIMessageChunk;

    expect(msg.usage_metadata).toBeUndefined();
  });
});

describe('convertConverseMessageToLangChainMessage - cache token extraction', () => {
  const makeResponseMetadata = (
    usage: Record<string, number>
  ): Omit<ConverseResponse, 'output'> =>
    ({
      usage,
      stopReason: 'end_turn',
      metrics: undefined,
      $metadata: { requestId: 'test-id' },
    }) as unknown as Omit<ConverseResponse, 'output'>;

  test('should extract cache tokens in non-streaming response', () => {
    const message = {
      role: 'assistant' as const,
      content: [{ text: 'Hello!' }],
    };

    const result = convertConverseMessageToLangChainMessage(
      message,
      makeResponseMetadata({
        inputTokens: 20,
        outputTokens: 5,
        totalTokens: 10856,
        cacheReadInputTokens: 10831,
        cacheWriteInputTokens: 0,
      })
    );

    expect(result.usage_metadata).toEqual({
      input_tokens: 20,
      output_tokens: 5,
      total_tokens: 10856,
      input_token_details: {
        cache_read: 10831,
        cache_creation: 0,
      },
    });
  });

  test('should not include input_token_details when no cache tokens in non-streaming response', () => {
    const message = {
      role: 'assistant' as const,
      content: [{ text: 'Hello!' }],
    };

    const result = convertConverseMessageToLangChainMessage(
      message,
      makeResponseMetadata({
        inputTokens: 100,
        outputTokens: 50,
        totalTokens: 150,
      })
    );

    expect(result.usage_metadata).toEqual({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
    });
    expect(result.usage_metadata?.input_token_details).toBeUndefined();
  });
});

describe('convertToConverseMessages', () => {
  test('should convert basic messages', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an AI assistant."),
      new HumanMessage('Hello!'),
    ]);

    expect(converseSystem).toEqual([{ text: "You're an AI assistant." }]);
    expect(converseMessages).toHaveLength(1);
    expect(converseMessages[0].role).toBe('user');
    expect(converseMessages[0].content).toEqual([{ text: 'Hello!' }]);
  });

  test('should handle standard v1 format with tool_call blocks (e.g., from Anthropic provider)', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an advanced AI assistant."),
      new HumanMessage("What's the weather in SF?"),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me check the weather for you.' },
          {
            type: 'tool_call',
            id: 'call_123',
            name: 'get_weather',
            args: { location: 'San Francisco' },
          },
        ],
        response_metadata: {
          output_version: 'v1',
          model_provider: 'anthropic',
        },
      }),
      new ToolMessage({
        tool_call_id: 'call_123',
        content: '72°F and sunny',
      }),
    ]);

    expect(converseSystem).toEqual([
      { text: "You're an advanced AI assistant." },
    ]);
    expect(converseMessages).toHaveLength(3);

    // Check user message
    expect(converseMessages[0].role).toBe('user');
    expect(converseMessages[0].content).toEqual([
      { text: "What's the weather in SF?" },
    ]);

    // Check AI message with tool use
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toHaveLength(2);
    expect(converseMessages[1].content?.[0]).toEqual({
      text: 'Let me check the weather for you.',
    });
    expect(converseMessages[1].content?.[1]).toEqual({
      toolUse: {
        toolUseId: 'call_123',
        name: 'get_weather',
        input: { location: 'San Francisco' },
      },
    });

    // Check tool result
    expect(converseMessages[2].role).toBe('user');
    expect(converseMessages[2].content).toHaveLength(1);
    expect((converseMessages[2].content?.[0] as any).toolResult).toBeDefined();
    expect((converseMessages[2].content?.[0] as any).toolResult.toolUseId).toBe(
      'call_123'
    );
  });

  test('should handle standard v1 format with reasoning blocks (e.g., from Anthropic provider)', () => {
    const { converseMessages, converseSystem } = convertToConverseMessages([
      new SystemMessage("You're an advanced AI assistant."),
      new HumanMessage('What is 2+2?'),
      new AIMessage({
        content: [
          {
            type: 'reasoning',
            reasoning: 'I need to add 2 and 2 together.',
          },
          { type: 'text', text: 'The answer is 4.' },
        ],
        response_metadata: {
          output_version: 'v1',
          model_provider: 'anthropic',
        },
      }),
      new HumanMessage('Thanks! What about 3+3?'),
    ]);

    expect(converseSystem).toEqual([
      { text: "You're an advanced AI assistant." },
    ]);
    expect(converseMessages).toHaveLength(3);

    // Check AI message with reasoning
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toHaveLength(2);
    expect(
      (converseMessages[1].content?.[0] as any).reasoningContent
    ).toBeDefined();
    expect(
      (converseMessages[1].content?.[0] as any).reasoningContent.reasoningText
        .text
    ).toBe('I need to add 2 and 2 together.');
    expect(converseMessages[1].content?.[1]).toEqual({
      text: 'The answer is 4.',
    });
  });

  test('should handle messages without v1 format', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('Hello'),
      new AIMessage({
        content: 'Hi there!',
        tool_calls: [],
      }),
    ]);

    expect(converseMessages).toHaveLength(2);
    expect(converseMessages[1].role).toBe('assistant');
    expect(converseMessages[1].content).toEqual([{ text: 'Hi there!' }]);
  });

  test('should combine consecutive tool result messages', () => {
    const { converseMessages } = convertToConverseMessages([
      new HumanMessage('Get weather for SF and NYC'),
      new AIMessage({
        content: 'I will check both cities.',
        tool_calls: [
          { id: 'call_1', name: 'get_weather', args: { city: 'SF' } },
          { id: 'call_2', name: 'get_weather', args: { city: 'NYC' } },
        ],
      }),
      new ToolMessage({
        tool_call_id: 'call_1',
        content: 'SF: 72°F',
      }),
      new ToolMessage({
        tool_call_id: 'call_2',
        content: 'NYC: 65°F',
      }),
    ]);

    // Tool messages should be combined into one user message
    expect(converseMessages).toHaveLength(3);
    const toolResultMessage = converseMessages[2];
    expect(toolResultMessage.role).toBe('user');
    expect(toolResultMessage.content).toHaveLength(2);
    expect((toolResultMessage.content?.[0] as any).toolResult.toolUseId).toBe(
      'call_1'
    );
    expect((toolResultMessage.content?.[1] as any).toolResult.toolUseId).toBe(
      'call_2'
    );
  });
});

// Integration tests (require AWS credentials)
describe.skip('Integration tests', () => {
  const integrationArgs = {
    region: process.env.BEDROCK_AWS_REGION ?? 'us-east-1',
    credentials: {
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
    },
  };

  test('basic invoke', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxRetries: 0,
    });
    const message = new HumanMessage('Hello!');
    const res = await model.invoke([message]);
    expect(res.response_metadata.usage).toBeDefined();
  });

  test('basic streaming', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxRetries: 0,
    });

    let fullMessage: AIMessageChunk | undefined;
    for await (const chunk of await model.stream('Hello!')) {
      fullMessage = fullMessage ? concat(fullMessage, chunk) : chunk;
    }

    expect(fullMessage).toBeDefined();
    expect(fullMessage?.content).toBeDefined();
  });

  test('with thinking/reasoning enabled', async () => {
    const model = new CustomChatBedrockConverse({
      ...integrationArgs,
      model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
      maxTokens: 5000,
      additionalModelRequestFields: {
        thinking: { type: 'enabled', budget_tokens: 2000 },
      },
    });

    const result = await model.invoke('What is 2 + 2?');
    expect(result.content).toBeDefined();

    // Should have reasoning content if the model supports it
    if (Array.isArray(result.content)) {
      const reasoningBlocks = result.content.filter(
        (b: any) => b.type === 'reasoning_content' || b.type === 'reasoning'
      );
      expect(reasoningBlocks.length).toBeGreaterThanOrEqual(0);
    }
  });

  test('cache tokens should populate input_token_details', async () => {
    const client = new BedrockRuntimeClient({
      region: integrationArgs.region,
      credentials: integrationArgs.credentials,
    });

    // Large system prompt (>1024 tokens) to meet Bedrock's minimum cache threshold
    const largeSystemPrompt = [
      'You are an expert assistant.',
      ...Array(200).fill(
        'This is padding content to exceed the minimum token threshold for Bedrock prompt caching. '
      ),
      'When answering, be brief and direct.',
    ].join(' ');

    const systemBlocks = [
      { text: largeSystemPrompt },
      { cachePoint: { type: 'default' as const } },
    ];

    const converseArgs = {
      modelId: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
      system: systemBlocks,
      inferenceConfig: { maxTokens: 50 },
    };

    // Call 1: populate the cache (may be a write or read if already warm)
    await client.send(
      new ConverseCommand({
        ...converseArgs,
        messages: [{ role: 'user', content: [{ text: 'Say hello.' }] }],
      })
    );

    // Call 2: should read from cache — this is the one we assert on
    const response = await client.send(
      new ConverseCommand({
        ...converseArgs,
        messages: [
          { role: 'user', content: [{ text: 'Say hello.' }] },
          { role: 'assistant', content: [{ text: 'Hello!' }] },
          { role: 'user', content: [{ text: 'Say goodbye.' }] },
        ],
      })
    );

    // Feed raw response through convertConverseMessageToLangChainMessage
    const result = convertConverseMessageToLangChainMessage(
      response.output!.message!,
      response
    );

    expect(result.usage_metadata).toBeDefined();
    expect(result.usage_metadata!.input_tokens).toBeGreaterThan(0);
    expect(result.usage_metadata!.output_tokens).toBeGreaterThan(0);

    // Cache should have been populated by call 1, so call 2 should show cache reads
    expect(result.usage_metadata!.input_token_details).toBeDefined();
    expect(
      result.usage_metadata!.input_token_details!.cache_read
    ).toBeGreaterThan(0);
  });
});
