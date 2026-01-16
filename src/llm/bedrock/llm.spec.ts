/* eslint-disable no-process-env */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import { expect, test, describe, jest } from '@jest/globals';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime';
import { CustomChatBedrockConverse, ServiceTierType } from './index';
import { convertToConverseMessages } from './utils';

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

    test('should detect contentBlockIndex at top level', () => {
      const model = getModelWithCleanMethods();
      const objWithIndex = { contentBlockIndex: 0, text: 'hello' };
      const objWithoutIndex = { text: 'hello' };

      expect(model.hasContentBlockIndex(objWithIndex)).toBe(true);
      expect(model.hasContentBlockIndex(objWithoutIndex)).toBe(false);
    });

    test('should detect contentBlockIndex in nested objects', () => {
      const model = getModelWithCleanMethods();
      const nestedWithIndex = {
        outer: {
          inner: {
            contentBlockIndex: 1,
            data: 'test',
          },
        },
      };
      const nestedWithoutIndex = {
        outer: {
          inner: {
            data: 'test',
          },
        },
      };

      expect(model.hasContentBlockIndex(nestedWithIndex)).toBe(true);
      expect(model.hasContentBlockIndex(nestedWithoutIndex)).toBe(false);
    });

    test('should return false for null, undefined, and primitives', () => {
      const model = getModelWithCleanMethods();

      expect(model.hasContentBlockIndex(null)).toBe(false);
      expect(model.hasContentBlockIndex(undefined)).toBe(false);
      expect(model.hasContentBlockIndex('string')).toBe(false);
      expect(model.hasContentBlockIndex(123)).toBe(false);
      expect(model.hasContentBlockIndex(true)).toBe(false);
    });

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

    test('cleanChunk should remove contentBlockIndex from AIMessageChunk response_metadata', () => {
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

      const cleaned = model.cleanChunk(chunkWithIndex);

      expect(cleaned.message.response_metadata).toEqual({
        stopReason: null,
      });
      expect(
        (cleaned.message.response_metadata as any).contentBlockIndex
      ).toBeUndefined();
      expect(cleaned.text).toBe('Hello');
    });

    test('cleanChunk should pass through chunks without contentBlockIndex unchanged', () => {
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

      const cleaned = model.cleanChunk(chunkWithoutIndex);

      expect(cleaned.message.response_metadata).toEqual({
        stopReason: 'end_turn',
        usage: { inputTokens: 10, outputTokens: 5 },
      });
    });

    test('cleanChunk should handle deeply nested contentBlockIndex in response_metadata', () => {
      const model = getModelWithCleanMethods();

      const chunkWithNestedIndex = new ChatGenerationChunk({
        text: 'Test',
        message: new AIMessageChunk({
          content: 'Test',
          response_metadata: {
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

      const cleaned = model.cleanChunk(chunkWithNestedIndex);

      expect(cleaned.message.response_metadata).toEqual({
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
});
