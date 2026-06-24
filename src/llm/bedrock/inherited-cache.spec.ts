/* eslint-disable @typescript-eslint/no-explicit-any */
import { expect, test, describe, jest } from '@jest/globals';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import type {
  ConverseCommandInput,
  ConverseStreamCommandInput,
} from '@aws-sdk/client-bedrock-runtime';
import { CustomChatBedrockConverse as ChatBedrockConverse } from './index';

jest.setTimeout(120000);

const baseConstructorArgs = {
  region: 'us-east-1',
  credentials: {
    secretAccessKey: 'test-secret',
    accessKeyId: 'test-key',
  },
  model: 'anthropic.claude-3-sonnet-20240229-v1:0',
};

function nonStreamingClient(): {
  client: BedrockRuntimeClient;
  send: ReturnType<typeof jest.fn>;
} {
  const send = jest.fn<any>().mockResolvedValue({
    output: {
      message: {
        role: 'assistant',
        content: [{ text: 'Response' }],
      },
    },
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
  });
  return { client: { send } as unknown as BedrockRuntimeClient, send };
}

function streamingClient(): {
  client: BedrockRuntimeClient;
  send: ReturnType<typeof jest.fn>;
} {
  const send = jest.fn<any>().mockResolvedValue({
    stream: (async function* streamChunks() {
      yield {
        contentBlockDelta: {
          contentBlockIndex: 0,
          delta: { text: 'Response' },
        },
      };
      yield {
        metadata: {
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        },
      };
    })(),
  });
  return { client: { send } as unknown as BedrockRuntimeClient, send };
}

describe('CustomChatBedrockConverse prompt caching request mapping', () => {
  test('invoke maps cache_control to system, messages, and tools', async () => {
    const { client, send } = nonStreamingClient();
    const model = new ChatBedrockConverse({ ...baseConstructorArgs, client });

    await model.invoke(
      [new SystemMessage('System prompt'), new HumanMessage('Hello')],
      {
        cache_control: { type: 'ephemeral', ttl: '1h' },
        tools: [
          {
            toolSpec: {
              name: 'get_weather',
              description: 'Get weather',
              inputSchema: {
                json: { type: 'object', properties: {} },
              },
            },
          },
        ],
      }
    );

    expect(send).toHaveBeenCalledTimes(1);
    const command = send.mock.calls[0][0] as ConverseCommand;
    expect(command).toBeInstanceOf(ConverseCommand);
    const input = command.input as ConverseCommandInput;

    expect(input.system).toEqual([
      { text: 'System prompt' },
      { cachePoint: { type: 'default', ttl: '1h' } },
    ]);
    expect(input.messages?.[0].content).toEqual([
      { text: 'Hello' },
      { cachePoint: { type: 'default', ttl: '1h' } },
    ]);
    expect(input.toolConfig?.tools).toEqual([
      {
        toolSpec: {
          name: 'get_weather',
          description: 'Get weather',
          inputSchema: {
            json: { type: 'object', properties: {} },
          },
        },
      },
      { cachePoint: { type: 'default', ttl: '1h' } },
    ]);
  });

  test('stream maps cache_control to system and last message', async () => {
    const { client, send } = streamingClient();
    const model = new ChatBedrockConverse({ ...baseConstructorArgs, client });

    const stream = await model.stream(
      [new SystemMessage('System prompt'), new HumanMessage('Hello')],
      {
        cache_control: { type: 'ephemeral' },
      }
    );
    for await (const _chunk of stream) {
      // Fully consume stream so the command is executed.
    }

    expect(send).toHaveBeenCalledTimes(1);
    const command = send.mock.calls[0][0] as ConverseStreamCommand;
    expect(command).toBeInstanceOf(ConverseStreamCommand);
    const input = command.input as ConverseStreamCommandInput;

    expect(input.system).toEqual([
      { text: 'System prompt' },
      { cachePoint: { type: 'default' } },
    ]);
    expect(input.messages?.[0].content).toEqual([
      { text: 'Hello' },
      { cachePoint: { type: 'default' } },
    ]);
  });

  // Dropped (inherited): live
});
