import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import type { BaseMessageChunk } from '@langchain/core/messages';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { ChatOpenAI, AzureChatOpenAI } from './index';

type DeltaConverter = {
  _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, unknown>,
    rawResponse: Record<string, unknown>
  ): BaseMessageChunk;
};

const rawResponse = {
  id: 'chatcmpl-1',
  object: 'chat.completion.chunk',
  created: 1,
  model: 'gpt-5.5',
  choices: [],
};

const toolCallDelta = {
  role: 'assistant',
  tool_calls: [
    {
      index: 0,
      id: 'call_1',
      type: 'function',
      function: { name: 'weather', arguments: '{"ci' },
    },
  ],
};

function convertDelta(
  model: unknown,
  delta: Record<string, unknown>
): AIMessageChunk {
  const converter = (model as { completions: DeltaConverter }).completions;
  const message = converter._convertCompletionsDeltaToBaseMessageChunk(
    delta,
    rawResponse
  );
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

function adapterOf(message: AIMessageChunk): unknown {
  return (message.response_metadata as Record<string, unknown>)[
    STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY
  ];
}

describe('Chat Completions sequential tool-call seal stamping', () => {
  test('stamps tool-call deltas when no baseURL is configured (official)', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('stamps tool-call deltas for an explicit api.openai.com baseURL', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5.5',
      apiKey: 'test',
      configuration: { baseURL: 'https://api.openai.com/v1' },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });

  test('does not stamp tool-call deltas for OpenAI-compatible endpoints', () => {
    const model = new ChatOpenAI({
      model: 'kimi-k2',
      apiKey: 'test',
      configuration: { baseURL: 'https://api.moonshot.ai/v1' },
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBeUndefined();
  });

  test('does not stamp text-only deltas', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.5', apiKey: 'test' });
    const message = convertDelta(model, {
      role: 'assistant',
      content: 'hello',
    });
    expect(adapterOf(message)).toBeUndefined();
  });

  test('stamps Azure OpenAI tool-call deltas (first-party endpoint)', () => {
    const model = new AzureChatOpenAI({
      azureOpenAIApiKey: 'test',
      azureOpenAIApiInstanceName: 'test-instance',
      azureOpenAIApiDeploymentName: 'test-deployment',
      azureOpenAIApiVersion: '2024-08-01-preview',
    });
    const message = convertDelta(model, toolCallDelta);
    expect(adapterOf(message)).toBe(
      OPENAI_CHAT_SEQUENTIAL_STREAMED_TOOL_CALL_ADAPTER
    );
  });
});
