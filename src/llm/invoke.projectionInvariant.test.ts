import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { afterEach, describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { AgentLogEvent } from '@/types';
import type * as t from '@/types';
import { prepareProviderRequest } from '@/llm/prepareProviderRequest';
import { attemptInvoke } from '@/llm/invoke';
import {
  ProviderMessageProjectionInvariantError,
  setProviderMessageProvenance,
} from '@/messages';
import { GraphEvents, Providers } from '@/common';

const originalInvariantMode =
  process.env.AGENT_MESSAGE_PROJECTION_INVARIANT;

afterEach(() => {
  if (originalInvariantMode == null) {
    delete process.env.AGENT_MESSAGE_PROJECTION_INVARIANT;
  } else {
    process.env.AGENT_MESSAGE_PROJECTION_INVARIANT = originalInvariantMode;
  }
  jest.restoreAllMocks();
});

function createStreamingRequest(messages: BaseMessage[]): {
  model: FakeListChatModel;
  request: ReturnType<typeof prepareProviderRequest>;
} {
  const model = new FakeListChatModel({ responses: ['ok'] });
  return {
    model,
    request: prepareProviderRequest({
      model,
      messages,
      provider: Providers.OPENAI,
    }),
  };
}

function createNonStreamingRequest(messages: BaseMessage[]): {
  model: FakeListChatModel;
  request: ReturnType<typeof prepareProviderRequest>;
} {
  const model = new FakeListChatModel({ responses: ['ok'] });
  Object.defineProperty(model, 'stream', { value: undefined });
  return {
    model,
    request: prepareProviderRequest({
      model: model as t.ChatModel,
      messages,
      provider: Providers.OPENAI,
    }),
  };
}

describe('attemptInvoke provider message projection invariant', () => {
  it('keeps the disabled path free of an added callback handler', async () => {
    delete process.env.AGENT_MESSAGE_PROJECTION_INVARIANT;
    const callbacks = [BaseCallbackHandler.fromMethods({})];
    const capturedCallbacks: unknown[] = [];
    const model: t.ChatModel = {
      invoke: async (_messages, config): Promise<AIMessageChunk> => {
        capturedCallbacks.push(config?.callbacks);
        return new AIMessageChunk('ok');
      },
    };
    const request = prepareProviderRequest({
      model,
      messages: [new HumanMessage('unstamped')],
      provider: Providers.OPENAI,
    });

    await attemptInvoke({ request }, { callbacks });

    expect(capturedCallbacks).toEqual([callbacks]);
  });

  it('observes gaps once and still invokes the streaming provider', async () => {
    process.env.AGENT_MESSAGE_PROJECTION_INVARIANT = 'observe';
    const logs: AgentLogEvent[] = [];
    const logHandler = BaseCallbackHandler.fromMethods({
      handleCustomEvent: (eventName: string, data: unknown): void => {
        if (eventName === GraphEvents.ON_AGENT_LOG) {
          logs.push(data as AgentLogEvent);
        }
      },
    });
    const { model, request } = createStreamingRequest([
      new HumanMessage('private-content'),
    ]);
    const providerCall = jest.spyOn(model, '_streamResponseChunks');

    await attemptInvoke({ request, onChunk: async () => {} }, {
      callbacks: [logHandler],
    });

    expect(providerCall).toHaveBeenCalledTimes(1);
    expect(logs).toHaveLength(1);
    expect(logs[0]).toMatchObject({
      level: 'warn',
      scope: 'projection',
      data: {
        provider: Providers.OPENAI,
        report: { valid: false, gapMessageCount: 1 },
      },
    });
    expect(JSON.stringify(logs[0])).not.toContain('private-content');
  });

  it('asserts before streaming provider I/O when provenance has a gap', async () => {
    process.env.AGENT_MESSAGE_PROJECTION_INVARIANT = 'assert';
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const { model, request } = createStreamingRequest([
      new HumanMessage('unstamped'),
    ]);
    const providerCall = jest.spyOn(model, '_streamResponseChunks');

    await expect(attemptInvoke({ request })).rejects.toBeInstanceOf(
      ProviderMessageProjectionInvariantError
    );
    expect(providerCall).not.toHaveBeenCalled();
  });

  it('asserts before non-streaming provider I/O when provenance has a gap', async () => {
    process.env.AGENT_MESSAGE_PROJECTION_INVARIANT = 'assert';
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const { model, request } = createNonStreamingRequest([
      new HumanMessage('unstamped'),
    ]);
    const providerCall = jest.spyOn(model, '_generate');

    await expect(attemptInvoke({ request })).rejects.toBeInstanceOf(
      ProviderMessageProjectionInvariantError
    );
    expect(providerCall).not.toHaveBeenCalled();
  });

  it('allows valid provenance through both invocation paths', async () => {
    process.env.AGENT_MESSAGE_PROJECTION_INVARIANT = 'assert';
    const streamingMessage = new HumanMessage('streaming');
    setProviderMessageProvenance(streamingMessage, [
      { attribution: 'user', sourceMessageId: 'streaming-source' },
    ]);
    const nonStreamingMessage = new HumanMessage('non-streaming');
    setProviderMessageProvenance(nonStreamingMessage, [
      { attribution: 'user', sourceMessageId: 'non-streaming-source' },
    ]);
    const streaming = createStreamingRequest([streamingMessage]);
    const nonStreaming = createNonStreamingRequest([nonStreamingMessage]);
    const streamCall = jest.spyOn(streaming.model, '_streamResponseChunks');
    const invokeCall = jest.spyOn(nonStreaming.model, '_generate');

    await attemptInvoke({
      request: streaming.request,
      onChunk: async () => {},
    });
    await attemptInvoke({ request: nonStreaming.request });

    expect(streamCall).toHaveBeenCalledTimes(1);
    expect(invokeCall).toHaveBeenCalledTimes(1);
  });
});
