import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  assertPreparedProviderRequestFor,
  prepareProviderRequest,
} from '@/llm/prepareProviderRequest';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { attemptInvoke } from '@/llm/invoke';
import { Providers } from '@/common';

type StubModel = {
  model?: string;
  invoke: (messages: BaseMessage[]) => Promise<AIMessage>;
};

interface CapturingModel {
  model: StubModel;
  invocations: BaseMessage[][];
}

function createCapturingModel(): CapturingModel {
  const invocations: BaseMessage[][] = [];
  return {
    invocations,
    model: {
      model: 'prepared-model',
      invoke: jest.fn(async (messages: BaseMessage[]): Promise<AIMessage> => {
        invocations.push(messages);
        return new AIMessage('ok');
      }),
    },
  };
}

describe('prepareProviderRequest', () => {
  it('measures and sends the exact prepared message array without re-projection', async () => {
    const { model, invocations } = createCapturingModel();
    const source = [
      new HumanMessage('first'),
      new HumanMessage('second'),
    ];
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: true,
      projectedMessageTokens: messages.length * 10,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: source,
      provider: Providers.MISTRAL,
      measure,
    });

    expect(Object.isFrozen(request)).toBe(true);
    const [brand] = Object.getOwnPropertySymbols(request);
    expect(Object.getOwnPropertyDescriptor(request, brand)).toMatchObject({
      configurable: false,
      enumerable: false,
      value: true,
      writable: false,
    });
    expect(Object.getOwnPropertySymbols({ ...request })).toHaveLength(0);
    expect(request.modelId).toBe('prepared-model');
    expect(request.messages).toHaveLength(1);
    expect(request.measurement).toEqual({
      fits: true,
      projectedMessageTokens: 10,
    });
    expect(measure).toHaveBeenCalledTimes(1);
    expect(measure).toHaveBeenCalledWith(request.messages);
    expect(source).toHaveLength(2);

    await attemptInvoke({ request });

    expect(invocations).toHaveLength(1);
    expect(invocations[0]).toBe(request.messages);
    expect(measure).toHaveBeenCalledTimes(1);
  });

  it('keeps tool-reference annotation transient and inside the measured request', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-1', 'tool0turn0', 'stored');
    const toolMessage = new ToolMessage({
      content: 'tool output',
      tool_call_id: 'call-1',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const { model } = createCapturingModel();
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: messages.length > 0,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [toolMessage],
      provider: Providers.ANTHROPIC,
      context: { getOrCreateToolOutputRegistry: () => registry },
      config: { configurable: { run_id: 'run-1' } },
      measure,
    });

    expect(request.messages[0].content).toBe(
      '[ref: tool0turn0]\ntool output'
    );
    expect(measure).toHaveBeenCalledWith(request.messages);
    expect(toolMessage.content).toBe('tool output');
    expect(toolMessage.additional_kwargs._refKey).toBe('tool0turn0');
  });

  it('includes serving-provider handoff shaping before measurement', () => {
    const { model } = createCapturingModel();
    const predecessor = new AIMessage({ id: 'previous-agent', content: 'done' });
    const measure = jest.fn((messages: BaseMessage[]) => ({
      fits: messages.length > 0,
    }));

    const request = prepareProviderRequest({
      model: model as t.ChatModel,
      messages: [predecessor],
      provider: Providers.ANTHROPIC,
      context: { isRunProducedMessage: (message) => message === predecessor },
      measure,
    });

    expect(request.messages).toHaveLength(2);
    expect(request.messages[1].getType()).toBe('human');
    expect(measure).toHaveBeenCalledWith(request.messages);
  });

  it('fails closed when a prepared artifact is used for another model or provider', () => {
    const first = createCapturingModel();
    const second = createCapturingModel();
    const request = prepareProviderRequest({
      model: first.model as t.ChatModel,
      messages: [new HumanMessage('hello')],
      provider: Providers.OPENAI,
    });

    expect(() =>
      assertPreparedProviderRequestFor(
        request,
        second.model as t.ChatModel,
        Providers.OPENAI
      )
    ).toThrow('does not match serving model');
    expect(() =>
      assertPreparedProviderRequestFor(
        request,
        first.model as t.ChatModel,
        Providers.ANTHROPIC
      )
    ).toThrow('does not match serving provider');
  });
});
