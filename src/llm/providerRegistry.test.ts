import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { FakeChatModel as CoreFakeChatModel } from '@langchain/core/utils/testing';
import type * as t from '@/types';
import {
  getProviderFamily,
  getChatModelClass,
  providerRequiresStrictAlternation,
  providerUsesManualToolStream,
  registerProvider,
} from './providers';
import { prepareProviderRequest } from './prepareProviderRequest';
import { toLangChainContent } from '@/messages/langchain';
import { attemptInvoke } from './invoke';
import { initializeModel } from './init';
import { FakeChatModel } from './fake';
import { isOpenAILike } from '@/utils';
import { Providers } from '@/common';

interface HostOptions {
  endpoint: string;
}

declare module '../provider-registration' {
  interface CustomProviderOptionsMap {
    'typed-host-test': HostOptions;
  }
}

class HostProvider extends FakeChatModel {
  readonly config: HostOptions;

  constructor(config: HostOptions) {
    super({ responses: ['ok'] });
    this.config = config;
  }
}

class HostProviderWithoutTools extends CoreFakeChatModel {
  constructor(readonly config: HostOptions) {
    super({});
  }
}

const lookupTool = new DynamicStructuredTool({
  name: 'lookup',
  description: 'Looks up a value',
  schema: z.object({ value: z.string() }),
  func: async ({ value }): Promise<string> => value,
});

const disposers: Array<() => void> = [];

function track(dispose: () => void): void {
  disposers.push(dispose);
}

afterEach(() => {
  for (let i = disposers.length - 1; i >= 0; i--) {
    disposers[i]();
  }
  disposers.length = 0;
});

describe('provider registry', () => {
  it('constructs a typed host provider through the normal model seam', () => {
    track(
      registerProvider({
        provider: 'typed-host-test',
        model: HostProvider,
        family: 'openai',
      })
    );

    const model = initializeModel({
      provider: 'typed-host-test',
      clientOptions: { endpoint: 'https://models.example.test' },
    });

    expect(model).toBeInstanceOf(HostProvider);
    expect((model as HostProvider).config.endpoint).toBe(
      'https://models.example.test'
    );
    expect(getProviderFamily('typed-host-test')).toBe('openai');
    expect(isOpenAILike('typed-host-test')).toBe(true);
  });

  it('applies registered streaming and message-shaping traits', () => {
    const provider = 'trait-host-test';
    track(
      registerProvider({
        provider,
        model: HostProvider,
        manualToolStream: true,
        strictAlternation: true,
      })
    );

    expect(providerUsesManualToolStream(provider)).toBe(true);
    expect(providerRequiresStrictAlternation(provider)).toBe(true);
    const request = prepareProviderRequest({
      model: new HostProvider({ endpoint: 'https://models.example.test' }),
      provider,
      messages: [new HumanMessage('one'), new HumanMessage('two')],
    });
    expect(request.messages).toHaveLength(1);
  });

  it('preserves built-in provider behavior in the same registry', () => {
    expect(getChatModelClass(Providers.OPENAI)).toBeDefined();
    expect(getProviderFamily(Providers.OPENAI)).toBe('openai');
    expect(providerUsesManualToolStream(Providers.ANTHROPIC)).toBe(true);
    expect(providerRequiresStrictAlternation(Providers.BEDROCK)).toBe(true);
  });

  it('shares host registrations with another package module graph', async () => {
    track(
      registerProvider({
        provider: 'isolated-module-host-test',
        model: HostProvider,
      })
    );
    await jest.isolateModulesAsync(async () => {
      const isolatedProviders = await import('./providers');
      expect(
        isolatedProviders.getChatModelClass('isolated-module-host-test')
      ).toBe(HostProvider);
    });
  });

  it('normalizes manual streams with the registered provider family', async () => {
    const cases: Array<{
      provider: string;
      family: 'anthropic' | 'bedrock';
      content: t.MessageContentComplex[];
      expectedType: string;
      expectedLength: number;
    }> = [
      {
        provider: 'anthropic-stream-host-test',
        family: 'anthropic',
        content: [
          {
            type: 'thinking',
            thinking: 'private chain',
            signature: 'signature',
          },
        ],
        expectedType: 'thinking',
        expectedLength: 1,
      },
      {
        provider: 'bedrock-stream-host-test',
        family: 'bedrock',
        content: [
          { type: 'reasoning_content', reasoningText: { text: 'one' } },
          { type: 'reasoning_content', reasoningText: { text: 'two' } },
        ],
        expectedType: 'reasoning_content',
        expectedLength: 1,
      },
    ];

    for (const testCase of cases) {
      track(
        registerProvider({
          provider: testCase.provider,
          model: HostProvider,
          family: testCase.family,
          manualToolStream: true,
        })
      );
      const chunk = new AIMessageChunk({
        content: toLangChainContent(testCase.content),
      });
      const model: t.ChatModel = {
        stream: async (): Promise<AsyncIterable<AIMessageChunk>> => {
          return (async function* (): AsyncGenerator<AIMessageChunk> {
            yield chunk;
          })();
        },
        invoke: async (): Promise<AIMessageChunk> => chunk,
      };

      const result = await attemptInvoke({
        model,
        messages: [new HumanMessage('hello')],
        provider: testCase.provider,
        onChunk: (): void => undefined,
      });
      const content = result.messages?.[0]?.content;

      expect(Array.isArray(content)).toBe(true);
      expect(content).toHaveLength(testCase.expectedLength);
      expect((content?.[0] as t.MessageContentComplex).type).toBe(
        testCase.expectedType
      );
    }
  });

  it('rejects duplicate built-in and host registrations', () => {
    expect(() =>
      registerProvider({
        provider: Providers.OPENAI,
        model: HostProvider,
      })
    ).toThrow('LLM provider already registered: openAI');

    const provider = 'duplicate-host-test';
    track(registerProvider({ provider, model: HostProvider }));
    expect(() => registerProvider({ provider, model: HostProvider })).toThrow(
      `LLM provider already registered: ${provider}`
    );
  });

  it('rejects invalid names and non-constructible registrations', () => {
    expect(() =>
      registerProvider({ provider: '' as t.ProviderName, model: HostProvider })
    ).toThrow('LLM provider name must be a non-empty string');
    expect(() =>
      registerProvider({
        provider: ' invalid-host-test',
        model: HostProvider,
      })
    ).toThrow('LLM provider name must not have surrounding whitespace');
    expect(() =>
      registerProvider({
        provider: 'arrow-host-test',
        model: (() => undefined) as never,
      })
    ).toThrow('LLM provider constructor is invalid: arrow-host-test');
  });

  it('uses map semantics for inherited object property names', () => {
    expect(() => getChatModelClass('constructor')).toThrow(
      'Unsupported LLM provider: constructor'
    );
    expect(() => getChatModelClass('__proto__')).toThrow(
      'Unsupported LLM provider: __proto__'
    );
  });

  it('keeps a newer registration when an old disposer runs again', () => {
    const provider = 'reload-host-test';
    const disposeFirst = registerProvider({ provider, model: HostProvider });
    disposeFirst();
    const disposeSecond = registerProvider({ provider, model: HostProvider });
    track(disposeSecond);

    disposeFirst();

    expect(getChatModelClass(provider)).toBe(HostProvider);
  });

  it('fails clearly when a registered model cannot bind requested tools', () => {
    const provider = 'toolless-host-test';
    track(registerProvider({ provider, model: HostProviderWithoutTools }));

    expect(() =>
      initializeModel({
        provider,
        clientOptions: { endpoint: 'https://models.example.test' },
        tools: [lookupTool],
      })
    ).toThrow(`LLM provider does not support tool binding: ${provider}`);
  });
});
