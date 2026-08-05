import { expect, test, describe, jest, afterEach } from '@jest/globals';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatMistralAI } from '@langchain/mistralai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { CustomChatMistralAI } from './index';

describe('Mistral stream smoothing', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  function stubParentStream(texts: string[]): void {
    jest
      .spyOn(
        ChatMistralAI.prototype as unknown as {
          _streamResponseChunks: (
            ...args: unknown[]
          ) => AsyncGenerator<ChatGenerationChunk>;
        },
        '_streamResponseChunks'
      )
      .mockImplementation(async function* () {
        for (const text of texts) {
          yield new ChatGenerationChunk({
            text,
            message: new AIMessageChunk({ content: text }),
          });
        }
      });
  }

  async function collect(
    model: CustomChatMistralAI,
    runManager?: CallbackManagerForLLMRun
  ): Promise<string[]> {
    const texts: string[] = [];
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage('hi')],
      {} as Parameters<CustomChatMistralAI['_streamResponseChunks']>[1],
      runManager
    )) {
      if (chunk.text) {
        texts.push(chunk.text);
      }
    }
    return texts;
  }

  test('splits large text chunks at stream boundaries with pacing', async () => {
    stubParentStream(['alpha beta gamma']);
    const model = new CustomChatMistralAI({
      model: 'mistral-large-latest',
      apiKey: 'test-key',
      _lc_stream_delay: 1,
    });

    const dispatchedTokens: string[] = [];
    const runManager = {
      handleLLMNewToken: jest.fn(async (token: string) => {
        dispatchedTokens.push(token);
      }),
    } as unknown as CallbackManagerForLLMRun;

    expect(await collect(model, runManager)).toEqual([
      'alpha ',
      'beta ',
      'gamma',
    ]);
    expect(dispatchedTokens.filter(Boolean)).toEqual([
      'alpha ',
      'beta ',
      'gamma',
    ]);
  });

  test('passes chunks through unsplit when smoothing is disabled', async () => {
    stubParentStream(['alpha beta gamma']);
    const model = new CustomChatMistralAI({
      model: 'mistral-large-latest',
      apiKey: 'test-key',
      _lc_stream_delay: 0,
    });

    expect(await collect(model)).toEqual(['alpha beta gamma']);
  });

  test('defaults to 25ms adaptive smoothing with 0 disabling', () => {
    const base = { model: 'mistral-large-latest', apiKey: 'test-key' };
    expect(new CustomChatMistralAI(base)._lc_stream_delay).toBe(25);
    expect(
      new CustomChatMistralAI({ ...base, _lc_stream_delay: 0 })
        ._lc_stream_delay
    ).toBe(0);
    expect(CustomChatMistralAI.lc_name()).toBe('LibreChatMistralAI');
  });
});
