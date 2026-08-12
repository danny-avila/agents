import { expect, test, describe, jest } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatVertexAI } from './index';

describe('Vertex stream smoothing', () => {
  function textOutput(text: string): Record<string, unknown> {
    return {
      candidates: [
        {
          content: { role: 'model', parts: [{ text }] },
          index: 0,
        },
      ],
    };
  }

  async function runStream(
    outputs: Record<string, unknown>[],
    modelFields: Record<string, unknown> = {}
  ): Promise<{
    yielded: ChatGenerationChunk[];
    dispatchedTokens: string[];
  }> {
    const model = new ChatVertexAI({
      model: 'gemini-2.5-flash',
      authOptions: {
        projectId: 'test-project',
        credentials: { client_email: 'test@test', private_key: 'test' },
      },
      ...modelFields,
    });

    let index = 0;
    const fakeStream = {
      get streamDone(): boolean {
        return index > outputs.length;
      },
      async nextChunk(): Promise<unknown> {
        const output = index < outputs.length ? outputs[index] : null;
        index += 1;
        return output;
      },
    };
    (
      model as unknown as {
        streamedConnection: { request: unknown };
      }
    ).streamedConnection.request = jest.fn(async () => ({ data: fakeStream }));

    const dispatchedTokens: string[] = [];
    const runManager = {
      handleCustomEvent: jest.fn(async () => undefined),
      handleLLMNewToken: jest.fn(async (token: string) => {
        dispatchedTokens.push(token);
      }),
    } as unknown as CallbackManagerForLLMRun;

    const yielded: ChatGenerationChunk[] = [];
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage('hi')],
      {} as Parameters<ChatVertexAI['_streamResponseChunks']>[1],
      runManager
    )) {
      yielded.push(chunk);
    }
    return { yielded, dispatchedTokens };
  }

  test('splits large text outputs at stream boundaries with pacing', async () => {
    const { yielded, dispatchedTokens } = await runStream(
      [textOutput('alpha beta gamma')],
      { _lc_stream_delay: 1 }
    );

    const texts = yielded.map((chunk) => chunk.text).filter(Boolean);
    expect(texts).toEqual(['alpha ', 'beta ', 'gamma']);
    expect(dispatchedTokens.filter(Boolean)).toEqual([
      'alpha ',
      'beta ',
      'gamma',
    ]);
  });

  test('passes chunks through unsplit when smoothing is disabled', async () => {
    const { yielded } = await runStream([textOutput('alpha beta gamma')], {
      _lc_stream_delay: 0,
    });

    expect(yielded.map((chunk) => chunk.text).filter(Boolean)).toEqual([
      'alpha beta gamma',
    ]);
  });

  test('defaults to 25ms adaptive smoothing with 0 disabling', () => {
    const base = {
      model: 'gemini-2.5-flash',
      authOptions: {
        projectId: 'test-project',
        credentials: { client_email: 'test@test', private_key: 'test' },
      },
    };
    expect(new ChatVertexAI(base)._lc_stream_delay).toBe(25);
    expect(
      new ChatVertexAI({ ...base, _lc_stream_delay: 0 })._lc_stream_delay
    ).toBe(0);
  });
});
