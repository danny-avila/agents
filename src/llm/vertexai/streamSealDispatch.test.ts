import { expect, test, describe, jest } from '@jest/globals';
import { HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { ChatVertexAI } from './index';

/**
 * Registered stream handlers consume chunks through `handleLLMNewToken`
 * callback events. `@langchain/google-common` yields each chunk BEFORE
 * dispatching that callback, and the generator only resumes (firing the
 * callback) after this package's `_streamResponseChunks` override has
 * stamped the seal on the same message object — so callback consumers must
 * observe sealed chunks. This drives the real google-common stream loop and
 * conversion with a stubbed connection to lock that ordering in.
 */
describe('Vertex stream seal dispatch', () => {
  async function runStream(outputs: unknown[]): Promise<{
    yielded: AIMessageChunk[];
    dispatched: AIMessageChunk[];
  }> {
    const model = new ChatVertexAI({
      model: 'gemini-2.5-flash',
      authOptions: {
        projectId: 'test-project',
        credentials: { client_email: 'test@test', private_key: 'test' },
      },
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

    const dispatched: AIMessageChunk[] = [];
    const runManager = {
      handleCustomEvent: jest.fn(async () => undefined),
      handleLLMNewToken: jest.fn(
        async (
          _token: string,
          _idx?: unknown,
          _runId?: unknown,
          _parentRunId?: unknown,
          _tags?: unknown,
          fields?: { chunk?: ChatGenerationChunk }
        ) => {
          const message = fields?.chunk?.message;
          if (message instanceof AIMessageChunk) {
            dispatched.push(message);
          }
        }
      ),
    } as unknown as CallbackManagerForLLMRun;

    const yielded: AIMessageChunk[] = [];
    for await (const chunk of model._streamResponseChunks(
      [new HumanMessage('hi')],
      {} as Parameters<ChatVertexAI['_streamResponseChunks']>[1],
      runManager
    )) {
      if (chunk.message instanceof AIMessageChunk) {
        yielded.push(chunk.message);
      }
    }
    return { yielded, dispatched };
  }

  test('callback consumers receive function-call chunks already sealed', async () => {
    const { yielded, dispatched } = await runStream([
      {
        candidates: [
          {
            content: {
              role: 'model',
              parts: [
                { functionCall: { name: 'weather', args: { city: 'NYC' } } },
              ],
            },
            index: 0,
          },
        ],
      },
    ]);

    const metadataOf = (m: AIMessageChunk): Record<string, unknown> =>
      m.response_metadata as Record<string, unknown>;

    const yieldedCall = yielded.find(
      (m) => (m.tool_call_chunks?.length ?? 0) > 0
    );
    expect(yieldedCall).toBeDefined();
    expect(
      metadataOf(yieldedCall!)[STREAMED_TOOL_CALL_SEAL_METADATA_KEY]
    ).toEqual({ kind: 'all' });

    const dispatchedCall = dispatched.find(
      (m) => (m.tool_call_chunks?.length ?? 0) > 0
    );
    expect(dispatchedCall).toBeDefined();
    expect(dispatchedCall!.tool_calls?.[0]).toMatchObject({
      name: 'weather',
      args: { city: 'NYC' },
    });
    expect(
      metadataOf(dispatchedCall!)[STREAMED_TOOL_CALL_SEAL_METADATA_KEY]
    ).toEqual({ kind: 'all' });
    expect(
      metadataOf(dispatchedCall!)[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]
    ).toBe(GOOGLE_STREAMED_TOOL_CALL_ADAPTER);
  });

  test('text-only chunks are not sealed on either path', async () => {
    const { yielded, dispatched } = await runStream([
      {
        candidates: [
          {
            content: { role: 'model', parts: [{ text: 'hello' }] },
            index: 0,
          },
        ],
      },
    ]);

    const hasSeal = (m: AIMessageChunk): boolean =>
      (m.response_metadata as Record<string, unknown>)[
        STREAMED_TOOL_CALL_SEAL_METADATA_KEY
      ] != null;

    expect(yielded.some(hasSeal)).toBe(false);
    expect(dispatched.some(hasSeal)).toBe(false);
  });
});
