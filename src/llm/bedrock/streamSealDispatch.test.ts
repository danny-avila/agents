import { expect, test, describe, jest } from '@jest/globals';
import { HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { CustomChatBedrockConverse } from './index';

/**
 * Registered stream handlers consume chunks through `handleLLMNewToken`
 * callback events, not the yielded generator (`attemptInvoke` skips manual
 * dispatch when a handler is registered). These tests drive the Converse
 * stream loop with a stubbed client and assert that toolUse start and stop
 * seal chunks reach BOTH paths.
 */
describe('Converse stream seal dispatch', () => {
  async function runStream(
    events: Array<Record<string, unknown>>,
    modelFields: Record<string, unknown> = {}
  ): Promise<{
    yielded: AIMessageChunk[];
    dispatched: AIMessageChunk[];
  }> {
    const model = new CustomChatBedrockConverse({
      model: 'anthropic.claude-3-5-sonnet-20240620-v1:0',
      region: 'us-east-1',
      credentials: { accessKeyId: 'test', secretAccessKey: 'test' },
      ...modelFields,
    });

    (model as unknown as { client: { send: unknown } }).client.send = jest.fn(
      async () => ({
        stream: (async function* () {
          yield* events;
        })(),
      })
    );

    const dispatched: AIMessageChunk[] = [];
    const runManager = {
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
      {} as Parameters<CustomChatBedrockConverse['_streamResponseChunks']>[1],
      runManager
    )) {
      if (chunk.message instanceof AIMessageChunk) {
        yielded.push(chunk.message);
      }
    }
    return { yielded, dispatched };
  }

  const toolUseEvents = [
    {
      contentBlockStart: {
        contentBlockIndex: 1,
        start: { toolUse: { toolUseId: 'call_1', name: 'weather' } },
      },
    },
    {
      contentBlockDelta: {
        contentBlockIndex: 1,
        delta: { toolUse: { input: '{"city":"NYC"}' } },
      },
    },
    { contentBlockStop: { contentBlockIndex: 1 } },
  ];

  test('dispatches toolUse start and stop seal chunks to callbacks', async () => {
    const { yielded, dispatched } = await runStream(toolUseEvents);

    const sealOf = (m: AIMessageChunk): unknown =>
      (m.response_metadata as Record<string, unknown>)[
        STREAMED_TOOL_CALL_SEAL_METADATA_KEY
      ];

    expect(yielded.some((m) => sealOf(m) != null)).toBe(true);

    expect(dispatched).toHaveLength(3);
    expect(dispatched[0].tool_call_chunks).toMatchObject([
      { id: 'call_1', name: 'weather', index: 1 },
    ]);
    expect(dispatched[1].tool_call_chunks).toMatchObject([
      { args: '{"city":"NYC"}', index: 1 },
    ]);
    expect(sealOf(dispatched[2])).toEqual({ kind: 'single', index: 1 });
    expect(
      (dispatched[2].response_metadata as Record<string, unknown>)[
        STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY
      ]
    ).toBe(BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER);
  });

  test('does not emit seal chunks when guardrails are configured', async () => {
    const { yielded, dispatched } = await runStream(toolUseEvents, {
      guardrailConfig: {
        guardrailIdentifier: 'guardrail_1',
        guardrailVersion: '1',
      },
    });

    const hasSeal = (m: AIMessageChunk): boolean =>
      (m.response_metadata as Record<string, unknown>)[
        STREAMED_TOOL_CALL_SEAL_METADATA_KEY
      ] != null;

    // Guardrails can reject the turn at messageStop after contentBlockStop,
    // so no eager seal may be emitted — but tool chunks still stream.
    expect(yielded.some(hasSeal)).toBe(false);
    expect(dispatched.some(hasSeal)).toBe(false);
    expect(dispatched).toHaveLength(2);
    expect(dispatched[0].tool_call_chunks).toMatchObject([
      { id: 'call_1', name: 'weather', index: 1 },
    ]);
  });

  test('does not emit seal chunks for non-toolUse block stops', async () => {
    const { yielded, dispatched } = await runStream([
      {
        contentBlockDelta: {
          contentBlockIndex: 0,
          delta: { text: 'hello' },
        },
      },
      { contentBlockStop: { contentBlockIndex: 0 } },
    ]);

    const hasSeal = (m: AIMessageChunk): boolean =>
      (m.response_metadata as Record<string, unknown>)[
        STREAMED_TOOL_CALL_SEAL_METADATA_KEY
      ] != null;

    expect(yielded.some(hasSeal)).toBe(false);
    expect(dispatched.some(hasSeal)).toBe(false);
    expect(dispatched).toHaveLength(1);
  });
});
