import { concat } from '@langchain/core/utils/stream';
import { expect, test, describe } from '@jest/globals';
import { HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import type {
  NewTokenIndices,
  HandleLLMNewTokenCallbackFields,
} from '@langchain/core/callbacks/base';
import type { LLMResult } from '@langchain/core/outputs';
import type { OpenAIClient } from '@langchain/openai';
import { ChatOpenAI } from './index';

/**
 * Regression for duplicated stream `response_metadata` (issue #14086).
 *
 * Providers that emit `finish_reason` on more than one streamed chunk (e.g.
 * OpenRouter / `openai/gpt-chat-latest`) made LangChain core's `_mergeDicts`
 * concatenate the scalar fields it stamps alongside finish_reason — producing
 * `finish_reason: "stopstop"` and duplicated `model_name`. That corruption
 * appeared in BOTH aggregations: core's `_streamIterator` (feeds `handleLLMEnd`
 * / the Langfuse GENERATION observation) and any host-side `concat` of the
 * yielded chunks (the graph message / trace output). Both are exercised here.
 */

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;

const MODEL = 'openai/gpt-chat-latest-20260505';

function stubStream(model: ChatOpenAI, chunks: RawChunk[]): void {
  (
    model as unknown as {
      completions: {
        completionWithRetry: (
          request: unknown
        ) => Promise<AsyncIterable<RawChunk>>;
      };
    }
  ).completions.completionWithRetry = async (): Promise<
    AsyncIterable<RawChunk>
  > => ({
    async *[Symbol.asyncIterator](): AsyncGenerator<RawChunk> {
      for (const chunk of chunks) {
        yield chunk;
      }
    },
  });
}

function rawChunk(
  delta: Record<string, unknown>,
  finishReason: 'stop' | null,
  extra: Partial<RawChunk> = {}
): RawChunk {
  return {
    id: 'chatcmpl-1',
    object: 'chat.completion.chunk',
    created: 1,
    model: MODEL,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
        logprobs: null,
      },
    ],
    ...extra,
  } as unknown as RawChunk;
}

async function collect(model: ChatOpenAI): Promise<{
  coreEnd?: AIMessageChunk;
  hostConcat?: AIMessageChunk;
  tokenFinishReasons: unknown[];
}> {
  let coreEnd: AIMessageChunk | undefined;
  const tokenFinishReasons: unknown[] = [];
  const stream = await model.stream([new HumanMessage('hi')], {
    callbacks: [
      {
        handleLLMEnd(output: LLMResult): void {
          const generation = output.generations?.[0]?.[0] as
            | { message?: AIMessageChunk }
            | undefined;
          coreEnd = generation?.message;
        },
        handleLLMNewToken(
          _token: string,
          _idx: NewTokenIndices,
          _runId: string,
          _parentRunId?: string,
          _tags?: string[],
          fields?: HandleLLMNewTokenCallbackFields
        ): void {
          // At emit time the scalars still live in `generationInfo` (core
          // folds them into `response_metadata` only after it receives the
          // chunk), so read the surface a token-callback consumer sees here.
          const chunk = fields?.chunk as
            | {
                generationInfo?: Record<string, unknown>;
                message?: AIMessageChunk;
              }
            | undefined;
          const finishReason =
            chunk?.generationInfo?.finish_reason ??
            chunk?.message?.response_metadata?.finish_reason;
          if (finishReason != null) {
            tokenFinishReasons.push(finishReason);
          }
        },
      },
    ],
  });
  let hostConcat: AIMessageChunk | undefined;
  for await (const chunk of stream) {
    hostConcat = hostConcat == null ? chunk : concat(hostConcat, chunk);
  }
  return { coreEnd, hostConcat, tokenFinishReasons };
}

describe('streamed response_metadata scalar de-duplication', () => {
  test('collapses repeated finish_reason / model_name across two finish chunks', async () => {
    const model = new ChatOpenAI({ model: MODEL, apiKey: 'test-key' });
    stubStream(model, [
      rawChunk({ role: 'assistant', content: 'Hi' }, 'stop'),
      rawChunk({ content: '' }, 'stop', {
        service_tier: 'default',
      } as Partial<RawChunk>),
    ]);

    const { coreEnd, hostConcat, tokenFinishReasons } = await collect(model);

    for (const meta of [
      coreEnd?.response_metadata,
      hostConcat?.response_metadata,
    ]) {
      expect(meta?.finish_reason).toBe('stop');
      expect(meta?.model_name).toBe(MODEL);
      // keep-first still preserves a field that only appears on the 2nd chunk
      expect(meta?.service_tier).toBe('default');
    }
    // token callbacks observe the same de-duplicated chunks (finish_reason
    // stays on the first finish chunk only, never re-emitted on the repeat)
    expect(tokenFinishReasons).toEqual(['stop']);
  });

  test('leaves a conformant single-finish stream untouched', async () => {
    const model = new ChatOpenAI({ model: MODEL, apiKey: 'test-key' });
    stubStream(model, [
      rawChunk({ role: 'assistant', content: 'Hi' }, null),
      rawChunk({ content: '' }, 'stop', {
        system_fingerprint: 'fp_1',
      } as Partial<RawChunk>),
    ]);

    const { coreEnd, hostConcat } = await collect(model);

    for (const meta of [
      coreEnd?.response_metadata,
      hostConcat?.response_metadata,
    ]) {
      expect(meta?.finish_reason).toBe('stop');
      expect(meta?.model_name).toBe(MODEL);
      expect(meta?.system_fingerprint).toBe('fp_1');
    }
  });
});
