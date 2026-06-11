import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import type { EnhancedGenerateContentResponse } from '@google/generative-ai';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { convertResponseContentToChatGenerationChunk } from './common';

function buildResponse(
  parts: Array<Record<string, unknown>>
): EnhancedGenerateContentResponse {
  return {
    candidates: [
      {
        content: { role: 'model', parts },
        index: 0,
      },
    ],
  } as unknown as EnhancedGenerateContentResponse;
}

function asAIMessageChunk(message: unknown): AIMessageChunk {
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

describe('convertResponseContentToChatGenerationChunk seal metadata', () => {
  test('stamps an on-arrival seal on function call chunks', () => {
    const chunk = convertResponseContentToChatGenerationChunk(
      buildResponse([
        {
          functionCall: { name: 'weather', args: { city: 'NYC' } },
        },
      ]),
      { usageMetadata: undefined, index: 0 }
    );

    const message = asAIMessageChunk(chunk?.message);
    expect(message.response_metadata).toMatchObject({
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
      [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
    });
    expect(message.tool_call_chunks).toHaveLength(1);
    expect(message.tool_calls?.[0]).toMatchObject({
      name: 'weather',
      args: { city: 'NYC' },
    });
  });

  test('does not stamp seal metadata on text-only chunks', () => {
    const chunk = convertResponseContentToChatGenerationChunk(
      buildResponse([{ text: 'hello' }]),
      { usageMetadata: undefined, index: 0 }
    );

    const metadata = asAIMessageChunk(chunk?.message)
      .response_metadata as Record<string, unknown>;
    expect(metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]).toBeUndefined();
    expect(metadata[STREAMED_TOOL_CALL_SEAL_METADATA_KEY]).toBeUndefined();
  });
});
