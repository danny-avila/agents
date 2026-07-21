import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import type { Content, EnhancedGenerateContentResponse } from '@google/generative-ai';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  convertResponseContentToChatGenerationChunk,
  dropUnsupportedModelTurnPrefill,
  rejectsModelTurnPrefill,
} from './common';

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

describe('rejectsModelTurnPrefill', () => {
  test('is true for models that reject a trailing model turn', () => {
    expect(rejectsModelTurnPrefill('gemini-3.6-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('gemini-3.5-flash-lite')).toBe(true);
    expect(rejectsModelTurnPrefill('models/gemini-3.6-flash')).toBe(true);
    expect(rejectsModelTurnPrefill('google/gemini-3.5-flash-lite-latest')).toBe(true);
  });

  test('is false for models that still accept prefill and for empty input', () => {
    expect(rejectsModelTurnPrefill('gemini-3.5-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-2.5-flash')).toBe(false);
    expect(rejectsModelTurnPrefill('gemini-3-pro-preview')).toBe(false);
    expect(rejectsModelTurnPrefill(undefined)).toBe(false);
    expect(rejectsModelTurnPrefill('')).toBe(false);
  });
});

describe('dropUnsupportedModelTurnPrefill', () => {
  const userTurn: Content = { role: 'user', parts: [{ text: 'Hi' }] };
  const modelTurn: Content = { role: 'model', parts: [{ text: 'Hello, I am' }] };

  test('drops a trailing model turn for no-prefill models', () => {
    const contents: Content[] = [userTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(contents, 'gemini-3.6-flash');
    expect(result).toEqual([userTurn]);
  });

  test('drops multiple consecutive trailing model turns but keeps one turn', () => {
    const contents: Content[] = [userTurn, modelTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(contents, 'gemini-3.5-flash-lite');
    expect(result).toEqual([userTurn]);
  });

  test('leaves a trailing model turn for models that accept prefill', () => {
    const contents: Content[] = [userTurn, modelTurn];
    const result = dropUnsupportedModelTurnPrefill(contents, 'gemini-3.5-flash');
    expect(result).toBe(contents);
  });

  test('is a no-op when the request already ends with a user turn', () => {
    const contents: Content[] = [modelTurn, userTurn];
    const result = dropUnsupportedModelTurnPrefill(contents, 'gemini-3.6-flash');
    expect(result).toBe(contents);
  });

  test('is a no-op for empty or undefined contents', () => {
    expect(dropUnsupportedModelTurnPrefill([], 'gemini-3.6-flash')).toEqual([]);
    expect(dropUnsupportedModelTurnPrefill(undefined, 'gemini-3.6-flash')).toBeUndefined();
  });
});
