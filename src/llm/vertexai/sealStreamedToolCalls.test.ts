import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import { sealCompleteStreamedToolCalls } from './index';

describe('sealCompleteStreamedToolCalls', () => {
  test('stamps an on-arrival seal when every tool-call chunk parsed cleanly', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          id: 'call_1',
          name: 'weather',
          args: '{"city":"NYC"}',
          type: 'tool_call_chunk',
        },
      ],
    });

    sealCompleteStreamedToolCalls(message);

    expect(message.response_metadata).toMatchObject({
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
      [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' },
    });
  });

  test('stamps multi-call chunks when all calls are complete', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          id: 'call_1',
          name: 'weather',
          args: '{"city":"NYC"}',
          type: 'tool_call_chunk',
        },
        {
          id: 'call_2',
          name: 'stock',
          args: '{"ticker":"CH"}',
          type: 'tool_call_chunk',
        },
      ],
    });

    sealCompleteStreamedToolCalls(message);

    expect(
      message.response_metadata[STREAMED_TOOL_CALL_SEAL_METADATA_KEY]
    ).toEqual({ kind: 'all' });
  });

  test('leaves chunks without tool calls unstamped', () => {
    const message = new AIMessageChunk({ content: 'hello' });

    sealCompleteStreamedToolCalls(message);

    expect(
      message.response_metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]
    ).toBeUndefined();
  });

  test('leaves chunks with unparsable tool calls unstamped', () => {
    // No id forces the parse into invalid_tool_calls.
    const message = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          name: 'weather',
          args: '{"city":',
          type: 'tool_call_chunk',
        },
      ],
    });

    sealCompleteStreamedToolCalls(message);

    expect(
      message.response_metadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]
    ).toBeUndefined();
  });
});
