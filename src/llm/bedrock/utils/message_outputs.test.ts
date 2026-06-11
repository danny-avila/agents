import { expect, test, describe } from '@jest/globals';
import { AIMessageChunk } from '@langchain/core/messages';
import type { ContentBlockDeltaEvent, ContentBlockStartEvent } from '../types';
import {
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  createConverseToolUseStopChunk,
  handleConverseStreamContentBlockStart,
  handleConverseStreamContentBlockDelta,
} from './message_outputs';

function asAIMessageChunk(message: unknown): AIMessageChunk {
  expect(message).toBeInstanceOf(AIMessageChunk);
  return message as AIMessageChunk;
}

describe('Converse streamed tool-call seal metadata', () => {
  test('stamps the adapter on toolUse content block starts', () => {
    const chunk = handleConverseStreamContentBlockStart({
      contentBlockIndex: 1,
      start: {
        toolUse: { toolUseId: 'call_1', name: 'weather' },
      },
    } as ContentBlockStartEvent);

    const message = asAIMessageChunk(chunk?.message);
    expect(message.response_metadata).toMatchObject({
      contentBlockIndex: 1,
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
    });
    expect(message.tool_call_chunks).toEqual([
      {
        id: 'call_1',
        name: 'weather',
        index: 1,
        type: 'tool_call_chunk',
      },
    ]);
  });

  test('stamps the adapter on toolUse deltas but not text deltas', () => {
    const toolChunk = handleConverseStreamContentBlockDelta({
      contentBlockIndex: 1,
      delta: { toolUse: { input: '{"city":' } },
    } as ContentBlockDeltaEvent);
    const toolMetadata = asAIMessageChunk(toolChunk.message)
      .response_metadata as Record<string, unknown>;
    expect(toolMetadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]).toBe(
      BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER
    );

    const textChunk = handleConverseStreamContentBlockDelta({
      contentBlockIndex: 0,
      delta: { text: 'hello' },
    } as ContentBlockDeltaEvent);
    const textMetadata = asAIMessageChunk(textChunk.message)
      .response_metadata as Record<string, unknown>;
    expect(
      textMetadata[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]
    ).toBeUndefined();
  });

  test('builds an explicit single seal chunk for a stopped toolUse block', () => {
    const chunk = createConverseToolUseStopChunk(2);

    const message = asAIMessageChunk(chunk.message);
    expect(message.response_metadata).toEqual({
      [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
        BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
      [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 2 },
    });
    expect(message.tool_call_chunks).toEqual([
      {
        args: '',
        index: 2,
        type: 'tool_call_chunk',
      },
    ]);
    expect(message.content).toBe('');
  });
});
