import { describe, expect, it } from '@jest/globals';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import {
  OutputTruncationError,
  assertNotTruncatedToolCall,
  getTruncationStopReason,
} from '@/llm/truncation';
import { Providers } from '@/common';

describe('getTruncationStopReason', () => {
  it('returns null when there is no response metadata', () => {
    expect(getTruncationStopReason(new AIMessage('hi'))).toBeNull();
    expect(getTruncationStopReason(undefined)).toBeNull();
  });

  it('detects Bedrock Converse streaming shape (messageStop.stopReason)', () => {
    const message = new AIMessageChunk({
      content: '',
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects Bedrock Converse non-streaming shape (stopReason)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { stopReason: 'max_tokens' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects Anthropic shape (stop_reason)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { stop_reason: 'max_tokens' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('detects OpenAI shape (finish_reason: length)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { finish_reason: 'length' },
    });
    expect(getTruncationStopReason(message)).toBe('length');
  });

  it('detects Google shape (finishReason: MAX_TOKENS, case-insensitive)', () => {
    const message = new AIMessage({
      content: '',
      response_metadata: { finishReason: 'MAX_TOKENS' },
    });
    expect(getTruncationStopReason(message)).toBe('max_tokens');
  });

  it('returns null for normal stop reasons', () => {
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { stopReason: 'end_turn' },
        })
      )
    ).toBeNull();
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { finish_reason: 'stop' },
        })
      )
    ).toBeNull();
    expect(
      getTruncationStopReason(
        new AIMessage({
          content: 'x',
          response_metadata: { stopReason: 'tool_use' },
        })
      )
    ).toBeNull();
  });
});

describe('assertNotTruncatedToolCall', () => {
  it('no-ops for a normal completion', () => {
    expect(() =>
      assertNotTruncatedToolCall(
        new AIMessage({
          content: 'done',
          response_metadata: { stopReason: 'end_turn' },
        })
      )
    ).not.toThrow();
  });

  it('no-ops for a truncated plain-text turn (no tool calls)', () => {
    expect(() =>
      assertNotTruncatedToolCall(
        new AIMessage({
          content: 'partial...',
          response_metadata: { stopReason: 'max_tokens' },
        })
      )
    ).not.toThrow();
  });

  it('no-ops for a complete tool call (normal stop reason)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: '1', name: 'create_file', args: { path: 'a', content: 'b' } },
      ],
      response_metadata: { stopReason: 'tool_use' },
    });
    expect(() => assertNotTruncatedToolCall(message)).not.toThrow();
  });

  it('throws when a tool call is present and the turn was truncated', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(() => assertNotTruncatedToolCall(message)).toThrow(
      OutputTruncationError
    );
    try {
      assertNotTruncatedToolCall(message);
    } catch (err) {
      expect(err).toBeInstanceOf(OutputTruncationError);
      expect((err as OutputTruncationError).stopReason).toBe('max_tokens');
      expect((err as OutputTruncationError).toolCallNames).toContain(
        'create_file'
      );
      expect((err as OutputTruncationError).message).toMatch(
        /max output tokens/i
      );
    }
  });

  it('throws when only incomplete tool_call_chunks survived truncation', () => {
    const message = new AIMessageChunk({
      content: '',
      tool_call_chunks: [
        {
          name: 'create_file',
          args: '{"path":"a"',
          index: 0,
          type: 'tool_call_chunk',
        },
      ],
      response_metadata: { messageStop: { stopReason: 'max_tokens' } },
    });
    expect(() => assertNotTruncatedToolCall(message)).toThrow(
      OutputTruncationError
    );
  });

  it('still throws for a streaming-arg provider (Anthropic)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [{ id: '1', name: 'create_file', args: { path: 'a' } }],
      response_metadata: { stop_reason: 'max_tokens' },
    });
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.ANTHROPIC)
    ).toThrow(OutputTruncationError);
  });

  it('does not throw for providers that deliver complete tool calls (Google/Vertex)', () => {
    const message = new AIMessage({
      content: '',
      tool_calls: [
        { id: '1', name: 'create_file', args: { path: 'a', content: 'b' } },
      ],
      response_metadata: { finishReason: 'MAX_TOKENS' },
    });
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.GOOGLE)
    ).not.toThrow();
    expect(() =>
      assertNotTruncatedToolCall(message, Providers.VERTEXAI)
    ).not.toThrow();
  });
});
