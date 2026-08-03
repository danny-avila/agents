import { describe, it, expect } from '@jest/globals';
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type { StreamLimitState } from '@/llm/streamLimits';
import {
  DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
  enforceStreamedToolCallArgLimit,
  enforceStreamDeltaEventLimit,
  StreamLimitExceededError,
  resolveStreamLimits,
} from '@/llm/streamLimits';

const chunk = (fields: Partial<ToolCallChunk>): ToolCallChunk =>
  ({ type: 'tool_call_chunk', ...fields }) as ToolCallChunk;

describe('resolveStreamLimits', () => {
  it('defaults the byte cap ON and the event cap OFF', () => {
    expect(resolveStreamLimits()).toEqual({
      maxToolCallArgBytes: DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
      maxDeltaEventsPerTurn: 0,
    });
    expect(resolveStreamLimits({})).toEqual({
      maxToolCallArgBytes: DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
      maxDeltaEventsPerTurn: 0,
    });
  });

  it('honors explicit values and floors fractions', () => {
    expect(
      resolveStreamLimits({ maxToolCallArgBytes: 100.9, maxDeltaEventsPerTurn: 5 })
    ).toEqual({ maxToolCallArgBytes: 100, maxDeltaEventsPerTurn: 5 });
  });

  it('treats 0, negatives, and Infinity as disabled', () => {
    expect(resolveStreamLimits({ maxToolCallArgBytes: 0 }).maxToolCallArgBytes).toBe(0);
    expect(resolveStreamLimits({ maxToolCallArgBytes: -5 }).maxToolCallArgBytes).toBe(0);
    expect(
      resolveStreamLimits({ maxToolCallArgBytes: Infinity }).maxToolCallArgBytes
    ).toBe(0);
    expect(
      resolveStreamLimits({ maxDeltaEventsPerTurn: -1 }).maxDeltaEventsPerTurn
    ).toBe(0);
  });

  it('falls back to the default on NaN', () => {
    expect(resolveStreamLimits({ maxToolCallArgBytes: NaN }).maxToolCallArgBytes).toBe(
      DEFAULT_MAX_TOOL_CALL_ARG_BYTES
    );
  });
});

describe('enforceStreamedToolCallArgLimit', () => {
  it('accumulates across chunk events and throws past the limit', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 10 }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'step',
      toolCallChunks: [chunk({ name: 'db_query', id: 'call_1', args: '', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'step',
      toolCallChunks: [chunk({ args: '12345', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'step',
      toolCallChunks: [chunk({ args: '67890', index: 0 })],
    });
    let caught: unknown;
    try {
      enforceStreamedToolCallArgLimit({
        graph,
        stepKey: 'step',
        toolCallChunks: [chunk({ args: '!', index: 0 })],
      });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(StreamLimitExceededError);
    const limitError = caught as StreamLimitExceededError;
    expect(limitError.kind).toBe('tool_call_args');
    expect(limitError.limit).toBe(10);
    expect(limitError.observed).toBe(11);
    expect(limitError.toolName).toBe('db_query');
    expect(limitError.message).toContain('10-byte');
    expect(limitError.message).toContain('(tool call: db_query)');
    expect(limitError.message).toContain('maxToolCallArgBytes');
  });

  it('counts UTF-8 bytes, not string length', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 5 }),
    };
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        stepKey: 'step',
        toolCallChunks: [chunk({ args: '€€', index: 0 })],
      })
    ).toThrow(StreamLimitExceededError);
    expect(graph.streamedToolCallArgTallies?.get('step:0')?.bytes).toBe(6);
  });

  it('tallies parallel tool calls and turns independently', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 8 }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'turn-1',
      toolCallChunks: [
        chunk({ name: 'first', args: '123456', index: 0 }),
        chunk({ name: 'second', args: '123456', index: 1 }),
      ],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'turn-2',
      toolCallChunks: [chunk({ args: '123456', index: 0 })],
    });
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        stepKey: 'turn-1',
        toolCallChunks: [chunk({ args: '789', index: 1 })],
      })
    ).toThrow('(tool call: second)');
  });

  it('does nothing when disabled', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 0 }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'step',
      toolCallChunks: [chunk({ args: 'x'.repeat(1_000_000), index: 0 })],
    });
    expect(graph.streamedToolCallArgTallies).toBeUndefined();
  });

  it('applies the default limit when graph state was never resolved', () => {
    const graph: StreamLimitState = {};
    enforceStreamedToolCallArgLimit({
      graph,
      stepKey: 'step',
      toolCallChunks: [
        chunk({ args: 'x'.repeat(DEFAULT_MAX_TOOL_CALL_ARG_BYTES), index: 0 }),
      ],
    });
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        stepKey: 'step',
        toolCallChunks: [chunk({ args: 'x', index: 0 })],
      })
    ).toThrow(StreamLimitExceededError);
  });
});

describe('enforceStreamDeltaEventLimit', () => {
  it('throws once a single turn exceeds the event limit', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 3 }),
    };
    enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-1' });
    enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-1' });
    enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-1' });
    let caught: unknown;
    try {
      enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-1' });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(StreamLimitExceededError);
    const limitError = caught as StreamLimitExceededError;
    expect(limitError.kind).toBe('delta_events');
    expect(limitError.limit).toBe(3);
    expect(limitError.observed).toBe(4);
    expect(limitError.message).toContain('maxDeltaEventsPerTurn');
    enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-2' });
    expect(graph.streamDeltaEventCounts?.get('turn-2')).toBe(1);
  });

  it('is disabled by default and allocates nothing', () => {
    const graph: StreamLimitState = {};
    for (let i = 0; i < 10; i++) {
      enforceStreamDeltaEventLimit({ graph, stepKey: 'turn-1' });
    }
    expect(graph.streamDeltaEventCounts).toBeUndefined();
  });
});
