import { describe, it, expect } from '@jest/globals';
import type { ToolCallChunk } from '@langchain/core/messages/tool';
import type { StreamLimitState } from '@/llm/streamLimits';
import {
  STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY,
  STREAMED_TOOL_CALL_SEAL_METADATA_KEY,
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
} from '@/tools/streamedToolCallSeals';
import {
  DEFAULT_MAX_TOOL_CALL_ARG_BYTES,
  enforceStreamedToolCallArgLimit,
  enforceStreamDeltaEventLimit,
  StreamLimitExceededError,
  resetStreamLimitTallies,
  resolveGenerationKey,
  resolveStreamLimits,
} from '@/llm/streamLimits';

const chunk = (fields: Partial<ToolCallChunk>): ToolCallChunk =>
  ({ type: 'tool_call_chunk', ...fields }) as ToolCallChunk;

const generation = (step: number): Record<string, unknown> => ({
  langgraph_checkpoint_ns: '',
  langgraph_node: 'agent',
  langgraph_step: step,
});

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

describe('resolveGenerationKey', () => {
  it('derives a stable key from langgraph node-execution metadata', () => {
    expect(resolveGenerationKey(generation(3))).toBe('|agent|3');
    expect(resolveGenerationKey(generation(3))).toBe(resolveGenerationKey(generation(3)));
  });

  it('separates supersteps, nodes, and namespaces', () => {
    expect(resolveGenerationKey(generation(3))).not.toBe(resolveGenerationKey(generation(4)));
    expect(
      resolveGenerationKey({ ...generation(3), langgraph_node: 'other_agent' })
    ).not.toBe(resolveGenerationKey(generation(3)));
    expect(
      resolveGenerationKey({ ...generation(3), langgraph_checkpoint_ns: 'child' })
    ).not.toBe(resolveGenerationKey(generation(3)));
  });

  it('degrades to a shared bucket when metadata is absent', () => {
    expect(resolveGenerationKey(undefined)).toBe('');
    expect(resolveGenerationKey({})).toBe('||');
  });
});

describe('enforceStreamedToolCallArgLimit', () => {
  it('accumulates across chunk events and throws past the limit', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 10 }),
    };
    const metadata = generation(1);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ name: 'db_query', id: 'call_1', args: '', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: '12345', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: '67890', index: 0 })],
    });
    let caught: unknown;
    try {
      enforceStreamedToolCallArgLimit({
        graph,
        metadata,
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
        metadata: generation(1),
        toolCallChunks: [chunk({ args: '€€', index: 0 })],
      })
    ).toThrow(StreamLimitExceededError);
    expect(graph.streamedToolCallArgTallies?.get('|agent|1:0')?.bytes).toBe(6);
  });

  it('tallies parallel tool calls and generations independently', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 8 }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(1),
      toolCallChunks: [
        chunk({ name: 'first', args: '123456', index: 0 }),
        chunk({ name: 'second', args: '123456', index: 1 }),
      ],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(2),
      toolCallChunks: [chunk({ args: '123456', index: 0 })],
    });
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        metadata: generation(1),
        toolCallChunks: [chunk({ args: '789', index: 1 })],
      })
    ).toThrow('(tool call: second)');
  });

  it('falls back to the chunk id when index is absent', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 8 }),
    };
    const metadata = generation(1);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ id: 'call_a', name: 'a_tool', args: '123456' })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ id: 'call_b', name: 'b_tool', args: '123456' })],
    });
    expect(graph.streamedToolCallArgTallies?.get('|agent|1:call_a')?.bytes).toBe(6);
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        metadata,
        toolCallChunks: [chunk({ id: 'call_b', args: '789' })],
      })
    ).toThrow('(tool call: b_tool)');
  });

  it('keys anonymous chunks by batch position so parallel calls stay separate', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 8 }),
    };
    const metadata = generation(1);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [
        chunk({ name: 'anon_a', args: '123456' }),
        chunk({ name: 'anon_b', args: '123456' }),
      ],
    });
    expect(graph.streamedToolCallArgTallies?.get('|agent|1:#0')?.bytes).toBe(6);
    expect(graph.streamedToolCallArgTallies?.get('|agent|1:#1')?.bytes).toBe(6);
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        metadata,
        toolCallChunks: [chunk({ args: '789' })],
      })
    ).toThrow(StreamLimitExceededError);
  });

  it('checks arrival-sealed complete calls standalone, keeping no tally', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 64 }),
    };
    const sealAll = { [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'all' } };
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(1),
      toolCallChunks: [
        chunk({ args: 'x'.repeat(40) }),
        chunk({ args: 'y'.repeat(40) }),
      ],
      responseMetadata: sealAll,
    });
    expect(graph.streamedToolCallArgTallies?.size).toBe(0);
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        metadata: generation(1),
        toolCallChunks: [chunk({ name: 'big_call', args: 'z'.repeat(70) })],
        responseMetadata: sealAll,
      })
    ).toThrow('(tool call: big_call)');
  });

  it('replaces the tally when a single-seal chunk restates complete args (OpenAI Responses)', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
    };
    const metadata = generation(1);
    const fullArgs = 'x'.repeat(40);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: fullArgs.slice(0, 20), index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: fullArgs.slice(20), index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ name: 'writer', args: fullArgs, index: 0 })],
      responseMetadata: {
        [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
          OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 0 },
      },
    });
    expect(graph.streamedToolCallArgTallies?.has('|agent|1:0')).toBe(false);
  });

  it('releases the tally on a Bedrock Converse stop chunk without counting it', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
    };
    const metadata = generation(1);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: '1234567890', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [chunk({ args: '', index: 0 })],
      responseMetadata: {
        [STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY]:
          BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
        [STREAMED_TOOL_CALL_SEAL_METADATA_KEY]: { kind: 'single', index: 0 },
      },
    });
    expect(graph.streamedToolCallArgTallies?.has('|agent|1:0')).toBe(false);
  });

  it('resets one generation on a new model attempt, keeping other generations', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({
        maxToolCallArgBytes: 10,
        maxDeltaEventsPerTurn: 5,
      }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(1),
      toolCallChunks: [chunk({ args: '12345678', index: 0 })],
    });
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(2),
      toolCallChunks: [chunk({ args: '1234', index: 0 })],
    });
    enforceStreamDeltaEventLimit({ graph, metadata: generation(1) });

    resetStreamLimitTallies({ graph, metadata: generation(1) });

    expect(graph.streamedToolCallArgTallies?.has('|agent|1:0')).toBe(false);
    expect(graph.streamedToolCallArgTallies?.get('|agent|2:0')?.bytes).toBe(4);
    expect(graph.streamDeltaEventCounts?.has('|agent|1')).toBe(false);

    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(1),
      toolCallChunks: [chunk({ args: '12345678', index: 0 })],
    });
    expect(graph.streamedToolCallArgTallies?.get('|agent|1:0')?.bytes).toBe(8);
  });

  it('reset is a no-op when nothing was counted', () => {
    const graph: StreamLimitState = {};
    resetStreamLimitTallies({ graph, metadata: generation(1) });
    resetStreamLimitTallies({ graph: undefined, metadata: generation(1) });
    expect(graph.streamedToolCallArgTallies).toBeUndefined();
    expect(graph.streamDeltaEventCounts).toBeUndefined();
  });

  it('does nothing when disabled', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 0 }),
    };
    enforceStreamedToolCallArgLimit({
      graph,
      metadata: generation(1),
      toolCallChunks: [chunk({ args: 'x'.repeat(1_000_000), index: 0 })],
    });
    expect(graph.streamedToolCallArgTallies).toBeUndefined();
  });

  it('applies the default limit when graph state was never resolved', () => {
    const graph: StreamLimitState = {};
    const metadata = generation(1);
    enforceStreamedToolCallArgLimit({
      graph,
      metadata,
      toolCallChunks: [
        chunk({ args: 'x'.repeat(DEFAULT_MAX_TOOL_CALL_ARG_BYTES), index: 0 }),
      ],
    });
    expect(() =>
      enforceStreamedToolCallArgLimit({
        graph,
        metadata,
        toolCallChunks: [chunk({ args: 'x', index: 0 })],
      })
    ).toThrow(StreamLimitExceededError);
  });
});

describe('enforceStreamDeltaEventLimit', () => {
  it('throws once a single generation exceeds the event limit', () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 3 }),
    };
    const metadata = generation(1);
    enforceStreamDeltaEventLimit({ graph, metadata });
    enforceStreamDeltaEventLimit({ graph, metadata });
    enforceStreamDeltaEventLimit({ graph, metadata });
    let caught: unknown;
    try {
      enforceStreamDeltaEventLimit({ graph, metadata });
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(StreamLimitExceededError);
    const limitError = caught as StreamLimitExceededError;
    expect(limitError.kind).toBe('delta_events');
    expect(limitError.limit).toBe(3);
    expect(limitError.observed).toBe(4);
    expect(limitError.message).toContain('maxDeltaEventsPerTurn');
    enforceStreamDeltaEventLimit({ graph, metadata: generation(2) });
    expect(graph.streamDeltaEventCounts?.get('|agent|2')).toBe(1);
  });

  it('is disabled by default and allocates nothing', () => {
    const graph: StreamLimitState = {};
    for (let i = 0; i < 10; i++) {
      enforceStreamDeltaEventLimit({ graph, metadata: generation(1) });
    }
    expect(graph.streamDeltaEventCounts).toBeUndefined();
  });
});
