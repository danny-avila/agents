/**
 * Summarization streams replace `ChatModelStreamHandler` with an `onChunk`
 * callback, so the stream circuit breakers are enforced inside
 * `createSummarizationChunkHandler`: the per-generation event cap, and the
 * argument byte cap for tool calls a tool-bound summarization model may
 * emit. Keys derive from the attempt metadata `attemptInvoke` hands each
 * chunk, so a fallback summary attempt never inherits the failed primary's
 * counts.
 */
import { AIMessageChunk } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { StreamLimitState } from '@/llm/streamLimits';
import {
  STREAM_LIMIT_ATTEMPT_KEY,
  StreamLimitExceededError,
  enforceStreamLimitsForWireChunk,
  resolveStreamLimits,
} from '@/llm/streamLimits';
import { createSummarizationChunkHandler } from '@/summarization/node';
import { Providers } from '@/common';

const makeHandler = (graph: StreamLimitState) =>
  createSummarizationChunkHandler({
    stepId: 'step_1',
    config: { metadata: { langgraph_node: 'summarize', langgraph_step: 4 } },
    provider: Providers.OPENAI,
    reasoningKey: 'reasoning_content',
    graph,
  });

const textChunk = (): AIMessageChunk => new AIMessageChunk({ content: 'part' });

describe('createSummarizationChunkHandler stream limits', () => {
  it('enforces the event cap on summary streams', async () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
    };
    const onChunk = makeHandler(graph);
    expect(onChunk).toBeDefined();
    await onChunk!(textChunk());
    await onChunk!(textChunk());
    await expect(async () => onChunk!(textChunk())).rejects.toThrow(
      StreamLimitExceededError
    );
  });

  it('keys each summary attempt by the metadata attemptInvoke hands the chunk', async () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
    };
    const onChunk = makeHandler(graph);
    const primary = {
      langgraph_node: 'summarize',
      langgraph_step: 4,
      [STREAM_LIMIT_ATTEMPT_KEY]: 1,
    };
    const fallback = { ...primary, [STREAM_LIMIT_ATTEMPT_KEY]: 2 };
    await onChunk!(textChunk(), primary);
    await onChunk!(textChunk(), primary);
    await onChunk!(textChunk(), fallback);
    await onChunk!(textChunk(), fallback);
    await expect(async () => onChunk!(textChunk(), fallback)).rejects.toThrow(
      StreamLimitExceededError
    );
  });

  it('enforces the argument byte cap on tool calls from tool-bound summary models', async () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
    };
    const onChunk = makeHandler(graph);
    const toolChunk = (args: string): AIMessageChunk =>
      new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          { type: 'tool_call_chunk', name: 'db_query', args, index: 0 },
        ],
      });
    await onChunk!(toolChunk('x'.repeat(40)));
    await expect(async () => onChunk!(toolChunk('x'.repeat(20)))).rejects.toThrow(
      '(tool call: db_query)'
    );
  });

  it('trips the run breaker when its own enforcement breaches', async () => {
    const breaker = new AbortController();
    const graph: StreamLimitState & {
      getBreakerController?: () => AbortController;
    } = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 50 }),
      getBreakerController: () => breaker,
    };
    const onChunk = createSummarizationChunkHandler({
      stepId: 'step_1',
      config: { metadata: { langgraph_node: 'summarize', langgraph_step: 4 } },
      provider: Providers.OPENAI,
      reasoningKey: 'reasoning_content',
      graph,
    });
    const toolChunk = (args: string): AIMessageChunk =>
      new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          { type: 'tool_call_chunk', name: 'db_query', args, index: 0 },
        ],
      });

    /** This producer claim wins the race, so the wire consumer's
     * breaker-aborting catch never fires for the same chunk — the trip has
     * to happen here or sibling branches keep consuming quota. */
    await expect(async () => onChunk!(toolChunk('x'.repeat(60)))).rejects.toThrow(
      StreamLimitExceededError
    );
    expect(breaker.signal.aborted).toBe(true);
    expect(breaker.signal.reason).toMatchObject({ kind: 'tool_call_args' });
  });

  it('charges a chunk once when the run wire consumer also claims it', async () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxToolCallArgBytes: 100 }),
    };
    const onChunk = makeHandler(graph);
    const metadata = { langgraph_node: 'summarize', langgraph_step: 4 };
    const toolChunk = (args: string): AIMessageChunk =>
      new AIMessageChunk({
        content: '',
        tool_call_chunks: [
          { type: 'tool_call_chunk', name: 'db_query', args, index: 0 },
        ],
      });

    /** Inside a live run the provider callbacks route the same chunk object
     * through the graph's registered stream consumer as well. Wire consumer
     * first, then the summarization handler: 60 bytes must be tallied once
     * (a second same-side claim would put the tally at 120 and trip). */
    const first = toolChunk('x'.repeat(60));
    enforceStreamLimitsForWireChunk({
      graph,
      metadata,
      chunk: first,
      side: 'consumer',
    });
    await onChunk!(first, metadata);

    /** Opposite order for the next chunk: handler (producer) before wire
     * consumer. 60 + 39 = 99 stays under the cap only when each chunk was
     * charged exactly once. */
    const second = toolChunk('x'.repeat(39));
    await onChunk!(second, metadata);
    enforceStreamLimitsForWireChunk({
      graph,
      metadata,
      chunk: second,
      side: 'consumer',
    });

    /** Enforcement stays armed: two more bytes cross the 100-byte cap. */
    await expect(async () => onChunk!(toolChunk('xx'), metadata)).rejects.toThrow(
      StreamLimitExceededError
    );
  });

  it('counts dual-consumed events once against the event cap', async () => {
    const graph: StreamLimitState = {
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 2 }),
    };
    const onChunk = makeHandler(graph);
    const metadata = { langgraph_node: 'summarize', langgraph_step: 4 };
    const passBoth = async (chunk: AIMessageChunk): Promise<void> => {
      enforceStreamLimitsForWireChunk({
        graph,
        metadata,
        chunk,
        side: 'consumer',
      });
      await onChunk!(chunk, metadata);
    };

    await passBoth(textChunk());
    await passBoth(textChunk());
    await expect(async () => passBoth(textChunk())).rejects.toThrow(
      StreamLimitExceededError
    );
  });

  it('stays inert without graph state', async () => {
    const onChunk = createSummarizationChunkHandler({
      stepId: 'step_1',
      config: { metadata: {} },
      provider: Providers.OPENAI,
    });
    for (let i = 0; i < 10; i++) {
      await onChunk!(textChunk());
    }
  });
});
