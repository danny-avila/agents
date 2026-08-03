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
