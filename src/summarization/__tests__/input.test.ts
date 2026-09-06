import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  createSummarizationInputBudget,
  estimateMessageTokens,
  MAX_SUMMARIZATION_CHUNKS,
  planSummarizationChunks,
} from '@/summarization/input';

describe('summarization input preparation', () => {
  it('accounts for tool-call metadata omitted by content token counters', () => {
    const message = new AIMessage({
      content: 'ok',
      tool_calls: [
        { id: 'call_1', name: 'read', args: { path: '/'.repeat(100) } },
      ],
    });

    expect(estimateMessageTokens(message, () => 2)).toBeGreaterThan(10);
  });

  it('subtracts instruction overhead and output reserve from the context window', () => {
    expect(
      createSummarizationInputBudget({
        maxContextTokens: 1_000_000,
        fixedOverheadTokens: 25_000,
        maxSummaryTokens: 80_000,
      })
    ).toEqual({
      contextTokens: 1_000_000,
      fixedOverheadTokens: 25_000,
      outputReserveTokens: 80_000,
      messageBudgetTokens: 895_000,
    });
  });

  it('applies calibration while preserving an explicit zero reserve ratio', () => {
    expect(
      createSummarizationInputBudget({
        maxContextTokens: 1_000,
        fixedOverheadTokens: 100,
        maxSummaryTokens: 100,
        reserveRatio: 0,
        calibrationRatio: 2,
      }).messageBudgetTokens
    ).toBe(400);
  });

  it('subtracts fixed overhead before expanding a calibrated raw budget', () => {
    expect(
      createSummarizationInputBudget({
        maxContextTokens: 1_000,
        fixedOverheadTokens: 200,
        maxSummaryTokens: 100,
        calibrationRatio: 0.5,
      }).messageBudgetTokens
    ).toBe(1_400);
  });

  it('never separates a tool call from its result', () => {
    const toolCall = new AIMessage({
      content: '',
      tool_calls: [{ id: 'call_1', name: 'read', args: {} }],
    });
    const toolResult = new ToolMessage({
      content: 'result'.repeat(40),
      tool_call_id: 'call_1',
      name: 'read',
    });
    const result = planSummarizationChunks({
      messages: [
        new HumanMessage('objective'.repeat(20)),
        toolCall,
        toolResult,
        new HumanMessage('next'.repeat(60)),
      ],
      messageBudgetTokens: 400,
      tokenCounter: (message) => String(message.content).length,
    });

    expect(result.error).toBeUndefined();
    const chunks = result.plan?.chunks ?? [];
    const callChunk = chunks.findIndex((chunk) => chunk.includes(toolCall));
    const resultChunk = chunks.findIndex((chunk) => chunk.includes(toolResult));
    expect(callChunk).toBeGreaterThanOrEqual(0);
    expect(resultChunk).toBe(callChunk);
  });

  it('keeps serialized tool calls with their results', () => {
    const toolCall = new AIMessage({ content: '' });
    toolCall.additional_kwargs.tool_calls = [
      {
        id: 'call_serialized',
        type: 'function',
        function: { name: 'read', arguments: '{}' },
      },
    ];
    const toolResult = new ToolMessage({
      content: 'result'.repeat(40),
      tool_call_id: 'call_serialized',
      name: 'read',
    });
    const result = planSummarizationChunks({
      messages: [
        new HumanMessage('objective'.repeat(20)),
        toolCall,
        toolResult,
        new HumanMessage('next'.repeat(60)),
      ],
      messageBudgetTokens: 600,
      tokenCounter: (message) => String(message.content).length,
    });

    expect(result.error).toBeUndefined();
    const chunks = result.plan?.chunks ?? [];
    const callChunk = chunks.findIndex((chunk) => chunk.includes(toolCall));
    const resultChunk = chunks.findIndex((chunk) => chunk.includes(toolResult));
    expect(callChunk).toBeGreaterThanOrEqual(0);
    expect(resultChunk).toBe(callChunk);
  });

  it('keeps an unresolved tool call with all following messages', () => {
    const openCall = new AIMessage({
      content: '',
      tool_calls: [{ id: 'open_call', name: 'exec', args: {} }],
    });
    const following = new HumanMessage('do not detach this');
    const result = planSummarizationChunks({
      messages: [new HumanMessage('objective'), openCall, following],
      messageBudgetTokens: 500,
      tokenCounter: () => 100,
    });

    expect(result.error).toBeUndefined();
    const chunks = result.plan?.chunks ?? [];
    const callChunk = chunks.findIndex((chunk) => chunk.includes(openCall));
    const followingChunk = chunks.findIndex((chunk) =>
      chunk.includes(following)
    );
    expect(callChunk).toBeGreaterThanOrEqual(0);
    expect(followingChunk).toBe(callChunk);
  });

  it('fails before invoking a model when the bounded chunk count is exceeded', () => {
    const result = planSummarizationChunks({
      messages: Array.from(
        { length: MAX_SUMMARIZATION_CHUNKS + 1 },
        (_, index) => new HumanMessage(`message-${index}`)
      ),
      messageBudgetTokens: 220,
      tokenCounter: () => 100,
    });

    expect(result.error).toContain('bounded limit');
    expect(result.plan).toBeUndefined();
  });
});
