import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const charCounter: t.TokenCounter = (msg: BaseMessage): number => {
  const content = msg.content;
  if (typeof content === 'string') {
    return content.length + 3;
  }
  return 3;
};

const llmConfig: t.LLMConfig = {
  provider: Providers.OPENAI,
  streaming: true,
  streamUsage: false,
};

const streamConfig = {
  configurable: { thread_id: 'context-usage-event' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

describe('ON_CONTEXT_USAGE event', () => {
  jest.setTimeout(15000);

  it('dispatches a post-prune context snapshot per model call', async () => {
    const received: t.ContextUsageEvent[] = [];
    const maxContextTokens = 4000;

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
        maxContextTokens,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            received.push(data as unknown as t.ContextUsageEvent);
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });

    run.Graph?.overrideTestModel(['Hello there!'], 1);
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      streamConfig
    );

    expect(received).toHaveLength(1);
    const event = received[0];
    expect(event.runId).toBe('test-context-usage-event');
    expect(event.agentId).toBeDefined();
    expect(event.breakdown.maxContextTokens).toBe(maxContextTokens);
    expect(event.breakdown.instructionTokens).toBeGreaterThan(0);
    expect(event.breakdown.toolTokenCounts).toEqual({});
    expect(event.contextBudget).toBeGreaterThan(0);
    expect(event.contextBudget).toBeLessThanOrEqual(maxContextTokens);
    expect(event.effectiveInstructionTokens).toBeGreaterThan(0);
    expect(event.prePruneContextTokens).toBeGreaterThan(0);
    expect(event.remainingContextTokens).toBeGreaterThan(0);
    expect(event.remainingContextTokens).toBeLessThan(
      event.contextBudget as number
    );
    expect(event.breakdown.instructionTokens).toBe(
      event.effectiveInstructionTokens
    );
    expect(event.breakdown.availableForMessages).toBe(
      (event.contextBudget as number) -
        (event.effectiveInstructionTokens as number)
    );
    expect(event.breakdown.messageTokens).toBe(
      (event.contextBudget as number) -
        (event.effectiveInstructionTokens as number) -
        (event.remainingContextTokens as number)
    );
  });

  it('does not dispatch when no tokenCounter is configured', async () => {
    const received: t.ContextUsageEvent[] = [];

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event-no-counter',
      graphConfig: {
        type: 'standard',
        llmConfig,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            received.push(data as unknown as t.ContextUsageEvent);
          },
        },
      },
    });

    run.Graph?.overrideTestModel(['Hello there!'], 1);
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      streamConfig
    );

    expect(received).toHaveLength(0);
  });
});
