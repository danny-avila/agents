import { z } from 'zod';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

class CapturingFakeChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

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
  it('accounts for the retained exchange in the final provider-facing snapshot', async () => {
    const received: t.ContextUsageEvent[] = [];
    const retainedResult = `tool-result:${'x'.repeat(10_000)}`;
    const lookup = new DynamicStructuredTool({
      name: 'lookup',
      description: 'Returns a deterministic retained result.',
      schema: z.object({ query: z.string() }),
      func: async () => retainedResult,
    });
    const toolCall = {
      id: 'lookup-call',
      name: 'lookup',
      args: { query: 'retained context' },
      type: 'tool_call' as const,
    };
    const model = new CapturingFakeChatModel({
      responses: ['', 'done'],
      toolCalls: [toolCall],
    });

    const run = await Run.create<t.IState>({
      runId: 'test-context-usage-event-tool-loop',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'Use lookup, then answer briefly.',
        maxContextTokens: 2_000,
        maxToolResultChars: 240,
        tools: [lookup],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_CONTEXT_USAGE]: {
          handle: (_event, data): void => {
            if (data != null && 'breakdown' in data) {
              received.push(data);
            }
          },
        },
      },
      tokenCounter: charCounter,
      indexTokenCountMap: {},
    });

    run.Graph!.overrideModel = model;
    await run.processStream(
      { messages: [new HumanMessage('Look this up.')] },
      {
        ...streamConfig,
        configurable: { thread_id: 'context-usage-event-tool-loop' },
      }
    );

    expect(model.invocations).toHaveLength(2);
    expect(received).toHaveLength(2);
    expect(received[0].breakdown.toolMessageTokens).toBe(0);
    expect(received[0].breakdown.toolMessageTokenCounts).toBeUndefined();

    const finalProviderMessages = model.invocations[1];
    const retainedToolMessages = finalProviderMessages.filter((message) => {
      if (message.getType() === 'tool') {
        return true;
      }
      if (message.getType() !== 'ai') {
        return false;
      }
      const aiMessage = message as AIMessage;
      return (
        (aiMessage.tool_calls?.length ?? 0) > 0 &&
        typeof aiMessage.content === 'string' &&
        !/\S/u.test(aiMessage.content)
      );
    });
    const providerToolTokens = retainedToolMessages.reduce(
      (total, message) => total + charCounter(message),
      0
    );
    const finalEvent = received[1];
    const retainedToolResult = retainedToolMessages.find(
      (message) => message.getType() === 'tool'
    );
    expect(retainedToolResult).toBeDefined();
    const resultTokenCount = charCounter(retainedToolResult!);
    expect(String(retainedToolResult?.content).length).toBeLessThan(
      retainedResult.length
    );
    expect(finalEvent.breakdown.toolMessageTokens).toBe(providerToolTokens);
    expect(resultTokenCount).toBeGreaterThan(0);
    expect(finalEvent.breakdown.toolMessageTokens).toBeGreaterThan(
      resultTokenCount
    );
    expect(finalEvent.breakdown.toolMessageTokenCounts).toEqual({
      lookup: resultTokenCount,
    });
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
