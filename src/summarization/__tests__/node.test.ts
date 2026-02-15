import { HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import {
  createSummarizeNode,
  DEFAULT_SUMMARIZATION_PROMPT,
} from '@/summarization/node';
import * as providers from '@/llm/providers';
import * as eventUtils from '@/utils/events';
import type { AgentContext } from '@/agents/AgentContext';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Creates a mock AgentContext with sensible defaults. */
function mockAgentContext(overrides: Partial<AgentContext> = {}): AgentContext {
  return {
    provider: Providers.OPENAI,
    agentId: 'agent_0',
    summaryVersion: 0,
    tokenCounter: undefined,
    summarizationConfig: undefined,
    getSummaryText: () => undefined,
    setSummary: jest.fn(),
    ...overrides,
  } as unknown as AgentContext;
}

/** Creates a mock graph container for createSummarizeNode. */
function mockGraph(): {
  contentData: t.RunStep[];
  contentIndexMap: Map<string, number>;
  config: RunnableConfig;
  runId: string;
  isMultiAgent: boolean;
  } {
  const contentData: t.RunStep[] = [];
  const contentIndexMap = new Map<string, number>();
  return {
    contentData,
    contentIndexMap,
    config: {} as RunnableConfig,
    runId: 'run_1',
    isMultiAgent: false,
  };
}

let stepCounter = 0;
function generateStepId(_stepKey: string): [string, number] {
  const id = `step_test_${stepCounter++}`;
  return [id, 0];
}

/** Collects custom events dispatched during the node execution. */
function captureEvents(): Array<{ event: string; data: unknown }> {
  const events: Array<{ event: string; data: unknown }> = [];
  jest.spyOn(eventUtils, 'safeDispatchCustomEvent').mockImplementation((async (
    ...args: unknown[]
  ) => {
    events.push({ event: args[0] as string, data: args[1] });
  }) as never);
  return events;
}

/** Creates a mock model that returns a canned response via invoke(). */
function mockInvokeModel(response: string): { invoke: jest.Mock } {
  return {
    invoke: jest.fn().mockResolvedValue({ content: response }),
  };
}

/**
 * Creates a mock model that streams text chunk-by-chunk.
 * invoke() returns the full text; stream() yields one chunk per word.
 */
function mockStreamingModel(response: string): {
  invoke: jest.Mock;
  stream: jest.Mock;
} {
  const words = response.split(' ');
  return {
    invoke: jest.fn().mockResolvedValue({ content: response }),
    stream: jest.fn().mockImplementation(async () => {
      return (async function* (): AsyncGenerator<{ content: string }> {
        for (const word of words) {
          // Add space back except for first word
          yield { content: word + ' ' };
        }
      })();
    }),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  stepCounter = 0;
  jest.restoreAllMocks();
});

describe('createSummarizeNode', () => {
  it('emits ON_SUMMARIZE_START and ON_SUMMARIZE_COMPLETE on success', async () => {
    const events = captureEvents();

    // Mock getChatModelClass to return our mock model
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Test summary output');
        }
      } as never
    );

    const agentContext = mockAgentContext();
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [
            new HumanMessage('Hello'),
            new HumanMessage('World'),
          ],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const eventNames = events.map((e) => e.event);
    expect(eventNames).toContain(GraphEvents.ON_RUN_STEP);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_START);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_COMPLETE);

    // Complete event should have the summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect((completeEvent?.data as t.SummarizeCompleteEvent).summary.text).toBe(
      'Test summary output'
    );
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).error
    ).toBeUndefined();
  });

  it('collects streamed text when model supports stream()', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockStreamingModel('one two three');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = mockAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [new HumanMessage('Test message')],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Node collects the full streamed text and calls setSummary.
    // Delta events are dispatched by ChatModelStreamHandler, not the node.
    expect(setSummary).toHaveBeenCalledWith(
      'one two three',
      expect.any(Number)
    );
  });

  it('falls back to invoke when model has no stream()', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Full summary text');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = mockAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [new HumanMessage('Test message')],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Falls back to invoke and still collects the text
    expect(setSummary).toHaveBeenCalledWith(
      'Full summary text',
      expect.any(Number)
    );
  });

  it('produces metadata stub when all LLM attempts fail', async () => {
    const events = captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockRejectedValue(new Error('Model error')),
          };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = mockAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    const result = await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [new HumanMessage('Test')],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(result).toEqual({ summarizationRequest: undefined });

    // Tier 3 fallback: metadata stub is used as summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).summary.text
    ).toMatch(/^\[Metadata summary:/);
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).error
    ).toBeUndefined();
  });

  it('retries with reduced budget (tier 2) when first attempt fails', async () => {
    captureEvents();

    let callCount = 0;
    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockImplementation(async () => {
              callCount++;
              if (callCount === 1) {
                throw new Error('First attempt failed');
              }
              return { content: 'Recovered summary' };
            }),
          };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = mockAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [new HumanMessage('Test message')],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should have recovered on tier 2
    expect(setSummary).toHaveBeenCalledWith(
      'Recovered summary',
      expect.any(Number)
    );
  });

  it('calls setSummary with the final text', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Final summary');
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = mockAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [new HumanMessage('Test')],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalledWith(
      'Final summary',
      expect.any(Number)
    );
  });

  it('uses multi-stage when parts > 1 and enough messages', async () => {
    const events = captureEvents();

    const invokeResponses = [
      'Part 1 summary',
      'Part 2 summary',
      'Merged summary',
    ];
    let callIndex = 0;

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockImplementation(async () => {
              return { content: invokeResponses[callIndex++] || '' };
            }),
          };
        }
      } as never
    );

    const agentContext = mockAgentContext({
      summarizationConfig: {
        parameters: { parts: 2, minMessagesForSplit: 2 },
      },
    } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello')],
        summarizationRequest: {
          messagesToRefine: [
            new HumanMessage('Message 1'),
            new HumanMessage('Message 2'),
            new HumanMessage('Message 3'),
            new HumanMessage('Message 4'),
          ],
          context: [],
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should have called invoke 3 times: 2 chunks + 1 merge
    // (all fall back to invoke since model has no stream method)
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect((completeEvent?.data as t.SummarizeCompleteEvent).summary.text).toBe(
      'Merged summary'
    );
  });
});

describe('DEFAULT_SUMMARIZATION_PROMPT', () => {
  it('is exported and non-empty', () => {
    expect(typeof DEFAULT_SUMMARIZATION_PROMPT).toBe('string');
    expect(DEFAULT_SUMMARIZATION_PROMPT.length).toBeGreaterThan(0);
  });
});
