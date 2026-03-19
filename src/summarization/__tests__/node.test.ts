import { HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import {
  createSummarizeNode,
  DEFAULT_SUMMARIZATION_PROMPT,
  DEFAULT_UPDATE_SUMMARIZATION_PROMPT,
} from '@/summarization/node';
import * as providers from '@/llm/providers';
import * as eventUtils from '@/utils/events';
import { AgentContext } from '@/agents/AgentContext';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Creates a real AgentContext via fromConfig with sensible defaults.
 *  Extra properties are assigned directly for test-specific overrides. */
function createAgentContext(
  overrides: Record<string, unknown> = {}
): AgentContext {
  const {
    // AgentInputs fields
    agentId = 'agent_0',
    provider = Providers.OPENAI,
    instructions = 'Test agent',
    summarizationEnabled = true,
    summarizationConfig,
    maxContextTokens,
    tools,
    ...extra
  } = overrides;

  const ctx = AgentContext.fromConfig({
    agentId: agentId as string,
    provider: provider as Providers,
    instructions: instructions as string,
    summarizationEnabled: summarizationEnabled as boolean,
    ...(summarizationConfig != null ? { summarizationConfig } : {}),
    ...(maxContextTokens != null ? { maxContextTokens } : {}),
    ...(tools != null ? { tools } : {}),
  } as import('@/types').AgentInputs);

  // Apply direct property overrides for test-specific internal state
  for (const [key, value] of Object.entries(extra)) {
    (ctx as unknown as Record<string, unknown>)[key] = value;
  }

  return ctx;
}

/** Creates a mock graph container for createSummarizeNode. */
function mockGraph(
  onStepCompleted?: (stepId: string, result: t.StepCompleted) => void
): {
  contentData: t.RunStep[];
  contentIndexMap: Map<string, number>;
  config: RunnableConfig;
  runId: string;
  isMultiAgent: boolean;
  dispatchRunStep: (
    runStep: t.RunStep,
    config?: RunnableConfig
  ) => Promise<void>;
  dispatchRunStepCompleted: (
    stepId: string,
    result: t.StepCompleted,
    config?: RunnableConfig
  ) => Promise<void>;
} {
  const contentData: t.RunStep[] = [];
  const contentIndexMap = new Map<string, number>();
  return {
    contentData,
    contentIndexMap,
    config: {} as RunnableConfig,
    runId: 'run_1',
    isMultiAgent: false,
    dispatchRunStep: async (runStep: t.RunStep): Promise<void> => {
      contentData.push(runStep);
      contentIndexMap.set(runStep.id, runStep.index);
    },
    dispatchRunStepCompleted: async (
      stepId: string,
      result: t.StepCompleted
    ): Promise<void> => {
      onStepCompleted?.(stepId, result);
    },
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

    const agentContext = createAgentContext();
    const graph = mockGraph((_stepId, result) => {
      if (result.type === 'summary') {
        events.push({
          event: GraphEvents.ON_SUMMARIZE_COMPLETE,
          data: { summary: result.summary },
        });
      }
    });
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Hello'), new HumanMessage('World')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    const eventNames = events.map((e) => e.event);
    // ON_RUN_STEP now goes through graph.dispatchRunStep, not safeDispatchCustomEvent
    expect(graph.contentData.length).toBeGreaterThan(0);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_START);
    expect(eventNames).toContain(GraphEvents.ON_SUMMARIZE_COMPLETE);

    // Complete event should have the summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(
      (
        (completeEvent?.data as t.SummarizeCompleteEvent).summary!
          .content?.[0] as { text: string } | undefined
      )?.text
    ).toBe('Test summary output');
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
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
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
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
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
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph((_stepId, result) => {
      if (result.type === 'summary') {
        events.push({
          event: GraphEvents.ON_SUMMARIZE_COMPLETE,
          data: { summary: result.summary },
        });
      }
    });
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    const result = await node(
      {
        messages: [new HumanMessage('Test')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(result.summarizationRequest).toBeUndefined();
    // After summarization, REMOVE_ALL + surviving context is returned
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThanOrEqual(1);
    expect(result.messages![0]._getType()).toBe('remove');

    // Tier 3 fallback: metadata stub is used as summary text
    const completeEvent = events.find(
      (e) => e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(
      (
        (completeEvent?.data as t.SummarizeCompleteEvent).summary!
          .content?.[0] as { text: string } | undefined
      )?.text
    ).toMatch(/^\[Metadata summary:/);
    expect(
      (completeEvent?.data as t.SummarizeCompleteEvent).error
    ).toBeUndefined();
  });

  it('falls back to metadata stub when primary LLM call fails', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest.fn().mockRejectedValue(new Error('LLM unavailable')),
          };
        }
      } as never
    );

    const setSummary = jest.fn();
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(setSummary).toHaveBeenCalledWith(
      expect.stringContaining('[Metadata summary:'),
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
    const agentContext = createAgentContext({ setSummary } as never);
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('Test')],
        summarizationRequest: {
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

  it('cache-hit path sends raw messages with instruction appended as final HumanMessage', async () => {
    captureEvents();

    const capturedMessages: Array<{ type: string; content: string }> = [];

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(async (messages: unknown[]) => {
                for (const msg of messages as {
                  getType: () => string;
                  content: string | unknown[];
                }[]) {
                  capturedMessages.push({
                    type: msg.getType(),
                    content:
                      typeof msg.content === 'string'
                        ? msg.content
                        : JSON.stringify(msg.content),
                  });
                }
                return {
                  content:
                    '## Goal\nTest goal\n\n<events>\n<event key="test" turn="0">value</event>\n</events>',
                };
              }),
          };
        }
      } as never
    );

    const agentContext = createAgentContext();
    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    await node(
      {
        messages: [
          new HumanMessage('Message 1'),
          new HumanMessage('Message 2'),
          new HumanMessage('Message 3'),
        ],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // The raw messages should be sent + instruction appended as the last HumanMessage
    // messagesToRefine has 3 HumanMessages, instruction adds 1 more
    expect(capturedMessages.length).toBe(4);
    expect(capturedMessages[0].type).toBe('human');
    expect(capturedMessages[0].content).toBe('Message 1');
    expect(capturedMessages[3].type).toBe('human');
    // The last message should contain the summarization prompt
    expect(capturedMessages[3].content).toContain(
      'context window is filling up'
    );
  });

  it('cache-hit path includes prior summary in the instruction message', async () => {
    captureEvents();

    const capturedMessages: Array<{ type: string; content: string }> = [];

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return {
            invoke: jest
              .fn()
              .mockImplementation(async (messages: unknown[]) => {
                for (const msg of messages as {
                  getType: () => string;
                  content: string | unknown[];
                }[]) {
                  capturedMessages.push({
                    type: msg.getType(),
                    content:
                      typeof msg.content === 'string'
                        ? msg.content
                        : JSON.stringify(msg.content),
                  });
                }
                return { content: '## Goal\nUpdated summary' };
              }),
          };
        }
      } as never
    );

    // Create context with a prior summary
    const agentContext = createAgentContext();
    agentContext.setSummary('## Goal\nPrior summary content.', 50);

    const graph = mockGraph();
    const node = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    await node(
      {
        messages: [new HumanMessage('New message')],
        summarizationRequest: {
          remainingContextTokens: 1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // The last message should contain the update prompt (prior summary exists)
    const lastMsg = capturedMessages[capturedMessages.length - 1];
    expect(lastMsg.type).toBe('human');
    expect(lastMsg.content).toContain('Merge the new messages');
    // Should include the prior summary
    expect(lastMsg.content).toContain('<previous-summary>');
    expect(lastMsg.content).toContain('Prior summary content');
  });
});

describe('DEFAULT_SUMMARIZATION_PROMPT', () => {
  it('is exported and non-empty', () => {
    expect(typeof DEFAULT_SUMMARIZATION_PROMPT).toBe('string');
    expect(DEFAULT_SUMMARIZATION_PROMPT.length).toBeGreaterThan(0);
  });

  it('contains structured checkpoint sections', () => {
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Goal');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Progress');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Key Decisions');
    expect(DEFAULT_SUMMARIZATION_PROMPT).toContain('## Next Steps');
  });
});

describe('DEFAULT_UPDATE_SUMMARIZATION_PROMPT', () => {
  it('is exported and non-empty', () => {
    expect(typeof DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toBe('string');
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT.length).toBeGreaterThan(0);
  });

  it('instructs merging new content', () => {
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(
      /Merge the new messages/i
    );
  });

  it('instructs updating progress tracking', () => {
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(/Done/);
    expect(DEFAULT_UPDATE_SUMMARIZATION_PROMPT).toMatch(/In Progress/);
  });
});

describe('budget check — instructions exceed context', () => {
  it('skips summarization when instructionTokens >= maxContextTokens', async () => {
    const events = captureEvents();
    const agentContext = createAgentContext({
      maxContextTokens: 4000,
      systemMessageTokens: 5000,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage('test')],
        summarizationRequest: {
          remainingContextTokens: -1000,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    expect(result.summarizationRequest).toBeUndefined();
    expect(result.messages).toBeUndefined();

    // No summarization events should have fired
    const summarizeEvents = events.filter(
      (e) =>
        e.event === GraphEvents.ON_SUMMARIZE_START ||
        e.event === GraphEvents.ON_SUMMARIZE_DELTA ||
        e.event === GraphEvents.ON_SUMMARIZE_COMPLETE
    );
    expect(summarizeEvents).toHaveLength(0);
  });

  it('proceeds normally when instructionTokens < maxContextTokens', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Budget is fine summary');
        }
      } as never
    );

    const agentContext = createAgentContext({
      maxContextTokens: 8000,
      systemMessageTokens: 2000,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage('hello')],
        summarizationRequest: {
          remainingContextTokens: 500,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should have summarized — messages returned for state replacement
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThan(0);
  });
});

describe('emoji-heavy content does not break summarization', () => {
  it('summarization completes without JSON errors on emoji-heavy messages', async () => {
    captureEvents();

    jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
      class {
        constructor() {
          return mockInvokeModel('Summary of emoji conversation');
        }
      } as never
    );

    const emojiContent = '👨‍💻 coding 🎉 party 🌍 world 🚀 rocket '.repeat(30);
    const agentContext = createAgentContext({
      maxContextTokens: 8000,
      systemMessageTokens: 100,
      formatTokenBudgetBreakdown: () => 'mock breakdown',
    });

    const graph = mockGraph();
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph as never,
      generateStepId,
    });

    const result = await summarizeNode(
      {
        messages: [new HumanMessage(emojiContent)],
        summarizationRequest: {
          remainingContextTokens: 500,
          agentId: 'agent_0',
        },
      },
      {} as RunnableConfig
    );

    // Should complete without throwing JSON serialization errors
    expect(result.messages).toBeDefined();
    expect(result.messages!.length).toBeGreaterThan(0);
  });
});
