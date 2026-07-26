import { nanoid } from 'nanoid';
import { MessageContentText } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents, StepTypes, ContentTypes } from '@/common';
import { createContentAggregator } from './stream';
import { SplitStreamHandler } from './splitStream';
import { createMockStream } from './mockStream';

// Mock sleep to speed up tests
jest.mock('@/utils', () => ({
  sleep: (): Promise<void> => Promise.resolve(),
}));

const createRunStep = (id: string): t.RunStep => ({
  id,
  stepIndex: 0,
  type: StepTypes.MESSAGE_CREATION,
  index: 0,
  stepDetails: {
    type: StepTypes.MESSAGE_CREATION,
    message_creation: { message_id: id },
  },
  usage: null,
});

describe('Stream Generation and Handling', () => {
  let mockHandlers: {
    [GraphEvents.ON_RUN_STEP]: jest.Mock;
    [GraphEvents.ON_MESSAGE_DELTA]: jest.Mock;
  };

  beforeEach(() => {
    mockHandlers = {
      [GraphEvents.ON_RUN_STEP]: jest.fn(),
      [GraphEvents.ON_MESSAGE_DELTA]: jest.fn(),
    };
  });

  it('should properly stream tokens including spaces', async () => {
    const stream = createMockStream({
      text: 'Hello world!',
      streamRate: 0,
    })();

    const tokens: string[] = [];
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta.content ?? '';
      if (content) tokens.push(content);
    }

    expect(tokens).toEqual(['Hello', ' ', 'world!']);
  });

  it('should handle code blocks without splitting them', async () => {
    const runId = nanoid();
    const handler = new SplitStreamHandler({
      runId,
      blockThreshold: 10,
      handlers: mockHandlers,
    });

    const codeText = `Code:
\`\`\`
const x = 1;
const y = 2;
const z = 2;
const a = 2;
const b = 2;
const c = 2;
const d = 2;
const e = 2;
const f = 2;
const g = 2;
const h = 2;
\`\`\`
End code.`;

    const stream = createMockStream({
      text: codeText,
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    // Verify that only one message block was created for the code section
    const runSteps = mockHandlers[GraphEvents.ON_RUN_STEP].mock.calls;
    expect(runSteps.length).toBe(2); // Should only create one message block
  });

  it('should split content when exceeding threshold', async () => {
    const runId = nanoid();
    const handler = new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
      // Set a very low threshold for testing
      blockThreshold: 10,
    });

    // Make the text longer and ensure it has clear breaking points
    const longText =
      'This is the first sentence. And here is another sentence. And yet another one here. Finally one more.';

    const stream = createMockStream({
      text: longText,
      streamRate: 0,
    })();

    // For debugging
    // let totalLength = 0;
    for await (const chunk of stream) {
      handler.handle(chunk);
      // For debugging
      // const content = chunk.choices?.[0]?.delta.content;
      // if (content) {
      //   totalLength += content.length;
      //   console.log(`Current length: ${totalLength}, Content: "${content}"`);
      // }
    }

    // Verify multiple message blocks were created
    const runSteps = mockHandlers[GraphEvents.ON_RUN_STEP].mock.calls;
    // console.log('Number of run steps:', runSteps.length);
    expect(runSteps.length).toEqual(handler.currentIndex + 1);
  });

  it('should handle reasoning text separately', async () => {
    const runId = nanoid();
    new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
    });

    const stream = createMockStream({
      text: 'Main content',
      reasoningText: 'Reasoning text',
      streamRate: 0,
    })();

    const reasoningTokens: string[] = [];
    const contentTokens: string[] = [];

    for await (const chunk of stream) {
      const reasoning = chunk.choices?.[0]?.delta.reasoning_content ?? '';
      const content = chunk.choices?.[0]?.delta.content ?? '';

      if (reasoning) reasoningTokens.push(reasoning);
      if (content) contentTokens.push(content);
    }

    expect(reasoningTokens.length).toBeGreaterThan(0);
    expect(contentTokens.length).toBeGreaterThan(0);
  });

  it('should preserve empty strings and whitespace', async () => {
    const stream = createMockStream({
      text: 'Hello  world', // Note double space
      streamRate: 0,
    })();

    const tokens: string[] = [];
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta.content ?? '';
      if (!content) {
        return;
      }
      tokens.push(content);
    }

    expect(tokens).toContain(' ');
    expect(tokens.join('')).toBe('Hello  world');
  });
});

describe('ContentAggregator empty deltas', () => {
  it('should ignore empty message delta content arrays', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: createRunStep('step_empty_message'),
      });

      aggregateContent({
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: {
          id: 'step_empty_message',
          delta: { content: [] },
        },
      });

      expect(warnSpy).not.toHaveBeenCalled();
      expect(contentParts).toEqual([]);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('should ignore empty reasoning delta content arrays', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: createRunStep('step_empty_reasoning'),
      });

      aggregateContent({
        event: GraphEvents.ON_REASONING_DELTA,
        data: {
          id: 'step_empty_reasoning',
          delta: { content: [] },
        },
      });

      expect(warnSpy).not.toHaveBeenCalled();
      expect(contentParts).toEqual([]);
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('ContentAggregator resumed tool completions', () => {
  it('matches a completion to seeded content when a rebuilt run has no step id', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: { value: 'original' },
        },
      }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: '',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_approval',
            name: 'approval_probe',
            args: '{"value":"original"}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Before approval',
    });
    expect(contentParts[1]).toEqual({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_approval',
        name: 'approval_probe',
        args: '{"value":"original"}',
        type: 'tool_call',
        output: 'approved',
        progress: 1,
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_continuation'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_continuation',
        delta: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'After approval',
            },
          ],
        },
      } as t.MessageDeltaEvent,
    });

    expect(contentParts[2]).toEqual({
      type: ContentTypes.TEXT,
      text: 'After approval',
    });
  });

  it('rebinds a replacement run step to seeded content by tool call id', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: { value: 'original' },
        },
      }
    );

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: {
          id: 'step_rebuilt',
          stepIndex: 0,
          type: StepTypes.TOOL_CALLS,
          index: 0,
          stepDetails: {
            type: StepTypes.TOOL_CALLS,
            tool_calls: [
              {
                id: 'call_approval',
                name: 'approval_probe',
                args: { value: 'original' },
              },
            ],
          },
          usage: null,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_rebuilt',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id: 'call_approval',
              name: 'approval_probe',
              args: '{"value":"original"}',
              output: 'approved',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(stepMap.get('step_rebuilt')?.index).toBe(1);
      expect(contentParts[0]).toEqual({
        type: ContentTypes.TEXT,
        text: 'Before approval',
      });
      expect(contentParts[1]).toMatchObject({
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          output: 'approved',
          progress: 1,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: { ...createRunStep('step_continuation'), index: 1 },
      });
      aggregateContent({
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: {
          id: 'step_continuation',
          delta: {
            content: [
              {
                type: ContentTypes.TEXT,
                text: 'After approval',
              },
            ],
          },
        } as t.MessageDeltaEvent,
      });
      expect(stepMap.get('step_continuation')?.index).toBe(2);
      expect(contentParts[2]).toEqual({
        type: ContentTypes.TEXT,
        text: 'After approval',
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('matches parallel completions to their own seeded tool cards', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_first',
          name: 'approval_probe',
          args: { value: 'first' },
        },
      },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_second',
          name: 'approval_probe',
          args: { value: 'second' },
        },
      }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_parallel',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_first',
              name: 'approval_probe',
              args: { value: 'first' },
            },
            {
              id: 'call_second',
              name: 'approval_probe',
              args: { value: 'second' },
            },
          ],
        },
        usage: null,
      },
    });

    for (const [id, value] of [
      ['call_first', 'first'],
      ['call_second', 'second'],
    ]) {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_parallel',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id,
              name: 'approval_probe',
              args: JSON.stringify({ value }),
              output: `${value} approved`,
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });
    }

    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_first',
        output: 'first approved',
        progress: 1,
      },
    });
    expect(contentParts[2]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_second',
        output: 'second approved',
        progress: 1,
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_after_parallel'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_parallel',
        delta: {
          content: [
            {
              type: ContentTypes.TEXT,
              text: 'Parallel tools complete',
            },
          ],
        },
      } as t.MessageDeltaEvent,
    });
    expect(contentParts[3]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Parallel tools complete',
    });
  });

  it('appends unmatched tools and continuation content after the entire seed', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Before approval' },
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'call_approval',
          name: 'approval_probe',
          args: {},
        },
      },
      { type: ContentTypes.TEXT, text: 'Seeded tail one' },
      { type: ContentTypes.TEXT, text: 'Seeded tail two' }
    );

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_rebuilt',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_approval',
              name: 'approval_probe',
              args: {},
            },
            {
              id: 'call_new',
              name: 'approval_probe',
              args: { value: 'new' },
            },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_after_seed'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_seed',
        delta: {
          content: [{ type: ContentTypes.TEXT, text: 'After seed' }],
        },
      } as t.MessageDeltaEvent,
    });

    expect(stepMap.get('step_rebuilt')?.index).toBe(1);
    expect(stepMap.get('step_after_seed')?.index).toBe(5);
    expect(contentParts.slice(2, 4)).toEqual([
      { type: ContentTypes.TEXT, text: 'Seeded tail one' },
      { type: ContentTypes.TEXT, text: 'Seeded tail two' },
    ]);
    expect(contentParts[4]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_new',
        args: { value: 'new' },
      },
    });
    expect(contentParts[5]).toEqual({
      type: ContentTypes.TEXT,
      text: 'After seed',
    });
  });

  it('preserves seeded metadata when a blank-id completion replaces a tool card', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push({
      type: ContentTypes.TOOL_CALL,
      agentId: 'researcher',
      groupId: 7,
      tool_call: {
        id: 'call_with_metadata',
        name: 'approval_probe',
        args: {},
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: '',
          index: 42,
          type: 'tool_call',
          tool_call: {
            id: 'call_with_metadata',
            name: 'approval_probe',
            args: '{}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toMatchObject({
      agentId: 'researcher',
      groupId: 7,
      tool_call: {
        id: 'call_with_metadata',
        output: 'approved',
      },
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_after_large_completion_index'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_after_large_completion_index',
        delta: {
          content: [{ type: ContentTypes.TEXT, text: 'Still dense' }],
        },
      } as t.MessageDeltaEvent,
    });
    expect(stepMap.get('step_after_large_completion_index')?.index).toBe(1);
    expect(contentParts[1]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Still dense',
    });
  });

  it('applies metadata from a hydrated run step to its seeded tool card', () => {
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();
    contentParts.push({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_hydrated',
        name: 'approval_probe',
        args: {},
      },
    });
    stepMap.set('step_hydrated', {
      id: 'step_hydrated',
      stepIndex: 0,
      type: StepTypes.TOOL_CALLS,
      index: 0,
      agentId: 'reviewer',
      groupId: 11,
      stepDetails: {
        type: StepTypes.TOOL_CALLS,
        tool_calls: [
          {
            id: 'call_hydrated',
            name: 'approval_probe',
            args: {},
          },
        ],
      },
      usage: null,
    });

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_COMPLETED,
      data: {
        result: {
          id: 'step_hydrated',
          index: 0,
          type: 'tool_call',
          tool_call: {
            id: 'call_hydrated',
            name: 'approval_probe',
            args: '{}',
            output: 'approved',
            progress: 1,
          } as t.ProcessedToolCall,
        },
      } as { result: t.ToolEndEvent },
    });

    expect(contentParts[0]).toMatchObject({
      agentId: 'reviewer',
      groupId: 11,
      tool_call: { output: 'approved' },
    });
  });

  it('drops an unknown completion for a parallel step', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();
    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP,
        data: {
          id: 'step_parallel_unknown',
          stepIndex: 0,
          type: StepTypes.TOOL_CALLS,
          index: 0,
          stepDetails: {
            type: StepTypes.TOOL_CALLS,
            tool_calls: [
              { id: 'call_known_first', name: 'first', args: {} },
              { id: 'call_known_second', name: 'second', args: {} },
            ],
          },
          usage: null,
        },
      });
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_parallel_unknown',
            index: 999,
            type: 'tool_call',
            tool_call: {
              id: 'call_unknown',
              name: 'unknown',
              args: '{}',
              output: 'must not overwrite',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(contentParts).toHaveLength(2);
      expect(contentParts[0]?.type).toBe(ContentTypes.TOOL_CALL);
      expect(contentParts[1]?.type).toBe(ContentTypes.TOOL_CALL);
      expect(warnSpy).toHaveBeenCalledWith(
        'No run step or tool call found for completed step event'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('drops a completion that matches neither a step nor seeded tool call', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { contentParts, aggregateContent } = createContentAggregator();

    try {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: '',
            index: 0,
            type: 'tool_call',
            tool_call: {
              id: 'call_unknown',
              name: 'unknown',
              args: '{}',
              output: 'untrusted',
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });

      expect(contentParts).toEqual([]);
      expect(warnSpy).toHaveBeenCalledWith(
        'No run step or tool call found for completed step event'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('ContentAggregator physical content indices', () => {
  it('allocates physical slots independently from streamed chunk keys', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_streamed_tool',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_streamed_tool',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              index: 2,
              id: 'call_streamed_first',
              name: 'approval_probe',
              args: '{"value":"first"}',
            },
            {
              index: 5,
              id: 'call_streamed_second',
              name: 'approval_probe',
              args: '{"value":"second"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(2);
    expect(contentParts[0]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_streamed_first',
        name: 'approval_probe',
      },
    });
    expect(contentParts[1]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_streamed_second',
        name: 'approval_probe',
      },
    });

    for (const [id, output] of [
      ['call_streamed_first', 'first complete'],
      ['call_streamed_second', 'second complete'],
    ]) {
      aggregateContent({
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: {
          result: {
            id: 'step_streamed_tool',
            index: 500,
            type: 'tool_call',
            tool_call: {
              id,
              name: 'approval_probe',
              args: '{}',
              output,
              progress: 1,
            } as t.ProcessedToolCall,
          },
        } as { result: t.ToolEndEvent },
      });
    }
    expect(contentParts[0]).toMatchObject({
      tool_call: { output: 'first complete' },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: { output: 'second complete' },
    });
  });

  it('keeps distinct chunk keys separate when tool ids arrive late', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_late_ids',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 2, name: 'first', args: '{"value":"first"}' },
          ],
        },
      } as t.RunStepDeltaEvent,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 5, name: 'second', args: '{"value":"second"}' },
          ],
        },
      } as t.RunStepDeltaEvent,
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_late_ids',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 2, id: 'call_late_first', name: 'first', args: {} },
            { index: 5, id: 'call_late_second', name: 'second', args: {} },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(2);
    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_late_first',
        name: 'first',
      },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: {
        id: 'call_late_second',
        name: 'second',
      },
    });
  });

  it('uses the reserved slot when a streamed tool id has no chunk index', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_without_chunk_index',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_without_chunk_index',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_without_chunk_index',
              name: 'approval_probe',
              args: '{}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(1);
    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_without_chunk_index',
        name: 'approval_probe',
      },
    });
  });

  it('routes an id-less chunk to the only known tool card', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_single_tool',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              id: 'call_single',
              name: 'approval_probe',
              args: {},
            },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_single_tool',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            {
              index: 9,
              name: 'approval_probe',
              args: '{"value":"updated"}',
            },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts).toHaveLength(1);
    expect(contentParts[0]).toMatchObject({
      type: ContentTypes.TOOL_CALL,
      tool_call: {
        id: 'call_single',
        args: '{"value":"updated"}',
      },
    });
  });

  it('routes indexed id-less chunks across known parallel tool cards', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: {
        id: 'step_parallel_chunks',
        stepIndex: 0,
        type: StepTypes.TOOL_CALLS,
        index: 0,
        stepDetails: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { id: 'call_chunk_first', name: 'first', args: {} },
            { id: 'call_chunk_second', name: 'second', args: {} },
          ],
        },
        usage: null,
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP_DELTA,
      data: {
        id: 'step_parallel_chunks',
        delta: {
          type: StepTypes.TOOL_CALLS,
          tool_calls: [
            { index: 7, name: 'first', args: '{"value":"first"}' },
            { index: 9, name: 'second', args: '{"value":"second"}' },
          ],
        },
      } as t.RunStepDeltaEvent,
    });

    expect(contentParts[0]).toMatchObject({
      tool_call: {
        id: 'call_chunk_first',
        args: '{"value":"first"}',
      },
    });
    expect(contentParts[1]).toMatchObject({
      tool_call: {
        id: 'call_chunk_second',
        args: '{"value":"second"}',
      },
    });
  });

  it('places a raw agent update after seeded content', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    contentParts.push(
      { type: ContentTypes.TEXT, text: 'Seed one' },
      { type: ContentTypes.TEXT, text: 'Seed two' }
    );

    aggregateContent({
      event: GraphEvents.ON_AGENT_UPDATE,
      data: {
        type: ContentTypes.AGENT_UPDATE,
        agent_update: {
          index: 0,
          runId: 'run_resumed',
          agentId: 'researcher',
        },
      },
    });

    expect(contentParts[2]).toEqual({
      type: ContentTypes.AGENT_UPDATE,
      agent_update: {
        index: 2,
        runId: 'run_resumed',
        agentId: 'researcher',
      },
    });
  });
});

describe('ContentAggregator provider-specific parts', () => {
  it('should preserve Gemini server-side tool content blocks', () => {
    const { contentParts, aggregateContent } = createContentAggregator();
    const toolCallPart: t.MessageContentComplex = {
      type: 'toolCall',
      toolCall: {
        id: 'server-search-1',
        name: 'google_search',
        args: {},
      },
    };
    const toolResponsePart: t.MessageContentComplex = {
      type: 'toolResponse',
      toolResponse: {
        id: 'server-search-1',
        name: 'google_search',
        response: { results: [] },
      },
    };

    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: createRunStep('step_tool_call'),
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_tool_call',
        delta: { content: [toolCallPart] },
      },
    });
    aggregateContent({
      event: GraphEvents.ON_RUN_STEP,
      data: { ...createRunStep('step_tool_response'), index: 1 },
    });
    aggregateContent({
      event: GraphEvents.ON_MESSAGE_DELTA,
      data: {
        id: 'step_tool_response',
        delta: { content: [toolResponsePart] },
      },
    });

    expect(contentParts[0]).toEqual(toolCallPart);
    expect(contentParts[1]).toEqual(toolResponsePart);
  });
});

describe('ContentAggregator with SplitStreamHandler', () => {
  it('should aggregate content from multiple message blocks', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'First sentence. Second sentence. Third sentence.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    expect(contentParts.length).toBeGreaterThan(0);
    contentParts.forEach((part) => {
      expect(part?.type).toBe(ContentTypes.TEXT);
      if (part?.type === ContentTypes.TEXT) {
        expect(typeof part.text).toBe('string');
        expect(part.text.length).toBeGreaterThan(0);
      }
    });

    const fullText = contentParts
      .filter((part) => part?.type === ContentTypes.TEXT)
      .map((part) => (part?.type === ContentTypes.TEXT ? part.text : ''))
      .join('');
    expect(fullText).toBe(text);
  });

  it('should maintain content order across splits', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 15,
    });

    const text = 'First part. Second part. Third part.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const texts = contentParts
      .filter((part) => part?.type === ContentTypes.TEXT)
      .map((part) => (part?.type === ContentTypes.TEXT ? part.text : ''));

    expect(texts[0]).toContain('First');
    expect(texts[texts.length - 1]).toContain('Third');
  });

  it('should handle code blocks as single content parts', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 10,
    });

    const text = `Before code.
\`\`\`python
def test():
    return True
\`\`\`
After code.`;

    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const codeBlockPart = contentParts.find(
      (part) =>
        part?.type === ContentTypes.TEXT &&
        part.text.includes('```python') === true
    );

    expect(codeBlockPart).toBeDefined();
    if (codeBlockPart?.type === ContentTypes.TEXT) {
      expect(codeBlockPart.text).toContain('def test()');
      expect(codeBlockPart.text).toContain('return True');
    }
  });

  it('should properly map steps to their content', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'Hi. Ok. Yes.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    Array.from(stepMap.entries()).forEach(([_stepId, step]) => {
      expect(step?.type).toBe(StepTypes.MESSAGE_CREATION);
      const currentIndex = step?.index ?? -1;
      const stepContent = contentParts[currentIndex];
      if (!stepContent && currentIndex > 0) {
        const prevStepContent = contentParts[currentIndex - 1];
        expect(
          (prevStepContent as MessageContentText | undefined)?.text
        ).toEqual(text);
      } else if (stepContent?.type === ContentTypes.TEXT) {
        expect(stepContent.text.length).toBeGreaterThan(0);
      }
    });

    contentParts.forEach((part, index) => {
      const hasMatchingStep = Array.from(stepMap.values()).some(
        (step) => step?.index === index
      );
      expect(hasMatchingStep).toBe(true);
    });
  });

  it('should aggregate content across multiple splits while preserving order', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 10,
    });

    const text = 'A. B. C. D. E. F.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const letters = ['A', 'B', 'C', 'D', 'E', 'F'];
    let letterIndex = 0;

    contentParts.forEach((part) => {
      if (part?.type === ContentTypes.TEXT) {
        while (
          letterIndex < letters.length &&
          part.text.includes(letters[letterIndex]) === true
        ) {
          letterIndex++;
        }
      }
    });

    expect(letterIndex).toBe(letters.length);
  });
});

describe('SplitStreamHandler with Reasoning Tokens', () => {
  it('should apply same splitting rules to both content types', async () => {
    const runId = nanoid();
    const mockHandlers: t.SplitStreamHandlers = {
      [GraphEvents.ON_RUN_STEP]: jest.fn(),
      [GraphEvents.ON_MESSAGE_DELTA]: jest.fn(),
      [GraphEvents.ON_REASONING_DELTA]: jest.fn(),
    };

    const handler = new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
      blockThreshold: 3,
    });

    const stream = createMockStream({
      text: 'First text. Second text. Third text.',
      reasoningText: 'First thought. Second thought. Third thought.',
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const runSteps = (mockHandlers[GraphEvents.ON_RUN_STEP] as jest.Mock).mock
      .calls;
    const reasoningDeltas = (
      mockHandlers[GraphEvents.ON_REASONING_DELTA] as jest.Mock
    ).mock.calls;
    const messageDeltas = (
      mockHandlers[GraphEvents.ON_MESSAGE_DELTA] as jest.Mock
    ).mock.calls;

    // Both content types should create multiple blocks
    expect(runSteps.length).toBeGreaterThan(1);
    expect(reasoningDeltas.length).toBeGreaterThan(0);
    expect(messageDeltas.length).toBeGreaterThan(0);

    // Verify splitting behavior for both types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const getStepTypes = (calls: any[]): string[] =>
      calls
        .map(([{ data }]) =>
          data.stepDetails?.type === StepTypes.MESSAGE_CREATION
            ? data.stepDetails.message_creation.message_id
            : null
        )
        .filter(Boolean);

    const messageSteps = getStepTypes(runSteps);
    expect(new Set(messageSteps).size).toBeGreaterThan(1);
  });

  it('should properly map steps to their reasoning content', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
        [GraphEvents.ON_REASONING_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'Main content.';
    const reasoningText = 'First thought. Second thought. Third thought.';
    const stream = createMockStream({
      text,
      reasoningText,
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    Array.from(stepMap.entries()).forEach(([_stepId, step]) => {
      expect(step?.type).toBe(StepTypes.MESSAGE_CREATION);
      const currentIndex = step?.index ?? -1;
      const stepContent = contentParts[currentIndex];

      if (stepContent?.type === ContentTypes.THINK) {
        // Verify reasoning content structure
        expect(stepContent).toHaveProperty('think');
        expect(typeof stepContent.think).toBe('string');
        expect(stepContent.think.length).toBeGreaterThan(0);
      }
    });

    // Verify at least one reasoning content part exists
    const reasoningParts = contentParts.filter(
      (part) => part?.type === ContentTypes.THINK
    );
    expect(reasoningParts.length).toBeGreaterThan(0);

    // Verify the content order (reasoning should come before main content)
    const contentTypes = contentParts
      .filter((part) => part !== undefined)
      .map((part) => part.type);

    expect(contentTypes).toContain(ContentTypes.THINK);
    expect(contentTypes).toContain(ContentTypes.TEXT);

    // Verify the complete reasoning content is preserved
    const fullReasoningText = reasoningParts
      .map((part) => (part?.type === ContentTypes.THINK ? part.think : ''))
      .join('');
    expect(fullReasoningText).toBe(reasoningText);
  });
});

describe('SplitStreamHandler', () => {
  it('should handle think blocks correctly', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];

    const streamHandler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: ({ data }): void => {
          messageDeltaEvents.push(data);
        },
        [GraphEvents.ON_REASONING_DELTA]: ({ data }): void => {
          reasoningDeltaEvents.push(data);
        },
      },
    });

    const content =
      'Here\'s some regular text. <think>Now I\'m thinking deeply about something important. This should all be reasoning.</think> Back to regular text.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Check that content before <think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Here\'s') === true
      )
    ).toBe(true);

    // Check that <think> tag was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('<think>') === true
      )
    ).toBe(true);

    // Check that content inside <think> tags was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('thinking') === true
      )
    ).toBe(true);

    // Check that </think> tag was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('</think>') === true
      )
    ).toBe(true);

    // Check that content after </think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Back') === true
      )
    ).toBe(true);
  });

  it('should ignore think tags inside code blocks', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];

    const streamHandler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: ({ data }): void => {
          messageDeltaEvents.push(data);
        },
        [GraphEvents.ON_REASONING_DELTA]: ({ data }): void => {
          reasoningDeltaEvents.push(data);
        },
      },
    });

    const content =
      'Regular text. ```<think>This should stay as code</think>``` More text.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Check that think tags inside code blocks were treated as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Regular') === true
      )
    ).toBe(true);

    // Verify no reasoning events were generated
    expect(reasoningDeltaEvents.length).toBe(0);
  });

  it('should properly split content with think tags while maintaining context', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];
    const runStepEvents: t.RunStep[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();

    const streamHandler = new SplitStreamHandler({
      runId,
      blockThreshold: 20, // Small threshold to force splits
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: (event): void => {
          messageDeltaEvents.push(event.data);
          aggregateContent(event);
        },
        [GraphEvents.ON_REASONING_DELTA]: (event): void => {
          reasoningDeltaEvents.push(event.data);
          aggregateContent(event);
        },
        [GraphEvents.ON_RUN_STEP]: (event): void => {
          runStepEvents.push(event.data);
          aggregateContent(event);
        },
      },
    });

    const content =
      'Here\'s some regular text. <think>Now I\'m thinking deeply about something important. This is a long thought that should be split into multiple parts. We want to ensure the splitting works correctly.</think> Back to regular text after thinking.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Verify that multiple message blocks were created
    expect(runStepEvents.length).toBeGreaterThan(2);

    // Check that content before <think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('regular') === true
      )
    ).toBe(true);

    // Verify that reasoning content was split into multiple parts
    const reasoningParts = reasoningDeltaEvents
      .map(
        (event) =>
          (event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined)
            ?.think
      )
      .filter(Boolean);
    expect(reasoningParts.length).toBeGreaterThan(1);

    // Verify that the complete reasoning content is preserved when joined
    const fullReasoningContent = reasoningParts.join('');
    expect(fullReasoningContent).toContain('thinking');
    expect(fullReasoningContent).toContain('should');
    expect(fullReasoningContent).toContain('be');
    expect(fullReasoningContent).toContain('split');

    // Check that each reasoning part maintains proper think context
    let seenThinkOpen = false;
    let seenThinkClose = false;
    reasoningParts.forEach((part) => {
      if (part == null) return;
      if (part.includes('<think>')) {
        seenThinkOpen = true;
      }
      if (part.includes('</think>')) {
        seenThinkClose = true;
      }
      // Middle parts should be handled as reasoning even without explicit think tags
      if (!part.includes('<think>') && !part.includes('</think>')) {
        expect(
          reasoningDeltaEvents.some(
            (event) =>
              (event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined)
                ?.think === part
          )
        ).toBe(true);
      }
    });
    expect(seenThinkOpen).toBe(true);
    expect(seenThinkClose).toBe(true);

    // Check that content after </think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Back') === true
      )
    ).toBe(true);

    const thinkingBlocks = contentParts.filter(
      (part) => part?.type === ContentTypes.THINK
    );
    expect(thinkingBlocks.length).toBeGreaterThan(0);
    expect(
      (thinkingBlocks[0] as t.ReasoningContentText).think.startsWith('<think>')
    ).toBeTruthy();
  });
});
