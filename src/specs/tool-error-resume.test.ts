import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { MemorySaver, Command } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { askUserQuestion } from '@/hitl';
import { StandardGraph } from '@/graphs';
import { Run } from '@/run';

/**
 * Regression: a DIRECT (graphTools) tool that fails schema validation during
 * the RESUME pass of an interrupted batch.
 *
 * LangGraph re-executes the whole interrupted batch on resume, and a
 * fast-failing sibling (e.g. a zod reject) errors BEFORE the rebuilt graph
 * has registered run steps for the replayed calls. `handleToolCallErrorStatic`
 * used to throw (`No config provided`) in that state, which surfaced as a
 * scary `Error in errorHandler` log on every such resume — the error itself
 * was already handled (the model receives the error ToolMessage either way).
 * It now reports `false` (nothing dispatched) so the ToolNode can decide, and
 * the client-facing contract stays: the error completion event dispatches
 * exactly ONCE, on the pass where the call's run step is live (the first
 * pass), never zero times and never twice.
 */
const askTool = tool(
  async (input) => {
    const { answer } = askUserQuestion(input as { question: string });
    return answer;
  },
  {
    name: 'ask_user_question',
    description:
      'Ask the user a clarifying question and wait for their answer.',
    schema: z.object({ question: z.string() }),
  }
);

const strictTool = tool(
  async ({ items }: { items: string[] }) => `got ${items.length}`,
  {
    name: 'strict_tool',
    // Array cap mirrors the field repro: an ask_user_question sibling whose
    // `options` array exceeded the 12-option schema cap.
    description: 'A tool with a strict array-cap schema.',
    schema: z.object({ items: z.array(z.string()).max(1) }),
  }
);

describe('Direct tool schema error across interrupt/resume', () => {
  jest.setTimeout(30000);

  const threadConfig: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    streamMode: string;
  } = {
    configurable: { thread_id: 'tool-error-resume-thread' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const completedToolEvents: Array<{ name?: string; output?: string }> = [];
  const customHandlers = {
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (_event: GraphEvents, data: t.StreamEventData): void => {
        const result = (data as { result?: t.ToolCompleteEvent }).result;
        if (result?.type === 'tool_call') {
          completedToolEvents.push({
            name: result.tool_call.name,
            output: result.tool_call.output,
          });
        }
      },
    },
  };

  async function buildRun(
    saver: MemorySaver,
    runId: string
  ): Promise<Run<t.IState>> {
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'resume-error-agent',
            provider: Providers.OPENAI,
            clientOptions: {
              model: 'gpt-4o-mini',
              streaming: true,
              streamUsage: false,
            },
            instructions: 'You are a helpful assistant.',
            maxContextTokens: 8000,
            /** Non-empty toolDefinitions flips the ToolNode to event dispatch —
             *  the production shape — while graphTools ride the direct path. */
            toolDefinitions: [
              { name: 'dummy_event_tool', description: 'host tool' },
            ],
            graphTools: [askTool, strictTool],
          },
        ],
        compileOptions: { checkpointer: saver },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      eagerEventToolExecution: {
        enabled: true,
        excludeToolNames: ['ask_user_question', 'strict_tool'],
      },
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['calling tools', 'done after resume'],
      toolCalls: [
        {
          name: 'ask_user_question',
          args: { question: 'pick one' },
          id: 'call_ask_1',
          type: 'tool_call',
        },
        {
          name: 'strict_tool',
          args: { items: ['a', 'b', 'c'] },
          id: 'call_strict_1',
          type: 'tool_call',
        },
      ],
    });
    return run;
  }

  test('error completion dispatches exactly once; resume-pass handler reports false instead of throwing', async () => {
    const saver = new MemorySaver();
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const staticSpy = jest.spyOn(StandardGraph, 'handleToolCallErrorStatic');

    try {
      /** Pass 1: the strict tool rejects its args while the ask tool
       *  interrupts — the run pauses AND the schema error completes. */
      const run1 = await buildRun(saver, 'run-pass-1');
      await run1.processStream(
        { messages: [new HumanMessage('go')] },
        threadConfig
      );

      const strictCompletions = completedToolEvents.filter(
        (e) =>
          e.name === 'strict_tool' &&
          e.output?.includes('Error processing tool') === true
      );
      expect(strictCompletions).toHaveLength(1);
      // Stable LangChain wrapper text (version-independent), proving the
      // completion carried the schema-validation failure.
      expect(strictCompletions[0].output).toContain(
        'did not match expected schema'
      );

      /** Pass 2: fresh instance (host restart shape), resume with the
       *  answer. The batch re-executes; the strict tool fails again before
       *  any run step is registered on the rebuilt graph. */
      const run2 = await buildRun(saver, 'run-pass-2');
      await run2.processStream(
        new Command({ resume: { answer: 'blue' } }) as unknown as t.IState,
        threadConfig
      );

      /** The handler reported "not dispatched" (no run step yet) rather than
       *  throwing — so nothing was logged and no duplicate completion event
       *  reached the client. */
      const resumeResults = await Promise.all(
        staticSpy.mock.results.map((r) => r.value as Promise<boolean>)
      );
      expect(resumeResults).toContain(false);
      const handlerFailures = errorSpy.mock.calls.filter((args) =>
        String(args[0]).includes('Error in errorHandler')
      );
      expect(handlerFailures).toHaveLength(0);
      expect(
        completedToolEvents.filter(
          (e) =>
            e.name === 'strict_tool' &&
            e.output?.includes('Error processing tool') === true
        )
      ).toHaveLength(1);
    } finally {
      errorSpy.mockRestore();
      staticSpy.mockRestore();
    }
  });
});
