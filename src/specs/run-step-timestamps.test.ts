import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { MemorySaver, Command } from '@langchain/langgraph';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { GraphEvents, StepTypes, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { askUserQuestion } from '@/hitl';
import { Run } from '@/run';

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

const llmConfig: t.LLMConfig = {
  provider: Providers.OPENAI,
  streaming: true,
  streamUsage: false,
};

const echoTool = tool(
  async ({ input }: { input: string }) => `echo: ${input}`,
  {
    name: 'echoTool',
    description: 'Echoes the input back',
    schema: z.object({ input: z.string() }),
  }
);

const toolCalls: ToolCall[] = [
  {
    name: 'echoTool',
    args: { input: 'ping' },
    id: 'call_timestamps_1',
    type: 'tool_call',
  },
];

const createStreamConfig = (
  threadId: string,
  signal?: AbortSignal
): t.RunStreamConfig => ({
  configurable: { thread_id: threadId },
  version: 'v2',
  ...(signal != null ? { signal } : {}),
});

type RecordedEvent = {
  event: string;
  data: unknown;
  /** Lifecycle fields captured at receipt — dispatched RunStep objects are
   *  shared references that later closes mutate in place. */
  statusAtReceipt?: t.RunStepStatus;
};

function createRecorder(): {
  sequence: RecordedEvent[];
  handlers: Record<string, t.EventHandler>;
  } {
  const sequence: RecordedEvent[] = [];
  const record = (event: string): t.EventHandler => ({
    handle: (_event, data): void => {
      sequence.push({
        event,
        data,
        ...(event === GraphEvents.ON_RUN_STEP
          ? { statusAtReceipt: (data as t.RunStep).status }
          : {}),
      });
    },
  });
  return {
    sequence,
    handlers: {
      [GraphEvents.ON_RUN_STEP]: record(GraphEvents.ON_RUN_STEP),
      [GraphEvents.ON_RUN_STEP_COMPLETED]: record(
        GraphEvents.ON_RUN_STEP_COMPLETED
      ),
      [GraphEvents.ON_RUN_STEP_CLOSED]: record(GraphEvents.ON_RUN_STEP_CLOSED),
    },
  };
}

function getSteps(sequence: RecordedEvent[]): t.RunStep[] {
  return sequence
    .filter((entry) => entry.event === GraphEvents.ON_RUN_STEP)
    .map((entry) => entry.data as t.RunStep);
}

function getClosed(sequence: RecordedEvent[]): t.RunStepClosedEvent[] {
  return sequence
    .filter((entry) => entry.event === GraphEvents.ON_RUN_STEP_CLOSED)
    .map((entry) => entry.data as t.RunStepClosedEvent);
}

describe('run step timestamps', () => {
  jest.setTimeout(20000);

  it('stamps starts and closes every step exactly once on a natural finish', async () => {
    const { sequence, handlers } = createRecorder();
    const run = await Run.create<t.IState>({
      runId: 'test-run-step-timestamps',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [echoTool],
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: handlers,
    });

    run.Graph?.overrideTestModel(
      ['Let me call the tool', 'The tool answered'],
      2,
      toolCalls
    );
    const started = Date.now();
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      createStreamConfig('run-step-timestamps')
    );

    const steps = getSteps(sequence);
    const closed = getClosed(sequence);
    expect(steps.length).toBeGreaterThanOrEqual(3);
    const stepReceipts = sequence.filter(
      (entry) => entry.event === GraphEvents.ON_RUN_STEP
    );
    for (const receipt of stepReceipts) {
      expect(receipt.statusAtReceipt).toBe('in_progress');
    }
    for (const step of steps) {
      expect(typeof step.created_at).toBe('number');
      expect(step.created_at as number).toBeGreaterThanOrEqual(started);
    }

    const stepIds = steps.map((step) => step.id);
    expect(closed.map((event) => event.id).sort()).toEqual([...stepIds].sort());
    for (const event of closed) {
      expect(event.status).toBe('completed');
      expect(event.closed_at).toBeGreaterThanOrEqual(
        event.created_at as number
      );
    }

    const toolStep = steps.find((step) => step.type === StepTypes.TOOL_CALLS);
    expect(toolStep).toBeDefined();
    const toolStepId = (toolStep as t.RunStep).id;
    const toolCompletedIndex = sequence.findIndex(
      (entry) =>
        entry.event === GraphEvents.ON_RUN_STEP_COMPLETED &&
        (entry.data as { result?: { id?: string } }).result?.id === toolStepId
    );
    const toolClosedIndex = sequence.findIndex(
      (entry) =>
        entry.event === GraphEvents.ON_RUN_STEP_CLOSED &&
        (entry.data as t.RunStepClosedEvent).id === toolStepId
    );
    expect(toolCompletedIndex).toBeGreaterThanOrEqual(0);
    expect(toolClosedIndex).toBeGreaterThan(toolCompletedIndex);

    const firstMessageStepId = steps[0].id;
    const toolStepIndex = sequence.findIndex(
      (entry) =>
        entry.event === GraphEvents.ON_RUN_STEP &&
        (entry.data as t.RunStep).id === toolStepId
    );
    const firstMessageClosedIndex = sequence.findIndex(
      (entry) =>
        entry.event === GraphEvents.ON_RUN_STEP_CLOSED &&
        (entry.data as t.RunStepClosedEvent).id === firstMessageStepId
    );
    expect(firstMessageClosedIndex).toBeGreaterThanOrEqual(0);
    expect(firstMessageClosedIndex).toBeLessThan(toolStepIndex);

    const completedEvent = sequence
      .filter((entry) => entry.event === GraphEvents.ON_RUN_STEP_COMPLETED)
      .map((entry) => (entry.data as { result: t.ToolCompleteEvent }).result)
      .find((result) => result.id === toolStepId);
    expect(typeof completedEvent?.completed_at).toBe('number');

    for (const step of run.Graph?.contentData ?? []) {
      expect(step.status).toBe('completed');
      expect(typeof step.completed_at).toBe('number');
    }
  });

  it('changes nothing for hosts that only register the legacy handlers', async () => {
    const received: string[] = [];
    const legacyHandler = (event: string): t.EventHandler => ({
      handle: (): void => {
        received.push(event);
      },
    });
    const run = await Run.create<t.IState>({
      runId: 'test-run-step-timestamps-legacy',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [echoTool],
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: legacyHandler(GraphEvents.ON_RUN_STEP),
        [GraphEvents.ON_RUN_STEP_COMPLETED]: legacyHandler(
          GraphEvents.ON_RUN_STEP_COMPLETED
        ),
        [GraphEvents.ON_MESSAGE_DELTA]: legacyHandler(
          GraphEvents.ON_MESSAGE_DELTA
        ),
      },
    });

    run.Graph?.overrideTestModel(
      ['Let me call the tool', 'The tool answered'],
      2,
      toolCalls
    );
    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      createStreamConfig('run-step-timestamps-legacy')
    );

    expect(new Set(received)).toEqual(
      new Set([
        GraphEvents.ON_RUN_STEP,
        GraphEvents.ON_RUN_STEP_COMPLETED,
        GraphEvents.ON_MESSAGE_DELTA,
      ])
    );
    for (const step of run.Graph?.contentData ?? []) {
      expect(step.status).toBe('completed');
    }
  });

  it('sweeps unfinished steps as cancelled when the caller aborts mid-stream', async () => {
    const { sequence, handlers } = createRecorder();
    const controller = new AbortController();
    const run = await Run.create<t.IState>({
      runId: 'test-run-step-timestamps-abort',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        ...handlers,
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            controller.abort();
          },
        },
      },
    });

    run.Graph?.overrideTestModel(
      ['This response streams slowly enough to abort in flight'],
      25
    );
    await expect(
      run.processStream(
        { messages: [new HumanMessage('hello')] },
        createStreamConfig('run-step-timestamps-abort', controller.signal)
      )
    ).rejects.toThrow();

    const closed = getClosed(sequence);
    expect(closed.length).toBeGreaterThanOrEqual(1);
    expect(closed.every((event) => event.status === 'cancelled')).toBe(true);
    for (const step of run.Graph?.contentData ?? []) {
      expect(step.status).toBe('cancelled');
      expect(typeof step.cancelled_at).toBe('number');
      expect(step.cancelled_at as number).toBeGreaterThanOrEqual(
        step.created_at as number
      );
    }
  });

  it('sweeps as failed when an AbortError surfaces with no signal aborted', async () => {
    const { sequence, handlers } = createRecorder();
    const run = await Run.create<t.IState>({
      runId: 'test-run-step-timestamps-uncorroborated-abort',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        ...handlers,
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            /** A provider/host rejection that merely borrows the name — no
             *  caller or construction signal was ever aborted. */
            const error = new Error('provider stream aborted internally');
            error.name = 'AbortError';
            throw error;
          },
        },
      },
    });

    run.Graph?.overrideTestModel(['some streamed content'], 2);
    await expect(
      run.processStream(
        { messages: [new HumanMessage('hello')] },
        createStreamConfig('run-step-timestamps-uncorroborated-abort')
      )
    ).rejects.toThrow();

    const closed = getClosed(sequence);
    expect(closed.length).toBeGreaterThanOrEqual(1);
    expect(closed.every((event) => event.status === 'failed')).toBe(true);
    for (const step of run.Graph?.contentData ?? []) {
      expect(step.status).toBe('failed');
      expect(typeof step.failed_at).toBe('number');
      expect(step.cancelled_at).toBeUndefined();
    }
  });

  it('keeps steps open across a HITL interrupt and closes them on resume with the original created_at', async () => {
    jest.setTimeout(30000);
    const { sequence, handlers } = createRecorder();
    const saver = new MemorySaver();
    const run = await Run.create<t.IState>({
      runId: 'test-run-step-timestamps-hitl',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'timestamps-hitl-agent',
            provider: Providers.OPENAI,
            clientOptions: {
              model: 'gpt-4o-mini',
              streaming: true,
              streamUsage: false,
            },
            instructions: 'You are a helpful assistant.',
            maxContextTokens: 8000,
            graphTools: [askTool],
          },
        ],
        compileOptions: { checkpointer: saver },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: handlers,
    });
    run.Graph!.overrideModel = new FakeChatModel({
      responses: ['asking the user', 'done after resume'],
      toolCalls: [
        {
          name: 'ask_user_question',
          args: { question: 'pick one' },
          id: 'call_ask_timestamps',
          type: 'tool_call',
        },
      ],
    });

    const streamConfig = createStreamConfig('run-step-timestamps-hitl');
    await run.processStream(
      { messages: [new HumanMessage('go')] },
      streamConfig
    );

    expect(run.getInterrupt()).toBeDefined();
    const toolStep = run.Graph?.contentData.find(
      (step) => step.type === StepTypes.TOOL_CALLS
    ) as t.RunStep;
    expect(toolStep).toBeDefined();
    expect(toolStep.status).toBe('in_progress');
    const createdAt = toolStep.created_at;
    expect(typeof createdAt).toBe('number');
    expect(getClosed(sequence).map((event) => event.id)).not.toContain(
      toolStep.id
    );

    await run.processStream(
      new Command({ resume: { answer: 'blue' } }) as unknown as t.IState,
      streamConfig
    );

    expect(toolStep.status).toBe('completed');
    expect(toolStep.created_at).toBe(createdAt);
    const closedForTool = getClosed(sequence).filter(
      (event) => event.id === toolStep.id
    );
    expect(closedForTool).toHaveLength(1);
    expect(closedForTool[0].closed_at).toBeGreaterThanOrEqual(
      createdAt as number
    );
    for (const step of run.Graph?.contentData ?? []) {
      expect(step.status).toBe('completed');
    }
  });

  it('rehydrates and closes open steps when HITL resumes in a fresh process', async () => {
    const { sequence, handlers } = createRecorder();
    const saver = new MemorySaver();
    const createRun = async (resume: boolean): Promise<Run<t.IState>> => {
      const run = await Run.create<t.IState>({
        runId: 'test-run-step-timestamps-cross-process',
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'timestamps-cross-process-agent',
              provider: Providers.OPENAI,
              clientOptions: {
                model: 'gpt-4o-mini',
                streaming: true,
                streamUsage: false,
              },
              instructions: 'You are a helpful assistant.',
              maxContextTokens: 8000,
              graphTools: [askTool],
            },
          ],
          compileOptions: { checkpointer: saver },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: handlers,
      });
      run.Graph!.overrideModel = new FakeChatModel(
        resume
          ? { responses: ['done after resume'] }
          : {
            responses: ['asking the user'],
            toolCalls: [
              {
                name: 'ask_user_question',
                args: { question: 'pick one' },
                id: 'call_ask_cross_process',
                type: 'tool_call',
              },
            ],
          }
      );
      return run;
    };

    const streamConfig = createStreamConfig(
      'run-step-timestamps-cross-process'
    );
    const firstRun = await createRun(false);
    await firstRun.processStream(
      { messages: [new HumanMessage('go')] },
      streamConfig
    );

    const openStep = firstRun.Graph?.contentData.find(
      (step) => step.type === StepTypes.TOOL_CALLS
    ) as t.RunStep;
    expect(firstRun.getInterrupt()?.payload).toMatchObject({
      type: 'ask_user_question',
    });
    expect(openStep.status).toBe('in_progress');
    expect(getClosed(sequence).map((event) => event.id)).not.toContain(
      openStep.id
    );

    const resumedRun = await createRun(true);
    await resumedRun.resume({ answer: 'blue' }, streamConfig);

    const restoredStep = resumedRun.Graph?.getRunStep(openStep.id);
    expect(restoredStep).toMatchObject({
      id: openStep.id,
      index: openStep.index,
      created_at: openStep.created_at,
      status: 'completed',
    });
    expect(
      getSteps(sequence).filter((step) => step.id === openStep.id)
    ).toHaveLength(1);
    expect(
      getClosed(sequence).filter((event) => event.id === openStep.id)
    ).toEqual([
      expect.objectContaining({
        id: openStep.id,
        index: openStep.index,
        created_at: openStep.created_at,
        status: 'completed',
      }),
    ]);
  });
});
