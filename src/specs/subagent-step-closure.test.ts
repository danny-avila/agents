import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
} from '@/index';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

const CHILD_RESPONSE = 'Research result: Paris is the capital of France.';

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'subagent-step-closure-thread' },
  streamMode: 'values',
  version: 'v2' as const,
};

const createParentAgent = (): t.AgentInputs => ({
  agentId: 'parent',
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: 'You are a supervisor. Delegate research using the subagent.',
  maxContextTokens: 8000,
  subagentConfigs: [
    {
      type: 'researcher',
      name: 'Research Agent',
      description: 'Researches and summarizes information',
      agentInputs: {
        agentId: 'researcher',
        provider: Providers.OPENAI,
        clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
        instructions: 'You are a research agent. Answer concisely.',
        maxContextTokens: 8000,
      },
    },
  ],
});

const subagentToolCall: ToolCall = {
  id: 'call_subagent_1',
  name: Constants.SUBAGENT,
  args: {
    description: 'What is the capital of France?',
    subagent_type: 'researcher',
  },
  type: 'tool_call',
};

type CapturedUpdate = {
  phase: t.SubagentUpdatePhase;
  data?: unknown;
};

type UpdateCapture = {
  updates: CapturedUpdate[];
  handlers: Record<string, t.EventHandler>;
};

/**
 * Collects the child-run lifecycle envelopes the executor forwards to the
 * parent. Child run steps only reach a host through these, so they are the
 * only place a subagent's closure is observable.
 */
function createUpdateCapture(): UpdateCapture {
  const updates: CapturedUpdate[] = [];
  const handlers: Record<string, t.EventHandler> = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_SUBAGENT_UPDATE]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const update = data as unknown as t.SubagentUpdateEvent;
        updates.push({ phase: update.phase, data: update.data });
      },
    },
  };
  return { updates, handlers };
}

describe('Subagent child-graph run step closure', () => {
  jest.setTimeout(30000);

  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            constructor(_options: any) {
              super({ responses: [CHILD_RESPONSE] });
            }
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  it('closes the child run steps it opened when the subagent completes', async () => {
    const { updates, handlers } = createUpdateCapture();

    const run = await Run.create<t.IState>({
      runId: `subagent-closure-${Date.now()}`,
      graphConfig: { type: 'standard', agents: [createParentAgent()] },
      returnContent: true,
      skipCleanup: true,
      customHandlers: handlers,
    });

    run.Graph?.overrideTestModel(
      ['Delegating this research.', `Based on the research: ${CHILD_RESPONSE}`],
      10,
      [subagentToolCall]
    );

    await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      callerConfig
    );

    const opened = updates.filter((update) => update.phase === 'run_step');
    const closed = updates.filter(
      (update) => update.phase === 'run_step_closed'
    );

    expect(opened.length).toBeGreaterThan(0);
    expect(closed.length).toBeGreaterThan(0);
    expect(
      closed.every(
        (update) =>
          (update.data as { status?: string } | undefined)?.status ===
          'completed'
      )
    ).toBe(true);
  });

  it('closes the child run steps as cancelled when the caller aborts mid-child', async () => {
    const updates: CapturedUpdate[] = [];
    const controller = new AbortController();

    const run = await Run.create<t.IState>({
      runId: `subagent-closure-abort-${Date.now()}`,
      graphConfig: { type: 'standard', agents: [createParentAgent()] },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        [GraphEvents.ON_SUBAGENT_UPDATE]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            const update = data as unknown as t.SubagentUpdateEvent;
            updates.push({ phase: update.phase, data: update.data });
            /** The child has opened a step — abort while it is still open. */
            if (update.phase === 'run_step') {
              controller.abort();
            }
          },
        },
      },
    });

    run.Graph?.overrideTestModel(
      ['Delegating this research.', `Based on the research: ${CHILD_RESPONSE}`],
      25,
      [subagentToolCall]
    );

    await run
      .processStream(
        { messages: [new HumanMessage('What is the capital of France?')] },
        {
          ...callerConfig,
          configurable: { thread_id: 'subagent-step-closure-abort' },
          signal: controller.signal,
        }
      )
      .catch(() => {
        /** The abort is the point; the rejection shape is not under test. */
      });

    const closed = updates.filter(
      (update) => update.phase === 'run_step_closed'
    );
    expect(closed.length).toBeGreaterThan(0);
    expect(
      closed.every(
        (update) =>
          (update.data as { status?: string } | undefined)?.status ===
          'cancelled'
      )
    ).toBe(true);
  });
});
