import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import type {
  HookCallback,
  SubagentStartHookInput,
  SubagentStartHookOutput,
  SubagentStopHookInput,
  SubagentStopHookOutput,
} from '@/hooks/types';
import { HookRegistry } from '@/hooks/HookRegistry';
import { Run } from '@/run';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
} from '@/index';
import * as providers from '@/llm/providers';

const CHILD_RESPONSE = 'Hook test child response.';

const callerConfig = {
  configurable: { thread_id: 'hook-test-thread' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

const originalGetChatModelClass = providers.getChatModelClass;

function makeSubagentToolCall(): ToolCall {
  return {
    name: Constants.SUBAGENT,
    args: {
      description: 'Test task for hook verification',
      subagent_type: 'researcher',
    },
    id: `call_sub_${Date.now()}`,
    type: 'tool_call',
  };
}

function createParentAgent(): t.AgentInputs {
  return {
    agentId: 'hook-parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: 'Delegate research tasks to subagents.',
    maxContextTokens: 8000,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher-child',
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Answer concisely.',
          maxContextTokens: 8000,
        },
      },
    ],
  };
}

async function createSubagentRun(
  hooks: HookRegistry,
  runId = `subagent-hook-${Date.now()}`
): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [createParentAgent()],
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    },
    hooks,
  });
}

describe('Subagent hook integration (end-to-end via Run)', () => {
  jest.setTimeout(15000);

  let getChatModelClassSpy: jest.SpyInstance;

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

  it('SubagentStart fires with correct payload through real Run pipeline', async () => {
    const registry = new HookRegistry();
    let captured: SubagentStartHookInput | undefined;

    const hook: HookCallback<'SubagentStart'> = async (
      input
    ): Promise<SubagentStartHookOutput> => {
      captured = input;
      return {};
    };
    registry.register('SubagentStart', { hooks: [hook] });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    expect(captured).toBeDefined();
    expect(captured!.hook_event_name).toBe('SubagentStart');
    expect(captured!.agentType).toBe('researcher');
    expect(captured!.parentAgentId).toBe('hook-parent');
    expect(captured!.threadId).toBe('hook-test-thread');
    expect(captured!.inputs).toHaveLength(1);
    expect(captured!.inputs[0].content).toContain(
      'Test task for hook verification'
    );
  });

  it('SubagentStop fires with messages from child execution', async () => {
    const registry = new HookRegistry();
    let captured: SubagentStopHookInput | undefined;

    const hook: HookCallback<'SubagentStop'> = async (
      input
    ): Promise<SubagentStopHookOutput> => {
      captured = input;
      return {};
    };
    registry.register('SubagentStop', { hooks: [hook] });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    expect(captured).toBeDefined();
    expect(captured!.hook_event_name).toBe('SubagentStop');
    expect(captured!.agentType).toBe('researcher');
    expect(captured!.threadId).toBe('hook-test-thread');
    expect(captured!.messages.length).toBeGreaterThan(0);
  });

  it('SubagentStart deny blocks subagent execution and returns blocked message', async () => {
    const registry = new HookRegistry();
    const denyHook: HookCallback<
      'SubagentStart'
    > = async (): Promise<SubagentStartHookOutput> => ({
      decision: 'deny',
      reason: 'policy violation',
    });
    registry.register('SubagentStart', {
      pattern: '^researcher$',
      hooks: [denyHook],
    });

    const tc = makeSubagentToolCall();
    const run = await createSubagentRun(registry);
    run.Graph!.overrideTestModel(
      ['Delegating...', 'The subagent was blocked.'],
      5,
      [tc]
    );

    await run.processStream(
      { messages: [new HumanMessage('research something')] },
      callerConfig
    );

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();

    const toolMessages = runMessages!.filter(
      (msg) =>
        msg._getType() === 'tool' &&
        'name' in msg &&
        msg.name === Constants.SUBAGENT
    );
    expect(toolMessages.length).toBe(1);
    expect(String(toolMessages[0].content)).toContain(
      'Blocked: policy violation'
    );
  });
});
