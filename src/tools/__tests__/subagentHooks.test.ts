import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { ToolCall } from '@langchain/core/messages/tool';
import type {
  HookCallback,
  PermissionDeniedHookOutput,
  PostToolBatchHookOutput,
  PostToolUseHookOutput,
  PreToolUseHookOutput,
  SubagentStartHookInput,
  SubagentStartHookOutput,
  SubagentStopHookInput,
  SubagentStopHookOutput,
} from '@/hooks/types';
import type * as t from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
} from '@/index';
import { HookRegistry } from '@/hooks/HookRegistry';
import * as providers from '@/llm/providers';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

const CHILD_RESPONSE = 'Hook test child response.';

const calculatorDef: t.LCTool = {
  name: 'calculator',
  description: 'Evaluate a math expression.',
  parameters: {
    type: 'object',
    properties: {
      expression: { type: 'string' },
    },
    required: ['expression'],
  },
};

const callerConfig = {
  configurable: { thread_id: 'hook-test-thread' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

const originalGetChatModelClass = providers.getChatModelClass;

function makeSubagentToolCall(
  id = `call_sub_${Date.now()}`,
  description = 'Test task for hook verification'
): ToolCall {
  return {
    name: Constants.SUBAGENT,
    args: {
      description,
      subagent_type: 'researcher',
    },
    id,
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

function createParentAgentWithChildTool(): t.AgentInputs {
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
          instructions: 'Use calculator for arithmetic, then answer concisely.',
          maxContextTokens: 8000,
          toolDefinitions: [calculatorDef],
        },
      },
    ],
  };
}

function createCalculatorToolCall(): ToolCall {
  return {
    name: 'calculator',
    args: { expression: '21 * 2' },
    id: 'call_child_calculator',
    type: 'tool_call',
  };
}

class HitlChildFakeChatModel extends FakeChatModel {
  constructor(_options: object) {
    super({ responses: [CHILD_RESPONSE], sleep: 1 });
  }

  _streamResponseChunks(
    messages: Parameters<FakeChatModel['_streamResponseChunks']>[0],
    options: Parameters<FakeChatModel['_streamResponseChunks']>[1],
    runManager?: Parameters<FakeChatModel['_streamResponseChunks']>[2]
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const hasToolResult = messages.some(
      (message) => message._getType() === 'tool'
    );
    return new FakeChatModel({
      responses: [hasToolResult ? CHILD_RESPONSE : 'Using calculator.'],
      sleep: 1,
      toolCalls: hasToolResult ? [] : [createCalculatorToolCall()],
    })._streamResponseChunks(messages, options, runManager);
  }

  bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
    const config = {
      tools,
    } as Parameters<FakeChatModel['withConfig']>[0];
    return this.withConfig(config);
  }
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

  it('PreToolUse and PostToolUse fire for event-driven tools inside subagents', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({
              responses: ['Using calculator.', CHILD_RESPONSE],
              sleep: 1,
              toolCalls: [createCalculatorToolCall()],
            });
          }
          bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
            const config = {
              tools,
            } as Parameters<FakeChatModel['withConfig']>[0];
            return this.withConfig(config);
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const preToolEvents: string[] = [];
    const postToolEvents: string[] = [];
    // `executingAgentId` is always the owning agent (incl. top-level), unlike
    // `agentId` which stays undefined outside a subagent scope.
    const preExecEvents: string[] = [];
    const postExecEvents: string[] = [];

    const preHook: HookCallback<'PreToolUse'> = async (
      input
    ): Promise<PreToolUseHookOutput> => {
      preToolEvents.push(`${input.agentId ?? '-'}:${input.toolName}`);
      preExecEvents.push(`${input.executingAgentId ?? '-'}:${input.toolName}`);
      return { decision: 'allow' };
    };
    registry.register('PreToolUse', { hooks: [preHook] });

    const postHook: HookCallback<'PostToolUse'> = async (
      input
    ): Promise<PostToolUseHookOutput> => {
      postToolEvents.push(`${input.agentId ?? '-'}:${input.toolName}`);
      postExecEvents.push(`${input.executingAgentId ?? '-'}:${input.toolName}`);
      return {};
    };
    registry.register('PostToolUse', { hooks: [postHook] });

    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          const results: t.ToolExecuteResult[] = request.toolCalls.map(
            (call) => ({
              toolCallId: call.id,
              status: 'success',
              content: '42',
            })
          );
          request.resolve(results);
        },
      },
    };

    const run = await Run.create<t.IState>({
      runId: `subagent-tool-hook-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgentWithChildTool()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      hooks: registry,
    });

    const tc = makeSubagentToolCall();
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

    await run.processStream(
      { messages: [new HumanMessage('calculate something')] },
      callerConfig
    );

    expect(preToolEvents).toContain('-:subagent');
    expect(preToolEvents).toContain('researcher-child:calculator');
    expect(postToolEvents).toContain('-:subagent');
    expect(postToolEvents).toContain('researcher-child:calculator');

    // `agentId` stays the subagent-scope marker (undefined at the top level),
    // but `executingAgentId` attributes every batch to its owning agent — incl.
    // the top-level parent, which a multi-agent host needs and `agentId` can't give.
    expect(preExecEvents).toContain('hook-parent:subagent');
    expect(preExecEvents).toContain('researcher-child:calculator');
    expect(postExecEvents).toContain('hook-parent:subagent');
    expect(postExecEvents).toContain('researcher-child:calculator');
  });

  it('top-level event-driven dispatches leave agentId unset (subagent-scope marker)', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({
              responses: ['Calculating.', 'All done.'],
              sleep: 1,
              toolCalls: [createCalculatorToolCall()],
            });
          }
          bindTools(tools: unknown): ReturnType<FakeChatModel['withConfig']> {
            const config = {
              tools,
            } as Parameters<FakeChatModel['withConfig']>[0];
            return this.withConfig(config);
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const scopeEvents: string[] = [];

    const preHook: HookCallback<'PreToolUse'> = async (
      input
    ): Promise<PreToolUseHookOutput> => {
      scopeEvents.push(
        `pre:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return { decision: 'allow' };
    };
    registry.register('PreToolUse', { hooks: [preHook] });

    const postHook: HookCallback<'PostToolUse'> = async (
      input
    ): Promise<PostToolUseHookOutput> => {
      scopeEvents.push(
        `post:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return {};
    };
    registry.register('PostToolUse', { hooks: [postHook] });

    const batchHook: HookCallback<'PostToolBatch'> = async (
      input
    ): Promise<PostToolBatchHookOutput> => {
      scopeEvents.push(
        `batch:${input.agentId ?? '-'}:${input.executingAgentId ?? '-'}`
      );
      return {};
    };
    registry.register('PostToolBatch', { hooks: [batchHook] });

    const dispatchAgentIds: Array<string | undefined> = [];
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          dispatchAgentIds.push(request.agentId);
          const results: t.ToolExecuteResult[] = request.toolCalls.map(
            (call) => ({
              toolCallId: call.id,
              status: 'success',
              content: '42',
            })
          );
          request.resolve(results);
        },
      },
    };

    /**
     * The LibreChat shape: a TOP-LEVEL agent whose tools ride
     * `toolDefinitions` (event-driven ToolNode, host executes via
     * ON_TOOL_EXECUTE). Hook inputs here must NOT carry the subagent-scope
     * marker — a host hook keying on `agentId != null` (e.g. a steering
     * drain that must never inject into child state) would otherwise skip
     * every top-level batch. The DISPATCH payload is the opposite: hosts
     * key tool/credential lookup on `request.agentId`, so it must keep
     * identifying the owning agent.
     */
    const run = await Run.create<t.IState>({
      runId: `toplevel-event-hook-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'hook-parent',
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'Use the calculator.',
            maxContextTokens: 8000,
            toolDefinitions: [calculatorDef],
          },
        ],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      hooks: registry,
    });

    run.Graph!.overrideTestModel(['Calculating.', 'All done.'], 5, [
      createCalculatorToolCall(),
    ]);

    await run.processStream(
      { messages: [new HumanMessage('what is 21 * 2?')] },
      callerConfig
    );

    expect(scopeEvents).toContain('pre:-:hook-parent');
    expect(scopeEvents).toContain('post:-:hook-parent');
    expect(scopeEvents).toContain('batch:-:hook-parent');
    expect(dispatchAgentIds).toEqual(['hook-parent']);
  });

  it.each([
    {
      label: 'approve',
      resumeDecision: { type: 'approve' } as const,
      shouldExecute: true,
      deniedReason: undefined,
    },
    {
      label: 'reject',
      resumeDecision: {
        type: 'reject',
        reason: 'host rejected child tool',
      } as const,
      shouldExecute: false,
      deniedReason: 'host rejected child tool',
    },
    {
      label: 'deny',
      resumeDecision: undefined,
      shouldExecute: false,
      deniedReason: 'policy denied child tool',
    },
  ])(
    'handles a child subagent tool $label through the parent Run',
    async ({ resumeDecision, shouldExecute, deniedReason }) => {
      getChatModelClassSpy.mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return HitlChildFakeChatModel;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);

      const registry = new HookRegistry();
      const deniedTools: string[] = [];
      const executedTools: string[] = [];
      let calculatorPreToolCalls = 0;
      let calculatorPostToolCalls = 0;

      const preHook: HookCallback<'PreToolUse'> = async (
        input
      ): Promise<PreToolUseHookOutput> => {
        if (input.toolName === 'calculator') {
          calculatorPreToolCalls += 1;
          if (resumeDecision == null) {
            return {
              decision: 'deny',
              reason: 'policy denied child tool',
            };
          }
          return { decision: 'ask', reason: 'review calculator' };
        }
        return { decision: 'allow' };
      };
      registry.register('PreToolUse', { hooks: [preHook] });
      registry.register('PostToolUse', {
        hooks: [
          async (input): Promise<PostToolUseHookOutput> => {
            if (input.toolName === 'calculator') {
              calculatorPostToolCalls += 1;
            }
            return {};
          },
        ],
      });

      const deniedHook: HookCallback<'PermissionDenied'> = async (
        input
      ): Promise<PermissionDeniedHookOutput> => {
        deniedTools.push(
          `${input.agentId ?? '-'}:${input.toolName}:${input.reason}`
        );
        return {};
      };
      registry.register('PermissionDenied', { hooks: [deniedHook] });

      const customHandlers: Record<string, t.EventHandler> = {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: (_event, rawData): void => {
            const request = rawData as t.ToolExecuteBatchRequest;
            executedTools.push(...request.toolCalls.map((call) => call.name));
            const results: t.ToolExecuteResult[] = request.toolCalls.map(
              (call) => ({
                toolCallId: call.id,
                status: 'success',
                content: '42',
              })
            );
            request.resolve(results);
          },
        },
      };

      const run = await Run.create<t.IState>({
        runId: `subagent-tool-ask-${Date.now()}`,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });

      const tc = makeSubagentToolCall();
      run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [tc]);

      await run.processStream(
        { messages: [new HumanMessage('calculate something')] },
        callerConfig
      );

      if (resumeDecision == null) {
        expect(run.getInterrupt()).toBeUndefined();
        expect(calculatorPreToolCalls).toBe(1);
        expect(executedTools).not.toContain('calculator');
        expect(deniedTools).toEqual([
          `researcher-child:calculator:${deniedReason}`,
        ]);
        return;
      }

      const pending = run.getInterrupt();
      expect(pending?.payload).toMatchObject({
        type: 'tool_approval',
        subagent: {
          agent_id: 'researcher-child',
          parent_tool_call_id: tc.id,
          subagent_type: 'researcher',
        },
      });
      expect(executedTools).not.toContain('calculator');

      await run.resume([resumeDecision], callerConfig);

      expect(run.getInterrupt()).toBeUndefined();
      expect(calculatorPreToolCalls).toBe(2);
      expect(
        executedTools.filter((name) => name === 'calculator')
      ).toHaveLength(shouldExecute ? 1 : 0);
      expect(calculatorPostToolCalls).toBe(shouldExecute ? 1 : 0);
      expect(deniedTools).toEqual(
        shouldExecute ? [] : [`researcher-child:calculator:${deniedReason}`]
      );
    }
  );

  it('resumes approvals across multiple subagents and keeps updates sanitized', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return HitlChildFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const registry = new HookRegistry();
    const deniedToolIds: string[] = [];
    const executedToolIds: string[] = [];
    const updates: t.SubagentUpdateEvent[] = [];
    const completedSubagentCalls: string[] = [];

    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> =>
          input.toolName === 'calculator'
            ? { decision: 'ask', reason: 'review child calculator' }
            : { decision: 'allow' },
      ],
    });
    registry.register('PostToolUse', {
      hooks: [
        async (input): Promise<PostToolUseHookOutput> => {
          if (input.toolName === Constants.SUBAGENT) {
            completedSubagentCalls.push(input.toolUseId);
            return { additionalContext: `context:${input.toolUseId}` };
          }
          return {};
        },
      ],
    });
    registry.register('PermissionDenied', {
      hooks: [
        async (input): Promise<PermissionDeniedHookOutput> => {
          deniedToolIds.push(input.toolUseId);
          return {};
        },
      ],
    });

    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          executedToolIds.push(...request.toolCalls.map((call) => call.id));
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: '42',
            }))
          );
        },
      },
      [GraphEvents.ON_SUBAGENT_UPDATE]: {
        handle: (_event, data): void => {
          updates.push(data as t.SubagentUpdateEvent);
        },
      },
    };

    const checkpointer = new MemorySaver();
    const baseRunId = `subagent-multi-hitl-${Date.now()}`;
    const createRun = (runId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
          compileOptions: { checkpointer, interruptBefore: [] },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
    const run = await createRun(`${baseRunId}-initial`);
    const first = makeSubagentToolCall(
      'call_sub_first',
      'Run the first calculation'
    );
    const second = makeSubagentToolCall(
      'call_sub_second',
      'Run the second calculation'
    );
    run.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      first,
      second,
    ]);
    const multiCallerConfig = {
      ...callerConfig,
      configurable: {
        thread_id: 'multi-subagent-hitl-thread',
        access_token: 'must-not-leak',
        requestBody: { currentTaskInput: 'must-not-leak-task' },
        userMCPAuthMap: { private: { token: 'must-not-leak-mcp' } },
      },
    };

    await run.processStream(
      { messages: [new HumanMessage('calculate twice')] },
      multiCallerConfig
    );
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: first.id },
    });
    expect(updates.some((event) => event.parentToolCallId === second.id)).toBe(
      true
    );

    await run.resume([{ type: 'approve' }], multiCallerConfig);
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: second.id },
    });

    const rebuiltRun = await createRun(`${baseRunId}-rebuilt`);
    rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
    await rebuiltRun.resume(
      [{ type: 'reject', reason: 'reject second child' }],
      multiCallerConfig
    );

    expect(rebuiltRun.getInterrupt()).toBeUndefined();
    expect(executedToolIds).toHaveLength(1);
    expect(deniedToolIds).toHaveLength(1);
    expect(completedSubagentCalls).toEqual([first.id, second.id]);
    const firstPhases = updates
      .filter((event) => event.parentToolCallId === first.id)
      .map((event) => event.phase);
    const secondPhases = updates
      .filter((event) => event.parentToolCallId === second.id)
      .map((event) => event.phase);
    expect(firstPhases[0]).toBe('start');
    expect(firstPhases[firstPhases.length - 1]).toBe('stop');
    expect(secondPhases[0]).toBe('start');
    expect(secondPhases[secondPhases.length - 1]).toBe('stop');
    const serializedUpdates = JSON.stringify(updates);
    expect(serializedUpdates).not.toContain('must-not-leak');
    expect(serializedUpdates).not.toContain('currentTaskInput');
    expect(serializedUpdates).not.toContain('userMCPAuthMap');
    expect(serializedUpdates).not.toContain('checkpoint_');
  });

  it('resumes a child approval after rebuilding Run with the same checkpointer', async () => {
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return HitlChildFakeChatModel;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const checkpointer = new MemorySaver();
    const registry = new HookRegistry();
    const executedTools: string[] = [];
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> =>
          input.toolName === 'calculator'
            ? { decision: 'ask', reason: 'review calculator' }
            : { decision: 'allow' },
      ],
    });
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
      [GraphEvents.ON_TOOL_EXECUTE]: {
        handle: (_event, rawData): void => {
          const request = rawData as t.ToolExecuteBatchRequest;
          executedTools.push(...request.toolCalls.map((call) => call.name));
          request.resolve(
            request.toolCalls.map((call) => ({
              toolCallId: call.id,
              status: 'success' as const,
              content: '42',
            }))
          );
        },
      },
    };
    const runId = `subagent-rebuild-hitl-${Date.now()}`;
    const createRun = (currentRunId: string): Promise<Run<t.IState>> =>
      Run.create<t.IState>({
        runId: currentRunId,
        graphConfig: {
          type: 'standard',
          agents: [createParentAgentWithChildTool()],
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
        hooks: registry,
        humanInTheLoop: { enabled: true },
      });
    const tc = makeSubagentToolCall('call_sub_rebuild');
    const initialRun = await createRun(`${runId}-initial`);
    initialRun.Graph!.overrideTestModel(['Delegating...', 'Final answer.'], 5, [
      tc,
    ]);

    await initialRun.processStream(
      { messages: [new HumanMessage('calculate after restart')] },
      callerConfig
    );
    const persistedInterrupt = initialRun.getInterrupt();
    expect(persistedInterrupt?.payload).toMatchObject({
      type: 'tool_approval',
      subagent: { parent_tool_call_id: tc.id },
    });

    const rebuiltRun = await createRun(`${runId}-rebuilt`);
    rebuiltRun.Graph!.overrideTestModel(['Final answer.'], 1);
    const warningSpy = jest
      .spyOn(console, 'warn')
      .mockImplementation((): void => undefined);
    try {
      await rebuiltRun.resume([{ type: 'approve' }], callerConfig);
    } finally {
      warningSpy.mockRestore();
    }

    expect(rebuiltRun.getInterrupt()).toBeUndefined();
    expect(executedTools).toEqual(['calculator']);
  });
});
