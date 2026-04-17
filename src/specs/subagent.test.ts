import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Run } from '@/run';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
  StandardGraph,
} from '@/index';
import * as providers from '@/llm/providers';

const CHILD_RESPONSE = 'Research result: Paris is the capital of France.';

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'subagent-test-thread' },
  streamMode: 'values',
  version: 'v2' as const,
};

const createParentAgent = (): t.AgentInputs => ({
  agentId: 'parent',
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions:
    'You are a supervisor. Delegate research tasks using the subagent tool.',
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

describe('Subagent Integration', () => {
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

  it('should create subagent tool on agent context', async () => {
    const run = await Run.create<t.IState>({
      runId: `subagent-test-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
    });

    expect(run.Graph).toBeDefined();
    const parentContext = (run.Graph as StandardGraph).agentContexts.get(
      'parent'
    );
    expect(parentContext).toBeDefined();
    expect(parentContext?.graphTools).toBeDefined();

    const subagentTool = (parentContext?.graphTools as t.GenericTool[]).find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();
  });

  it('should execute subagent and return filtered result to parent', async () => {
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    };

    const run = await Run.create<t.IState>({
      runId: `subagent-exec-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
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

    run.Graph?.overrideTestModel(
      [
        'Let me delegate this research task.',
        `Based on the research: ${CHILD_RESPONSE}`,
      ],
      10,
      [subagentToolCall]
    );

    const result = await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      callerConfig
    );

    expect(result).toBeDefined();

    const runMessages = run.getRunMessages();
    expect(runMessages).toBeDefined();
    expect(runMessages!.length).toBeGreaterThan(0);

    const toolMessages = runMessages!.filter(
      (msg) => msg._getType() === 'tool'
    );
    const subagentResult = toolMessages.find(
      (msg) => 'name' in msg && msg.name === Constants.SUBAGENT
    );
    expect(subagentResult).toBeDefined();
    expect(String(subagentResult!.content)).toContain('Paris');
  });

  it('should not create subagent tool when no subagentConfigs', async () => {
    const agentWithoutSubagents: t.AgentInputs = {
      agentId: 'plain',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Plain agent without subagents.',
      maxContextTokens: 8000,
    };

    const run = await Run.create<t.IState>({
      runId: `no-subagent-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithoutSubagents],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get('plain');
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeUndefined();
  });

  it('should handle self-spawn subagent config', async () => {
    const agentWithSelfSpawn: t.AgentInputs = {
      agentId: 'self-parent',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Agent with self-spawn for context isolation.',
      maxContextTokens: 8000,
      subagentConfigs: [
        {
          type: 'isolated',
          name: 'Isolated Worker',
          description: 'Runs a task with isolated context',
          self: true,
        },
      ],
    };

    const run = await Run.create<t.IState>({
      runId: `self-spawn-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSelfSpawn],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'self-parent'
    );
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();
  });

  it('should not create subagent tool when maxSubagentDepth is 0', async () => {
    const agentWithZeroDepth: t.AgentInputs = {
      ...createParentAgent(),
      agentId: 'zero-depth',
      maxSubagentDepth: 0,
    };

    const run = await Run.create<t.IState>({
      runId: `zero-depth-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithZeroDepth],
      },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'zero-depth'
    );
    const tools = context?.graphTools as t.GenericTool[] | undefined;
    const subagentTool = tools?.find(
      (t) => 'name' in t && t.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeUndefined();
  });

  it('should account for subagent tool schema in toolSchemaTokens', async () => {
    /** Simple char-count tokenizer — deterministic, lets us assert presence. */
    const tokenCounter: t.TokenCounter = (message) => {
      const content = message.content;
      if (typeof content === 'string') return content.length;
      if (Array.isArray(content)) return JSON.stringify(content).length;
      return JSON.stringify(content).length;
    };

    const agentWithSubagent = createParentAgent();
    const runWith = await Run.create<t.IState>({
      runId: `with-sub-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSubagent],
      },
      tokenCounter,
      returnContent: true,
      skipCleanup: true,
    });

    const agentWithoutSubagent: t.AgentInputs = {
      agentId: 'plain',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions:
        'You are a supervisor. Delegate research tasks using the subagent tool.',
      maxContextTokens: 8000,
    };
    const runWithout = await Run.create<t.IState>({
      runId: `without-sub-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithoutSubagent],
      },
      tokenCounter,
      returnContent: true,
      skipCleanup: true,
    });

    const contextWith = (runWith.Graph as StandardGraph).agentContexts.get(
      'parent'
    );
    const contextWithout = (
      runWithout.Graph as StandardGraph
    ).agentContexts.get('plain');

    await contextWith?.tokenCalculationPromise;
    await contextWithout?.tokenCalculationPromise;

    /** Subagent tool schema is ~600 chars; expect measurable difference. */
    expect(contextWith!.toolSchemaTokens).toBeGreaterThan(
      contextWithout!.toolSchemaTokens
    );
  });
});
