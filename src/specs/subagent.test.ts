import { ChatGenerationChunk } from '@langchain/core/outputs';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { UsageMetadata } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
  StandardGraph,
} from '@/index';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

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

  it('inherits eager event-tool settings into self-spawn child graphs', async () => {
    const originalCreateWorkflow = StandardGraph.prototype.createWorkflow;
    const observedChildGraphs: Array<{
      eagerEventToolExecution: StandardGraph['eagerEventToolExecution'];
      toolOutputReferences: StandardGraph['toolOutputReferences'];
      eventToolExecutionAvailable: boolean;
    }> = [];
    const createWorkflowSpy = jest
      .spyOn(StandardGraph.prototype, 'createWorkflow')
      .mockImplementation(function (this: StandardGraph) {
        if (this.runId?.includes('_sub_') === true) {
          observedChildGraphs.push({
            eagerEventToolExecution: this.eagerEventToolExecution,
            toolOutputReferences: this.toolOutputReferences,
            eventToolExecutionAvailable: this.eventToolExecutionAvailable,
          });
          return {
            invoke: jest.fn(async () => ({
              messages: [new AIMessage('child done')],
            })),
          } as unknown as ReturnType<StandardGraph['createWorkflow']>;
        }
        return originalCreateWorkflow.call(this);
      });

    const agentWithSelfSpawn: t.AgentInputs = {
      agentId: 'self-parent',
      provider: Providers.OPENAI,
      clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
      instructions: 'Agent with self-spawn for context isolation.',
      maxContextTokens: 8000,
      toolDefinitions: [{ name: 'mcp_lookup' }],
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
      runId: `self-spawn-eager-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [agentWithSelfSpawn],
      },
      customHandlers: {
        [GraphEvents.ON_TOOL_EXECUTE]: {
          handle: async () => undefined,
        },
      },
      eagerEventToolExecution: { enabled: true },
      toolOutputReferences: { enabled: true },
      returnContent: true,
      skipCleanup: true,
    });

    const context = (run.Graph as StandardGraph).agentContexts.get(
      'self-parent'
    );
    const subagentTool = (context?.graphTools as t.GenericTool[]).find(
      (tool) => 'name' in tool && tool.name === Constants.SUBAGENT
    );
    expect(subagentTool).toBeDefined();

    await subagentTool!.invoke(
      {
        description: 'Use your MCP tool.',
        subagent_type: 'isolated',
      },
      callerConfig
    );

    expect(observedChildGraphs).toEqual([
      {
        eagerEventToolExecution: { enabled: true },
        toolOutputReferences: { enabled: true },
        eventToolExecutionAvailable: true,
      },
    ]);

    createWorkflowSpy.mockRestore();
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

  it('reports child model usage through subagentUsageSink', async () => {
    const CHILD_USAGE = {
      input_tokens: 11,
      output_tokens: 7,
      total_tokens: 18,
    };
    /**
     * The default mock (FakeListChatModel) reports no usage. Re-mock with a
     * subclass that reports `usage_metadata` the way live providers do:
     * stamped on the generation in the invoke path, and carried on a final
     * zero-content chunk in the stream path (the graph's `attemptInvoke`
     * prefers `model.stream()`, and chunk concatenation folds the usage
     * into the aggregated message that `handleLLMEnd` receives).
     */
    getChatModelClassSpy.mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeListChatModel {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          constructor(_options: any) {
            super({ responses: [CHILD_RESPONSE] });
          }
          async _generate(
            ...args: Parameters<FakeListChatModel['_generate']>
          ): ReturnType<FakeListChatModel['_generate']> {
            const result = await super._generate(...args);
            for (const generation of result.generations) {
              (generation.message as AIMessage).usage_metadata = {
                ...CHILD_USAGE,
              };
            }
            return result;
          }
          async *_streamResponseChunks(
            ...args: Parameters<FakeListChatModel['_streamResponseChunks']>
          ): ReturnType<FakeListChatModel['_streamResponseChunks']> {
            yield* super._streamResponseChunks(...args);
            yield new ChatGenerationChunk({
              text: '',
              message: new AIMessageChunk({
                content: '',
                usage_metadata: { ...CHILD_USAGE },
              }),
            });
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);

    const collectedUsage: UsageMetadata[] = [];
    const sunkEvents: t.SubagentUsageEvent[] = [];
    const customHandlers: Record<string, t.EventHandler> = {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    };

    const runId = `subagent-usage-${Date.now()}`;
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [createParentAgent()],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
      subagentUsageSink: (event) => {
        sunkEvents.push(event);
      },
    });

    const subagentToolCall: ToolCall = {
      id: 'call_subagent_usage',
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

    await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      callerConfig
    );

    /** Child made exactly one model call; all events are child-tagged. */
    expect(sunkEvents).toHaveLength(1);
    const event = sunkEvents[0];
    /** Chunk concat adds empty `*_token_details` — match on the counts. */
    expect(event.usage).toMatchObject(CHILD_USAGE);
    expect(event.subagentType).toBe('researcher');
    expect(event.subagentAgentId).toBe('researcher');
    expect(event.provider).toBe(Providers.OPENAI);
    /** FakeListChatModel emits no ls_model_name → config fallback. */
    expect(event.model).toBe('gpt-4o-mini');
    expect(event.runId).toBe(runId);
    expect(event.subagentRunId).toContain(`${runId}_sub_`);
    /**
     * The parent's own calls must NOT be routed through the sink — they
     * flow through the registered CHAT_MODEL_END handler. (The fake
     * override model reports no usage, so collectedUsage stays empty;
     * the load-bearing assertion is that the sink saw no parent calls.)
     */
    expect(sunkEvents.every((e) => e.subagentType === 'researcher')).toBe(true);
  });
});
