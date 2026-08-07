import { MemorySaver } from '@langchain/langgraph';
import { AIMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type {
  AgentInputs,
  ExecutableSubagentConfigEntry,
  GraphSubagentConfig,
  LazySingleAgentSubagentConfig,
  MultiAgentGraphState,
  ResolvedSubagentConfig,
  StandardGraphInput,
  SubagentResolveContext,
} from '@/types';
import type { GraphFactoryRequest } from '@/graphs/graphFactory';
import type { StandardGraph } from '@/graphs/Graph';
import {
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_MANIFEST_CONFIG_KEY,
} from '@/tools/subagent/SubagentReplay';
import { normalizeSubagentConfigEntries } from '@/tools/subagent/childGraphConfig';
import {
  HookRegistry,
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
} from '@/hooks';
import { SubagentExecutor } from '@/tools/subagent/SubagentExecutor';
import { RUN_BREAKER_SCOPE_CONFIG_KEY } from '@/llm/streamLimits';
import { AgentContext } from '@/agents/AgentContext';
import { Providers } from '@/common';

const makeAgent = (agentId = 'child-agent'): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: `You are ${agentId}.`,
  maxContextTokens: 8000,
});

const makeLazyConfig = (
  type: string,
  resolveAgentInputs: LazySingleAgentSubagentConfig['resolveAgentInputs'],
  overrides: Partial<LazySingleAgentSubagentConfig> = {}
): LazySingleAgentSubagentConfig => ({
  type,
  name: `${type} worker`,
  description: `Handles ${type} tasks.`,
  configId: `${type}@v1`,
  resolveAgentInputs,
  ...overrides,
});

const makeConfigMap = (
  ...configs: ExecutableSubagentConfigEntry[]
): Map<string, ExecutableSubagentConfigEntry> =>
  new Map(configs.map((config) => [config.type, config]));

const makeGraph = (
  result: MultiAgentGraphState = {
    messages: [new AIMessage('Task completed')],
  }
): StandardGraph =>
  ({
    createWorkflow: () => ({
      invoke: jest.fn(async (): Promise<MultiAgentGraphState> => result),
    }),
    clearHeavyState: jest.fn(),
  }) as unknown as StandardGraph;

const makeRecoveredGraph = (): StandardGraph =>
  ({
    createWorkflow: () => ({
      getState: jest.fn(async () => ({
        values: { messages: [new AIMessage('persisted result')] },
        next: [],
        tasks: [],
      })),
      invoke: jest.fn(),
      updateState: jest.fn(async () => ({})),
    }),
    clearHeavyState: jest.fn(),
  }) as unknown as StandardGraph;

const createExecutor = (
  configs: ExecutableSubagentConfigEntry[],
  overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
): SubagentExecutor =>
  new SubagentExecutor({
    configs: makeConfigMap(...configs),
    parentRunId: 'parent-run',
    parentAgentId: 'parent-agent',
    createChildGraph: () => makeGraph(),
    ...overrides,
  });

describe('SubagentExecutor lazy selected-subagent resolution', () => {
  it('resolves only the selected descriptor', async () => {
    const researcherResolver = jest.fn(async () =>
      makeAgent('lazy-researcher')
    );
    const coderResolver = jest.fn(async () => makeAgent('lazy-coder'));
    const researcher = makeLazyConfig('researcher', researcherResolver);
    const coder = makeLazyConfig('coder', coderResolver);
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([researcher, coder], {
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Research this.',
      subagentType: researcher.type,
      parentToolCallId: 'call_research',
    });

    expect(researcherResolver).toHaveBeenCalledTimes(1);
    expect(coderResolver).not.toHaveBeenCalled();
    expect(observedInput?.agents[0].agentId).toBe('lazy-researcher');
  });

  it('redacts a selected resolver failure without affecting a healthy sibling', async () => {
    const failingConfig = makeLazyConfig('failing', async () => {
      throw new Error('oauth-token=private-selected-resolver-secret');
    });
    const healthyResolver = jest.fn(async () => makeAgent('healthy-child'));
    const healthyConfig = makeLazyConfig('healthy', healthyResolver);
    const executor = createExecutor([failingConfig, healthyConfig]);

    const failed = await executor.execute({
      description: 'Fail this selected child.',
      subagentType: failingConfig.type,
      parentToolCallId: 'call_failing',
    });
    const succeeded = await executor.execute({
      description: 'Run the healthy child.',
      subagentType: healthyConfig.type,
      parentToolCallId: 'call_healthy',
    });

    expect(failed.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(failed.content).not.toContain('oauth-token');
    expect(failed.content).not.toContain('private-selected-resolver-secret');
    expect(succeeded.content).toBe('Task completed');
    expect(healthyResolver).toHaveBeenCalledTimes(1);
  });

  it('clears lazy resolution and execution identity when SubagentStart denies', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const hookRegistry = new HookRegistry();
    hookRegistry.register('SubagentStart', {
      hooks: [
        async (): Promise<{ decision: 'deny'; reason: string }> => ({
          decision: 'deny',
          reason: 'blocked by policy',
        }),
      ],
    });
    const executor = createExecutor([config], { hookRegistry });
    const params = {
      description: 'Block this selected child.',
      subagentType: config.type,
      parentToolCallId: 'call_blocked',
    };

    const first = await executor.execute(params);
    const blockedState = executor as unknown as {
      resolvedConfigs: Map<string, ResolvedSubagentConfig>;
      childExecutionIdentities: Map<string, object>;
    };

    expect(first.content).toBe('Blocked: blocked by policy');
    expect(blockedState.resolvedConfigs.size).toBe(0);
    expect(blockedState.childExecutionIdentities.size).toBe(0);

    const retry = await executor.execute(params);

    expect(retry.content).toBe('Blocked: blocked by policy');
    expect(resolver).toHaveBeenCalledTimes(2);
  });

  it('coalesces concurrent resolution and releases resolved inputs after completion', async () => {
    let finishResolution = (_inputs: AgentInputs): void => undefined;
    const resolverResult = new Promise<AgentInputs>((resolve) => {
      finishResolution = resolve;
    });
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const resolver = jest.fn(async () => {
      markResolutionStarted();
      return resolverResult;
    });
    const config = makeLazyConfig('researcher', resolver);
    const executor = createExecutor([config], {
      parentRunId: 'coalesced-run',
    });
    const params = {
      description: 'Run the same selected execution.',
      subagentType: config.type,
      parentToolCallId: 'call_same',
    };

    const first = executor.execute(params);
    const duplicate = executor.execute(params);
    await resolutionStarted;
    expect(resolver).toHaveBeenCalledTimes(1);

    finishResolution(makeAgent('coalesced-child'));
    await Promise.all([first, duplicate]);
    await executor.execute({
      ...params,
      parentToolCallId: 'call_fresh',
    });

    expect(resolver).toHaveBeenCalledTimes(2);
    const resolutionState = executor as unknown as {
      resolvedConfigs: Map<string, ResolvedSubagentConfig>;
      pendingConfigResolutions: Map<string, Promise<ResolvedSubagentConfig>>;
    };
    expect(resolutionState.resolvedConfigs.size).toBe(0);
    expect(resolutionState.pendingConfigResolutions.size).toBe(0);
  });

  it('sanitizes private runtime state before calling the resolver', async () => {
    let configurable: Readonly<Record<string, unknown>> | undefined;
    const config = makeLazyConfig('researcher', async (context) => {
      configurable = context.configurable;
      return makeAgent();
    });
    const executor = createExecutor([config]);

    await executor.execute({
      description: 'Resolve safely.',
      subagentType: config.type,
      parentToolCallId: 'call_sanitized',
      parentConfigurable: {
        requestBody: { messageId: 'allowed-message' },
        user: { id: 'allowed-user' },
        __pregel_scratchpad: { secret: 'pregel-secret' },
        checkpoint_id: 'checkpoint-secret',
        [SUBAGENT_PARENT_BATCH_CONFIG_KEY]: 'batch-secret',
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: { secret: 'resume-secret' },
        [TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY]: 'approval-secret',
        [RUN_BREAKER_SCOPE_CONFIG_KEY]: { secret: 'breaker-secret' },
      },
    });

    expect(configurable).toEqual({
      requestBody: { messageId: 'allowed-message' },
      user: { id: 'allowed-user' },
    });
    expect(JSON.stringify(configurable)).not.toContain('secret');
  });

  it('uses a per-call abort signal while awaiting a non-cooperative resolver', async () => {
    const callAbort = new AbortController();
    let observedContext: SubagentResolveContext | undefined;
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const config = makeLazyConfig('researcher', async (context) => {
      observedContext = context;
      markResolutionStarted();
      return new Promise<AgentInputs>(() => undefined);
    });
    const executor = createExecutor([config]);

    const execution = executor.execute({
      description: 'Cancel this selection.',
      subagentType: config.type,
      parentToolCallId: 'call_cancel',
      signal: callAbort.signal,
    });
    await resolutionStarted;
    callAbort.abort(new Error('private cancellation reason'));
    const result = await execution;

    expect(observedContext?.signal.aborted).toBe(true);
    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(result.content).not.toContain('private cancellation reason');
  });

  it('prefers eager agent inputs without calling an attached resolver', async () => {
    const resolveAgentInputs = jest.fn(async () => makeAgent('unexpected'));
    const eagerInputs = makeAgent('eager-child');
    const eagerConfig: ResolvedSubagentConfig = {
      type: 'eager',
      name: 'Eager Worker',
      description: 'Already has complete inputs.',
      configId: 'eager@v1',
      agentInputs: eagerInputs,
      resolveAgentInputs,
    };
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([eagerConfig], {
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Use eager inputs.',
      subagentType: eagerConfig.type,
      parentToolCallId: 'call_eager',
    });

    expect(resolveAgentInputs).not.toHaveBeenCalled();
    expect(observedInput?.agents[0].agentId).toBe(eagerInputs.agentId);
  });

  it('preserves nested lazy descriptors and decrements the child depth', async () => {
    const nestedDescriptor = makeLazyConfig('grandchild', async () =>
      makeAgent('grandchild')
    );
    const config = makeLazyConfig(
      'nested',
      async () => ({
        ...makeAgent('nested-child'),
        subagentConfigs: [nestedDescriptor],
        maxSubagentDepth: 3,
      }),
      { allowNested: true }
    );
    let observedInput: StandardGraphInput | undefined;
    const executor = createExecutor([config], {
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedInput = input;
        return makeGraph();
      },
    });

    await executor.execute({
      description: 'Delegate recursively.',
      subagentType: config.type,
      parentToolCallId: 'call_nested',
    });

    expect(observedInput?.agents[0].maxSubagentDepth).toBe(2);
    expect(observedInput?.agents[0].subagentConfigs).toEqual([
      nestedDescriptor,
    ]);
  });

  it('keeps resolver identity stable after executor reconstruction', async () => {
    const contexts: SubagentResolveContext[] = [];
    const config = makeLazyConfig('researcher', async (context) => {
      contexts.push(context);
      return makeAgent();
    });
    const makeRebuiltExecutor = (): SubagentExecutor =>
      createExecutor([config], {
        parentRunId: 'rebuilt-parent-run',
        humanInTheLoop: { enabled: true },
        createChildGraph: makeRecoveredGraph,
      });
    const params = {
      description: 'Recover the deterministic child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId: 'call_rebuild',
    };

    await makeRebuiltExecutor().execute(params);
    await makeRebuiltExecutor().execute(params);

    expect(contexts).toHaveLength(2);
    expect(contexts[0].executionId).toBe(contexts[1].executionId);
    expect(contexts[0].descriptor).toEqual(contexts[1].descriptor);
    expect(contexts[0].descriptor.configId).toBe('researcher@v1');
    expect(contexts[0].threadId).toBe('durable-thread');
    expect(contexts[0].parentToolCallId).toBe('call_rebuild');
  });

  it('rejects a resumed execution with a different configId before resolution', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver, {
      configId: 'researcher@v2',
    });
    const parentRunId = 'rebuilt-parent-run';
    const parentAgentId = 'parent-agent';
    const threadId = 'durable-thread';
    const parentToolCallId = 'call_rebuild';
    const branchThreadId = `subagent:${Buffer.from(
      JSON.stringify([
        threadId,
        'root',
        parentAgentId,
        parentToolCallId,
        'batch',
        parentRunId,
      ])
    ).toString('base64url')}`;
    const executor = createExecutor([config], {
      parentRunId,
      parentAgentId,
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the original child.',
      subagentType: config.type,
      threadId,
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [
            {
              parentToolCallId,
              childRunId: 'original-child-run',
              configId: 'researcher@v1',
              approvalExecutionScope: 'original-approval-scope',
              checkpoints: [
                {
                  threadId: branchThreadId,
                  checkpointId: 'original-checkpoint',
                  checkpointNs: '',
                },
              ],
              graphState: {
                toolCallSteps: [],
                toolSessions: [],
                toolNodes: [],
                eagerToolUsage: [],
                eagerToolSuppressions: [],
              },
              approvalReplays: [],
            },
          ],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(result.content).not.toContain('researcher@v1');
    expect(result.content).not.toContain('researcher@v2');
    expect(resolver).not.toHaveBeenCalled();
  });

  it('uses the standard factory for lazy agents and the polymorphic factory for graphs', async () => {
    const lazyConfig = makeLazyConfig('researcher', async () =>
      makeAgent('lazy-researcher')
    );
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'research-team',
      name: 'Research Team',
      description: 'Runs a graph child.',
      agents: [makeAgent('result')],
      edges: [],
      entryAgentId: 'result',
      resultAgentId: 'result',
    };
    const standardInputs: StandardGraphInput[] = [];
    const graphRequests: GraphFactoryRequest[] = [];
    const createChildGraph = jest.fn((input: StandardGraphInput) => {
      standardInputs.push(input);
      return makeGraph({ messages: [new AIMessage('lazy result')] });
    });
    const createChildGraphByKind = jest.fn((request: GraphFactoryRequest) => {
      graphRequests.push(request);
      return makeGraph({
        messages: [new AIMessage('graph result')],
        subagentResult: {
          agentId: 'result',
          message: new AIMessage('graph result'),
        },
      });
    });
    const executor = createExecutor([lazyConfig, graphConfig], {
      createChildGraph,
      createChildGraphByKind,
    });

    const lazyResult = await executor.execute({
      description: 'Run one agent.',
      subagentType: lazyConfig.type,
      parentToolCallId: 'call_lazy',
    });
    const graphResult = await executor.execute({
      description: 'Run the team.',
      subagentType: graphConfig.type,
      parentToolCallId: 'call_graph',
    });

    expect(lazyResult.content).toBe('lazy result');
    expect(graphResult.content).toBe('graph result');
    expect(createChildGraph).toHaveBeenCalledTimes(1);
    expect(standardInputs[0].agents[0].agentId).toBe('lazy-researcher');
    expect(createChildGraphByKind).toHaveBeenCalledTimes(1);
    expect(graphRequests[0]).toMatchObject({
      kind: 'multi-agent',
      input: {
        agents: [{ agentId: 'result' }],
        resultAgentId: 'result',
      },
    });
  });

  it('rejects graph lazy fields during normalization before resolver or factory invocation', async () => {
    const resolveAgentInputs = jest.fn(async () => makeAgent('unexpected'));
    const graphConfig: GraphSubagentConfig = {
      kind: 'graph',
      type: 'invalid-graph',
      name: 'Invalid Graph',
      description: 'Illegally carries lazy single-agent fields.',
      agents: [makeAgent('result')],
      edges: [],
      entryAgentId: 'result',
      resultAgentId: 'result',
    };
    Reflect.set(graphConfig, 'configId', 'invalid-graph@v1');
    Reflect.set(graphConfig, 'resolveAgentInputs', resolveAgentInputs);
    const createChildGraph = jest.fn((_input: StandardGraphInput) =>
      makeGraph()
    );
    const createChildGraphByKind = jest.fn((_request: GraphFactoryRequest) =>
      makeGraph()
    );
    const parentContext = AgentContext.fromConfig(makeAgent('parent'));

    const initializeAndExecute = async (): Promise<void> => {
      const configs = normalizeSubagentConfigEntries(
        [graphConfig],
        parentContext
      );
      const executor = createExecutor(configs, {
        createChildGraph,
        createChildGraphByKind,
      });
      await executor.execute({
        description: 'This must never execute.',
        subagentType: graphConfig.type,
        parentToolCallId: 'call_invalid_graph',
      });
    };

    await expect(initializeAndExecute()).rejects.toThrow(
      /lazy fields configId\/resolveAgentInputs/
    );
    expect(resolveAgentInputs).not.toHaveBeenCalled();
    expect(createChildGraph).not.toHaveBeenCalled();
    expect(createChildGraphByKind).not.toHaveBeenCalled();
  });
});
