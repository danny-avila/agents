import { MemorySaver } from '@langchain/langgraph';
import { describe, expect, it, jest } from '@jest/globals';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type {
  AgentInputs,
  ExecutableSubagentConfigEntry,
  GraphSubagentConfig,
  LazySingleAgentSubagentConfig,
  MultiAgentGraphState,
  ResolvedSubagentConfig,
  SubagentResolveConfigurable,
  StandardGraphInput,
  SubagentResolveContext,
} from '@/types';
import type { SubagentResumeExecution } from '@/tools/subagent/SubagentReplay';
import type { GraphFactoryRequest } from '@/graphs/graphFactory';
import type { StandardGraph } from '@/graphs/Graph';
import {
  SUBAGENT_PARENT_BATCH_CONFIG_KEY,
  SUBAGENT_RESUME_MANIFEST_CONFIG_KEY,
} from '@/tools/subagent/SubagentReplay';
import {
  RUN_BREAKER_SCOPE_CONFIG_KEY,
  StreamLimitExceededError,
} from '@/llm/streamLimits';
import { normalizeSubagentConfigEntries } from '@/tools/subagent/childGraphConfig';
import {
  HookRegistry,
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
} from '@/hooks';
import { SubagentExecutor } from '@/tools/subagent/SubagentExecutor';
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

const makeResumeExecution = (
  parentToolCallId: string,
  configId: string,
  checkpointThreadId = 'resume-source-thread'
): SubagentResumeExecution => ({
  parentToolCallId,
  childRunId: 'original-child-run',
  configId,
  approvalExecutionScope: 'original-approval-scope',
  checkpoints: [
    {
      threadId: checkpointThreadId,
      checkpointId: 'resume-checkpoint',
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
});

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

  it('coalesces concurrent execution and releases resolved inputs after completion', async () => {
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
    const invoke = jest.fn(
      async (): Promise<MultiAgentGraphState> => ({
        messages: [new AIMessage('Task completed')],
      })
    );
    const executor = createExecutor([config], {
      parentRunId: 'coalesced-run',
      createChildGraph: () =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
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
    expect(invoke).toHaveBeenCalledTimes(1);
    await executor.execute({
      ...params,
      parentToolCallId: 'call_fresh',
    });

    expect(resolver).toHaveBeenCalledTimes(2);
    expect(invoke).toHaveBeenCalledTimes(2);
    const resolutionState = executor as unknown as {
      resolvedConfigs: Map<string, ResolvedSubagentConfig>;
      pendingConfigResolutions: Map<string, Promise<ResolvedSubagentConfig>>;
      pendingExecutions: Map<string, Promise<object>>;
    };
    expect(resolutionState.resolvedConfigs.size).toBe(0);
    expect(resolutionState.pendingConfigResolutions.size).toBe(0);
    expect(resolutionState.pendingExecutions.size).toBe(0);
  });

  it('rejects a stale resolver completion after cleanup and preserves a newer retry', async () => {
    let finishFirst = (_inputs: AgentInputs): void => undefined;
    let finishSecond = (_inputs: AgentInputs): void => undefined;
    const firstResult = new Promise<AgentInputs>((resolve) => {
      finishFirst = resolve;
    });
    const secondResult = new Promise<AgentInputs>((resolve) => {
      finishSecond = resolve;
    });
    let resolverInvocation = 0;
    const config = makeLazyConfig('researcher', () => {
      resolverInvocation += 1;
      return resolverInvocation === 1 ? firstResult : secondResult;
    });
    const executor = createExecutor([config]);
    const resolutionState = executor as unknown as {
      resolveExecutionConfig: (
        executableConfig: ExecutableSubagentConfigEntry,
        context: {
          childExecutionKey: string;
          childRunId: string;
          childSignal: AbortSignal;
        }
      ) => Promise<ResolvedSubagentConfig>;
      resolvedConfigs: Map<string, ResolvedSubagentConfig>;
      pendingConfigResolutions: Map<string, Promise<ResolvedSubagentConfig>>;
    };
    const childExecutionKey = 'stable-child-execution';
    const context = {
      childExecutionKey,
      childRunId: 'stable-child-run',
      childSignal: new AbortController().signal,
    };

    const staleResolution = resolutionState.resolveExecutionConfig(
      config,
      context
    );
    executor.clearHeavyState();
    const currentResolution = resolutionState.resolveExecutionConfig(
      config,
      context
    );

    finishFirst(makeAgent('stale-child'));
    await expect(staleResolution).rejects.toThrow(
      'Subagent execution was invalidated.'
    );
    expect(resolutionState.resolvedConfigs.has(childExecutionKey)).toBe(false);
    expect(
      resolutionState.pendingConfigResolutions.get(childExecutionKey)
    ).toBe(currentResolution);

    finishSecond(makeAgent('current-child'));
    await expect(currentResolution).resolves.toMatchObject({
      agentInputs: { agentId: 'current-child' },
    });
    expect(
      resolutionState.resolvedConfigs.get(childExecutionKey)?.agentInputs
        .agentId
    ).toBe('current-child');
    expect(resolutionState.pendingConfigResolutions.size).toBe(0);
  });

  it('does not execute a child after its lazy resolution was invalidated', async () => {
    let finishFirst = (_inputs: AgentInputs): void => undefined;
    let finishSecond = (_inputs: AgentInputs): void => undefined;
    const firstResult = new Promise<AgentInputs>((resolve) => {
      finishFirst = resolve;
    });
    const secondResult = new Promise<AgentInputs>((resolve) => {
      finishSecond = resolve;
    });
    let markFirstStarted = (): void => undefined;
    let markSecondStarted = (): void => undefined;
    const firstStarted = new Promise<void>((resolve) => {
      markFirstStarted = resolve;
    });
    const secondStarted = new Promise<void>((resolve) => {
      markSecondStarted = resolve;
    });
    let resolverInvocation = 0;
    const config = makeLazyConfig('researcher', () => {
      resolverInvocation += 1;
      if (resolverInvocation === 1) {
        markFirstStarted();
        return firstResult;
      }
      markSecondStarted();
      return secondResult;
    });
    const observedAgentIds: string[] = [];
    const createChildGraph = jest.fn((input: StandardGraphInput) => {
      observedAgentIds.push(input.agents[0].agentId);
      return makeGraph();
    });
    const executor = createExecutor([config], { createChildGraph });
    const params = {
      description: 'Run the selected child.',
      subagentType: config.type,
      parentToolCallId: 'call_cleanup_retry',
    };

    const staleExecution = executor.execute(params);
    await firstStarted;
    executor.clearHeavyState();
    const currentExecution = executor.execute(params);
    await secondStarted;

    finishFirst(makeAgent('stale-child'));
    await expect(staleExecution).resolves.toMatchObject({
      content: 'Subagent error: Unable to initialize the selected subagent.',
    });
    expect(createChildGraph).not.toHaveBeenCalled();

    finishSecond(makeAgent('current-child'));
    await expect(currentExecution).resolves.toMatchObject({
      content: 'Task completed',
    });
    expect(createChildGraph).toHaveBeenCalledTimes(1);
    expect(observedAgentIds).toEqual(['current-child']);
  });

  it('keeps concurrent checkpoint forks as distinct child executions', async () => {
    const resolverContexts: SubagentResolveContext[] = [];
    let releaseInvocations = (): void => undefined;
    const invocationGate = new Promise<void>((resolve) => {
      releaseInvocations = resolve;
    });
    let invocationCount = 0;
    const invoke = jest.fn(async (): Promise<MultiAgentGraphState> => {
      invocationCount += 1;
      const current = invocationCount;
      await invocationGate;
      return { messages: [new AIMessage(`result-${current}`)] };
    });
    const config = makeLazyConfig('researcher', async (context) => {
      resolverContexts.push(context);
      return makeAgent();
    });
    const executor = createExecutor([config], {
      createChildGraph: () =>
        ({
          createWorkflow: () => ({ invoke }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph,
    });
    const common = {
      description: 'Run this selected execution.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId: 'call_reused',
    };

    const first = executor.execute({
      ...common,
      parentConfigurable: { checkpoint_id: 'fork-a' },
    });
    const second = executor.execute({
      ...common,
      parentConfigurable: { checkpoint_id: 'fork-b' },
    });
    for (
      let attempt = 0;
      attempt < 20 && invoke.mock.calls.length < 2;
      attempt += 1
    ) {
      await Promise.resolve();
    }
    const callsBeforeRelease = invoke.mock.calls.length;
    releaseInvocations();
    const [firstResult, secondResult] = await Promise.all([first, second]);

    expect(callsBeforeRelease).toBe(2);
    expect(resolverContexts).toHaveLength(2);
    expect(
      new Set(resolverContexts.map((context) => context.executionId)).size
    ).toBe(2);
    expect(new Set([firstResult.content, secondResult.content])).toEqual(
      new Set(['result-1', 'result-2'])
    );
  });

  it('sanitizes private runtime state before calling the resolver', async () => {
    let configurable: Readonly<SubagentResolveConfigurable> | undefined;
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
        user: {
          id: 'allowed-user',
          role: 'member',
          tenantId: 'tenant-1',
          privateToken: 'principal-secret',
        },
        user_id: 'allowed-user',
        userMCPAuthMap: { private: { token: 'mcp-secret' } },
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
      user: {
        id: 'allowed-user',
        role: 'member',
        tenantId: 'tenant-1',
      },
      user_id: 'allowed-user',
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

  it('rethrows a stream-limit trip while awaiting lazy resolution', async () => {
    const breaker = new AbortController();
    let markResolutionStarted = (): void => undefined;
    const resolutionStarted = new Promise<void>((resolve) => {
      markResolutionStarted = resolve;
    });
    const config = makeLazyConfig('researcher', async () => {
      markResolutionStarted();
      return new Promise<AgentInputs>(() => undefined);
    });
    const executor = createExecutor([config]);
    const trip = new StreamLimitExceededError({
      kind: 'tool_call_args',
      limit: 10,
      observed: 11,
      toolName: 'db_query',
    });

    const execution = executor.execute({
      description: 'Stop this selection.',
      subagentType: config.type,
      parentToolCallId: 'call_limit',
      breaker,
    });
    await resolutionStarted;
    breaker.abort(trip);

    await expect(execution).rejects.toBe(trip);
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

  it('keeps the persisted executionId when resuming a lazy child', async () => {
    let resolverContext: SubagentResolveContext | undefined;
    const config = makeLazyConfig('researcher', async (context) => {
      resolverContext = context;
      return makeAgent();
    });
    const parentToolCallId = 'call_persisted_identity';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    resumeExecution.childRunId = 'persisted-execution-id';
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
      createChildGraph: makeRecoveredGraph,
    });
    const forkTarget = executor as unknown as {
      forkCheckpointSnapshot: () => Promise<void>;
    };
    jest
      .spyOn(forkTarget, 'forkCheckpointSnapshot')
      .mockResolvedValue(undefined);

    await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(resolverContext?.executionId).toBe('persisted-execution-id');
  });

  it('rejects a changed configId before restoring or recording resume state', async () => {
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
    const rejectedState = executor as unknown as {
      childExecutionIdentities: Map<string, object>;
      checkpointThreadIds: Set<string>;
    };
    expect(rejectedState.childExecutionIdentities.size).toBe(0);
    expect(rejectedState.checkpointThreadIds.size).toBe(0);
  });

  it('rejects a versionless resume manifest for a lazy config', async () => {
    const resolver = jest.fn(async () => makeAgent());
    const config = makeLazyConfig('researcher', resolver);
    const parentToolCallId = 'call_versionless_resume';
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId
    );
    delete resumeExecution.configId;
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });

    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });

    expect(result.content).toBe(
      'Subagent error: Subagent configuration changed since this execution was paused.'
    );
    expect(resolver).not.toHaveBeenCalled();
    const rejectedState = executor as unknown as {
      childExecutionIdentities: Map<string, object>;
      checkpointThreadIds: Set<string>;
    };
    expect(rejectedState.childExecutionIdentities.size).toBe(0);
    expect(rejectedState.checkpointThreadIds.size).toBe(0);
  });

  it('rejects replay lifecycle revision mismatches before restoring state', async () => {
    const config = makeLazyConfig('researcher', async () => makeAgent(), {
      configId: 'researcher@v2',
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_lifecycle_mismatch';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Resume the selected child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1 as const,
          executions: [makeResumeExecution(parentToolCallId, 'researcher@v1')],
        },
      },
    };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Do not persist this mismatch.',
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
    });

    const rejectedState = executor as unknown as {
      childExecutionIdentities: Map<string, object>;
      checkpointThreadIds: Set<string>;
    };
    expect(rejectedState.childExecutionIdentities.size).toBe(0);
    expect(rejectedState.checkpointThreadIds.size).toBe(0);
  });

  it('retains configId in regenerated manifests after resolver failure', async () => {
    const config = makeLazyConfig('researcher', async () => {
      throw new Error('transient resolver failure');
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_resolver_retry';
    const branchThreadId = `subagent:${Buffer.from(
      JSON.stringify([
        'durable-thread',
        'root',
        'parent-agent',
        parentToolCallId,
        'batch',
        'parent-run',
      ])
    ).toString('base64url')}`;
    const resumeExecution = makeResumeExecution(
      parentToolCallId,
      config.configId,
      branchThreadId
    );
    const result = await executor.execute({
      description: 'Resume the selected child.',
      subagentType: config.type,
      threadId: 'durable-thread',
      parentToolCallId,
      parentConfigurable: {
        [SUBAGENT_RESUME_MANIFEST_CONFIG_KEY]: {
          version: 1,
          executions: [resumeExecution],
        },
      },
    });
    const checkpointTarget = executor as unknown as {
      getLatestCheckpointSnapshot: (
        threadId: string
      ) => Promise<SubagentResumeExecution['checkpoints']>;
    };
    jest
      .spyOn(checkpointTarget, 'getLatestCheckpointSnapshot')
      .mockResolvedValue([
        {
          threadId: 'restored-child-thread',
          checkpointId: 'restored-checkpoint',
          checkpointNs: '',
        },
      ]);
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBe(config.configId);
  });

  it('retains configId through fresh lifecycle persistence after resolver failure', async () => {
    const config = makeLazyConfig('researcher', async () => {
      throw new Error('transient resolver failure');
    });
    const executor = createExecutor([config], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_fresh_lifecycle_retry';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the selected child.',
        subagent_type: config.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    const result = await executor.execute({
      description: 'Start the selected child.',
      subagentType: config.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: result.content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBe(config.configId);
  });

  it('persists the configId for the effective rewritten subagent type', async () => {
    const researcher = makeLazyConfig('researcher', async () =>
      makeAgent('researcher')
    );
    const coder = makeLazyConfig('coder', async () => {
      throw new Error('transient coder resolver failure');
    });
    const executor = createExecutor([researcher, coder], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_rewritten_type';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the original child.',
        subagent_type: researcher.type,
      },
      type: 'tool_call' as const,
    };
    const parentConfigurable = {
      thread_id: 'durable-thread',
      checkpoint_id: 'parent-checkpoint',
      run_id: 'parent-run',
    };
    const runnableConfig = { configurable: parentConfigurable };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    const result = await executor.execute({
      description: 'Run the rewritten child.',
      subagentType: coder.type,
      threadId: parentConfigurable.thread_id,
      parentToolCallId,
      parentConfigurable,
    });
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: result.content,
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the rewritten child.',
        subagent_type: coder.type,
      },
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(result.content).toBe(
      'Subagent error: Unable to initialize the selected subagent.'
    );
    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBe(coder.configId);
  });

  it('does not retain the original configId after rewriting to an eager config', async () => {
    const original = makeLazyConfig('researcher', async () =>
      makeAgent('researcher')
    );
    const eager: ResolvedSubagentConfig = {
      type: 'eager',
      name: 'Eager worker',
      description: 'Runs from eager inputs.',
      agentInputs: makeAgent('eager-child'),
    };
    const executor = createExecutor([original, eager], {
      checkpointer: new MemorySaver(),
      humanInTheLoop: { enabled: true },
    });
    const parentToolCallId = 'call_rewritten_eager';
    const call = {
      id: parentToolCallId,
      name: 'spawn_subagent',
      args: {
        description: 'Start the original child.',
        subagent_type: original.type,
      },
      type: 'tool_call' as const,
    };
    const runnableConfig = {
      configurable: {
        thread_id: 'durable-thread',
        checkpoint_id: 'parent-checkpoint',
        run_id: 'parent-run',
      },
    };

    await expect(
      executor.getSettledToolOutput(call, runnableConfig)
    ).resolves.toBeUndefined();
    await executor.persistSettledToolOutput(call, runnableConfig, {
      output: new ToolMessage({
        content: 'Eager child completed.',
        name: call.name,
        tool_call_id: parentToolCallId,
      }),
      additionalContexts: [],
      resolvedArgs: {
        description: 'Run the eager child.',
        subagent_type: eager.type,
      },
    });
    const manifest = await executor.getResumeManifest(
      new Set([parentToolCallId])
    );

    expect(manifest?.executions).toHaveLength(1);
    expect(manifest?.executions[0].configId).toBeUndefined();
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
