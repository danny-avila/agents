import { describe, expect, it, jest } from '@jest/globals';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  EventActorAdapterPrepareRequest,
  EventActorCheckpointFork,
  EventActorCommitRequest,
  EventActorDiscardRequest,
  EventActorHead,
  EventActorHostAdapter,
  EventActorInvocation,
  EventActorInvocationContext,
  EventActorPrepareRequest,
  EventActorTerminalResult,
} from '@/eventActor';
import { EventActorExecutor } from '@/eventActor';

type TestEvent = { text: string };

const head = (
  generation = 0,
  checkpointNs = 'committed',
  checkpointId = `checkpoint-${generation}`
): EventActorHead => {
  const actorHead: EventActorHead = {
    actorThreadId: 'actor-thread',
    generation,
  };
  if (generation === 0) {
    return actorHead;
  }
  return {
    ...actorHead,
    checkpoint: {
      threadId: 'actor-checkpoints',
      checkpointNs,
      checkpointId,
    },
  };
};

const invocation = (
  request: EventActorAdapterPrepareRequest<TestEvent>,
  continuation: 'warm' | 'cold' = 'warm',
  generation = 0
): EventActorInvocation<TestEvent> => ({
  ...request,
  continuation,
  base: head(generation),
  fork: {
    invocationId: request.invocationId,
    threadId: 'actor-checkpoints',
    checkpointNs: request.checkpointNs,
  },
});

class TestHost implements EventActorHostAdapter<TestEvent, string> {
  generation = 0;
  checkpointAvailable = true;
  committedCheckpointNs = 'committed';
  committedCheckpointId = 'checkpoint-0';
  coldContinues = 0;
  invokes = 0;
  activeInvocations = 0;
  maxActiveInvocations = 0;
  commitError?: Error;
  readonly commits: EventActorCommitRequest<string>[] = [];
  readonly discards: EventActorDiscardRequest[] = [];
  readonly invokeSignals: AbortSignal[] = [];
  readonly invokeConfigs: RunnableConfig[] = [];
  readonly forkNamespaces: string[] = [];
  invokeImpl?: (
    prepared: EventActorInvocation<TestEvent>,
    signal: AbortSignal
  ) => Promise<EventActorTerminalResult<string>>;

  async prepare(request: EventActorAdapterPrepareRequest<TestEvent>) {
    if (!this.checkpointAvailable) {
      return {
        status: 'checkpoint_unavailable' as const,
        head: head(
          this.generation,
          this.committedCheckpointNs,
          this.committedCheckpointId
        ),
      };
    }
    const actorHead = head(
      this.generation,
      this.committedCheckpointNs,
      this.committedCheckpointId
    );
    const prepared = invocation(request, 'warm', this.generation);
    return {
      status: 'ready' as const,
      invocation: {
        ...prepared,
        base: actorHead,
        fork: {
          ...prepared.fork,
          ...(actorHead.checkpoint?.checkpointId == null
            ? {}
            : { checkpointId: actorHead.checkpoint.checkpointId }),
        },
      },
    };
  }

  async coldContinue(
    request: EventActorAdapterPrepareRequest<TestEvent>,
    _head: EventActorHead
  ) {
    this.coldContinues += 1;
    const actorHead = head(
      this.generation,
      this.committedCheckpointNs,
      this.committedCheckpointId
    );
    const prepared = invocation(request, 'cold', this.generation);
    return {
      ...prepared,
      base: actorHead,
      fork: {
        ...prepared.fork,
        ...(actorHead.checkpoint?.checkpointId == null
          ? {}
          : { checkpointId: actorHead.checkpoint.checkpointId }),
      },
    };
  }

  async invoke(
    prepared: EventActorInvocation<TestEvent>,
    context: EventActorInvocationContext
  ) {
    this.invokes += 1;
    this.activeInvocations += 1;
    this.maxActiveInvocations = Math.max(
      this.maxActiveInvocations,
      this.activeInvocations
    );
    this.invokeSignals.push(context.signal);
    this.invokeConfigs.push(context.config);
    this.forkNamespaces.push(prepared.fork.checkpointNs);
    try {
      if (this.invokeImpl != null) {
        return await this.invokeImpl(prepared, context.signal);
      }
      return {
        status: 'applied' as const,
        result: prepared.event.text,
        checkpoint: {
          ...prepared.fork,
          checkpointId: `result-${prepared.invocationId}`,
        },
      };
    } finally {
      this.activeInvocations -= 1;
    }
  }

  async commit(request: EventActorCommitRequest<string>) {
    this.commits.push(request);
    if (this.commitError != null) {
      throw this.commitError;
    }
    if (request.expectedHead.generation !== this.generation) {
      return {
        status: 'stale' as const,
        head: head(
          this.generation,
          this.committedCheckpointNs,
          this.committedCheckpointId
        ),
      };
    }
    this.generation += 1;
    this.committedCheckpointNs = request.checkpoint.checkpointNs;
    this.committedCheckpointId = request.checkpoint.checkpointId ?? '';
    return {
      status: 'committed' as const,
      head: head(
        this.generation,
        this.committedCheckpointNs,
        this.committedCheckpointId
      ),
    };
  }

  async discard(request: EventActorDiscardRequest) {
    this.discards.push(request);
  }
}

const request = (
  invocationId: string,
  overrides: Partial<{ depth: number; signal: AbortSignal }> = {}
) => ({
  actorThreadId: 'actor-thread',
  invocationId,
  event: { text: invocationId },
  ...(overrides.depth == null ? {} : { depth: overrides.depth }),
  ...(overrides.signal == null ? {} : { signal: overrides.signal }),
});

describe('EventActorExecutor', () => {
  it('exposes the lifecycle seam for host-driven mailbox orchestration', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const prepareRequest: EventActorPrepareRequest<TestEvent> = {
      actorThreadId: 'actor-thread',
      invocationId: 'host-driven',
      depth: 1,
      event: { text: 'host-driven' },
    };

    const preparation = await executor.prepare(prepareRequest);
    expect(preparation.status).toBe('ready');
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const terminal = await executor.invoke(preparation.invocation);
    expect(terminal.status).toBe('applied');
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }
    await expect(
      executor.commit(preparation.invocation, terminal)
    ).resolves.toMatchObject({
      status: 'committed',
      head: { generation: 1 },
    });
    expect(host.activeInvocations).toBe(0);
  });

  it('commits applied work with current-plus-previous retention', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host, {
      dormantCheckpointTtlMs: 60_000,
    });

    await expect(executor.execute(request('move-1'))).resolves.toMatchObject({
      status: 'applied',
      result: 'move-1',
      continuation: 'warm',
      head: { generation: 1 },
    });
    expect(host.commits).toHaveLength(1);
    expect(host.commits[0]).toMatchObject({
      expectedHead: { generation: 0 },
      retention: {
        committedCheckpoints: 2,
        dormantCheckpointTtlMs: 60_000,
      },
    });
    expect(host.discards).toHaveLength(0);
  });

  it('cold-continues the same logical actor when its checkpoint is unavailable', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('cold-1'))).resolves.toMatchObject({
      status: 'applied',
      continuation: 'cold',
    });
    expect(host.commits[0].invocation).toMatchObject({
      actorThreadId: 'actor-thread',
      continuation: 'cold',
    });
  });

  it('accepts a committed actor head in LangGraph root checkpoint namespace', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.committedCheckpointNs = '';
    host.committedCheckpointId = 'root-checkpoint';
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('root-namespace'))
    ).resolves.toMatchObject({
      status: 'applied',
      head: { generation: 2 },
    });
    expect(host.commits[0].expectedHead.checkpoint).toMatchObject({
      checkpointNs: '',
      checkpointId: 'root-checkpoint',
    });
  });

  it('pauses and warm-resumes one logical actor without retaining an executor', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('turn-1'))).resolves.toMatchObject({
      status: 'applied',
      head: { actorThreadId: 'actor-thread', generation: 1 },
    });
    expect(host.activeInvocations).toBe(0);
    await expect(executor.execute(request('turn-2'))).resolves.toMatchObject({
      status: 'applied',
      continuation: 'warm',
      head: { actorThreadId: 'actor-thread', generation: 2 },
    });
    expect(host.commits[1].expectedHead).toMatchObject({ generation: 1 });
    expect(host.activeInvocations).toBe(0);
  });

  it('discards normal completions without qualifying action evidence', async () => {
    const host = new TestHost();
    host.invokeImpl = async () => ({
      status: 'completed_no_action',
      result: 'observation recorded',
    });
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('no-action'))).resolves.toEqual({
      status: 'completed_no_action',
      result: 'observation recorded',
      continuation: 'warm',
    });
    expect(host.commits).toHaveLength(0);
    expect(host.discards).toEqual([
      expect.objectContaining({ reason: 'completed_no_action' }),
    ]);
  });

  it('discards failures and explicit cancellations without advancing the head', async () => {
    const failedHost = new TestHost();
    failedHost.invokeImpl = async () => {
      throw new Error('provider failed');
    };
    const cancelledHost = new TestHost();
    cancelledHost.invokeImpl = async (_prepared, signal) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), {
          once: true,
        });
      });
    const controller = new AbortController();

    await expect(
      new EventActorExecutor(failedHost).execute(request('failed'))
    ).resolves.toMatchObject({ status: 'failed', continuation: 'warm' });
    const cancelled = new EventActorExecutor(cancelledHost).execute(
      request('cancelled', { signal: controller.signal })
    );
    controller.abort(new Error('cancelled'));
    await expect(cancelled).resolves.toEqual({
      status: 'cancelled',
      continuation: 'warm',
    });
    expect(failedHost.commits).toHaveLength(0);
    expect(cancelledHost.commits).toHaveLength(0);
    expect(failedHost.discards[0].reason).toBe('failed');
    expect(cancelledHost.discards[0].reason).toBe('cancelled');
  });

  it('isolates duplicate deliveries and retains an applied CAS conflict', async () => {
    const host = new TestHost();
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.invokeImpl = async (prepared) => {
      await gate;
      return {
        status: 'applied',
        result: prepared.invocationId,
        checkpoint: {
          ...prepared.fork,
          checkpointId: `result-${prepared.invocationId}`,
        },
      };
    };
    const executor = new EventActorExecutor(host);
    const first = executor.execute(request('same-event'));
    const second = executor.execute(request('same-event'));
    await Promise.resolve();
    release();

    const results = await Promise.all([first, second]);
    expect(results.map((result) => result.status).sort()).toEqual([
      'applied',
      'commit_conflict',
    ]);
    expect(
      results.find((result) => result.status === 'commit_conflict')
    ).toMatchObject({
      result: 'same-event',
      checkpoint: { invocationId: 'same-event' },
      head: { actorThreadId: 'actor-thread', generation: 1 },
    });
    expect(host.generation).toBe(1);
    expect(new Set(host.forkNamespaces).size).toBe(2);
    expect(host.discards).toHaveLength(0);
  });

  it('retains a fork when commit acknowledgement is indeterminate', async () => {
    const host = new TestHost();
    host.commitError = new Error('connection dropped after commit');
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('uncertain'))).resolves.toMatchObject(
      {
        status: 'commit_indeterminate',
        result: 'uncertain',
        checkpoint: {
          invocationId: 'uncertain',
          checkpointId: 'result-uncertain',
        },
        error: { message: 'connection dropped after commit' },
      }
    );
    expect(host.commits).toHaveLength(1);
    expect(host.discards).toHaveLength(0);
  });

  it('rejects a committed head that does not promote the terminal checkpoint', async () => {
    const host = new TestHost();
    const commit = host.commit.bind(host);
    host.commit = async (commitRequest) => {
      const result = await commit(commitRequest);
      if (result.status === 'committed' && result.head.checkpoint != null) {
        result.head.checkpoint.checkpointId = 'different-checkpoint';
      }
      return result;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('wrong-head'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      error: { message: 'Event actor commit returned an invalid logical head' },
    });
    expect(host.discards).toHaveLength(0);
  });

  it('replaces an ambient parent signal with independent task-owned signals', async () => {
    const host = new TestHost();
    const parentController = new AbortController();
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.invokeImpl = async (prepared) => {
      await gate;
      return {
        status: 'applied',
        result: prepared.invocationId,
        checkpoint: {
          ...prepared.fork,
          checkpointId: `result-${prepared.invocationId}`,
        },
      };
    };
    const executor = new EventActorExecutor(host);
    const ambientStore = { get: jest.fn() };
    const ambientContext = { tenantId: 'tenant-1' };
    const ambientConfig = {
      signal: parentController.signal,
      callbacks: [],
      tags: ['parent-trace'],
      metadata: {
        userId: 'user-1',
        sessionId: 'session-1',
        run_id: 'parent-run',
        thread_id: 'parent-thread',
        checkpoint_ns: 'parent-namespace',
        checkpoint_id: 'parent-checkpoint',
        checkpoint_map: { parent: 'checkpoint' },
        langgraph_checkpoint_ns: 'parent-langgraph-namespace',
        __pregel_scratchpad: { parent: true },
      },
      recursionLimit: 80,
      maxConcurrency: 4,
      timeout: 30_000,
      runId: 'parent-run',
      runName: 'parent-run-name',
      store: ambientStore,
      context: ambientContext,
      configurable: {
        user_id: 'user-1',
        run_id: 'parent-configurable-run',
        requestBody: { conversationId: 'conversation-1' },
        userMCPAuthMap: { chess: 'credential' },
        thread_id: 'parent-thread',
        checkpoint_ns: 'parent-namespace',
        checkpoint_id: 'parent-checkpoint',
        checkpoint_map: { parent: 'checkpoint' },
        __pregel_checkpointer: { parent: true },
        __pregel_scratchpad: { parent: true },
        __librechat_subagent_resume_manifest: { parent: true },
        __librechat_tool_approval_execution_scope: { parent: true },
        lc_run_breaker_scope: { parent: true },
      },
    } as RunnableConfig & {
      store: typeof ambientStore;
      context: typeof ambientContext;
    };

    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue(ambientConfig);
    const executions = [
      executor.execute(request('sibling-a')),
      executor.execute(request('sibling-b')),
    ];
    runnableConfigSpy.mockRestore();
    while (host.invokes < 2) {
      await Promise.resolve();
    }
    parentController.abort();
    expect(host.invokeSignals).toHaveLength(2);
    expect(new Set(host.invokeSignals).size).toBe(2);
    expect(new Set(host.forkNamespaces).size).toBe(2);
    expect(
      host.forkNamespaces.every((namespace) =>
        /^event-actor\/[a-f0-9]{32}$/.test(namespace)
      )
    ).toBe(true);
    expect(host.invokeSignals).not.toContain(parentController.signal);
    expect(host.invokeSignals.every((signal) => !signal.aborted)).toBe(true);
    expect(host.invokeConfigs).toHaveLength(2);
    expect(host.invokeConfigs[0]).toMatchObject({
      callbacks: [],
      tags: ['parent-trace'],
      recursionLimit: 80,
      maxConcurrency: 4,
      timeout: 30_000,
      store: ambientStore,
      context: ambientContext,
      metadata: {
        userId: 'user-1',
        sessionId: 'session-1',
        thread_id: 'actor-checkpoints',
        checkpoint_ns: expect.stringMatching(/^event-actor\/[a-f0-9]{32}$/),
        eventActorThreadId: 'actor-thread',
      },
      configurable: {
        user_id: 'user-1',
        requestBody: { conversationId: 'conversation-1' },
        userMCPAuthMap: { chess: 'credential' },
        event_actor_thread_id: 'actor-thread',
      },
    });
    expect(host.invokeConfigs[0].configurable?.thread_id).toBe(
      'actor-checkpoints'
    );
    expect(host.invokeConfigs[0].configurable?.checkpoint_ns).toMatch(
      /^event-actor\/[a-f0-9]{32}$/
    );
    expect(host.invokeConfigs[0].configurable).not.toHaveProperty(
      'checkpoint_id'
    );
    expect(host.invokeConfigs[0].configurable).not.toHaveProperty(
      'checkpoint_map'
    );
    expect(
      Object.keys(host.invokeConfigs[0].configurable ?? {}).some(
        (key) => key.startsWith('__pregel_') || key.startsWith('__librechat_')
      )
    ).toBe(false);
    expect(host.invokeConfigs[0].configurable).not.toHaveProperty(
      'lc_run_breaker_scope'
    );
    expect(host.invokeConfigs[0].configurable).not.toHaveProperty('run_id');
    expect(host.invokeConfigs[0]).not.toHaveProperty('runId');
    expect(host.invokeConfigs[0]).not.toHaveProperty('runName');
    expect(host.invokeConfigs[0].metadata).not.toHaveProperty('run_id');
    expect(host.invokeConfigs[0].metadata).not.toHaveProperty('checkpoint_id');
    expect(host.invokeConfigs[0].metadata).not.toHaveProperty('checkpoint_map');
    expect(host.invokeConfigs[0].metadata).not.toHaveProperty(
      'langgraph_checkpoint_ns'
    );
    expect(
      Object.keys(host.invokeConfigs[0].metadata ?? {}).some((key) =>
        key.startsWith('__pregel_')
      )
    ).toBe(false);
    release();

    const results = await Promise.all(executions);
    expect(results).toHaveLength(2);
    expect(host.activeInvocations).toBe(0);
    expect(host.maxActiveInvocations).toBe(2);
  });

  it('permits depth-one siblings and rejects event-actor grandchildren', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);

    await Promise.all([
      executor.execute(request('sibling-1')),
      executor.execute(request('sibling-2')),
    ]);
    await expect(
      executor.execute(request('grandchild', { depth: 2 }))
    ).rejects.toThrow('exceeds maximum 1');
    expect(host.invokes).toBe(2);
  });

  it('enforces maxDepth for host-driven invocation', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host, { maxDepth: 1 });
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'host-driven-depth',
      depth: 1,
      event: { text: 'host-driven-depth' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    preparation.invocation.depth = 2;

    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'exceeds maximum 1'
    );
    expect(host.invokes).toBe(0);
  });

  it('enforces ambient actor depth for host-driven invocation', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'ambient-host-driven-depth',
      depth: 1,
      event: { text: 'ambient-host-driven-depth' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue({ configurable: { event_actor_depth: 1 } });
    const execution = executor.invoke(preparation.invocation);
    runnableConfigSpy.mockRestore();

    await expect(execution).rejects.toThrow('must advance parent depth 1');
    expect(host.invokes).toBe(0);
  });

  it('binds ambient actor depth during public preparation', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue({ configurable: { event_actor_depth: 1 } });
    const warmPreparation = executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'ambient-warm-prepare',
      depth: 1,
      event: { text: 'ambient-warm-prepare' },
    });
    const coldPreparation = executor.coldContinue({
      status: 'checkpoint_unavailable',
      request: {
        actorThreadId: 'actor-thread',
        invocationId: 'ambient-cold-prepare',
        depth: 1,
        event: { text: 'ambient-cold-prepare' },
      },
      head: head(0),
    });
    runnableConfigSpy.mockRestore();

    await expect(warmPreparation).rejects.toThrow(
      'must advance parent depth 1'
    );
    await expect(coldPreparation).rejects.toThrow(
      'must advance parent depth 1'
    );
    expect(host.coldContinues).toBe(0);
    expect(host.invokes).toBe(0);
  });

  it('preserves prepared ancestry after async-local context ends', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue({ configurable: { event_actor_depth: 1 } });
    const preparationPromise = executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'persisted-cold-ancestry',
      depth: 2,
      event: { text: 'persisted-cold-ancestry' },
    });
    runnableConfigSpy.mockRestore();
    const preparation = await preparationPromise;
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }

    const invocation = await executor.coldContinue(preparation);

    expect(preparation.request.depth).toBe(2);
    expect(invocation).toMatchObject({
      actorThreadId: 'actor-thread',
      invocationId: 'persisted-cold-ancestry',
      depth: 2,
      continuation: 'cold',
    });
  });

  it('derives nested depth from actor-owned ambient context', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const ambientConfig: RunnableConfig = {
      configurable: { event_actor_depth: 1 },
    };
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue(ambientConfig);

    await expect(
      executor.execute(request('implicit-grandchild'))
    ).rejects.toThrow('exceeds maximum 1');
    await expect(
      executor.execute(request('forged-grandchild', { depth: 1 }))
    ).rejects.toThrow('must advance parent depth 1');
    runnableConfigSpy.mockRestore();
    expect(host.invokes).toBe(0);
  });

  it('cancels before unavailable checkpoints enter cold continuation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const prepare = host.prepare.bind(host);
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.prepare = async (prepareRequest) => {
      await gate;
      return prepare(prepareRequest);
    };
    const controller = new AbortController();
    const execution = new EventActorExecutor(host).execute(
      request('cancel-before-cold', { signal: controller.signal })
    );
    controller.abort(new Error('cancelled'));
    release();

    await expect(execution).resolves.toEqual({
      status: 'cancelled',
      continuation: 'cold',
    });
    expect(host.coldContinues).toBe(0);
    expect(host.invokes).toBe(0);
    expect(host.discards).toHaveLength(0);
  });

  it('retains an applied result with a checkpoint outside the invocation fork', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.fork.checkpointId = 'starting-checkpoint';
      }
      return prepared;
    };
    host.invokeImpl = async () => ({
      status: 'applied',
      result: 'bad',
      checkpoint: {
        invocationId: 'another-invocation',
        threadId: 'actor-checkpoints',
        checkpointNs: 'invocations/another-invocation',
        checkpointId: 'bad-result',
      },
    });
    const executor = new EventActorExecutor(host);

    const result = await executor.execute(request('escape'));
    expect(result).toMatchObject({
      status: 'commit_indeterminate',
      result: 'bad',
      checkpoint: {
        invocationId: 'escape',
        threadId: 'actor-checkpoints',
      },
      error: expect.objectContaining({
        message: 'Event actor result escaped its invocation checkpoint fork',
      }),
    });
    expect(result).toHaveProperty('checkpoint');
    if (result.status !== 'commit_indeterminate') {
      throw new Error('Expected indeterminate commit');
    }
    expect(result.checkpoint).not.toHaveProperty('checkpointId');
    expect(host.commits).toHaveLength(0);
    expect(host.discards).toHaveLength(0);
  });

  it('retains an applied result that reports the fork starting checkpoint', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.fork.checkpointId = 'starting-checkpoint';
      }
      return prepared;
    };
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: 'action-claimed',
      checkpoint: { ...prepared.fork },
    });
    const executor = new EventActorExecutor(host);

    const result = await executor.execute(request('unchanged-checkpoint'));

    expect(result).toMatchObject({
      status: 'commit_indeterminate',
      result: 'action-claimed',
      checkpoint: {
        invocationId: 'unchanged-checkpoint',
        threadId: 'actor-checkpoints',
      },
      error: {
        message: 'Event actor result escaped its invocation checkpoint fork',
      },
    });
    if (result.status !== 'commit_indeterminate') {
      throw new Error('Expected indeterminate commit');
    }
    expect(result.checkpoint).not.toHaveProperty('checkpointId');
    expect(host.commits).toHaveLength(0);
    expect(host.discards).toHaveLength(0);
  });

  it('snapshots validated terminal evidence before awaiting commit', async () => {
    const host = new TestHost();
    let terminalCheckpoint: EventActorCheckpointFork | undefined;
    host.invokeImpl = async (prepared) => {
      terminalCheckpoint = {
        ...prepared.fork,
        checkpointId: 'trusted-terminal',
      };
      return {
        status: 'applied',
        result: 'action-completed',
        checkpoint: terminalCheckpoint,
      };
    };
    host.commit = async () => {
      if (terminalCheckpoint == null) {
        throw new Error('Expected terminal checkpoint');
      }
      terminalCheckpoint.threadId = 'mutated-thread';
      terminalCheckpoint.checkpointNs = 'mutated-namespace';
      terminalCheckpoint.checkpointId = 'mutated-checkpoint';
      return { status: 'stale' as const, head: head(1) };
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('mutable-terminal'))
    ).resolves.toMatchObject({
      status: 'commit_conflict',
      result: 'action-completed',
      checkpoint: {
        invocationId: 'mutable-terminal',
        threadId: 'actor-checkpoints',
        checkpointNs: expect.stringMatching(/^event-actor\/[a-f0-9]{32}$/),
        checkpointId: 'trusted-terminal',
      },
    });
  });

  it('invokes with the authoritative request event after warm preparation', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.event = { text: 'stale-event' };
      }
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('authoritative-event'))
    ).resolves.toMatchObject({
      status: 'applied',
      result: 'authoritative-event',
    });
  });

  it('invokes with the authoritative request event after cold continuation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead) => {
      const prepared = await coldContinue(prepareRequest, actorHead);
      prepared.event = { text: 'stale-event' };
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('authoritative-cold-event'))
    ).resolves.toMatchObject({
      status: 'applied',
      result: 'authoritative-cold-event',
      continuation: 'cold',
    });
  });

  it('rejects an advanced head without committed checkpoint identity', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.prepare = async () => ({
      status: 'checkpoint_unavailable' as const,
      head: { actorThreadId: 'actor-thread', generation: 1 },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('missing-head-checkpoint'))
    ).rejects.toThrow('Advanced event actor head has no checkpoint');
    expect(host.coldContinues).toBe(0);
    expect(host.invokes).toBe(0);
  });

  it('rejects a fork that leaves the committed logical checkpoint thread', async () => {
    const host = new TestHost();
    host.generation = 1;
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.fork.threadId = 'another-checkpoint-thread';
      }
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('foreign-checkpoint-thread'))
    ).rejects.toThrow('changed its logical checkpoint thread');
    expect(host.invokes).toBe(0);
  });

  it('requires a warm resumed fork to identify its starting checkpoint', async () => {
    const host = new TestHost();
    host.generation = 1;
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        delete prepared.invocation.fork.checkpointId;
      }
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('missing-warm-start'))
    ).rejects.toThrow('fork.checkpointId for resumed actor');
    expect(host.invokes).toBe(0);
  });

  it('binds a warm resumed fork to the committed checkpoint', async () => {
    const host = new TestHost();
    host.generation = 1;
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.fork.checkpointId = 'another-checkpoint';
      }
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('wrong-warm-start'))).rejects.toThrow(
      'did not start from the committed checkpoint'
    );
    expect(host.invokes).toBe(0);
  });

  it('allows a cold resumed fork to start from reconstructed state', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead) => {
      const prepared = await coldContinue(prepareRequest, actorHead);
      prepared.fork.checkpointId = 'reconstructed-start';
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('reconstructed-cold-start'))
    ).resolves.toMatchObject({ status: 'applied', continuation: 'cold' });
  });

  it('requires a cold resumed fork to identify its starting checkpoint', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead) => {
      const prepared = await coldContinue(prepareRequest, actorHead);
      delete prepared.fork.checkpointId;
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('missing-cold-start'))
    ).rejects.toThrow('fork.checkpointId for resumed actor');
    expect(host.invokes).toBe(0);
  });

  it('retains the SDK-generated namespace across warm adapter mutation', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      prepareRequest.checkpointNs = 'shared-namespace';
      return prepare(prepareRequest);
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('mutated-warm-namespace'))
    ).rejects.toThrow('mismatched checkpoint ownership');
    expect(host.invokes).toBe(0);
  });

  it('retains the SDK-generated namespace across cold adapter mutation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead) => {
      prepareRequest.checkpointNs = 'shared-namespace';
      return coldContinue(prepareRequest, actorHead);
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('mutated-cold-namespace'))
    ).rejects.toThrow('mismatched checkpoint ownership');
    expect(host.invokes).toBe(0);
  });

  it('rejects a stale commit result for another actor', async () => {
    const host = new TestHost();
    host.commit = async () => ({
      status: 'stale' as const,
      head: { actorThreadId: 'another-actor', generation: 0 },
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'foreign-stale-head',
      depth: 1,
      event: { text: 'foreign-stale-head' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const terminal = await executor.invoke(preparation.invocation);
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }

    await expect(
      executor.commit(preparation.invocation, terminal)
    ).rejects.toThrow('Event actor head is invalid');
  });

  it('rejects a stale commit head that did not advance past its base', async () => {
    const host = new TestHost();
    host.commit = async () => ({ status: 'stale' as const, head: head(0) });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'non-advanced-stale-head',
      depth: 1,
      event: { text: 'non-advanced-stale-head' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const terminal = await executor.invoke(preparation.invocation);
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }

    await expect(
      executor.commit(preparation.invocation, terminal)
    ).rejects.toThrow('did not advance past its base');
  });

  it('rejects a stale commit head that switches checkpoint threads', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.commit = async () => ({
      status: 'stale' as const,
      head: {
        actorThreadId: 'actor-thread',
        generation: 2,
        checkpoint: {
          threadId: 'another-checkpoint-thread',
          checkpointNs: 'other-namespace',
          checkpointId: 'other-checkpoint',
        },
      },
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'switched-stale-thread',
      depth: 1,
      event: { text: 'switched-stale-thread' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const terminal = await executor.invoke(preparation.invocation);
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }

    await expect(
      executor.commit(preparation.invocation, terminal)
    ).rejects.toThrow('changed its checkpoint thread');
  });

  it('retains applied work when stale moves the base checkpoint to another namespace', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.commit = async () => ({
      status: 'stale' as const,
      head: {
        ...head(1, 'another-namespace', 'checkpoint-0'),
        generation: 2,
      },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('stale-base-checkpoint'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'stale-base-checkpoint',
      error: {
        message:
          'Stale event actor head does not identify a competing checkpoint',
      },
    });
    expect(host.discards).toHaveLength(0);
  });

  it('retains applied work when stale names the submitted checkpoint', async () => {
    const host = new TestHost();
    host.commit = async (commitRequest) => ({
      status: 'stale' as const,
      head: {
        actorThreadId: 'actor-thread',
        generation: 1,
        checkpoint: { ...commitRequest.checkpoint },
      },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('stale-terminal-checkpoint'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'stale-terminal-checkpoint',
      error: {
        message:
          'Stale event actor head does not identify a competing checkpoint',
      },
    });
    expect(host.discards).toHaveLength(0);
  });

  it('retains cold-applied work when stale names its reconstructed start', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead) => {
      const prepared = await coldContinue(prepareRequest, actorHead);
      prepared.fork.checkpointId = 'reconstructed-start';
      return prepared;
    };
    host.commit = async () => ({
      status: 'stale' as const,
      head: {
        actorThreadId: 'actor-thread',
        generation: 2,
        checkpoint: {
          threadId: 'actor-checkpoints',
          checkpointNs: 'winner-namespace',
          checkpointId: 'reconstructed-start',
        },
      },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('stale-reconstructed-start'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'stale-reconstructed-start',
      error: {
        message:
          'Stale event actor head does not identify a competing checkpoint',
      },
    });
    expect(host.discards).toHaveLength(0);
  });

  it('accepts an initial stale winner that establishes another checkpoint thread', async () => {
    const host = new TestHost();
    host.commit = async () => ({
      status: 'stale' as const,
      head: {
        actorThreadId: 'actor-thread',
        generation: 1,
        checkpoint: {
          threadId: 'winner-checkpoint-thread',
          checkpointNs: 'winner-namespace',
          checkpointId: 'winner-checkpoint',
        },
      },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('initial-thread-race'))
    ).resolves.toMatchObject({
      status: 'commit_conflict',
      head: {
        generation: 1,
        checkpoint: { threadId: 'winner-checkpoint-thread' },
      },
    });
  });

  it('commits an applied result even when cancellation arrives as invoke returns', async () => {
    const host = new TestHost();
    const controller = new AbortController();
    host.invokeImpl = async (prepared) => {
      controller.abort(new Error('parent settled'));
      return {
        status: 'applied',
        result: 'move submitted',
        checkpoint: {
          ...prepared.fork,
          checkpointId: 'applied-before-abort',
        },
      };
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(
        request('applied-before-abort', { signal: controller.signal })
      )
    ).resolves.toMatchObject({ status: 'applied', head: { generation: 1 } });
    expect(host.commits).toHaveLength(1);
    expect(host.discards).toHaveLength(0);
  });

  it('rejects a preparation without discarding its untrusted fork reference', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    host.prepare = async (prepareRequest) => {
      const prepared = await prepare(prepareRequest);
      if (prepared.status === 'ready') {
        prepared.invocation.fork.checkpointNs = 'shared';
      }
      return prepared;
    };
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('wrong-namespace'))).rejects.toThrow(
      'mismatched checkpoint ownership'
    );
    expect(host.discards).toHaveLength(0);
  });

  it('validates a cold-continuation head before adapter work', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);

    await expect(
      executor.coldContinue({
        status: 'checkpoint_unavailable',
        request: {
          actorThreadId: 'actor-thread',
          invocationId: 'foreign-head',
          depth: 1,
          event: { text: 'foreign-head' },
        },
        head: { actorThreadId: 'another-actor', generation: 0 },
      })
    ).rejects.toThrow('Event actor head is invalid');
    expect(host.coldContinues).toBe(0);
  });

  it('rejects malformed ownership before destructive host operations', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'malformed-ownership',
      depth: 1,
      event: { text: 'malformed-ownership' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const malformed = {
      ...preparation.invocation,
      base: { ...preparation.invocation.base, actorThreadId: 'another-actor' },
    };
    const terminal = await executor.invoke(preparation.invocation);
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }

    expect(() => executor.discard(malformed, 'failed')).toThrow(
      'Event actor head is invalid'
    );
    await expect(executor.commit(malformed, terminal)).rejects.toThrow(
      'Event actor head is invalid'
    );
    expect(host.discards).toHaveLength(0);
    expect(host.commits).toHaveLength(0);
  });

  it('detects a cold adapter mutating its copy of the prepared head', async () => {
    const host = new TestHost();
    const validHead = head(1);
    host.coldContinue = async (prepareRequest, adapterHead) => {
      adapterHead.generation = 2;
      return {
        ...invocation(prepareRequest, 'cold', 2),
        base: adapterHead,
        fork: {
          ...invocation(prepareRequest, 'cold', 2).fork,
          checkpointId: adapterHead.checkpoint?.checkpointId,
        },
      };
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.coldContinue({
        status: 'checkpoint_unavailable',
        request: {
          actorThreadId: 'actor-thread',
          invocationId: 'mutated-head',
          depth: 1,
          event: { text: 'mutated-head' },
        },
        head: validHead,
      })
    ).rejects.toThrow('Cold continuation did not use the prepared actor head');
    expect(validHead.generation).toBe(1);
  });

  it('uses the validated ownership snapshot after adapter invocation', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => {
      const checkpoint = {
        ...prepared.fork,
        checkpointId: 'trusted-terminal',
      };
      prepared.actorThreadId = 'another-actor';
      prepared.base.generation = 99;
      prepared.fork.checkpointNs = 'another-fork';
      return {
        status: 'applied',
        result: 'applied-before-mutation',
        checkpoint,
      };
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('mutating-adapter'))
    ).resolves.toMatchObject({
      status: 'applied',
      head: { actorThreadId: 'actor-thread', generation: 1 },
    });
    expect(host.commits[0].invocation).toMatchObject({
      actorThreadId: 'actor-thread',
      invocationId: 'mutating-adapter',
      base: { generation: 0 },
      fork: { checkpointNs: expect.stringMatching(/^event-actor\//) },
    });
  });

  it('keeps a host-driven prepared invocation immutable across invoke', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => {
      const checkpoint = {
        ...prepared.fork,
        checkpointId: 'host-driven-terminal',
      };
      prepared.base.generation = 99;
      prepared.fork.checkpointNs = 'mutated-by-adapter';
      return { status: 'applied', result: 'done', checkpoint };
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'host-driven-mutation',
      depth: 1,
      event: { text: 'host-driven-mutation' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const originalNamespace = preparation.invocation.fork.checkpointNs;

    const terminal = await executor.invoke(preparation.invocation);

    expect(preparation.invocation.base.generation).toBe(0);
    expect(preparation.invocation.fork.checkpointNs).toBe(originalNamespace);
    expect(terminal.status).toBe('applied');
    if (terminal.status !== 'applied') {
      throw new Error('Expected applied terminal result');
    }
    await expect(
      executor.commit(preparation.invocation, terminal)
    ).resolves.toMatchObject({ status: 'committed' });
  });

  it('snapshots applied checkpoint evidence returned by public invoke', async () => {
    const host = new TestHost();
    let adapterCheckpoint: EventActorCheckpointFork | undefined;
    host.invokeImpl = async (prepared) => {
      adapterCheckpoint = {
        ...prepared.fork,
        checkpointId: 'public-invoke-terminal',
      };
      return {
        status: 'applied',
        result: 'public-invoke-action',
        checkpoint: adapterCheckpoint,
      };
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-invoke-snapshot',
      depth: 1,
      event: { text: 'public-invoke-snapshot' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    const terminal = await executor.invoke(preparation.invocation);
    if (terminal.status !== 'applied' || adapterCheckpoint == null) {
      throw new Error('Expected applied terminal result');
    }
    adapterCheckpoint.checkpointId = 'mutated-checkpoint';
    adapterCheckpoint.checkpointNs = 'mutated-namespace';

    expect(terminal.checkpoint).toMatchObject({
      checkpointId: 'public-invoke-terminal',
      checkpointNs: expect.stringMatching(/^event-actor\/[a-f0-9]{32}$/),
    });
    await expect(
      executor.commit(preparation.invocation, terminal)
    ).resolves.toMatchObject({ status: 'committed' });
  });
});
