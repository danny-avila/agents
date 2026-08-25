import { describe, expect, it } from '@jest/globals';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type {
  EventActorAdapterPrepareRequest,
  EventActorCommitRequest,
  EventActorDiscardRequest,
  EventActorHead,
  EventActorHostAdapter,
  EventActorInvocation,
  EventActorPrepareRequest,
  EventActorTerminalResult,
} from '@/eventActor';
import { EventActorExecutor } from '@/eventActor';

type TestEvent = { text: string };

const head = (generation = 0): EventActorHead => {
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
      checkpointNs: 'committed',
      checkpointId: `checkpoint-${generation}`,
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
  invokes = 0;
  activeInvocations = 0;
  maxActiveInvocations = 0;
  commitError?: Error;
  readonly commits: EventActorCommitRequest<string>[] = [];
  readonly discards: EventActorDiscardRequest[] = [];
  readonly invokeSignals: AbortSignal[] = [];
  readonly forkNamespaces: string[] = [];
  invokeImpl?: (
    prepared: EventActorInvocation<TestEvent>,
    signal: AbortSignal
  ) => Promise<EventActorTerminalResult<string>>;

  async prepare(request: EventActorAdapterPrepareRequest<TestEvent>) {
    if (!this.checkpointAvailable) {
      return {
        status: 'checkpoint_unavailable' as const,
        head: head(this.generation),
      };
    }
    return {
      status: 'ready' as const,
      invocation: invocation(request, 'warm', this.generation),
    };
  }

  async coldContinue(
    request: EventActorAdapterPrepareRequest<TestEvent>,
    _head: EventActorHead
  ) {
    return invocation(request, 'cold', this.generation);
  }

  async invoke(
    prepared: EventActorInvocation<TestEvent>,
    context: { signal: AbortSignal }
  ) {
    this.invokes += 1;
    this.activeInvocations += 1;
    this.maxActiveInvocations = Math.max(
      this.maxActiveInvocations,
      this.activeInvocations
    );
    this.invokeSignals.push(context.signal);
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
      return { status: 'stale' as const, head: head(this.generation) };
    }
    this.generation += 1;
    return { status: 'committed' as const, head: head(this.generation) };
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
    host.invokeImpl = async () => ({ status: 'completed_no_action' });
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('no-action'))).resolves.toEqual({
      status: 'completed_no_action',
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

  it('allows only one concurrent invocation from a committed generation to advance', async () => {
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
    const first = executor.execute(request('event-a'));
    const second = executor.execute(request('event-b'));
    await Promise.resolve();
    release();

    const results = await Promise.all([first, second]);
    expect(results.map((result) => result.status).sort()).toEqual([
      'applied',
      'stale',
    ]);
    expect(host.generation).toBe(1);
    expect(host.discards).toEqual([
      expect.objectContaining({ reason: 'stale' }),
    ]);
  });

  it('retains a fork when commit acknowledgement is indeterminate', async () => {
    const host = new TestHost();
    host.commitError = new Error('connection dropped after commit');
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('uncertain'))).resolves.toMatchObject(
      {
        status: 'commit_indeterminate',
        error: { message: 'connection dropped after commit' },
      }
    );
    expect(host.commits).toHaveLength(1);
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

    const executions = await AsyncLocalStorageProviderSingleton.runWithConfig(
      { signal: parentController.signal },
      async () => [
        executor.execute(request('sibling-a')),
        executor.execute(request('sibling-b')),
      ]
    );
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

  it('discards a checkpoint outside the invocation fork', async () => {
    const host = new TestHost();
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

    await expect(executor.execute(request('escape'))).resolves.toMatchObject({
      status: 'failed',
      error: expect.objectContaining({
        message: 'Event actor result escaped its invocation checkpoint fork',
      }),
    });
    expect(host.commits).toHaveLength(0);
    expect(host.discards[0].reason).toBe('failed');
  });

  it('rejects and discards a preparation that reuses another fork namespace', async () => {
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
    expect(host.discards).toEqual([
      expect.objectContaining({ reason: 'failed' }),
    ]);
  });
});
