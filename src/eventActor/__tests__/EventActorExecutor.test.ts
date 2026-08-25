import { createHash } from 'node:crypto';
import { describe, expect, it, jest } from '@jest/globals';
import { Command, GraphInterrupt, ParentCommand } from '@langchain/langgraph';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  EventActorAdapterPrepareRequest,
  EventActorCheckpointFork,
  EventActorCommitRequest,
  EventActorDiscardRequest,
  EventActorEvent,
  EventActorHead,
  EventActorHostAdapter,
  EventActorInvocation,
  EventActorInvocationContext,
  EventActorPreparationContext,
  EventActorPrepareRequest,
  EventActorTerminalResult,
} from '@/eventActor';
import { EventActorExecutor } from '@/eventActor';

type TestEvent = { text: string; value?: number };

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
  readonly prepareSignals: AbortSignal[] = [];
  readonly coldContinueSignals: AbortSignal[] = [];
  readonly invokeConfigs: RunnableConfig[] = [];
  readonly forkNamespaces: string[] = [];
  invokeImpl?: (
    prepared: EventActorInvocation<TestEvent>,
    signal: AbortSignal
  ) => Promise<EventActorTerminalResult<string>>;

  async prepare(
    request: EventActorAdapterPrepareRequest<TestEvent>,
    context?: EventActorPreparationContext
  ) {
    if (context != null) {
      this.prepareSignals.push(context.signal);
    }
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
    _head: EventActorHead,
    context: EventActorPreparationContext
  ) {
    this.coldContinues += 1;
    this.coldContinueSignals.push(context.signal);
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
    const currentHead = head(
      this.generation,
      this.committedCheckpointNs,
      this.committedCheckpointId
    );
    if (
      request.expectedHead.generation !== currentHead.generation ||
      request.expectedHead.checkpoint?.threadId !==
        currentHead.checkpoint?.threadId ||
      request.expectedHead.checkpoint?.checkpointNs !==
        currentHead.checkpoint?.checkpointNs ||
      request.expectedHead.checkpoint?.checkpointId !==
        currentHead.checkpoint?.checkpointId
    ) {
      return {
        status: 'stale' as const,
        head: currentHead,
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
  it('keeps lifecycle capability state runtime-private', () => {
    const executor = new EventActorExecutor(new TestHost());

    expect(Object.getOwnPropertyNames(executor)).toEqual([]);
    expect(Reflect.get(executor, 'preparationSigningKey')).toBeUndefined();
    expect(Reflect.get(executor, 'preparationPhases')).toBeUndefined();
    expect(Reflect.get(executor, 'issuedSettlements')).toBeUndefined();
    expect(Reflect.get(executor, 'signPreparation')).toBeUndefined();
    expect(Reflect.get(executor, 'issueSettlement')).toBeUndefined();
    expect(Reflect.get(executor, 'invokeWithConfig')).toBeUndefined();
    expect(Reflect.get(executor, 'discardInvocationReference')).toBeUndefined();
  });

  it('rejects unknown adapter preparation statuses', async () => {
    const host = new TestHost();
    host.prepare = async () =>
      ({
        status: 'unknown',
        head: head(0),
      }) as unknown as Awaited<ReturnType<TestHost['prepare']>>;
    const executor = new EventActorExecutor(host);

    await expect(
      executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'unknown-preparation-status',
        depth: 1,
        event: { text: 'unknown-preparation-status' },
      })
    ).rejects.toThrow('preparation returned an invalid status');
    expect(host.coldContinues).toBe(0);
    expect(host.invokes).toBe(0);
  });

  it('rejects preparation signing keys below 256 bits', () => {
    expect(
      () =>
        new EventActorExecutor(new TestHost(), {
          preparationSigningKey: 'short-key',
        })
    ).toThrow('preparationSigningKey must contain at least 32 bytes');
  });

  it('expires signed prepared invocation authority with its dormant fork', async () => {
    const now = jest.spyOn(Date, 'now').mockReturnValue(1_000);
    try {
      const host = new TestHost();
      const executor = new EventActorExecutor(host, {
        dormantCheckpointTtlMs: 10,
      });
      const preparation = await executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'expired-preparation',
        depth: 1,
        event: { text: 'expired-preparation' },
      });
      if (preparation.status !== 'ready') {
        throw new Error('Expected warm preparation');
      }

      now.mockReturnValue(1_010);
      await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
        'prepared invocation binding has expired'
      );
      await expect(
        executor.discard(preparation.invocation, 'failed')
      ).rejects.toThrow('prepared invocation binding has expired');
      expect(host.invokes).toBe(0);
      expect(host.discards).toHaveLength(0);

      now.mockReturnValue(2_000);
      host.invokeImpl = async () => ({ status: 'completed_no_action' });
      const consumed = await executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'expired-terminal-phase',
        depth: 1,
        event: { text: 'expired-terminal-phase' },
      });
      if (consumed.status !== 'ready') {
        throw new Error('Expected warm preparation');
      }
      await expect(executor.invoke(consumed.invocation)).resolves.toEqual({
        status: 'completed_no_action',
      });

      now.mockReturnValue(2_010);
      await expect(
        executor.discard(consumed.invocation, 'completed_no_action')
      ).rejects.toThrow('prepared invocation binding has expired');
      expect(host.invokes).toBe(1);
      expect(host.discards).toHaveLength(1);
    } finally {
      now.mockRestore();
    }
  });

  it('retains a terminal fence through signed authority expiry after clock rollback', async () => {
    const now = jest.spyOn(Date, 'now').mockReturnValue(1_000);
    try {
      const host = new TestHost();
      const executor = new EventActorExecutor(host, {
        dormantCheckpointTtlMs: 100,
      });
      const preparation = await executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'rollback-terminal-fence',
        depth: 1,
        event: { text: 'rollback-terminal-fence' },
      });
      if (preparation.status !== 'ready') {
        throw new Error('Expected warm preparation');
      }

      now.mockReturnValue(900);
      await expect(
        executor.invoke(preparation.invocation)
      ).resolves.toMatchObject({ status: 'applied' });
      now.mockReturnValue(1_050);
      await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
        'already consumed'
      );
      expect(host.invokes).toBe(1);
    } finally {
      now.mockRestore();
    }
  });

  it('time-bounds and consumes checkpoint-unavailable handoffs', async () => {
    const now = jest.spyOn(Date, 'now').mockReturnValue(1_000);
    try {
      const host = new TestHost();
      host.checkpointAvailable = false;
      const executor = new EventActorExecutor(host, {
        dormantCheckpointTtlMs: 10,
      });
      const consumed = await executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'consumed-unavailable',
        depth: 1,
        event: { text: 'consumed-unavailable' },
      });
      if (consumed.status !== 'checkpoint_unavailable') {
        throw new Error('Expected unavailable checkpoint');
      }
      await expect(executor.coldContinue(consumed)).resolves.toMatchObject({
        invocationId: 'consumed-unavailable',
        continuation: 'cold',
      });
      await expect(executor.coldContinue(consumed)).rejects.toThrow(
        'unavailable preparation was already consumed'
      );

      const expired = await executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'expired-unavailable',
        depth: 1,
        event: { text: 'expired-unavailable' },
      });
      if (expired.status !== 'checkpoint_unavailable') {
        throw new Error('Expected unavailable checkpoint');
      }
      now.mockReturnValue(1_010);
      await expect(executor.coldContinue(expired)).rejects.toThrow(
        'unavailable preparation binding has expired'
      );
      expect(host.coldContinues).toBe(1);
    } finally {
      now.mockRestore();
    }
  });

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
    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'committed',
      head: { generation: 1 },
    });
    expect(host.activeInvocations).toBe(0);
  });

  it('binds host-driven commit to the exact invocation that executed', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
      prepared.fork.checkpointId = 'reconstructed-start';
      return prepared;
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'sealed-settlement',
      depth: 1,
      event: { text: 'sealed-settlement' },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    const prepared = await executor.coldContinue(preparation);
    const settlement = await executor.invoke(prepared);
    if (settlement.status !== 'applied') {
      throw new Error('Expected applied settlement');
    }

    expect(Object.isFrozen(settlement)).toBe(true);
    expect(Object.isFrozen(settlement.invocation)).toBe(true);
    expect(() => {
      settlement.invocation.base.generation = 2;
    }).toThrow();
    const forgedSettlement = {
      ...settlement,
      invocation: {
        ...settlement.invocation,
        base: head(2, 'later', 'later-checkpoint'),
      },
    };
    await expect(executor.commit(forgedSettlement)).rejects.toThrow(
      'settlement was not issued by this executor'
    );
    await expect(executor.commit(settlement)).resolves.toMatchObject({
      status: 'committed',
      head: { generation: 2 },
    });
  });

  it('snapshots mutable settlement results before commit', async () => {
    type ObjectResult = { outcome: string; detail: { code: number } };
    const host = new TestHost();
    const adapterResult: ObjectResult = {
      outcome: 'applied',
      detail: { code: 200 },
    };
    let committedResult: ObjectResult | undefined;
    const adapter: EventActorHostAdapter<TestEvent, ObjectResult> = {
      prepare: host.prepare.bind(host),
      coldContinue: host.coldContinue.bind(host),
      invoke: async (prepared) => ({
        status: 'applied',
        result: adapterResult,
        checkpoint: {
          ...prepared.fork,
          checkpointId: 'object-result-terminal',
        },
      }),
      commit: async (commitRequest) => {
        committedResult = commitRequest.result;
        return {
          status: 'committed',
          head: {
            actorThreadId: commitRequest.invocation.actorThreadId,
            generation: commitRequest.expectedHead.generation + 1,
            checkpoint: { ...commitRequest.checkpoint },
          },
        };
      },
      discard: host.discard.bind(host),
    };
    const executor = new EventActorExecutor(adapter);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'immutable-result',
      depth: 1,
      event: { text: 'immutable-result' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const settlement = await executor.invoke(preparation.invocation);
    if (settlement.status !== 'applied') {
      throw new Error('Expected applied settlement');
    }

    adapterResult.outcome = 'mutated';
    adapterResult.detail.code = 500;
    expect(() => {
      settlement.result.outcome = 'forged';
    }).toThrow();
    await executor.commit(settlement);

    expect(committedResult).toEqual({
      outcome: 'applied',
      detail: { code: 200 },
    });
  });

  it('snapshots applied results before reporting an invalid terminal checkpoint', async () => {
    type ObjectResult = { outcome: string; detail: { code: number } };
    const host = new TestHost();
    const adapterResult: ObjectResult = {
      outcome: 'applied',
      detail: { code: 200 },
    };
    const adapter: EventActorHostAdapter<TestEvent, ObjectResult> = {
      prepare: host.prepare.bind(host),
      coldContinue: host.coldContinue.bind(host),
      invoke: async () => ({
        status: 'applied',
        result: adapterResult,
        checkpoint: {
          invocationId: 'invalid-terminal',
          threadId: 'foreign-checkpoint-thread',
          checkpointNs: 'foreign-checkpoint-namespace',
        },
      }),
      commit: async () => {
        throw new Error('commit must not run');
      },
      discard: host.discard.bind(host),
    };
    const executor = new EventActorExecutor(adapter);

    const execution = await executor.execute({
      actorThreadId: 'actor-thread',
      invocationId: 'invalid-terminal',
      event: { text: 'invalid-terminal' },
    });
    if (execution.status !== 'commit_indeterminate') {
      throw new Error('Expected an indeterminate commit');
    }
    if (execution.result == null) {
      throw new Error('Expected preserved applied result evidence');
    }
    adapterResult.outcome = 'mutated';
    adapterResult.detail.code = 500;

    expect(execution.result).toEqual({
      outcome: 'applied',
      detail: { code: 200 },
    });
    expect(Object.isFrozen(execution.result)).toBe(true);
    expect(Object.isFrozen(execution.result.detail)).toBe(true);
  });

  it('returns indeterminate evidence when an applied result is not JSON-safe', async () => {
    const host = new TestHost();
    const sparseResult = new Array<EventActorEvent>(1);
    const adapter: EventActorHostAdapter<TestEvent, EventActorEvent> = {
      prepare: host.prepare.bind(host),
      coldContinue: host.coldContinue.bind(host),
      invoke: async (prepared) => ({
        status: 'applied',
        result: sparseResult,
        checkpoint: {
          ...prepared.fork,
          checkpointId: 'sparse-result-terminal',
        },
      }),
      commit: async () => {
        throw new Error('commit must not run');
      },
      discard: host.discard.bind(host),
    };
    const executor = new EventActorExecutor(adapter);

    const execution = await executor.execute(request('sparse-result'));
    expect(execution).toEqual(
      expect.objectContaining({
        status: 'commit_indeterminate',
        error: expect.objectContaining({
          message: expect.stringContaining('arrays must not contain holes'),
        }),
        continuation: 'warm',
      })
    );
    expect(execution).not.toHaveProperty('result');
    expect(host.discards).toHaveLength(0);
  });

  it('validates and commits the same terminal checkpoint snapshot', async () => {
    const host = new TestHost();
    let namespaceReads = 0;
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: 'single-read-terminal',
      checkpoint: Object.defineProperties(
        {},
        {
          invocationId: {
            enumerable: true,
            value: prepared.invocationId,
          },
          threadId: { enumerable: true, value: prepared.fork.threadId },
          checkpointNs: {
            enumerable: true,
            get: () => {
              namespaceReads += 1;
              return namespaceReads === 1
                ? prepared.fork.checkpointNs
                : 'forged-namespace';
            },
          },
          checkpointId: {
            enumerable: true,
            value: 'single-read-terminal-checkpoint',
          },
        }
      ) as EventActorCheckpointFork,
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('single-read-terminal'))
    ).resolves.toMatchObject({
      status: 'applied',
      result: 'single-read-terminal',
      head: { generation: 1 },
    });
    expect(namespaceReads).toBe(1);
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

  it('models checkpoint identity in the test adapter CAS', async () => {
    const host = new TestHost();
    host.generation = 1;
    const fork: EventActorCheckpointFork = {
      invocationId: 'same-generation-mismatch',
      threadId: 'actor-checkpoints',
      checkpointNs: 'attempt',
      checkpointId: 'terminal',
    };

    await expect(
      host.commit({
        invocation: {
          actorThreadId: 'actor-thread',
          invocationId: 'same-generation-mismatch',
          depth: 1,
          continuation: 'warm',
          base: head(1, 'wrong-namespace', 'checkpoint-0'),
          fork,
        },
        expectedHead: head(1, 'wrong-namespace', 'checkpoint-0'),
        checkpoint: fork,
        result: 'must-not-commit',
        retention: {
          committedCheckpoints: 2,
          dormantCheckpointTtlMs: 60_000,
        },
      })
    ).resolves.toMatchObject({
      status: 'stale',
      head: {
        generation: 1,
        checkpoint: {
          checkpointNs: 'committed',
          checkpointId: 'checkpoint-0',
        },
      },
    });
    expect(host.generation).toBe(1);
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

  it('discards a no-action fork when its result cannot be snapshotted', async () => {
    const host = new TestHost();
    host.invokeImpl = async () => ({
      status: 'completed_no_action',
      result: new Array<EventActorEvent>(1) as unknown as string,
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('invalid-no-action-result'))
    ).resolves.toMatchObject({
      status: 'failed',
      error: {
        message: expect.stringContaining('arrays must not contain holes'),
      },
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

  it.each([
    ['GraphInterrupt', new GraphInterrupt([])],
    ['ParentCommand', new ParentCommand(new Command({ goto: 'parent-node' }))],
  ])('propagates %s without discarding its resumable fork', async (_, flow) => {
    const host = new TestHost();
    host.invokeImpl = async () => {
      throw flow;
    };
    const executor = new EventActorExecutor(host);

    await expect(executor.execute(request('control-flow'))).rejects.toBe(flow);
    expect(host.discards).toHaveLength(0);
    expect(host.commits).toHaveLength(0);
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

  it('returns indeterminate evidence from the public commit seam', async () => {
    const host = new TestHost();
    host.commitError = new Error('response lost after public commit');
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-uncertain',
      depth: 1,
      event: { text: 'public-uncertain' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const settlement = await executor.invoke(preparation.invocation);
    if (settlement.status !== 'applied') {
      throw new Error('Expected applied settlement');
    }

    await expect(executor.commit(settlement)).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'public-uncertain',
      checkpoint: { checkpointId: 'result-public-uncertain' },
      error: { message: 'response lost after public commit' },
    });
    await expect(executor.commit(settlement)).rejects.toThrow(
      'settlement was not issued by this executor'
    );
    expect(host.commits).toHaveLength(1);
    expect(host.discards).toHaveLength(0);
  });

  it('converts hostile post-action errors without throwing', async () => {
    const hostileError = {
      [Symbol.toPrimitive]: () => {
        throw new Error('string conversion failed');
      },
    };
    const host = new TestHost();
    host.commit = async () => {
      throw hostileError;
    };
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('hostile-commit-error'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'hostile-commit-error',
      error: { message: 'Unknown event actor error' },
    });
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

  it('rejects forged host-driven invocation depth', async () => {
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
    expect(() => {
      preparation.invocation.depth = 2;
    }).toThrow();
    const forgedInvocation = {
      ...preparation.invocation,
      depth: 2,
    };

    await expect(executor.invoke(forgedInvocation)).rejects.toThrow(
      'prepared invocation binding is invalid'
    );
    expect(host.invokes).toBe(0);
  });

  it('rejects a publicly recomputable digest as preparation authority', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'plain-digest-forgery',
      depth: 1,
      event: { text: 'plain-digest-forgery' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const forgedEvent = { text: 'forged-event' };
    const forgedDigest = createHash('sha256')
      .update(
        JSON.stringify({
          kind: 'invocation',
          actorThreadId: preparation.invocation.actorThreadId,
          invocationId: preparation.invocation.invocationId,
          depth: preparation.invocation.depth,
          continuation: preparation.invocation.continuation,
          base: {
            actorThreadId: preparation.invocation.base.actorThreadId,
            generation: preparation.invocation.base.generation,
            checkpoint: null,
          },
          fork: {
            invocationId: preparation.invocation.fork.invocationId,
            threadId: preparation.invocation.fork.threadId,
            checkpointId: null,
            checkpointNs: preparation.invocation.fork.checkpointNs,
          },
          event: forgedEvent,
        })
      )
      .digest('hex');

    await expect(
      executor.invoke({
        ...preparation.invocation,
        event: forgedEvent,
        preparationDigest: forgedDigest,
      })
    ).rejects.toThrow('prepared invocation binding is invalid');
    expect(host.invokes).toBe(0);
  });

  it('restores canonical prepared authority with the same private key', async () => {
    const host = new TestHost();
    host.generation = 1;
    const signingKey = 'stable-preparation-key'.repeat(2);
    const firstExecutor = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });
    const preparation = await firstExecutor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'restored-warm-authority',
      depth: 1,
      event: { text: 'restored-warm-authority' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const baseCheckpoint = preparation.invocation.base.checkpoint;
    if (baseCheckpoint == null) {
      throw new Error('Expected a committed base checkpoint');
    }
    const restored = {
      preparationDigest: preparation.invocation.preparationDigest,
      event: { text: preparation.invocation.event.text },
      fork: {
        checkpointNs: preparation.invocation.fork.checkpointNs,
        checkpointId: preparation.invocation.fork.checkpointId,
        threadId: preparation.invocation.fork.threadId,
        invocationId: preparation.invocation.fork.invocationId,
      },
      base: {
        checkpoint: {
          checkpointNs: baseCheckpoint.checkpointNs,
          checkpointId: baseCheckpoint.checkpointId,
          threadId: baseCheckpoint.threadId,
        },
        generation: preparation.invocation.base.generation,
        actorThreadId: preparation.invocation.base.actorThreadId,
      },
      continuation: preparation.invocation.continuation,
      depth: preparation.invocation.depth,
      invocationId: preparation.invocation.invocationId,
      actorThreadId: preparation.invocation.actorThreadId,
    };
    const extension = Symbol('checkpoint-extension');
    Object.assign(restored.fork, { untrustedExtension: 'fork-extension' });
    Object.assign(restored.base.checkpoint, {
      untrustedExtension: 'checkpoint-extension',
    });
    Object.defineProperty(restored.fork, extension, {
      enumerable: true,
      value: 'symbol-extension',
    });
    host.invokeImpl = async (prepared) => {
      expect(prepared.fork).not.toHaveProperty('untrustedExtension');
      expect(prepared.base.checkpoint).not.toHaveProperty('untrustedExtension');
      expect(Object.getOwnPropertySymbols(prepared.fork)).toHaveLength(0);
      return {
        status: 'applied',
        result: prepared.event.text,
        checkpoint: {
          ...prepared.fork,
          checkpointId: 'restored-authority-terminal',
        },
      };
    };
    const restoredExecutor = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });

    await expect(restoredExecutor.invoke(restored)).resolves.toMatchObject({
      status: 'applied',
      result: 'restored-warm-authority',
    });
  });

  it('rejects restored prepared authority signed by another executor', async () => {
    const host = new TestHost();
    const issuer = new EventActorExecutor(host, {
      preparationSigningKey: 'issuer-key'.repeat(4),
    });
    const preparation = await issuer.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'foreign-authority',
      depth: 1,
      event: { text: 'foreign-authority' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const verifier = new EventActorExecutor(host, {
      preparationSigningKey: 'different-key'.repeat(4),
    });

    await expect(verifier.invoke(preparation.invocation)).rejects.toThrow(
      'prepared invocation binding is invalid'
    );
    expect(host.invokes).toBe(0);
  });

  it('authenticates and executes the same immutable invocation snapshot', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'single-read-invocation',
      depth: 1,
      event: { text: 'single-read-invocation' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    let reads = 0;
    const event = Object.defineProperty({}, 'text', {
      enumerable: true,
      get: () => {
        reads += 1;
        return reads === 1 ? 'single-read-invocation' : 'forged-event';
      },
    }) as TestEvent;

    await expect(
      executor.invoke({ ...preparation.invocation, event })
    ).resolves.toMatchObject({
      status: 'applied',
      result: 'single-read-invocation',
    });
    expect(reads).toBe(1);
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
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const unavailablePreparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'ambient-cold-prepare',
      depth: 1,
      event: { text: 'ambient-cold-prepare' },
    });
    if (unavailablePreparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue({ configurable: { event_actor_depth: 1 } });
    const warmPreparation = executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'ambient-warm-prepare',
      depth: 1,
      event: { text: 'ambient-warm-prepare' },
    });
    const coldPreparation = executor.coldContinue(unavailablePreparation);
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

  it('snapshots prepared ancestry before adapter work can yield', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const prepare = host.prepare.bind(host);
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.prepare = async (adapterRequest) => {
      await gate;
      return prepare(adapterRequest);
    };
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const mutableRequest: EventActorPrepareRequest<TestEvent> = {
      actorThreadId: 'actor-thread',
      invocationId: 'mutable-prepared-ancestry',
      depth: 2,
      event: { text: 'mutable-prepared-ancestry' },
    };
    const runnableConfigSpy = jest
      .spyOn(AsyncLocalStorageProviderSingleton, 'getRunnableConfig')
      .mockReturnValue({ configurable: { event_actor_depth: 1 } });

    const preparationPromise = executor.prepare(mutableRequest);
    runnableConfigSpy.mockRestore();
    mutableRequest.actorThreadId = 'mutated-actor';
    mutableRequest.invocationId = 'mutated-invocation';
    mutableRequest.depth = 1;
    mutableRequest.event.text = 'mutated-event';
    release();

    const preparation = await preparationPromise;
    expect(preparation).toMatchObject({
      status: 'checkpoint_unavailable',
      request: {
        actorThreadId: 'actor-thread',
        invocationId: 'mutable-prepared-ancestry',
        depth: 2,
        event: { text: 'mutable-prepared-ancestry' },
      },
      head: { actorThreadId: 'actor-thread', generation: 0 },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    expect(() => {
      preparation.request.event.text = 'mutated-after-prepare';
    }).toThrow();
    await expect(executor.coldContinue(preparation)).resolves.toMatchObject({
      event: { text: 'mutable-prepared-ancestry' },
    });
  });

  it('rejects sparse arrays at the immutable JSON event boundary', async () => {
    const adapter: EventActorHostAdapter<EventActorEvent, string> = {
      prepare: async () => {
        throw new Error('prepare must not run');
      },
      coldContinue: async () => {
        throw new Error('coldContinue must not run');
      },
      invoke: async () => {
        throw new Error('invoke must not run');
      },
      commit: async () => {
        throw new Error('commit must not run');
      },
      discard: async () => undefined,
    };
    const prepareSpy = jest.spyOn(adapter, 'prepare');
    const executor = new EventActorExecutor(adapter);

    await expect(
      executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'sparse-event',
        depth: 1,
        event: new Array<EventActorEvent>(1),
      })
    ).rejects.toThrow('event arrays must not contain holes');
    expect(prepareSpy).not.toHaveBeenCalled();
  });

  it('normalizes signed zero at the immutable JSON boundary', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: Object.is(prepared.event.value, -0) ? 'negative-zero' : 'zero',
      checkpoint: {
        ...prepared.fork,
        checkpointId: 'signed-zero-terminal',
      },
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'signed-zero',
      depth: 1,
      event: { text: 'signed-zero', value: -0 },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const restored = {
      ...preparation.invocation,
      event: { text: 'signed-zero', value: 0 },
    };

    expect(Object.is(preparation.invocation.event.value, -0)).toBe(false);
    await expect(executor.invoke(restored)).resolves.toMatchObject({
      status: 'applied',
      result: 'zero',
    });
  });

  it('normalizes signed zero in authenticated actor heads', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: Object.is(prepared.base.generation, -0)
        ? 'negative-zero-generation'
        : 'zero-generation',
      checkpoint: {
        ...prepared.fork,
        checkpointId: 'signed-zero-head-terminal',
      },
    });
    const signingKey = 's'.repeat(32);
    const issuer = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });
    const preparation = await issuer.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'signed-zero-head',
      depth: 1,
      event: { text: 'signed-zero-head' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const restored = {
      ...preparation.invocation,
      base: { ...preparation.invocation.base, generation: -0 },
    };
    const verifier = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });

    await expect(verifier.invoke(restored)).resolves.toMatchObject({
      status: 'applied',
      result: 'zero-generation',
    });
  });

  it('rejects recombined unavailable request and head evidence', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host);
    const first = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'first-unavailable',
      depth: 1,
      event: { text: 'first-unavailable' },
    });
    host.generation = 1;
    const second = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'second-unavailable',
      depth: 1,
      event: { text: 'second-unavailable' },
    });
    if (
      first.status !== 'checkpoint_unavailable' ||
      second.status !== 'checkpoint_unavailable'
    ) {
      throw new Error('Expected unavailable checkpoints');
    }

    await expect(
      executor.coldContinue({
        status: 'checkpoint_unavailable',
        request: first.request,
        head: second.head,
        preparationDigest: first.preparationDigest,
      })
    ).rejects.toThrow('unavailable preparation binding is invalid');
    expect(host.coldContinues).toBe(0);
  });

  it('restores canonical unavailable authority with the same private key', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const signingKey = 'stable-unavailable-key'.repeat(2);
    const issuer = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });
    const preparation = await issuer.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'restored-unavailable-authority',
      depth: 1,
      event: { text: 'restored-unavailable-authority' },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    const restored = {
      preparationDigest: preparation.preparationDigest,
      head: {
        checkpoint: {
          checkpointNs: preparation.head.checkpoint?.checkpointNs ?? '',
          checkpointId: preparation.head.checkpoint?.checkpointId,
          threadId: preparation.head.checkpoint?.threadId ?? '',
        },
        generation: preparation.head.generation,
        actorThreadId: preparation.head.actorThreadId,
      },
      request: {
        event: { text: preparation.request.event.text },
        depth: preparation.request.depth,
        invocationId: preparation.request.invocationId,
        actorThreadId: preparation.request.actorThreadId,
      },
      status: 'checkpoint_unavailable' as const,
    };
    Object.assign(restored.head.checkpoint, {
      untrustedExtension: 'checkpoint-extension',
    });
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (adapterRequest, actorHead, context) => {
      expect(actorHead).not.toHaveProperty('untrustedExtension');
      expect(actorHead.checkpoint).not.toHaveProperty('untrustedExtension');
      return coldContinue(adapterRequest, actorHead, context);
    };
    const verifier = new EventActorExecutor(host, {
      preparationSigningKey: signingKey,
    });

    await expect(verifier.coldContinue(restored)).resolves.toMatchObject({
      actorThreadId: 'actor-thread',
      invocationId: 'restored-unavailable-authority',
      continuation: 'cold',
    });
  });

  it('authenticates and cold-continues the same immutable handoff snapshot', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'single-read-cold-handoff',
      depth: 1,
      event: { text: 'single-read-cold-handoff' },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    let reads = 0;
    const event = Object.defineProperty({}, 'text', {
      enumerable: true,
      get: () => {
        reads += 1;
        return reads === 1 ? 'single-read-cold-handoff' : 'forged-event';
      },
    }) as TestEvent;

    await expect(
      executor.coldContinue({
        ...preparation,
        request: { ...preparation.request, event },
      })
    ).resolves.toMatchObject({ event: { text: 'single-read-cold-handoff' } });
    expect(reads).toBe(1);
  });

  it('snapshots the cold-continuation handoff before adapter work can yield', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host, { maxDepth: 2 });
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'mutable-cold-handoff',
      depth: 2,
      event: { text: 'mutable-cold-handoff' },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
    const coldContinue = host.coldContinue.bind(host);
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.coldContinue = async (adapterRequest, actorHead, context) => {
      await gate;
      return coldContinue(adapterRequest, actorHead, context);
    };

    const mutablePreparation = {
      status: 'checkpoint_unavailable' as const,
      request: {
        ...preparation.request,
        event: { ...preparation.request.event },
      },
      head: {
        ...preparation.head,
        ...(preparation.head.checkpoint == null
          ? {}
          : { checkpoint: { ...preparation.head.checkpoint } }),
      },
      preparationDigest: preparation.preparationDigest,
    };
    const continuationPromise = executor.coldContinue(mutablePreparation);
    mutablePreparation.request.actorThreadId = 'mutated-actor';
    mutablePreparation.request.invocationId = 'mutated-invocation';
    mutablePreparation.request.depth = 1;
    mutablePreparation.head.actorThreadId = 'mutated-actor';
    release();

    await expect(continuationPromise).resolves.toMatchObject({
      actorThreadId: 'actor-thread',
      invocationId: 'mutable-cold-handoff',
      depth: 2,
      continuation: 'cold',
      base: { actorThreadId: 'actor-thread', generation: 0 },
    });
  });

  it('snapshots the execution identity and signal before preparation yields', async () => {
    const host = new TestHost();
    const prepare = host.prepare.bind(host);
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.prepare = async (adapterRequest) => {
      await gate;
      return prepare(adapterRequest);
    };
    const executor = new EventActorExecutor(host);
    const originalController = new AbortController();
    const replacementController = new AbortController();
    const mutableRequest = request('mutable-execution', {
      signal: originalController.signal,
    });

    const execution = executor.execute(mutableRequest);
    mutableRequest.actorThreadId = 'mutated-actor';
    mutableRequest.invocationId = 'mutated-invocation';
    mutableRequest.signal = replacementController.signal;
    originalController.abort(new Error('cancel original execution'));
    release();

    await expect(execution).resolves.toMatchObject({
      status: 'cancelled',
      continuation: 'warm',
    });
    expect(host.invokes).toBe(0);
    expect(host.discards).toEqual([
      expect.objectContaining({
        invocation: expect.objectContaining({
          actorThreadId: 'actor-thread',
          invocationId: 'mutable-execution',
        }),
        reason: 'cancelled',
      }),
    ]);
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

  it('cancels in-flight initial preparation before cold continuation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const prepare = host.prepare.bind(host);
    let release = (): void => undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.prepare = async (prepareRequest, context) => {
      await gate;
      return prepare(prepareRequest, context);
    };
    const controller = new AbortController();
    const execution = new EventActorExecutor(host).execute(
      request('cancel-before-cold', { signal: controller.signal })
    );
    controller.abort(new Error('cancelled'));
    release();

    await expect(execution).resolves.toEqual({
      status: 'cancelled',
      continuation: 'warm',
    });
    expect(host.prepareSignals).toHaveLength(1);
    expect(host.prepareSignals[0].aborted).toBe(true);
    expect(host.coldContinues).toBe(0);
    expect(host.invokes).toBe(0);
    expect(host.discards).toHaveLength(0);
  });

  it('propagates task cancellation into in-flight cold continuation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    let started = (): void => undefined;
    const coldStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    host.coldContinue = async (_prepareRequest, _actorHead, context) => {
      host.coldContinueSignals.push(context.signal);
      started();
      return await new Promise<EventActorInvocation<TestEvent>>(
        (_resolve, reject) => {
          context.signal.addEventListener(
            'abort',
            () => reject(context.signal.reason),
            { once: true }
          );
        }
      );
    };
    const controller = new AbortController();
    const executor = new EventActorExecutor(host);
    const execution = executor.execute(
      request('cancel-during-cold', { signal: controller.signal })
    );
    await coldStarted;

    controller.abort(new Error('cancel cold reconstruction'));

    await expect(execution).resolves.toEqual({
      status: 'cancelled',
      continuation: 'cold',
    });
    expect(host.coldContinueSignals).toHaveLength(1);
    expect(host.coldContinueSignals[0].aborted).toBe(true);
    expect(host.invokes).toBe(0);
    expect(host.discards).toHaveLength(0);
  });

  it('surfaces cleanup failure after cold continuation cancellation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    let started = (): void => undefined;
    const coldStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      started();
      await new Promise<void>((resolve) => {
        context.signal.addEventListener('abort', () => resolve(), {
          once: true,
        });
      });
      return coldContinue(prepareRequest, actorHead, context);
    };
    host.discard = async () => {
      throw new Error('cold cleanup failed');
    };
    const controller = new AbortController();
    const executor = new EventActorExecutor(host);
    const execution = executor.execute(
      request('cleanup-failure', { signal: controller.signal })
    );
    await coldStarted;

    controller.abort(new Error('cancel cold reconstruction'));

    await expect(execution).rejects.toThrow('cold cleanup failed');
    expect(host.invokes).toBe(0);
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

  it('retains a cold-applied result that reports the committed base checkpoint', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
      prepared.fork.checkpointId = 'reconstructed-start';
      return prepared;
    };
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: 'action-claimed',
      checkpoint: {
        ...prepared.fork,
        checkpointId: prepared.base.checkpoint?.checkpointId,
      },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.execute(request('cold-base-terminal'))
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      result: 'action-claimed',
      error: {
        message: 'Event actor result escaped its invocation checkpoint fork',
      },
    });
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

  it('rejects an event recombined with another prepared invocation', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'bound-event',
      depth: 1,
      event: { text: 'bound-event' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }
    const forgedInvocation = {
      ...preparation.invocation,
      event: { text: 'replacement-event' },
    };

    await expect(executor.invoke(forgedInvocation)).rejects.toThrow(
      'prepared invocation binding is invalid'
    );
    expect(host.invokes).toBe(0);
  });

  it('invokes with the authoritative request event after cold continuation', async () => {
    const host = new TestHost();
    host.checkpointAvailable = false;
    const coldContinue = host.coldContinue.bind(host);
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
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
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
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
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
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
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      prepareRequest.checkpointNs = 'shared-namespace';
      return coldContinue(prepareRequest, actorHead, context);
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

    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'commit_indeterminate',
      error: { message: 'Event actor head is invalid' },
    });
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

    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'commit_indeterminate',
      error: { message: expect.stringContaining('did not advance past') },
    });
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

    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'commit_indeterminate',
      error: { message: expect.stringContaining('changed its checkpoint') },
    });
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
    host.coldContinue = async (prepareRequest, actorHead, context) => {
      const prepared = await coldContinue(prepareRequest, actorHead, context);
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
    host.prepare = async () => ({
      status: 'checkpoint_unavailable' as const,
      head: { actorThreadId: 'another-actor', generation: 0 },
    });
    const executor = new EventActorExecutor(host);

    await expect(
      executor.prepare({
        actorThreadId: 'actor-thread',
        invocationId: 'foreign-head',
        depth: 1,
        event: { text: 'foreign-head' },
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

    await expect(executor.discard(malformed, 'failed')).rejects.toThrow(
      'prepared invocation binding is invalid'
    );
    const forgedSettlement = {
      ...terminal,
      invocation: malformed,
    };
    await expect(executor.commit(forgedSettlement)).rejects.toThrow(
      'settlement was not issued by this executor'
    );
    expect(host.discards).toHaveLength(0);
    expect(host.commits).toHaveLength(0);
  });

  it('requires executor-issued authority before public discard', async () => {
    const host = new TestHost();
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'authorized-discard',
      depth: 1,
      event: { text: 'authorized-discard' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(
      executor.discard(preparation.invocation, 'applied' as unknown as 'failed')
    ).rejects.toThrow('discard reason is invalid');
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).resolves.toBeUndefined();
    expect(host.discards).toHaveLength(1);
    await expect(
      executor.discard(
        {
          ...preparation.invocation,
          preparationDigest: '0'.repeat(64),
        },
        'failed'
      )
    ).rejects.toThrow('prepared invocation binding is invalid');
    expect(host.discards).toHaveLength(1);
  });

  it('reclaims a public no-action fork before returning and prevents reinvocation', async () => {
    const host = new TestHost();
    host.invokeImpl = async () => ({
      status: 'completed_no_action',
      result: 'nothing-applied',
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-no-action-phase',
      depth: 1,
      event: { text: 'public-no-action-phase' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(executor.invoke(preparation.invocation)).resolves.toEqual({
      status: 'completed_no_action',
      result: 'nothing-applied',
    });
    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'already consumed'
    );
    await expect(
      executor.discard(preparation.invocation, 'completed_no_action')
    ).rejects.toThrow('no longer discardable');
    expect(host.invokes).toBe(1);
    expect(host.discards).toHaveLength(1);
  });

  it('reclaims a definitely failed public invocation before rethrowing', async () => {
    const host = new TestHost();
    host.invokeImpl = async () => {
      throw new Error('provider failed before action');
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-failure-cleanup',
      depth: 1,
      event: { text: 'public-failure-cleanup' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'provider failed before action'
    );
    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'already consumed'
    );
    expect(host.discards).toEqual([
      expect.objectContaining({ reason: 'failed' }),
    ]);
  });

  it('preserves cleanup-only authority after public failure cleanup rejects', async () => {
    const host = new TestHost();
    host.invokeImpl = async () => {
      throw new Error('provider failed before action');
    };
    host.discard = async (discardRequest) => {
      host.discards.push(discardRequest);
      throw new Error('cleanup failed');
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-failure-cleanup-retry',
      depth: 1,
      event: { text: 'public-failure-cleanup-retry' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'cleanup failed'
    );
    await expect(executor.invoke(preparation.invocation)).rejects.toThrow(
      'already consumed'
    );
    host.discard = async (discardRequest) => {
      host.discards.push(discardRequest);
    };
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).resolves.toBeUndefined();
    expect(host.discards).toHaveLength(2);
  });

  it('revokes public discard authority while invocation is active and after action', async () => {
    const host = new TestHost();
    let started = (): void => undefined;
    let release = (): void => undefined;
    const invocationStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    host.invokeImpl = async (prepared) => {
      started();
      await gate;
      return {
        status: 'applied',
        result: 'action-applied',
        checkpoint: {
          ...prepared.fork,
          checkpointId: 'phase-bound-terminal',
        },
      };
    };
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'phase-bound-discard',
      depth: 1,
      event: { text: 'phase-bound-discard' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    const invocationResult = executor.invoke(preparation.invocation);
    await invocationStarted;
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).rejects.toThrow('no longer discardable');
    release();
    await expect(invocationResult).resolves.toMatchObject({
      status: 'applied',
    });
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).rejects.toThrow('no longer discardable');
    expect(host.discards).toHaveLength(0);
  });

  it('revokes public discard authority for indeterminate applied evidence', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: 'action-applied',
      checkpoint: {
        ...prepared.fork,
        checkpointNs: 'foreign-namespace',
        checkpointId: 'indeterminate-terminal',
      },
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'indeterminate-discard',
      depth: 1,
      event: { text: 'indeterminate-discard' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(
      executor.invoke(preparation.invocation)
    ).resolves.toMatchObject({ status: 'commit_indeterminate' });
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).rejects.toThrow('no longer discardable');
    expect(host.discards).toHaveLength(0);
  });

  it('returns indeterminate evidence when a public status read throws', async () => {
    const host = new TestHost();
    host.invokeImpl = async () =>
      Object.defineProperty({}, 'status', {
        get: () => {
          throw new Error('terminal status unavailable');
        },
      }) as EventActorTerminalResult<string>;
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'throwing-public-status',
      depth: 1,
      event: { text: 'throwing-public-status' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    await expect(
      executor.invoke(preparation.invocation)
    ).resolves.toMatchObject({
      status: 'commit_indeterminate',
      error: { message: 'terminal status unavailable' },
    });
    await expect(
      executor.discard(preparation.invocation, 'failed')
    ).rejects.toThrow('no longer discardable');
    expect(host.discards).toHaveLength(0);
  });

  it('detects a cold adapter mutating its copy of the prepared head', async () => {
    const host = new TestHost();
    host.generation = 1;
    host.checkpointAvailable = false;
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'mutated-head',
      depth: 1,
      event: { text: 'mutated-head' },
    });
    if (preparation.status !== 'checkpoint_unavailable') {
      throw new Error('Expected unavailable checkpoint');
    }
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

    await expect(executor.coldContinue(preparation)).rejects.toThrow(
      'Cold continuation did not use the prepared actor head'
    );
    expect(preparation.head.generation).toBe(1);
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
    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'committed',
    });
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
    await expect(executor.commit(terminal)).resolves.toMatchObject({
      status: 'committed',
    });
  });

  it('returns indeterminate evidence for an invalid public terminal checkpoint', async () => {
    const host = new TestHost();
    host.invokeImpl = async (prepared) => ({
      status: 'applied',
      result: 'public-invalid-checkpoint',
      checkpoint: {
        ...prepared.fork,
        checkpointNs: 'foreign-checkpoint-namespace',
        checkpointId: 'public-invalid-terminal',
      },
    });
    const executor = new EventActorExecutor(host);
    const preparation = await executor.prepare({
      actorThreadId: 'actor-thread',
      invocationId: 'public-invalid-checkpoint',
      depth: 1,
      event: { text: 'public-invalid-checkpoint' },
    });
    if (preparation.status !== 'ready') {
      throw new Error('Expected warm preparation');
    }

    const terminal = await executor.invoke(preparation.invocation);

    expect(terminal).toEqual(
      expect.objectContaining({
        status: 'commit_indeterminate',
        result: 'public-invalid-checkpoint',
        error: expect.objectContaining({
          message: expect.stringContaining('escaped its invocation'),
        }),
      })
    );
    expect(host.commits).toHaveLength(0);
    expect(host.discards).toHaveLength(0);
  });
});
