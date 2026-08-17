import { afterEach, describe, expect, it, jest } from '@jest/globals';
import type { SubagentTaskRuntime } from '@/types';
import { InMemorySubagentTaskStore } from '@/tools/subagent/InMemorySubagentTaskStore';

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const start = (
  store: InMemorySubagentTaskStore,
  run: (runtime: SubagentTaskRuntime) => Promise<{ content: string }>,
  overrides: Partial<{
    scopeId: string;
    idempotencyKey: string;
    requestFingerprint: string;
    threadId: string;
    subagentType: string;
  }> = {}
) =>
  store.start({
    scopeId: overrides.scopeId ?? 'user:conversation',
    idempotencyKey: overrides.idempotencyKey ?? 'run:agent:call',
    parentRunId: 'run',
    parentAgentId: 'parent-agent',
    parentToolCallId: 'call',
    ...(overrides.requestFingerprint == null
      ? {}
      : { requestFingerprint: overrides.requestFingerprint }),
    ...(overrides.threadId == null
      ? {}
      : { threadId: overrides.threadId }),
    input: 'Research the question.',
    subagentKind: 'agent',
    subagentType: overrides.subagentType ?? 'researcher',
    run,
  });

describe('InMemorySubagentTaskStore', () => {
  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns a handle immediately, coalesces replays, and claims once', async () => {
    const store = new InMemorySubagentTaskStore();
    let finish = (_result: { content: string }): void => undefined;
    const result = new Promise<{ content: string }>((resolve) => {
      finish = resolve;
    });
    const run = jest.fn(() => result);

    const first = start(store, run);
    const duplicate = start(store, run);

    expect(first).toMatchObject({ accepted: true, isNew: true });
    expect(duplicate).toMatchObject({ accepted: true, isNew: false });
    if (!first.accepted || !duplicate.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }
    expect(duplicate.task.taskId).toBe(first.task.taskId);
    expect(first.task.threadId).toBe(first.task.taskId);
    expect(run).not.toHaveBeenCalled();

    await settle();
    expect(run).toHaveBeenCalledTimes(1);
    finish({ content: 'finished research' });
    await settle();

    expect(store.claim('user:conversation', first.task.taskId)).toMatchObject({
      status: 'completed',
      result: 'finished research',
    });
    expect(store.claim('user:conversation', first.task.taskId)).toMatchObject({
      status: 'claimed',
    });
  });

  it('carries a host-owned thread identity without retaining its transcript', async () => {
    const store = new InMemorySubagentTaskStore();
    const started = start(store, async () => ({ content: 'continued' }), {
      threadId: 'child-thread',
    });
    if (!started.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }

    expect(started.task.threadId).toBe('child-thread');
    await settle();
    expect(store.get('user:conversation', started.task.taskId)).toMatchObject({
      threadId: 'child-thread',
      status: 'completed',
    });
  });

  it('rejects a conflicting replay of the same operation', async () => {
    const store = new InMemorySubagentTaskStore();
    const run = async (): Promise<{ content: string }> => ({ content: 'ok' });
    const first = start(store, run, { requestFingerprint: 'fingerprint-a' });
    const conflict = start(store, run, {
      requestFingerprint: 'fingerprint-b',
    });

    expect(first).toMatchObject({ accepted: true, isNew: true });
    expect(conflict).toMatchObject({ accepted: false, reason: 'conflict' });
    await settle();
  });

  it('does not launch work cancelled before its dispatch microtask', async () => {
    const store = new InMemorySubagentTaskStore();
    const run = jest.fn(async () => ({ content: 'should not run' }));
    const started = start(store, run);
    if (!started.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }

    expect(
      store.control('user:conversation', started.task.taskId, {
        action: 'cancel',
      })
    ).toMatchObject({ status: 'cancelled' });
    await settle();

    expect(run).not.toHaveBeenCalled();
    expect(store.get('user:conversation', started.task.taskId)).toMatchObject({
      status: 'cancelled',
    });
  });

  it('keeps scopes isolated and enforces running capacity', async () => {
    const store = new InMemorySubagentTaskStore({ maxRunningPerScope: 1 });
    const never = async (
      runtime: SubagentTaskRuntime
    ): Promise<{ content: string }> =>
      new Promise((_resolve, reject) => {
        runtime.signal.addEventListener(
          'abort',
          () => reject(runtime.signal.reason),
          { once: true }
        );
      });
    const first = start(store, never);
    const saturated = start(store, never, { idempotencyKey: 'another-call' });
    const anotherScope = start(store, never, {
      scopeId: 'other-user:conversation',
      idempotencyKey: 'another-call',
    });

    expect(saturated).toEqual({ accepted: false, reason: 'capacity' });
    expect(anotherScope.accepted).toBe(true);
    if (!first.accepted || !anotherScope.accepted) {
      throw new Error('Expected tasks to be accepted.');
    }
    expect(
      store.get('other-user:conversation', first.task.taskId)
    ).toBeUndefined();
    expect(
      store.control('user:conversation', first.task.taskId, {
        action: 'cancel',
      })
    ).toMatchObject({ status: 'cancelled' });
    expect(
      store.control('other-user:conversation', anotherScope.task.taskId, {
        action: 'cancel',
      })
    ).toMatchObject({ status: 'cancelled' });
    await settle();
  });

  it('enforces total capacity across independent scopes', async () => {
    const store = new InMemorySubagentTaskStore({ maxRunningTotal: 1 });
    const never = async (
      runtime: SubagentTaskRuntime
    ): Promise<{ content: string }> =>
      new Promise((_resolve, reject) => {
        runtime.signal.addEventListener(
          'abort',
          () => reject(runtime.signal.reason),
          { once: true }
        );
      });
    const first = start(store, never);
    const saturated = start(store, never, {
      scopeId: 'other-user:conversation',
      idempotencyKey: 'other-call',
    });

    expect(saturated).toEqual({ accepted: false, reason: 'capacity' });
    if (!first.accepted) {
      throw new Error('Expected first task to be accepted.');
    }
    store.control('user:conversation', first.task.taskId, {
      action: 'cancel',
    });
    await settle();
  });

  it('evicts the oldest terminal task to honor the total retained-task bound', async () => {
    const store = new InMemorySubagentTaskStore({
      maxRunningTotal: 2,
      maxTasksTotal: 1,
    });
    const first = start(store, async () => ({ content: 'first' }));
    if (!first.accepted) {
      throw new Error('Expected first task to be accepted.');
    }
    await settle();
    const second = start(store, async () => ({ content: 'second' }), {
      scopeId: 'other-user:conversation',
      idempotencyKey: 'other-call',
    });

    expect(second).toMatchObject({ accepted: true, isNew: true });
    expect(store.get('user:conversation', first.task.taskId)).toBeUndefined();
    await settle();
  });

  it('drains interrupt, steer, and queued messages at their intended boundaries', async () => {
    const store = new InMemorySubagentTaskStore();
    let runtime: SubagentTaskRuntime | undefined;
    let finish = (_result: { content: string }): void => undefined;
    const result = new Promise<{ content: string }>((resolve) => {
      finish = resolve;
    });
    const started = start(store, async (taskRuntime) => {
      runtime = taskRuntime;
      return result;
    });
    if (!started.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }
    await settle();

    const interrupt = store.control('user:conversation', started.task.taskId, {
      action: 'interrupt',
      message: 'Stop searching and summarize now.',
    });
    const steer = store.control('user:conversation', started.task.taskId, {
      action: 'steer',
      message: 'Also check the primary source.',
    });
    store.control('user:conversation', started.task.taskId, {
      action: 'queue',
      message: 'After this turn, compare both results.',
    });
    expect(interrupt).toMatchObject({ status: 'accepted' });
    expect(steer).toMatchObject({ status: 'accepted' });
    expect(runtime?.shouldPreempt()).toBe(true);
    expect(runtime?.drain('preempt').map((message) => message.content)).toEqual(
      ['Stop searching and summarize now.']
    );
    expect(runtime?.shouldPreempt()).toBe(false);
    expect(runtime?.drain('tool').map((message) => message.content)).toEqual([
      'Also check the primary source.',
    ]);
    expect(runtime?.closeTurn()).toEqual({
      closed: false,
      messages: [
        {
          role: 'user',
          content: 'After this turn, compare both results.',
          source: 'steer',
        },
      ],
    });
    expect(runtime?.closeTurn()).toEqual({ closed: true, messages: [] });

    finish({ content: 'done' });
    await settle();
  });

  it('cancels a pending message before it drains', async () => {
    const store = new InMemorySubagentTaskStore();
    let runtime: SubagentTaskRuntime | undefined;
    const started = start(
      store,
      async (taskRuntime) =>
        new Promise((_resolve, reject) => {
          runtime = taskRuntime;
          taskRuntime.signal.addEventListener(
            'abort',
            () => reject(taskRuntime.signal.reason),
            { once: true }
          );
        })
    );
    if (!started.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }
    await settle();
    const queued = store.control('user:conversation', started.task.taskId, {
      action: 'steer',
      message: 'This should not be delivered.',
    });
    if (queued.status !== 'accepted' || queued.controlId == null) {
      throw new Error('Expected a cancellable control receipt.');
    }
    expect(
      store.control('user:conversation', started.task.taskId, {
        action: 'cancel_message',
        controlId: queued.controlId,
      })
    ).toMatchObject({ status: 'accepted' });
    expect(runtime?.drain('tool')).toEqual([]);
    store.control('user:conversation', started.task.taskId, {
      action: 'cancel',
    });
    await settle();
  });

  it('times out a non-cooperative task and frees its running slot', async () => {
    jest.useFakeTimers();
    const store = new InMemorySubagentTaskStore({
      maxRunningPerScope: 1,
      taskTimeoutMs: 100,
    });
    const started = start(
      store,
      async () => new Promise<{ content: string }>(() => undefined)
    );
    if (!started.accepted) {
      throw new Error('Expected task dispatch to be accepted.');
    }
    await jest.advanceTimersByTimeAsync(101);

    expect(store.get('user:conversation', started.task.taskId)).toMatchObject({
      status: 'error',
      error: 'Detached subagent task timed out.',
    });
    expect(
      start(store, async () => ({ content: 'replacement' }), {
        idempotencyKey: 'replacement-call',
      })
    ).toMatchObject({ accepted: true, isNew: true });
  });
});
