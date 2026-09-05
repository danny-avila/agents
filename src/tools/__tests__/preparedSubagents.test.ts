import type { ToolCall } from '@langchain/core/messages/tool';
import { PreparedSubagents, PreparedSubagentError } from '../preparedSubagents';

const call: ToolCall = {
  id: 'call-1',
  name: 'subagent',
  args: { description: 'research' },
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe('Prepared subagent invocation ownership', () => {
  it('captures configuration separately for parallel model attempts', () => {
    const owner = new PreparedSubagents();
    const firstSignal = new AbortController().signal;
    const config = {
      configurable: { run_id: 'first' },
      metadata: { step: 1 },
      signal: firstSignal,
    };
    owner.begin('first', config);
    config.configurable.run_id = 'second';
    config.metadata.step = 2;
    owner.begin('second', config);
    expect(owner.getConfig('first')).toEqual({
      configurable: { run_id: 'first' },
      metadata: { step: 1 },
      signal: firstSignal,
    });
    expect(owner.getConfig('second')?.configurable?.run_id).toBe('second');
    owner.finish('first', []);
    expect(owner.getConfig('first')).toBeUndefined();
    owner.clear();
    expect(owner.getConfig('second')).toBeUndefined();
  });

  it('starts before attempt completion and adopts the same result once', async () => {
    const owner = new PreparedSubagents();
    const result = deferred<string>();
    let starts = 0;
    owner.begin('attempt');
    expect(
      owner.start('attempt', 'agent', call, 4, () => {
        starts++;
        return result.promise;
      })
    ).toBe(true);
    await Promise.resolve();
    expect(starts).toBe(1);
    expect(
      owner.start('attempt', 'agent', call, 4, () => {
        throw new Error('duplicate');
      })
    ).toBe(false);
    owner.finish('attempt', [call]);
    const adopted = owner.take('agent', call);
    result.resolve('child result');
    await expect(adopted).resolves.toBe('child result');
    await expect(owner.take('agent', call)).rejects.toThrow(
      PreparedSubagentError
    );
    expect(starts).toBe(1);
  });

  it('bounds pending and settled-but-unadopted results', async () => {
    const owner = new PreparedSubagents();
    owner.begin('attempt');
    expect(owner.start('attempt', 'agent', call, 1, async () => 'done')).toBe(
      true
    );
    await Promise.resolve();
    expect(
      owner.start(
        'attempt',
        'agent',
        { ...call, id: 'second' },
        1,
        async () => 'overflow'
      )
    ).toBe(false);
    owner.finish('attempt', [call]);
    await expect(owner.take('agent', call)).resolves.toBe('done');
  });

  it.each([undefined, [], [{ ...call, args: { description: 'different' } }]])(
    'aborts and fails closed on a failed, discarded or revised attempt (%p)',
    async (finalCalls) => {
      const owner = new PreparedSubagents();
      let signal!: AbortSignal;
      owner.begin('attempt');
      owner.start('attempt', 'agent', call, 1, async (value) => {
        signal = value;
        return 'done';
      });
      await Promise.resolve();
      expect(() => owner.finish('attempt', finalCalls)).toThrow(
        PreparedSubagentError
      );
      expect(signal.aborted).toBe(true);
      expect(owner.isOpen('attempt')).toBe(false);
      expect(owner.start('attempt', 'agent', call, 1, async () => 'late')).toBe(
        false
      );
      expect(owner.owns('agent', call)).toBe(false);
    }
  );

  it('cancels adopted work during graph cleanup', async () => {
    const owner = new PreparedSubagents();
    let signal!: AbortSignal;
    const result = deferred<string>();
    owner.begin('attempt');
    owner.start('attempt', 'agent', call, 1, (value) => {
      signal = value;
      return result.promise;
    });
    await Promise.resolve();
    owner.finish('attempt', [call]);
    const adopted = owner.take('agent', call);
    owner.clear();
    expect(signal.aborted).toBe(true);
    result.resolve('late result');
    await expect(adopted).rejects.toThrow(PreparedSubagentError);
  });

  it('contains rejected child promises until adoption', async () => {
    const owner = new PreparedSubagents();
    owner.begin('attempt');
    owner.start('attempt', 'agent', call, 1, async () => {
      throw new Error('child failed');
    });
    await new Promise<void>((resolve) => setImmediate(resolve));
    owner.finish('attempt', [call]);
    await expect(owner.take('agent', call)).rejects.toThrow('child failed');
  });

  it('rejects changed final invocation arguments without rerunning', async () => {
    const owner = new PreparedSubagents();
    owner.begin('attempt');
    owner.start('attempt', 'agent', call, 1, async () => 'done');
    owner.finish('attempt', [call]);
    await expect(owner.take('agent', { ...call, args: {} })).rejects.toThrow(
      PreparedSubagentError
    );
  });
  it('fences a reset while a cancellation-ignoring model attempt drains', async () => {
    const owner = new PreparedSubagents();
    const result = deferred<string>();
    owner.begin('old');
    owner.start('old', 'agent', call, 1, () => result.promise);
    await Promise.resolve();
    owner.clear();
    expect(owner.isOpen('old')).toBe(false);
    owner.begin('new');
    expect(owner.start('new', 'agent', call, 1, async () => 'new')).toBe(false);
    expect(() => owner.finish('old', [call])).toThrow(PreparedSubagentError);
    result.resolve('late result');
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(owner.start('new', 'agent', call, 1, async () => 'new')).toBe(true);
    owner.finish('new', [call]);
    await expect(owner.take('agent', call)).resolves.toBe('new');
  });

  it('keeps separate agents isolated even with identical provider call IDs', async () => {
    const owner = new PreparedSubagents();
    owner.begin('left');
    owner.begin('right');
    owner.start('left', 'left-agent', call, 4, async () => 'left');
    owner.start('right', 'right-agent', call, 4, async () => 'right');
    const left = { ...call };
    const right = { ...call };
    owner.finish('left', [left]);
    owner.finish('right', [right]);
    await expect(owner.take('left-agent', left)).resolves.toBe('left');
    await expect(owner.take('right-agent', right)).resolves.toBe('right');
  });
  it('never lets an old attempt cancel a new reservation with the same call ID', async () => {
    const owner = new PreparedSubagents();
    owner.begin('old');
    owner.start('old', 'agent', call, 4, async () => 'old');
    await new Promise<void>((resolve) => setImmediate(resolve));
    owner.clear();
    owner.begin('new');
    owner.start('new', 'agent', call, 4, async () => 'new');
    expect(() => owner.finish('old', [call])).toThrow(PreparedSubagentError);
    owner.finish('new', [call]);
    await expect(owner.take('agent', call)).resolves.toBe('new');
  });
  it('does not let a stale finalized call consume a new invocation', async () => {
    const owner = new PreparedSubagents();
    const oldCall = { ...call };
    const newCall = { ...call };
    owner.begin('old');
    owner.start('old', 'agent', oldCall, 4, async () => 'old');
    owner.finish('old', [oldCall]);
    owner.clear();
    owner.begin('new');
    owner.start('new', 'agent', newCall, 4, async () => 'new');
    owner.finish('new', [newCall]);
    await expect(owner.take('agent', oldCall)).rejects.toThrow(
      PreparedSubagentError
    );
    await expect(owner.take('agent', newCall)).resolves.toBe('new');
  });

  it('fences a settled result adopted just before reset', async () => {
    const owner = new PreparedSubagents();
    owner.begin('attempt');
    owner.start('attempt', 'agent', call, 4, async () => 'done');
    owner.finish('attempt', [call]);
    await new Promise<void>((resolve) => setImmediate(resolve));
    const result = owner.take('agent', call);
    owner.clear();
    await expect(result).rejects.toThrow(PreparedSubagentError);
  });
});
