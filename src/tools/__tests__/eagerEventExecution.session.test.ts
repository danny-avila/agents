import { describe, it, expect } from '@jest/globals';
import type * as t from '@/types';
import {
  buildToolExecutionRequestPlan,
  resolveRuntimeSessionHint,
} from '../eagerEventExecution';

describe('buildToolExecutionRequestPlan — runtimeSessionHint', () => {
  const usageCount = () => new Map<string, number>();

  it('carries runtimeSessionHint onto the built ToolCallRequest', () => {
    const plan = buildToolExecutionRequestPlan({
      toolCalls: [
        {
          id: 'call_1',
          name: 'execute_code',
          args: { lang: 'py', code: 'print(1)' },
          runtimeSessionHint: 'conv-42',
        },
      ],
      usageCount: usageCount(),
    });
    expect(plan?.requests[0].runtimeSessionHint).toBe('conv-42');
  });

  it('omits the field entirely when the hint is absent or empty', () => {
    const plan = buildToolExecutionRequestPlan({
      toolCalls: [{ id: 'c1', name: 'execute_code', args: {} }],
      usageCount: usageCount(),
    });
    expect('runtimeSessionHint' in (plan?.requests[0] as object)).toBe(false);

    const empty = buildToolExecutionRequestPlan({
      toolCalls: [
        { id: 'c2', name: 'execute_code', args: {}, runtimeSessionHint: '' },
      ],
      usageCount: usageCount(),
    });
    expect('runtimeSessionHint' in (empty?.requests[0] as object)).toBe(false);
  });

  it('carries the hint onto invalid-arg (rejected) requests too', () => {
    const plan = buildToolExecutionRequestPlan({
      toolCalls: [
        {
          id: 'c1',
          name: 'execute_code',
          args: 'not-an-object',
          runtimeSessionHint: 'conv-9',
        },
      ],
      usageCount: usageCount(),
      invalidArgsBehavior: 'error-result',
    });
    expect(plan?.allRequests[0].runtimeSessionHint).toBe('conv-9');
    expect(plan?.rejectedResults).toHaveLength(1);
  });
});

describe('resolveRuntimeSessionHint', () => {
  const sandbox = (
    o: Partial<t.SandboxExecutionConfig>
  ): t.ToolExecutionConfig => ({
    sandbox: o,
  });

  it('returns undefined unless statefulSessions is on', () => {
    expect(resolveRuntimeSessionHint(undefined, 'thread-1')).toBeUndefined();
    expect(resolveRuntimeSessionHint(sandbox({}), 'thread-1')).toBeUndefined();
    expect(
      resolveRuntimeSessionHint(
        sandbox({ statefulSessions: false }),
        'thread-1'
      )
    ).toBeUndefined();
  });

  it('prefers an explicit hint, else falls back to thread_id', () => {
    expect(
      resolveRuntimeSessionHint(
        sandbox({ statefulSessions: true, runtimeSessionHint: 'explicit' }),
        'thread-1'
      )
    ).toBe('explicit');
    expect(
      resolveRuntimeSessionHint(sandbox({ statefulSessions: true }), 'thread-1')
    ).toBe('thread-1');
    expect(
      resolveRuntimeSessionHint(sandbox({ statefulSessions: true }), '')
    ).toBeUndefined();
  });
});
