import { describe, it, expect } from '@jest/globals';
import type * as t from '@/types';
import {
  buildToolExecutionRequestPlan,
  coerceArgsForSchema,
  resolveRuntimeSessionHint,
} from '../eagerEventExecution';

describe('coerceArgsForSchema', () => {
  const issueSchema: t.JsonSchemaType = {
    type: 'object',
    properties: {
      issue_number: { type: 'integer' },
      dry_run: { type: 'boolean' },
      retries: { type: 'array', items: { type: 'integer' } },
      settings: {
        type: 'object',
        properties: { delay_seconds: { type: 'number' } },
      },
    },
  };

  it('repairs canonical scalar strings only where the schema permits it', () => {
    expect(
      coerceArgsForSchema(
        {
          issue_number: '2365',
          dry_run: 'false',
          retries: ['1', '2'],
          settings: { delay_seconds: '1.25' },
          label: '001',
        },
        issueSchema
      )
    ).toEqual({
      issue_number: 2365,
      dry_run: false,
      retries: [1, 2],
      settings: { delay_seconds: 1.25 },
      label: '001',
    });
  });

  it('does not coerce lossy, ambiguous, or unsafe numeric strings', () => {
    expect(
      coerceArgsForSchema(
        {
          issue_number: '02365',
          dry_run: 'False',
          retries: ['9007199254740992'],
        },
        issueSchema
      )
    ).toEqual({
      issue_number: '02365',
      dry_run: 'False',
      retries: ['9007199254740992'],
    });
  });
});

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

  it('normalizes arguments before event dispatch using the matching tool schema', () => {
    const plan = buildToolExecutionRequestPlan({
      toolCalls: [
        {
          id: 'c1',
          name: 'issue_write',
          args: { issue_number: '2365' },
        },
      ],
      usageCount: usageCount(),
      getToolSchema: () => ({
        type: 'object',
        properties: { issue_number: { type: 'integer' } },
      }),
    });
    expect(plan?.requests[0].args).toEqual({ issue_number: 2365 });
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
