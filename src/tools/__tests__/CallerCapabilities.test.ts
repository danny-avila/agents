import { describe, expect, it } from '@jest/globals';
import type * as t from '@/types';
import {
  allowsToolCaller,
  mergeCallerCapabilityDefinitions,
  resolveCallerCapabilityProjection,
} from '../CallerCapabilities';

describe('Caller Capability Projection', () => {
  const toolDefs: t.LCTool[] = [
    { name: 'default_direct' },
    { name: 'direct_only', allowed_callers: ['direct'] },
    { name: 'code_only', allowed_callers: ['code_execution'] },
    {
      name: 'both',
      allowed_callers: ['direct', 'code_execution'],
    },
    {
      name: 'deferred_code',
      allowed_callers: ['code_execution'],
      defer_loading: true,
    },
  ];

  it('classifies every caller combination in one pass', () => {
    const projection = resolveCallerCapabilityProjection(toolDefs);

    expect(projection.directTools.map((toolDef) => toolDef.name)).toEqual([
      'default_direct',
      'direct_only',
      'both',
    ]);
    expect(
      projection.codeExecutionTools.map((toolDef) => toolDef.name)
    ).toEqual(['code_only', 'both', 'deferred_code']);
    expect(projection.directOnlyTools.map((toolDef) => toolDef.name)).toEqual([
      'default_direct',
      'direct_only',
    ]);
    expect(
      projection.codeExecutionOnlyTools.map((toolDef) => toolDef.name)
    ).toEqual(['code_only', 'deferred_code']);
  });

  it('applies caller classification after effective-activity filtering', () => {
    const projection = resolveCallerCapabilityProjection(
      toolDefs,
      (toolDef) => toolDef.defer_loading !== true
    );

    expect(
      projection.codeExecutionTools.map((toolDef) => toolDef.name)
    ).not.toContain('deferred_code');
  });

  it('defaults omitted caller metadata to direct-only', () => {
    expect(allowsToolCaller(toolDefs[0], 'direct')).toBe(true);
    expect(allowsToolCaller(toolDefs[0], 'code_execution')).toBe(false);
  });

  it('merges schema-only and runtime definitions with runtime precedence', () => {
    expect(
      mergeCallerCapabilityDefinitions(
        [{ name: 'shared' }, { name: 'schema_only' }],
        [
          { name: 'shared', allowed_callers: ['code_execution'] },
          { name: 'runtime_only' },
        ]
      )
    ).toEqual([
      { name: 'shared', allowed_callers: ['code_execution'] },
      { name: 'schema_only' },
      { name: 'runtime_only' },
    ]);
  });
});
