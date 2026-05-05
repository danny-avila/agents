import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect } from '@jest/globals';
import {
  makeIsDeferred,
  partitionAndMarkAnthropicToolCache,
} from '../anthropicToolCache';

function fakeTool(name: string): unknown {
  return tool(async () => 'ok', {
    name,
    description: `tool ${name}`,
    schema: z.object({}),
  });
}

describe('partitionAndMarkAnthropicToolCache', () => {
  it('returns input unchanged when there are no tools', () => {
    expect(
      partitionAndMarkAnthropicToolCache(undefined, () => false)
    ).toBeUndefined();
    const empty = [] as unknown as Parameters<
      typeof partitionAndMarkAnthropicToolCache
    >[0];
    expect(partitionAndMarkAnthropicToolCache(empty, () => false)).toBe(empty);
  });

  it('returns input unchanged when every tool is deferred', () => {
    const tools = [fakeTool('a'), fakeTool('b')] as never;
    const result = partitionAndMarkAnthropicToolCache(tools, () => true);
    expect(result).toBe(tools);
  });

  it('partitions static-first, deferred-last and stamps cache_control on the last static tool', () => {
    const a = fakeTool('a-static');
    const b = fakeTool('b-deferred');
    const c = fakeTool('c-static');
    const d = fakeTool('d-deferred');
    const isDeferred = (n: string): boolean => n.endsWith('-deferred');
    const out = partitionAndMarkAnthropicToolCache(
      [a, b, c, d] as never,
      isDeferred
    ) as Array<{ name: string; extras?: { cache_control?: { type: string } } }>;

    expect(out.map((t) => t.name)).toEqual([
      'a-static',
      'c-static',
      'b-deferred',
      'd-deferred',
    ]);
    expect(out[1].extras?.cache_control).toEqual({ type: 'ephemeral' });
    expect(out[0].extras?.cache_control).toBeUndefined();
    expect(out[2].extras?.cache_control).toBeUndefined();
    expect(out[3].extras?.cache_control).toBeUndefined();
  });

  it('does not mutate the original tool instance', () => {
    const a = fakeTool('a-static') as { extras?: unknown };
    const out = partitionAndMarkAnthropicToolCache(
      [a] as never,
      () => false
    ) as Array<{ extras?: unknown }>;
    expect(out[0]).not.toBe(a);
    expect((a as { extras?: unknown }).extras).toBeUndefined();
    expect(out[0].extras).toBeDefined();
  });

  it('preserves the prototype chain so instanceof checks survive', () => {
    const a = fakeTool('a-static');
    const ctor = (a as object).constructor;
    const out = partitionAndMarkAnthropicToolCache(
      [a] as never,
      () => false
    ) as object[];
    expect(out[0].constructor).toBe(ctor);
  });

  it('keeps existing extras keys intact when stamping', () => {
    const a = fakeTool('a-static') as { extras?: Record<string, unknown> };
    a.extras = { providerToolDefinition: { foo: 'bar' } };
    const out = partitionAndMarkAnthropicToolCache(
      [a] as never,
      () => false
    ) as Array<{ extras?: Record<string, unknown> }>;
    expect(out[0].extras?.providerToolDefinition).toEqual({ foo: 'bar' });
    expect(out[0].extras?.cache_control).toEqual({ type: 'ephemeral' });
  });

  it('is idempotent when re-marking a tool that already has the marker', () => {
    const a = fakeTool('a-static') as { extras?: Record<string, unknown> };
    a.extras = { cache_control: { type: 'ephemeral' } };
    const input = [a] as never;
    // No deferred tools and the only static tool is already marked → input
    // is returned unchanged (same reference) so we don't churn the array.
    expect(partitionAndMarkAnthropicToolCache(input, () => false)).toBe(input);
  });
});

describe('makeIsDeferred', () => {
  it('returns false for everything when no defs are supplied', () => {
    const isDeferred = makeIsDeferred(undefined);
    expect(isDeferred('anything')).toBe(false);
  });

  it('returns false for everything when no def has defer_loading=true', () => {
    const isDeferred = makeIsDeferred([
      { name: 'a' },
      { name: 'b', defer_loading: false },
    ]);
    expect(isDeferred('a')).toBe(false);
    expect(isDeferred('b')).toBe(false);
  });

  it('returns true only for names declared as deferred', () => {
    const isDeferred = makeIsDeferred([
      { name: 'a' },
      { name: 'b', defer_loading: true },
      { name: 'c', defer_loading: false },
    ]);
    expect(isDeferred('a')).toBe(false);
    expect(isDeferred('b')).toBe(true);
    expect(isDeferred('c')).toBe(false);
    expect(isDeferred('unknown')).toBe(false);
  });
});
