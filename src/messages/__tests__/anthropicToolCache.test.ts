import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { describe, it, expect } from '@jest/globals';
import {
  makeIsDeferred,
  partitionAndMarkAnthropicToolCache,
} from '../anthropicToolCache';
import { CustomAnthropic } from '@/llm/anthropic';

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

  it('stamps Anthropic built-in tools with direct cache_control', () => {
    const webSearch = {
      type: 'web_search_20250305',
      name: 'web_search',
      max_uses: 3,
    };
    const out = partitionAndMarkAnthropicToolCache(
      [webSearch] as never,
      () => false
    ) as Array<{
      cache_control?: { type: string };
      extras?: { cache_control?: { type: string } };
    }>;

    expect(out[0]).not.toBe(webSearch);
    expect(out[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(out[0].extras).toBeUndefined();
  });

  it('does not serialize extras on Anthropic built-in tools', () => {
    const model = new CustomAnthropic({
      model: 'claude-haiku-4-5',
      apiKey: 'testing',
    });
    const webSearch = {
      type: 'web_search_20250305',
      name: 'web_search',
      max_uses: 3,
    };
    const tools = partitionAndMarkAnthropicToolCache(
      [webSearch] as never,
      () => false
    );
    const formattedTools = model.formatStructuredToolToAnthropic(tools);
    const formatted = formattedTools?.[0];

    expect(formatted).toEqual({
      type: 'web_search_20250305',
      name: 'web_search',
      max_uses: 3,
      cache_control: { type: 'ephemeral' },
    });
    expect(formatted).not.toHaveProperty('extras');
  });

  it('is idempotent when re-marking a tool that already has the marker', () => {
    const a = fakeTool('a-static') as { extras?: Record<string, unknown> };
    a.extras = { cache_control: { type: 'ephemeral' } };
    const input = [a] as never;
    // No deferred tools and the only static tool is already marked → input
    // is returned unchanged (same reference) so we don't churn the array.
    expect(partitionAndMarkAnthropicToolCache(input, () => false)).toBe(input);
  });

  it('stamps the resolved 1h ttl on the last static tool', () => {
    const out = partitionAndMarkAnthropicToolCache(
      [fakeTool('a-static'), fakeTool('b-static')] as never,
      () => false,
      '1h'
    ) as Array<{
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    expect(out[1].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(out[0].extras?.cache_control).toBeUndefined();
  });

  it('re-stamps a pre-marked 5m tool to 1h so it does not precede a 1h breakpoint', () => {
    const a = fakeTool('a-static') as {
      extras?: { cache_control?: { type: string; ttl?: string } };
    };
    a.extras = { cache_control: { type: 'ephemeral' } };
    const out = partitionAndMarkAnthropicToolCache(
      [a] as never,
      () => false,
      '1h'
    ) as Array<{
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    expect(out[0].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('strips a pre-marked earlier static tool so only the tail carries the 1h marker', () => {
    const a = fakeTool('a-static') as {
      extras?: { cache_control?: { type: string; ttl?: string } };
    };
    a.extras = { cache_control: { type: 'ephemeral' } };
    const b = fakeTool('b-static');
    const out = partitionAndMarkAnthropicToolCache(
      [a, b] as never,
      () => false,
      '1h'
    ) as Array<{
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    // Earlier tool's stray 5m marker is removed so it can't precede the tail.
    expect(out[0].extras?.cache_control).toBeUndefined();
    expect(out[1].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('strips stale markers off deferred tools so they do not precede the system/message breakpoint', () => {
    const staticTool = fakeTool('a-static');
    const deferred = fakeTool('b-deferred') as {
      extras?: { cache_control?: { type: string } };
    };
    deferred.extras = { cache_control: { type: 'ephemeral' } };
    const out = partitionAndMarkAnthropicToolCache(
      [staticTool, deferred] as never,
      (name) => name === 'b-deferred',
      '1h'
    ) as Array<{
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    expect(out[0].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(out[1].extras?.cache_control).toBeUndefined();
  });

  it('strips stale markers in the all-deferred case', () => {
    const deferred = fakeTool('only-deferred') as {
      extras?: { cache_control?: { type: string } };
    };
    deferred.extras = { cache_control: { type: 'ephemeral' } };
    const out = partitionAndMarkAnthropicToolCache(
      [deferred] as never,
      () => true,
      '1h'
    ) as Array<{ extras?: { cache_control?: unknown } }>;
    expect(out[0].extras?.cache_control).toBeUndefined();
  });

  it('strips a direct cache_control on an earlier native (non-built-in) tool', () => {
    const nativeWithMarker = {
      name: 'native_a',
      input_schema: { type: 'object', properties: {} },
      cache_control: { type: 'ephemeral' },
    };
    const out = partitionAndMarkAnthropicToolCache(
      [nativeWithMarker, fakeTool('native_b')] as never,
      () => false,
      '1h'
    ) as Array<{
      cache_control?: unknown;
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    expect(out[0].cache_control).toBeUndefined();
    expect(out[1].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('upgrades a raw Anthropic tail tool to a direct 1h marker (not extras)', () => {
    const nativeTail = {
      name: 'native_tail',
      input_schema: { type: 'object', properties: {} },
      cache_control: { type: 'ephemeral' },
    };
    const out = partitionAndMarkAnthropicToolCache(
      [nativeTail] as never,
      () => false,
      '1h'
    ) as Array<{
      cache_control?: { type: string; ttl?: string };
      extras?: { cache_control?: unknown };
    }>;
    // Raw Anthropic tools carry cache_control directly (extras is not promoted
    // for them), so the stale marker is upgraded in place to 1h — not moved.
    expect(out[0].cache_control).toEqual({ type: 'ephemeral', ttl: '1h' });
    expect(out[0].extras?.cache_control).toBeUndefined();
  });

  it('reorders deferred tools after a correctly pre-marked static tool', () => {
    const deferred = fakeTool('z-deferred');
    const staticTool = fakeTool('a-static') as {
      extras?: { cache_control?: { type: string; ttl?: string } };
    };
    // Already carries the resolved 1h marker, so nothing is mutated...
    staticTool.extras = { cache_control: { type: 'ephemeral', ttl: '1h' } };
    const out = partitionAndMarkAnthropicToolCache(
      [deferred, staticTool] as never, // deferred precedes static in the input
      (name) => name === 'z-deferred',
      '1h'
    ) as Array<{ name?: string }>;
    // ...but the static (cached) tool must still be hoisted ahead of the
    // deferred tool so the breakpoint precedes discovered tools.
    expect(out.map((t) => t.name)).toEqual(['a-static', 'z-deferred']);
  });

  it('re-stamps a LangChain tool whose marker sits only on the direct block', () => {
    // A StructuredTool (custom) pre-marked directly — not under extras — does
    // not reach the payload, so the breakpoint must be re-stamped under extras.
    const t = fakeTool('a-static') as {
      cache_control?: unknown;
      extras?: { cache_control?: { type: string; ttl?: string } };
    };
    t.cache_control = { type: 'ephemeral', ttl: '1h' };
    const out = partitionAndMarkAnthropicToolCache(
      [t] as never,
      () => false,
      '1h'
    ) as Array<{
      cache_control?: unknown;
      extras?: { cache_control?: { type: string; ttl?: string } };
    }>;
    expect(out[0].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(out[0].cache_control).toBeUndefined();
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
