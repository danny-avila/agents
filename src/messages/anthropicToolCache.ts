/**
 * Anthropic prompt-caching helper for the `tools[]` request field.
 *
 * Anthropic accepts `cache_control: { type: 'ephemeral' }` on individual
 * tool definitions. Whichever tool carries the marker becomes the end of
 * a cached prefix: `tools[0..N]` (everything up to and including the
 * marked tool) is cached and rebated on subsequent matching requests.
 *
 * For agents that mix static and deferred (lazily-discovered) tools, the
 * winning configuration is:
 *
 *   1. Stable-partition tools so all *static* (non-deferred) tools come
 *      first and discovered-deferred tools come last.
 *   2. Stamp `cache_control` on the LAST static tool.
 *
 * That way, the cached prefix covers exactly the static tool inventory.
 * Discovered tools that show up later (or vary turn-to-turn as new ones
 * get discovered) never invalidate the prefix because they sit *after*
 * the breakpoint.
 *
 * LangChain's Anthropic adapter passes the marker through via
 * `tool.extras.cache_control` (`AnthropicToolExtrasSchema`), so we set
 * it as an `extras` field on a fresh wrapper around the tool — never
 * mutating the original tool instance, since callers may share them
 * across runs.
 */

import type { GraphTools } from '@/types';

/**
 * Returns a callable that reports whether a given tool *name* is deferred
 * according to the supplied registry of tool definitions. Tools without
 * a registry entry are treated as non-deferred (i.e. static), matching
 * the host-supplied `graphTools` semantics elsewhere in the SDK.
 */
export function makeIsDeferred(
  toolDefinitions:
    | ReadonlyArray<{ name: string; defer_loading?: boolean }>
    | undefined
): (toolName: string) => boolean {
  if (toolDefinitions == null || toolDefinitions.length === 0) {
    return () => false;
  }
  const deferred = new Set<string>();
  for (const def of toolDefinitions) {
    if (def.defer_loading === true) deferred.add(def.name);
  }
  if (deferred.size === 0) return () => false;
  return (name) => deferred.has(name);
}

/**
 * Stable-partition `tools` into [static..., deferred...] and stamp the
 * Anthropic `cache_control: ephemeral` marker on the last static tool.
 *
 * If `tools` is undefined or empty, or no entry has a usable `.name`,
 * returns the input unchanged. If there are no static tools at all,
 * also returns unchanged (nothing to cache).
 *
 * The original tool instances are never mutated. The marked entry is a
 * shallow wrapper that preserves the prototype chain so downstream
 * `instanceof` checks still pass. `extras` is merged so any existing
 * `providerToolDefinition` / other extras the host attached are kept.
 */
export function partitionAndMarkAnthropicToolCache(
  tools: GraphTools | undefined,
  isDeferred: (toolName: string) => boolean
): GraphTools | undefined {
  if (tools == null || tools.length === 0) return tools;

  // Use unknown[] internally to avoid GraphTools' union-array variance
  // (each member is one of three array types). We cast back to
  // GraphTools when returning.
  const staticTools: unknown[] = [];
  const deferredTools: unknown[] = [];

  for (const tool of tools) {
    const name = (tool as { name?: unknown }).name;
    if (typeof name === 'string' && isDeferred(name)) {
      deferredTools.push(tool);
    } else {
      staticTools.push(tool);
    }
  }

  if (staticTools.length === 0) {
    return tools;
  }

  const last = staticTools[staticTools.length - 1] as {
    extras?: Record<string, unknown>;
  };
  // Already marked? Don't double-clone.
  if (
    last.extras != null &&
    'cache_control' in last.extras &&
    (last.extras as { cache_control?: unknown }).cache_control != null
  ) {
    if (deferredTools.length === 0) return tools;
    return [...staticTools, ...deferredTools] as GraphTools;
  }

  const wrapped = Object.assign(
    Object.create(Object.getPrototypeOf(last) ?? Object.prototype),
    last,
    {
      extras: {
        ...((last.extras as Record<string, unknown> | undefined) ?? {}),
        cache_control: { type: 'ephemeral' as const },
      },
    }
  );

  staticTools[staticTools.length - 1] = wrapped;
  return [...staticTools, ...deferredTools] as GraphTools;
}
