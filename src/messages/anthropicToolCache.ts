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
 * `tool.extras.cache_control` for custom tools, while Anthropic built-ins
 * require direct `cache_control`. Either way, we stamp a fresh wrapper —
 * never mutating the original tool instance, since callers may share them
 * across runs.
 */

import type { GraphTools } from '@/types';

const ANTHROPIC_BUILT_IN_TOOL_PREFIXES = [
  'text_editor_',
  'computer_',
  'bash_',
  'web_search_',
  'web_fetch_',
  'str_replace_editor_',
  'str_replace_based_edit_tool_',
  'code_execution_',
  'memory_',
  'tool_search_',
  'mcp_toolset',
] as const;

const CACHE_CONTROL = { type: 'ephemeral' as const };

type AnthropicToolCacheCandidate = {
  name?: unknown;
  type?: unknown;
  extras?: Record<string, unknown>;
  cache_control?: unknown;
};

function isAnthropicBuiltInTool(
  tool: AnthropicToolCacheCandidate
): tool is AnthropicToolCacheCandidate & { type: string } {
  const { type } = tool;
  return (
    typeof type === 'string' &&
    ANTHROPIC_BUILT_IN_TOOL_PREFIXES.some((prefix) => type.startsWith(prefix))
  );
}

function hasCacheControl(tool: AnthropicToolCacheCandidate): boolean {
  if (isAnthropicBuiltInTool(tool)) {
    return tool.cache_control != null;
  }
  return tool.extras?.cache_control != null;
}

function markCacheControl(
  tool: AnthropicToolCacheCandidate
): AnthropicToolCacheCandidate {
  const prototype = Object.getPrototypeOf(tool) ?? Object.prototype;
  if (isAnthropicBuiltInTool(tool)) {
    const wrapped = { ...tool };
    delete wrapped.extras;
    return Object.assign(Object.create(prototype), wrapped, {
      cache_control: CACHE_CONTROL,
    });
  }

  return Object.assign(Object.create(prototype), tool, {
    extras: {
      ...(tool.extras ?? {}),
      cache_control: CACHE_CONTROL,
    },
  });
}

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
 * `instanceof` checks still pass. For custom tools, `extras` is merged
 * so any existing `providerToolDefinition` / other extras are kept.
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

  const last = staticTools[
    staticTools.length - 1
  ] as AnthropicToolCacheCandidate;
  // Already marked? Don't double-clone.
  if (hasCacheControl(last)) {
    if (deferredTools.length === 0) return tools;
    return [...staticTools, ...deferredTools] as GraphTools;
  }

  staticTools[staticTools.length - 1] = markCacheControl(last);
  return [...staticTools, ...deferredTools] as GraphTools;
}
