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
import {
  buildAnthropicCacheControl,
  type PromptCacheTtl,
} from '@/messages/cache';

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

/**
 * Whether a tool already carries a cache breakpoint. Built-ins use a direct
 * `cache_control`; custom tools normally carry it under `extras`, but a
 * caller-supplied Anthropic-native tool object can also put it directly on the
 * block — check both so stale markers are never missed.
 */
function hasCacheControl(tool: AnthropicToolCacheCandidate): boolean {
  if (isAnthropicBuiltInTool(tool)) {
    return tool.cache_control != null;
  }
  return tool.cache_control != null || tool.extras?.cache_control != null;
}

/**
 * Read the extended-cache TTL (`'1h'`) carried by a tool's existing
 * `cache_control`, or `undefined` for the legacy 5-minute marker (no `ttl`).
 * Checks the direct block first, then `extras` (custom tools).
 */
function getCacheControlTtl(
  tool: AnthropicToolCacheCandidate
): '1h' | undefined {
  const cacheControl = (
    isAnthropicBuiltInTool(tool)
      ? tool.cache_control
      : (tool.cache_control ?? tool.extras?.cache_control)
  ) as { ttl?: unknown } | undefined;
  return cacheControl?.ttl === '1h' ? '1h' : undefined;
}

/**
 * Return a clone of `tool` with any `cache_control` removed — both the direct
 * block marker and the `extras` marker — preserving the prototype chain. Used
 * to clear stray markers off tools that must not anchor a competing breakpoint.
 */
function stripCacheControl(
  tool: AnthropicToolCacheCandidate
): AnthropicToolCacheCandidate {
  const prototype = Object.getPrototypeOf(tool) ?? Object.prototype;
  const wrapped = { ...tool };
  delete wrapped.cache_control;
  if (wrapped.extras != null) {
    wrapped.extras = { ...wrapped.extras };
    delete wrapped.extras.cache_control;
  }
  return Object.assign(Object.create(prototype), wrapped);
}

/**
 * Whether `tool` is already in the Anthropic provider payload shape — an
 * Anthropic built-in or a raw Anthropic tool object (has `input_schema`). These
 * carry `cache_control` directly on the block; the LangChain adapter does NOT
 * promote `extras.cache_control` for them. LangChain StructuredTools, by
 * contrast, expose the marker via `extras`.
 */
function isProviderShapedTool(tool: AnthropicToolCacheCandidate): boolean {
  return (
    isAnthropicBuiltInTool(tool) ||
    'input_schema' in (tool as Record<string, unknown>)
  );
}

function markCacheControl(
  tool: AnthropicToolCacheCandidate,
  ttl?: PromptCacheTtl
): AnthropicToolCacheCandidate {
  const cacheControl = buildAnthropicCacheControl(ttl);
  const prototype = Object.getPrototypeOf(tool) ?? Object.prototype;
  if (isProviderShapedTool(tool)) {
    // Built-ins and raw Anthropic tool objects carry cache_control directly on
    // the block; `extras` is not promoted onto the payload for these shapes.
    const wrapped = { ...tool };
    delete wrapped.extras;
    return Object.assign(Object.create(prototype), wrapped, {
      cache_control: cacheControl,
    });
  }

  // LangChain custom tools: drop any direct marker and expose the breakpoint via
  // `extras`, which the Anthropic adapter promotes onto the payload.
  const wrapped = { ...tool };
  delete wrapped.cache_control;
  return Object.assign(Object.create(prototype), wrapped, {
    extras: {
      ...(tool.extras ?? {}),
      cache_control: cacheControl,
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
  isDeferred: (toolName: string) => boolean,
  ttl?: PromptCacheTtl
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

  // Anthropic serializes ALL tools before system/messages, so a stray
  // cache_control on any tool — static or deferred — that survives the resolved
  // breakpoint would violate the longer-TTL-first ordering. Strip stale markers
  // off the deferred tools first (they sit after the breakpoint but still before
  // system/messages, and the all-deferred case has no breakpoint of its own).
  let mutated = false;
  for (let i = 0; i < deferredTools.length; i++) {
    const candidate = deferredTools[i] as AnthropicToolCacheCandidate;
    if (hasCacheControl(candidate)) {
      deferredTools[i] = stripCacheControl(candidate);
      mutated = true;
    }
  }

  if (staticTools.length === 0) {
    return mutated ? ([...deferredTools] as GraphTools) : tools;
  }

  // Strip any stray cache_control off the earlier static tools so a leftover
  // 5-minute marker never sits ahead of the resolved breakpoint, then stamp (or
  // re-stamp) only the last static tool with the resolved TTL.
  for (let i = 0; i < staticTools.length - 1; i++) {
    const candidate = staticTools[i] as AnthropicToolCacheCandidate;
    if (hasCacheControl(candidate)) {
      staticTools[i] = stripCacheControl(candidate);
      mutated = true;
    }
  }

  const last = staticTools[
    staticTools.length - 1
  ] as AnthropicToolCacheCandidate;
  const desiredTtl: '1h' | undefined = ttl === '1h' ? '1h' : undefined;
  const lastAlreadyCorrect =
    hasCacheControl(last) && getCacheControlTtl(last) === desiredTtl;
  if (!lastAlreadyCorrect) {
    staticTools[staticTools.length - 1] = markCacheControl(last, ttl);
    mutated = true;
  }

  // Return the original reference only when nothing changed AND partitioning
  // moved nothing. When deferred tools exist they must end up after the cache
  // breakpoint, so the partitioned array is returned even if no marker changed.
  if (!mutated && deferredTools.length === 0) {
    return tools;
  }
  return [...staticTools, ...deferredTools] as GraphTools;
}
