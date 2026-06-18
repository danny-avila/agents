import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { OpenAIClient } from '@langchain/openai';
import type { GraphTools } from '@/types';
import {
  buildAnthropicCacheControl,
  type PromptCacheTtl,
} from '@/messages/cache';
import { _convertToOpenAITool } from '@/llm/openai';

type OpenRouterCacheControl = { type: 'ephemeral'; ttl?: '1h' };

type OpenRouterToolWithCacheControl = OpenAIClient.ChatCompletionTool & {
  cache_control?: OpenRouterCacheControl;
  defer_loading?: boolean;
};

type ToolNameCandidate = {
  name?: unknown;
  function?: {
    name?: unknown;
  };
  defer_loading?: unknown;
};

function getToolName(tool: unknown): string | undefined {
  const candidate = tool as ToolNameCandidate;
  if (typeof candidate.name === 'string') {
    return candidate.name;
  }
  if (typeof candidate.function?.name === 'string') {
    return candidate.function.name;
  }
  return undefined;
}

function hasDeferredMarker(tool: unknown): boolean {
  return (tool as ToolNameCandidate).defer_loading === true;
}

function toOpenRouterTool(tool: unknown): OpenRouterToolWithCacheControl {
  const converted = _convertToOpenAITool(
    tool as BindToolsInput
  ) as OpenRouterToolWithCacheControl;

  if (hasDeferredMarker(tool)) {
    return { ...converted, defer_loading: true };
  }

  return converted;
}

function markCacheControl(
  tool: OpenRouterToolWithCacheControl,
  ttl?: PromptCacheTtl
): OpenRouterToolWithCacheControl {
  return {
    ...tool,
    cache_control: buildAnthropicCacheControl(ttl),
  };
}

/**
 * Drop any existing `cache_control` from a tool. Reused/caller-supplied tools
 * can carry a stale marker (e.g. from a prior `promptCacheTtl: '5m'` run); since
 * all tools serialize before system/messages, a leftover 5-minute marker ahead
 * of the resolved breakpoint would violate the longer-TTL-first ordering.
 */
function stripCacheControl(
  tool: OpenRouterToolWithCacheControl
): OpenRouterToolWithCacheControl {
  if (tool.cache_control == null) {
    return tool;
  }
  const { cache_control: _omit, ...rest } = tool;
  return rest;
}

export function partitionAndMarkOpenRouterToolCache(
  tools: GraphTools | undefined,
  isDeferred: (toolName: string) => boolean,
  ttl?: PromptCacheTtl
): GraphTools | undefined {
  if (tools == null || tools.length === 0) {
    return tools;
  }

  const staticTools: OpenRouterToolWithCacheControl[] = [];
  const deferredTools: OpenRouterToolWithCacheControl[] = [];

  for (const tool of tools as readonly unknown[]) {
    const converted = toOpenRouterTool(tool);
    const name = getToolName(converted) ?? getToolName(tool);

    if (name != null && isDeferred(name)) {
      deferredTools.push(converted);
      continue;
    }

    staticTools.push(converted);
  }

  // Deferred tools sit after the breakpoint but still before system/messages,
  // so strip any stale marker off them.
  for (let i = 0; i < deferredTools.length; i++) {
    deferredTools[i] = stripCacheControl(deferredTools[i]);
  }

  if (staticTools.length === 0) {
    return [...deferredTools] as GraphTools;
  }

  // Strip stale markers off the earlier static tools, then stamp only the last
  // static tool with the resolved TTL (markCacheControl overwrites any marker).
  for (let i = 0; i < staticTools.length - 1; i++) {
    staticTools[i] = stripCacheControl(staticTools[i]);
  }
  staticTools[staticTools.length - 1] = markCacheControl(
    staticTools[staticTools.length - 1],
    ttl
  );

  return [...staticTools, ...deferredTools] as GraphTools;
}
