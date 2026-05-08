import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { OpenAIClient } from '@langchain/openai';
import type { GraphTools } from '@/types';
import { _convertToOpenAITool } from '@/llm/openai';

const CACHE_CONTROL = { type: 'ephemeral' as const };

type OpenRouterToolWithCacheControl = OpenAIClient.ChatCompletionTool & {
  cache_control?: typeof CACHE_CONTROL;
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
  tool: OpenRouterToolWithCacheControl
): OpenRouterToolWithCacheControl {
  return {
    ...tool,
    cache_control: CACHE_CONTROL,
  };
}

export function partitionAndMarkOpenRouterToolCache(
  tools: GraphTools | undefined,
  isDeferred: (toolName: string) => boolean
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

  if (staticTools.length === 0) {
    return [...deferredTools] as GraphTools;
  }

  staticTools[staticTools.length - 1] = markCacheControl(
    staticTools[staticTools.length - 1]
  );

  return [...staticTools, ...deferredTools] as GraphTools;
}
