import type { Tool, ToolConfiguration } from '@aws-sdk/client-bedrock-runtime';
import type { BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { OpenAIClient } from '@langchain/openai';
import type { DocumentType } from '@smithy/types';
import type { GraphTools } from '@/types';
import { _convertToOpenAITool } from '@/llm/openai';

const CACHE_POINT: Tool.CachePointMember = {
  cachePoint: { type: 'default' },
};

const BEDROCK_TOOL_CACHE_MARKER = '__lc_bedrock_cache_point_after';
const BEDROCK_TOOL_CACHE_DISABLED_MARKER = '__lc_bedrock_skip_tool_cache';

type BedrockToolWithCacheMarker = Tool & {
  [BEDROCK_TOOL_CACHE_MARKER]?: true;
  [BEDROCK_TOOL_CACHE_DISABLED_MARKER]?: true;
};

type OpenAIFunctionTool = Extract<
  OpenAIClient.ChatCompletionTool,
  { type: 'function' }
>;

type ToolNameCandidate = {
  name?: unknown;
  function?: {
    name?: unknown;
  };
  toolSpec?: {
    name?: unknown;
  };
};

function isBedrockToolSpec(tool: unknown): tool is Tool.ToolSpecMember {
  return (
    typeof tool === 'object' &&
    tool != null &&
    'toolSpec' in tool &&
    typeof (tool as ToolNameCandidate).toolSpec?.name === 'string'
  );
}

function isBedrockCachePoint(tool: Tool): tool is Tool.CachePointMember {
  return 'cachePoint' in tool && tool.cachePoint != null;
}

function getToolName(tool: unknown): string | undefined {
  const candidate = tool as ToolNameCandidate;
  if (typeof candidate.toolSpec?.name === 'string') {
    return candidate.toolSpec.name;
  }
  if (typeof candidate.name === 'string') {
    return candidate.name;
  }
  if (typeof candidate.function?.name === 'string') {
    return candidate.function.name;
  }
  return undefined;
}

function openAIToBedrockTool(tool: OpenAIFunctionTool): Tool.ToolSpecMember {
  return {
    toolSpec: {
      name: tool.function.name,
      description: tool.function.description,
      inputSchema: { json: tool.function.parameters as DocumentType },
    },
  };
}

function toBedrockTool(tool: unknown): BedrockToolWithCacheMarker {
  if (isBedrockToolSpec(tool)) {
    return { ...tool };
  }

  return openAIToBedrockTool(
    _convertToOpenAITool(tool as BindToolsInput) as OpenAIFunctionTool
  ) as BedrockToolWithCacheMarker;
}

function markCachePointAfter(
  tool: BedrockToolWithCacheMarker
): BedrockToolWithCacheMarker {
  return {
    ...tool,
    [BEDROCK_TOOL_CACHE_MARKER]: true,
  };
}

function markToolCacheDisabled(
  tool: BedrockToolWithCacheMarker
): BedrockToolWithCacheMarker {
  return {
    ...tool,
    [BEDROCK_TOOL_CACHE_DISABLED_MARKER]: true,
  };
}

function stripCachePointMarker(tool: BedrockToolWithCacheMarker): Tool {
  const {
    [BEDROCK_TOOL_CACHE_MARKER]: _marker,
    [BEDROCK_TOOL_CACHE_DISABLED_MARKER]: _disabled,
    ...rest
  } = tool;
  return rest as Tool;
}

export function partitionAndMarkBedrockToolCache(
  tools: GraphTools | undefined,
  isDeferred: (toolName: string) => boolean
): GraphTools | undefined {
  if (tools == null || tools.length === 0) {
    return tools;
  }

  const staticTools: BedrockToolWithCacheMarker[] = [];
  const deferredTools: BedrockToolWithCacheMarker[] = [];

  for (const tool of tools as readonly unknown[]) {
    const converted = toBedrockTool(tool);
    const name = getToolName(converted) ?? getToolName(tool);

    if (name != null && isDeferred(name)) {
      deferredTools.push(converted);
      continue;
    }

    staticTools.push(converted);
  }

  if (staticTools.length === 0) {
    deferredTools[0] = markToolCacheDisabled(deferredTools[0]);
    return [...deferredTools] as GraphTools;
  }

  staticTools[staticTools.length - 1] = markCachePointAfter(
    staticTools[staticTools.length - 1]
  );

  return [...staticTools, ...deferredTools] as GraphTools;
}

export function insertBedrockToolCachePoint(
  toolConfig: ToolConfiguration | undefined,
  fallbackToEnd: boolean
): ToolConfiguration | undefined {
  const tools = toolConfig?.tools as BedrockToolWithCacheMarker[] | undefined;
  if (tools == null || tools.length === 0) {
    return toolConfig;
  }

  let markerIndex = -1;
  let hasCachePoint = false;
  let hasDisabledMarker = false;
  const cleanedTools: Tool[] = [];

  for (let i = 0; i < tools.length; i++) {
    const tool = tools[i];
    if (isBedrockCachePoint(tool)) {
      hasCachePoint = true;
      cleanedTools.push(tool);
      continue;
    }
    if (tool[BEDROCK_TOOL_CACHE_MARKER] === true) {
      markerIndex = cleanedTools.length;
    }
    if (tool[BEDROCK_TOOL_CACHE_DISABLED_MARKER] === true) {
      hasDisabledMarker = true;
    }
    cleanedTools.push(stripCachePointMarker(tool));
  }

  if (hasCachePoint || hasDisabledMarker) {
    return { ...toolConfig, tools: cleanedTools };
  }

  const insertionIndex = markerIndex >= 0 ? markerIndex : tools.length - 1;
  if (markerIndex < 0 && !fallbackToEnd) {
    return { ...toolConfig, tools: cleanedTools };
  }

  return {
    ...toolConfig,
    tools: [
      ...cleanedTools.slice(0, insertionIndex + 1),
      CACHE_POINT,
      ...cleanedTools.slice(insertionIndex + 1),
    ],
  };
}
