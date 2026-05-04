import { Constants, CODE_EXECUTION_TOOLS } from '@/common';
import {
  createLocalBashExecutionTool,
  createLocalCodeExecutionTool,
} from './LocalExecutionTools';
import {
  createLocalCodingTools,
  createLocalCodingToolDefinitions,
} from './LocalCodingTools';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from './LocalProgrammaticToolCalling';
import type * as t from '@/types';

type ResolveLocalToolsResult = {
  toolMap: t.ToolMap;
  directToolNames: Set<string>;
};

function shouldUseLocalExecution(config?: t.ToolExecutionConfig): boolean {
  return config?.engine === 'local';
}

function shouldIncludeCodingTools(config?: t.ToolExecutionConfig): boolean {
  return (
    shouldUseLocalExecution(config) &&
    config?.local?.includeCodingTools !== false
  );
}

function createLocalExecutionTool(
  name: string,
  config: t.LocalExecutionConfig
): t.GenericTool | undefined {
  switch (name) {
  case Constants.EXECUTE_CODE:
    return createLocalCodeExecutionTool(config);
  case Constants.BASH_TOOL:
    return createLocalBashExecutionTool({ config });
  case Constants.PROGRAMMATIC_TOOL_CALLING:
    return createLocalProgrammaticToolCallingTool(config);
  case Constants.BASH_PROGRAMMATIC_TOOL_CALLING:
    return createLocalBashProgrammaticToolCallingTool(config);
  default:
    return undefined;
  }
}

function mergeToolsByName(
  baseTools: t.GraphTools | undefined,
  localTools: t.GenericTool[]
): t.GraphTools {
  const orderedTools: t.GenericTool[] = [];
  const indexByName = new Map<string, number>();

  for (const tool of (baseTools as t.GenericTool[] | undefined) ?? []) {
    if ('name' in tool && typeof tool.name === 'string') {
      indexByName.set(tool.name, orderedTools.length);
    }
    orderedTools.push(tool);
  }

  for (const tool of localTools) {
    const existingIndex = indexByName.get(tool.name);
    if (existingIndex == null) {
      indexByName.set(tool.name, orderedTools.length);
      orderedTools.push(tool);
      continue;
    }
    orderedTools[existingIndex] = tool;
  }

  return orderedTools;
}

export function resolveLocalToolsForBinding(args: {
  tools?: t.GraphTools;
  toolExecution?: t.ToolExecutionConfig;
}): t.GraphTools | undefined {
  if (!shouldUseLocalExecution(args.toolExecution)) {
    return args.tools;
  }

  const localConfig = args.toolExecution?.local ?? {};
  if (shouldIncludeCodingTools(args.toolExecution)) {
    return mergeToolsByName(args.tools, createLocalCodingTools(localConfig));
  }

  const replacements = ((args.tools as t.GenericTool[] | undefined) ?? [])
    .filter(
      (existingTool): existingTool is t.GenericTool & { name: string } =>
        'name' in existingTool &&
        typeof existingTool.name === 'string' &&
        CODE_EXECUTION_TOOLS.has(existingTool.name)
    )
    .map((existingTool) =>
      createLocalExecutionTool(existingTool.name, localConfig)
    )
    .filter((localTool): localTool is t.GenericTool => localTool != null);

  return replacements.length === 0
    ? args.tools
    : mergeToolsByName(args.tools, replacements);
}

export function resolveLocalToolRegistry(args: {
  toolRegistry?: t.LCToolRegistry;
  toolExecution?: t.ToolExecutionConfig;
}): t.LCToolRegistry | undefined {
  if (!shouldIncludeCodingTools(args.toolExecution)) {
    return args.toolRegistry;
  }

  const registry = new Map(args.toolRegistry ?? []);
  for (const definition of createLocalCodingToolDefinitions()) {
    registry.set(definition.name, definition);
  }
  return registry;
}

export function resolveLocalExecutionTools(args: {
  toolMap: t.ToolMap;
  toolExecution?: t.ToolExecutionConfig;
}): ResolveLocalToolsResult {
  const directToolNames = new Set<string>();
  if (!shouldUseLocalExecution(args.toolExecution)) {
    return {
      toolMap: args.toolMap,
      directToolNames,
    };
  }

  const localConfig = args.toolExecution?.local ?? {};
  const toolMap = new Map(args.toolMap);

  if (shouldIncludeCodingTools(args.toolExecution)) {
    for (const localTool of createLocalCodingTools(localConfig)) {
      toolMap.set(localTool.name, localTool);
      directToolNames.add(localTool.name);
    }
  }

  for (const name of CODE_EXECUTION_TOOLS) {
    if (!toolMap.has(name) && !shouldIncludeCodingTools(args.toolExecution)) {
      continue;
    }

    const localTool = createLocalExecutionTool(name, localConfig);
    if (localTool == null) {
      continue;
    }

    toolMap.set(name, localTool);
    directToolNames.add(name);
  }

  return { toolMap, directToolNames };
}
