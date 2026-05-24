import {
  Constants,
  CODE_EXECUTION_TOOLS,
  LOCAL_CODING_BUNDLE_NAMES,
} from '@/common';
import {
  createLocalBashExecutionTool,
  createLocalCodeExecutionTool,
} from './LocalExecutionTools';
import {
  createLocalCodingToolBundle,
  createLocalCodingToolDefinitions,
  createLocalCodingTools,
} from './LocalCodingTools';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from './LocalProgrammaticToolCalling';
import {
  createCloudflareCodingToolBundle,
  createCloudflareCodingTools,
  createCloudflareExecutionTool,
} from '@/tools/cloudflare';
import type * as t from '@/types';

type ResolveLocalToolsResult = {
  toolMap: t.ToolMap;
  directToolNames: Set<string>;
  /**
   * Set when `local.fileCheckpointing === true` AND the auto-bind
   * coding suite is in use. ToolNode stashes this on the node and
   * exposes it via `getFileCheckpointer()` so the host can call
   * `rewind()` after a failed batch. Manual review (finding E)
   * flagged that the config flag was previously a no-op in the
   * Run/ToolNode auto-bind path — only direct
   * `createLocalCodingToolBundle()` callers could access the
   * checkpointer.
   */
  fileCheckpointer?: t.LocalFileCheckpointer;
};

function shouldUseLocalExecution(config?: t.ToolExecutionConfig): boolean {
  return config?.engine === 'local';
}

function shouldUseCloudflareSandboxExecution(
  config?: t.ToolExecutionConfig
): boolean {
  return config?.engine === 'cloudflare-sandbox';
}

function shouldIncludeCodingTools(config?: t.ToolExecutionConfig): boolean {
  return (
    (shouldUseLocalExecution(config) &&
      config?.local?.includeCodingTools !== false) ||
    (shouldUseCloudflareSandboxExecution(config) &&
      config?.cloudflare?.includeCodingTools !== false)
  );
}

function getCloudflareConfig(
  config?: t.ToolExecutionConfig
): t.CloudflareSandboxExecutionConfig {
  if (config?.cloudflare == null) {
    throw new Error(
      'toolExecution.cloudflare is required when engine is "cloudflare-sandbox".'
    );
  }
  return config.cloudflare;
}

function getSelectedCloudflareCodingToolNames(
  config: t.CloudflareSandboxExecutionConfig
): Set<string> {
  return new Set(config.codingToolNames ?? LOCAL_CODING_BUNDLE_NAMES);
}

function filterCloudflareCodingToolAllowlist(
  tools: t.GraphTools | undefined,
  selectedNames: Set<string>
): t.GraphTools | undefined {
  const existingTools = (tools as t.GenericTool[] | undefined) ?? [];
  if (existingTools.length === 0) {
    return tools;
  }
  return existingTools.filter((existingTool) => {
    if (!('name' in existingTool) || typeof existingTool.name !== 'string') {
      return true;
    }
    return (
      !LOCAL_CODING_BUNDLE_NAMES.includes(existingTool.name) ||
      selectedNames.has(existingTool.name)
    );
  });
}

function pruneCloudflareCodingToolAllowlist(
  toolMap: t.ToolMap,
  selectedNames: Set<string>
): void {
  for (const name of LOCAL_CODING_BUNDLE_NAMES) {
    if (!selectedNames.has(name)) {
      toolMap.delete(name);
    }
  }
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
  if (
    !shouldUseLocalExecution(args.toolExecution) &&
    !shouldUseCloudflareSandboxExecution(args.toolExecution)
  ) {
    return args.tools;
  }

  if (shouldUseCloudflareSandboxExecution(args.toolExecution)) {
    const cloudflareConfig = getCloudflareConfig(args.toolExecution);
    if (shouldIncludeCodingTools(args.toolExecution)) {
      const selectedNames =
        getSelectedCloudflareCodingToolNames(cloudflareConfig);
      return mergeToolsByName(
        filterCloudflareCodingToolAllowlist(args.tools, selectedNames),
        createCloudflareCodingTools(cloudflareConfig)
      );
    }

    const replacements = ((args.tools as t.GenericTool[] | undefined) ?? [])
      .filter(
        (existingTool): existingTool is t.GenericTool & { name: string } =>
          'name' in existingTool &&
          typeof existingTool.name === 'string' &&
          CODE_EXECUTION_TOOLS.has(existingTool.name)
      )
      .map((existingTool) =>
        createCloudflareExecutionTool(existingTool.name, cloudflareConfig)
      )
      .filter(
        (cloudflareTool): cloudflareTool is t.GenericTool =>
          cloudflareTool != null
      );

    return replacements.length === 0
      ? args.tools
      : mergeToolsByName(args.tools, replacements);
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
  const selectedNames = shouldUseCloudflareSandboxExecution(args.toolExecution)
    ? getSelectedCloudflareCodingToolNames(
      getCloudflareConfig(args.toolExecution)
    )
    : undefined;
  for (const definition of createLocalCodingToolDefinitions()) {
    if (selectedNames != null && !selectedNames.has(definition.name)) {
      registry.delete(definition.name);
      continue;
    }
    registry.set(definition.name, definition);
  }
  return registry;
}

export function resolveLocalExecutionTools(args: {
  toolMap: t.ToolMap;
  toolExecution?: t.ToolExecutionConfig;
  /**
   * Caller-provided checkpointer that overrides the bundle's
   * auto-created one. The Graph layer threads a single per-Run
   * instance so every ToolNode it compiles shares one snapshot
   * store — without that, a multi-agent graph would each get a
   * private checkpointer and `Run.rewindFiles()` couldn't reach
   * any of them.
   */
  fileCheckpointer?: t.LocalFileCheckpointer;
}): ResolveLocalToolsResult {
  const directToolNames = new Set<string>();
  if (
    !shouldUseLocalExecution(args.toolExecution) &&
    !shouldUseCloudflareSandboxExecution(args.toolExecution)
  ) {
    return {
      toolMap: args.toolMap,
      directToolNames,
    };
  }

  const toolMap = new Map(args.toolMap);
  let fileCheckpointer: t.LocalFileCheckpointer | undefined;

  if (shouldUseCloudflareSandboxExecution(args.toolExecution)) {
    const cloudflareConfig = getCloudflareConfig(args.toolExecution);
    if (shouldIncludeCodingTools(args.toolExecution)) {
      const selectedNames =
        getSelectedCloudflareCodingToolNames(cloudflareConfig);
      pruneCloudflareCodingToolAllowlist(toolMap, selectedNames);
      if (
        cloudflareConfig.fileCheckpointing === true ||
        args.fileCheckpointer != null
      ) {
        const bundle = createCloudflareCodingToolBundle(cloudflareConfig, {
          checkpointer: args.fileCheckpointer,
        });
        fileCheckpointer = bundle.checkpointer;
        for (const cloudflareTool of bundle.tools) {
          toolMap.set(cloudflareTool.name, cloudflareTool);
          directToolNames.add(cloudflareTool.name);
        }
      } else {
        for (const cloudflareTool of createCloudflareCodingTools(
          cloudflareConfig
        )) {
          toolMap.set(cloudflareTool.name, cloudflareTool);
          directToolNames.add(cloudflareTool.name);
        }
      }
    }

    const includeCodingTools = shouldIncludeCodingTools(args.toolExecution);
    for (const name of CODE_EXECUTION_TOOLS) {
      if (includeCodingTools) continue;
      if (!toolMap.has(name)) continue;

      const cloudflareTool = createCloudflareExecutionTool(
        name,
        cloudflareConfig
      );
      if (cloudflareTool == null) {
        continue;
      }

      toolMap.set(name, cloudflareTool);
      directToolNames.add(name);
    }

    return { toolMap, directToolNames, fileCheckpointer };
  }

  const localConfig = args.toolExecution?.local ?? {};

  if (shouldIncludeCodingTools(args.toolExecution)) {
    // Use the bundle factory when fileCheckpointing is on so we can
    // surface the checkpointer back to the caller — without this, the
    // execution-path tools each captured into a checkpointer that was
    // immediately discarded, making the public `fileCheckpointing`
    // config flag a silent no-op outside of direct
    // `createLocalCodingToolBundle()` use.
    if (
      localConfig.fileCheckpointing === true ||
      args.fileCheckpointer != null
    ) {
      const bundle = createLocalCodingToolBundle(localConfig, {
        checkpointer: args.fileCheckpointer,
      });
      fileCheckpointer = bundle.checkpointer;
      for (const localTool of bundle.tools) {
        toolMap.set(localTool.name, localTool);
        directToolNames.add(localTool.name);
      }
    } else {
      for (const localTool of createLocalCodingTools(localConfig)) {
        toolMap.set(localTool.name, localTool);
        directToolNames.add(localTool.name);
      }
    }
  }

  // When the coding-tool bundle was already installed above, it
  // already created `bash_tool` / `execute_code` / programmatic-tool
  // variants. Skip re-creating them here — the audit-of-audit (manual
  // finding #4) flagged that the original loop overwrote those bundle
  // instances with fresh ones via `createLocalExecutionTool`, wasting
  // work and (more importantly) replacing tools the bundle had
  // already wired up with shared state. The CODE_EXECUTION_TOOLS
  // loop is now only relevant when the host pre-bound a tool with
  // one of these names (the `toolMap.has(name)` branch) and coding
  // tools are off.
  const includeCodingTools = shouldIncludeCodingTools(args.toolExecution);
  for (const name of CODE_EXECUTION_TOOLS) {
    if (includeCodingTools) continue;
    if (!toolMap.has(name)) continue;

    const localTool = createLocalExecutionTool(name, localConfig);
    if (localTool == null) {
      continue;
    }

    toolMap.set(name, localTool);
    directToolNames.add(name);
  }

  return { toolMap, directToolNames, fileCheckpointer };
}
