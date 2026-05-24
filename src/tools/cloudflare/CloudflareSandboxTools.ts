import { tool } from '@langchain/core/tools';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  CodeExecutionToolName,
  CodeExecutionToolSchema,
} from '@/tools/CodeExecutor';
import {
  BashExecutionToolName,
  BashExecutionToolSchema,
  BashToolOutputReferencesGuide,
} from '@/tools/BashExecutor';
import {
  createLocalReadFileTool,
  createLocalWriteFileTool,
  createLocalEditFileTool,
  createLocalGrepSearchTool,
  createLocalGlobSearchTool,
  createLocalListDirectoryTool,
} from '@/tools/local/LocalCodingTools';
import { createLocalFileCheckpointer } from '@/tools/local/FileCheckpointer';
import { createCompileCheckTool } from '@/tools/local/CompileCheckTool';
import { Constants } from '@/common';
import {
  createCloudflareLocalExecutionConfig,
  createCloudflareWorkspaceFS,
  executeCloudflareBash,
  executeCloudflareCode,
  formatCloudflareOutput,
  getCloudflareWorkspaceRoot,
} from './CloudflareSandboxExecutionEngine';
import {
  createCloudflareBashProgrammaticToolCallingTool,
  createCloudflareProgrammaticToolCallingTool,
} from './CloudflareProgrammaticToolCalling';

export const CLOUDFLARE_CODING_TOOL_NAMES: readonly string[] = [
  Constants.READ_FILE,
  Constants.WRITE_FILE,
  Constants.EDIT_FILE,
  Constants.GREP_SEARCH,
  Constants.GLOB_SEARCH,
  Constants.LIST_DIRECTORY,
  Constants.COMPILE_CHECK,
  Constants.BASH_TOOL,
  Constants.EXECUTE_CODE,
  Constants.PROGRAMMATIC_TOOL_CALLING,
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
];

export const CLOUDFLARE_BASH_CODING_TOOL_NAMES: readonly string[] = [
  Constants.READ_FILE,
  Constants.WRITE_FILE,
  Constants.EDIT_FILE,
  Constants.GREP_SEARCH,
  Constants.GLOB_SEARCH,
  Constants.LIST_DIRECTORY,
  Constants.COMPILE_CHECK,
  Constants.BASH_TOOL,
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
];

export const CloudflareCodeExecutionToolDescription = `
Runs code inside the configured Cloudflare Sandbox workspace. The sandbox can see files and installed runtimes available inside the Cloudflare Sandbox container.

Usage:
- Commands execute in the Cloudflare Sandbox workspace and may modify sandbox files.
- Input code is already displayed to the user, so do not repeat it unless asked.
- Output is not displayed unless you print it explicitly.
`.trim();

export const CloudflareBashExecutionToolDescription = `
Runs bash commands inside the configured Cloudflare Sandbox workspace.

Usage:
- Commands execute in the Cloudflare Sandbox workspace and may modify sandbox files.
- Output is not displayed unless you print it explicitly.
- Prefer project-native commands and inspect files before changing them.
`.trim();

export function createCloudflareCodeExecutionTool(
  config: t.CloudflareSandboxExecutionConfig
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        lang: string;
        code: string;
        args?: string[];
      };
      const cwd = getCloudflareWorkspaceRoot(config);
      const result = await executeCloudflareCode(input, config);
      return [
        formatCloudflareOutput(result, cwd),
        {
          session_id: 'cloudflare-sandbox',
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    {
      name: CodeExecutionToolName,
      description: CloudflareCodeExecutionToolDescription,
      schema: CodeExecutionToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createCloudflareBashExecutionTool(options: {
  config: t.CloudflareSandboxExecutionConfig;
  enableToolOutputReferences?: boolean;
}): DynamicStructuredTool {
  const { config } = options;
  return tool(
    async (rawInput) => {
      const input = rawInput as { command: string; args?: string[] };
      const cwd = getCloudflareWorkspaceRoot(config);
      const result = await executeCloudflareBash(
        input.command,
        config,
        input.args ?? []
      );
      return [
        formatCloudflareOutput(result, cwd),
        {
          session_id: 'cloudflare-sandbox',
          files: [],
        } satisfies t.CodeExecutionArtifact,
      ];
    },
    {
      name: BashExecutionToolName,
      description:
        options.enableToolOutputReferences === true
          ? `${CloudflareBashExecutionToolDescription}\n\n${BashToolOutputReferencesGuide}`
          : CloudflareBashExecutionToolDescription,
      schema: BashExecutionToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export type CloudflareCodingToolBundle = {
  tools: DynamicStructuredTool[];
  checkpointer?: t.LocalFileCheckpointer;
};

function getSelectedCodingToolNames(
  config: t.CloudflareSandboxExecutionConfig
): Set<string> {
  return new Set(config.codingToolNames ?? CLOUDFLARE_CODING_TOOL_NAMES);
}

export function createCloudflareCodingTools(
  config: t.CloudflareSandboxExecutionConfig,
  options: { checkpointer?: t.LocalFileCheckpointer } = {}
): DynamicStructuredTool[] {
  const localConfig = createCloudflareLocalExecutionConfig(config);
  const checkpointer =
    options.checkpointer ??
    (config.fileCheckpointing === true
      ? createLocalFileCheckpointer({
        fs: createCloudflareWorkspaceFS(config),
      })
      : undefined);
  const tools = [
    [Constants.READ_FILE, createLocalReadFileTool(localConfig)],
    [Constants.WRITE_FILE, createLocalWriteFileTool(localConfig, checkpointer)],
    [Constants.EDIT_FILE, createLocalEditFileTool(localConfig, checkpointer)],
    [Constants.GREP_SEARCH, createLocalGrepSearchTool(localConfig)],
    [Constants.GLOB_SEARCH, createLocalGlobSearchTool(localConfig)],
    [Constants.LIST_DIRECTORY, createLocalListDirectoryTool(localConfig)],
    [Constants.COMPILE_CHECK, createCompileCheckTool(localConfig)],
    [Constants.BASH_TOOL, createCloudflareBashExecutionTool({ config })],
    [Constants.EXECUTE_CODE, createCloudflareCodeExecutionTool(config)],
    [
      Constants.PROGRAMMATIC_TOOL_CALLING,
      createCloudflareProgrammaticToolCallingTool(config),
    ],
    [
      Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      createCloudflareBashProgrammaticToolCallingTool(config),
    ],
  ] satisfies Array<[string, DynamicStructuredTool]>;
  const selectedNames = getSelectedCodingToolNames(config);
  return tools
    .filter(([name]) => selectedNames.has(name))
    .map(([, selectedTool]) => selectedTool);
}

export function createCloudflareCodingToolBundle(
  config: t.CloudflareSandboxExecutionConfig,
  options: { checkpointer?: t.LocalFileCheckpointer } = {}
): CloudflareCodingToolBundle {
  const checkpointer =
    options.checkpointer ??
    (config.fileCheckpointing === true
      ? createLocalFileCheckpointer({
        fs: createCloudflareWorkspaceFS(config),
      })
      : undefined);
  return {
    tools: createCloudflareCodingTools(config, { checkpointer }),
    checkpointer,
  };
}

export function createCloudflareExecutionTool(
  name: string,
  config: t.CloudflareSandboxExecutionConfig
): t.GenericTool | undefined {
  switch (name) {
  case Constants.EXECUTE_CODE:
    return createCloudflareCodeExecutionTool(config);
  case Constants.BASH_TOOL:
    return createCloudflareBashExecutionTool({ config });
  case Constants.PROGRAMMATIC_TOOL_CALLING:
    return createCloudflareProgrammaticToolCallingTool(config);
  case Constants.BASH_PROGRAMMATIC_TOOL_CALLING:
    return createCloudflareBashProgrammaticToolCallingTool(config);
  default:
    return undefined;
  }
}
