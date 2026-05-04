import { dirname } from 'path';
import { readdir, readFile, stat, writeFile, mkdir } from 'fs/promises';
import { tool } from '@langchain/core/tools';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  createLocalBashExecutionTool,
  createLocalCodeExecutionTool,
} from './LocalExecutionTools';
import {
  createLocalBashProgrammaticToolCallingTool,
  createLocalProgrammaticToolCallingTool,
} from './LocalProgrammaticToolCalling';
import {
  resolveWorkspacePath,
  spawnLocalProcess,
  truncateLocalOutput,
} from './LocalExecutionEngine';
import { Constants } from '@/common';

const MAX_READ_CHARS = 256000;
const DEFAULT_MAX_RESULTS = 200;

export const LocalWriteFileToolName = 'write_file';
export const LocalEditFileToolName = 'edit_file';
export const LocalGrepSearchToolName = 'grep_search';
export const LocalGlobSearchToolName = 'glob_search';
export const LocalListDirectoryToolName = 'list_directory';

export const LocalReadFileToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    file_path: {
      type: 'string',
      description: 'Path to a local file, relative to the configured cwd unless absolute paths are allowed.',
    },
    offset: {
      type: 'integer',
      description: 'Optional 1-indexed line offset for large files.',
    },
    limit: {
      type: 'integer',
      description: 'Optional maximum number of lines to return.',
    },
  },
  required: ['file_path'],
};

export const LocalWriteFileToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    file_path: {
      type: 'string',
      description: 'Path to write, relative to the configured cwd unless absolute paths are allowed.',
    },
    content: {
      type: 'string',
      description: 'Complete file contents to write.',
    },
  },
  required: ['file_path', 'content'],
};

export const LocalEditFileToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    file_path: {
      type: 'string',
      description: 'Path to edit, relative to the configured cwd unless absolute paths are allowed.',
    },
    old_text: {
      type: 'string',
      description: 'Exact text to replace. Must appear exactly once.',
    },
    new_text: {
      type: 'string',
      description: 'Replacement text.',
    },
    edits: {
      type: 'array',
      description: 'Optional batch of exact replacements. Each old_text must appear exactly once in the original file.',
      items: {
        type: 'object',
        properties: {
          old_text: { type: 'string' },
          new_text: { type: 'string' },
        },
        required: ['old_text', 'new_text'],
      },
    },
  },
  required: ['file_path'],
};

export const LocalGrepSearchToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    pattern: {
      type: 'string',
      description: 'Regex pattern to search for.',
    },
    path: {
      type: 'string',
      description: 'Directory or file to search. Defaults to cwd.',
    },
    glob: {
      type: 'string',
      description: 'Optional file glob passed to rg -g.',
    },
    max_results: {
      type: 'integer',
      description: 'Maximum matching lines to return.',
    },
  },
  required: ['pattern'],
};

export const LocalGlobSearchToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    pattern: {
      type: 'string',
      description: 'File glob pattern, for example "src/**/*.ts".',
    },
    path: {
      type: 'string',
      description: 'Directory to search. Defaults to cwd.',
    },
    max_results: {
      type: 'integer',
      description: 'Maximum file paths to return.',
    },
  },
  required: ['pattern'],
};

export const LocalListDirectoryToolSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    path: {
      type: 'string',
      description: 'Directory to list. Defaults to cwd.',
    },
  },
};

function lineWindow(
  content: string,
  offset?: number,
  limit?: number
): { text: string; truncated: boolean } {
  const lines = content.split('\n');
  const start = Math.max((offset ?? 1) - 1, 0);
  const end = limit != null && limit > 0 ? start + limit : lines.length;
  const selected = lines.slice(start, end);
  const numbered = selected
    .map((line, index) => `${String(start + index + 1).padStart(6, ' ')}\t${line}`)
    .join('\n');
  return {
    text: truncateLocalOutput(numbered, MAX_READ_CHARS),
    truncated: end < lines.length || numbered.length > MAX_READ_CHARS,
  };
}

function countOccurrences(content: string, needle: string): number {
  if (needle === '') {
    return 0;
  }
  let count = 0;
  let index = content.indexOf(needle);
  while (index !== -1) {
    count++;
    index = content.indexOf(needle, index + needle.length);
  }
  return count;
}

function normalizeEdits(input: {
  old_text?: string;
  new_text?: string;
  edits?: Array<{ old_text?: string; new_text?: string }>;
}): Array<{ oldText: string; newText: string }> {
  const edits = Array.isArray(input.edits)
    ? input.edits.map((edit) => ({
      oldText: edit.old_text ?? '',
      newText: edit.new_text ?? '',
    }))
    : [];

  if (input.old_text != null || input.new_text != null) {
    edits.push({
      oldText: input.old_text ?? '',
      newText: input.new_text ?? '',
    });
  }

  return edits;
}

function toolDefinition(
  name: string,
  description: string,
  parameters: t.JsonSchemaType
): t.LCTool {
  return {
    name,
    description,
    parameters,
    allowed_callers: ['direct', 'code_execution'],
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
    toolType: 'builtin',
  };
}

export function createLocalReadFileTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        file_path: string;
        offset?: number;
        limit?: number;
      };
      const path = resolveWorkspacePath(input.file_path, config);
      const fileStat = await stat(path);
      if (!fileStat.isFile()) {
        throw new Error(`Path is not a file: ${input.file_path}`);
      }
      const content = await readFile(path, 'utf8');
      const result = lineWindow(content, input.offset, input.limit);
      return [
        result.truncated ? `${result.text}\n[truncated]` : result.text,
        { path, bytes: fileStat.size },
      ];
    },
    {
      name: Constants.READ_FILE,
      description:
        'Read a local text file from the configured working directory with line numbers.',
      schema: LocalReadFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalWriteFileTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as { file_path: string; content: string };
      if (config.readOnly === true) {
        throw new Error('write_file is blocked in read-only local mode.');
      }
      const path = resolveWorkspacePath(input.file_path, config);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, input.content, 'utf8');
      return [`Wrote ${input.content.length} characters to ${path}`, { path }];
    },
    {
      name: LocalWriteFileToolName,
      description:
        'Create or overwrite a local text file in the configured working directory.',
      schema: LocalWriteFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalEditFileTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        file_path: string;
        old_text?: string;
        new_text?: string;
        edits?: Array<{ old_text?: string; new_text?: string }>;
      };
      if (config.readOnly === true) {
        throw new Error('edit_file is blocked in read-only local mode.');
      }
      const edits = normalizeEdits(input);
      if (edits.length === 0) {
        throw new Error('edit_file requires old_text/new_text or edits[].');
      }

      const path = resolveWorkspacePath(input.file_path, config);
      const original = await readFile(path, 'utf8');
      let next = original;
      for (const edit of edits) {
        const count = countOccurrences(next, edit.oldText);
        if (count !== 1) {
          throw new Error(
            `Expected old_text to appear exactly once in ${input.file_path}, found ${count}.`
          );
        }
        next = next.replace(edit.oldText, edit.newText);
      }
      await writeFile(path, next, 'utf8');
      return [`Applied ${edits.length} edit(s) to ${path}`, { path }];
    },
    {
      name: LocalEditFileToolName,
      description:
        'Apply exact text replacements to a local file. Each old_text must match exactly once.',
      schema: LocalEditFileToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalGrepSearchTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        pattern: string;
        path?: string;
        glob?: string;
        max_results?: number;
      };
      const target = resolveWorkspacePath(input.path ?? '.', config);
      const maxResults = Math.max(input.max_results ?? DEFAULT_MAX_RESULTS, 1);
      const args = [
        '--line-number',
        '--column',
        '--hidden',
        '--glob',
        '!.git/**',
        ...(input.glob != null && input.glob !== '' ? ['--glob', input.glob] : []),
        input.pattern,
        target,
      ];
      const result = await spawnLocalProcess('rg', args, {
        ...config,
        timeoutMs: config.timeoutMs ?? 30000,
      });
      const lines = result.stdout.split('\n').filter(Boolean).slice(0, maxResults);
      const output =
        lines.length > 0
          ? lines.join('\n')
          : result.stderr.trim() || 'No matches found.';
      return [output, { matches: lines.length }];
    },
    {
      name: LocalGrepSearchToolName,
      description: 'Search local files with ripgrep and return matching lines.',
      schema: LocalGrepSearchToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalGlobSearchTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        pattern: string;
        path?: string;
        max_results?: number;
      };
      const target = resolveWorkspacePath(input.path ?? '.', config);
      const maxResults = Math.max(input.max_results ?? DEFAULT_MAX_RESULTS, 1);
      const result = await spawnLocalProcess(
        'rg',
        ['--files', '--hidden', '--glob', '!.git/**', '--glob', input.pattern, target],
        { ...config, timeoutMs: config.timeoutMs ?? 30000 }
      );
      const lines = result.stdout.split('\n').filter(Boolean).slice(0, maxResults);
      return [lines.length > 0 ? lines.join('\n') : 'No files found.', { files: lines }];
    },
    {
      name: LocalGlobSearchToolName,
      description: 'Find local files matching a glob pattern.',
      schema: LocalGlobSearchToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalListDirectoryTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as { path?: string };
      const path = resolveWorkspacePath(input.path ?? '.', config);
      const entries = await readdir(path, { withFileTypes: true });
      const output = entries
        .map((entry) => `${entry.isDirectory() ? 'dir ' : 'file'}\t${entry.name}`)
        .join('\n');
      return [output || 'Directory is empty.', { path, count: entries.length }];
    },
    {
      name: LocalListDirectoryToolName,
      description: 'List files and directories in a local directory.',
      schema: LocalListDirectoryToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalCodingTools(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool[] {
  return [
    createLocalReadFileTool(config),
    createLocalWriteFileTool(config),
    createLocalEditFileTool(config),
    createLocalGrepSearchTool(config),
    createLocalGlobSearchTool(config),
    createLocalListDirectoryTool(config),
    createLocalBashExecutionTool({ config }),
    createLocalCodeExecutionTool(config),
    createLocalProgrammaticToolCallingTool(config),
    createLocalBashProgrammaticToolCallingTool(config),
  ];
}

export function createLocalCodingToolDefinitions(): t.LCTool[] {
  return [
    toolDefinition(
      Constants.READ_FILE,
      'Read a local text file from the configured working directory with line numbers.',
      LocalReadFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalWriteFileToolName,
      'Create or overwrite a local text file in the configured working directory.',
      LocalWriteFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalEditFileToolName,
      'Apply exact text replacements to a local file.',
      LocalEditFileToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalGrepSearchToolName,
      'Search local files with ripgrep and return matching lines.',
      LocalGrepSearchToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalGlobSearchToolName,
      'Find local files matching a glob pattern.',
      LocalGlobSearchToolSchema as t.JsonSchemaType
    ),
    toolDefinition(
      LocalListDirectoryToolName,
      'List files and directories in a local directory.',
      LocalListDirectoryToolSchema as t.JsonSchemaType
    ),
  ];
}

export function createLocalCodingToolRegistry(): t.LCToolRegistry {
  return new Map(
    createLocalCodingToolDefinitions().map((definition) => [
      definition.name,
      definition,
    ])
  );
}
