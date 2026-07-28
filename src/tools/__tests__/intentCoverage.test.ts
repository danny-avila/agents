import { describe, it, expect } from '@jest/globals';
import type { CloudflareSandboxRuntime } from '@/types';
import {
  createLocalCodingTools,
  createLocalCodingToolDefinitions,
} from '../local/LocalCodingTools';
import {
  CLOUDFLARE_CODING_TOOL_NAMES,
  createCloudflareCodingTools,
} from '../cloudflare/CloudflareSandboxTools';
import {
  BashExecutionToolSchema,
  buildBashExecutionToolSchema,
} from '../BashExecutor';
import {
  CodeExecutionToolSchema,
  buildCodeExecutionToolSchema,
} from '../CodeExecutor';
import {
  SubagentToolSchema,
  createSubagentToolDefinition,
} from '../SubagentTool';
import { LOCAL_CODING_BUNDLE_NAMES } from '@/common';
import { BashProgrammaticToolCallingSchema } from '../BashProgrammaticToolCalling';
import { ProgrammaticToolCallingSchema } from '../ProgrammaticToolCalling';
import { WebSearchToolSchema } from '../search/schema';
import { ReadFileToolDefinition } from '../ReadFile';
import { ToolSearchToolSchema } from '../ToolSearch';
import { SkillToolDefinition } from '../SkillTool';
import { INTENT_ARG } from '../intentArg';

type SchemaLike = {
  properties?: Record<string, unknown>;
  required?: readonly string[];
};

function firstKey(schema: SchemaLike | undefined): string | undefined {
  return Object.keys(schema?.properties ?? {})[0];
}

function expectIntentFirst(name: string, schema: SchemaLike | undefined): void {
  expect(`${name}:${firstKey(schema)}`).toBe(`${name}:${INTENT_ARG}`);
  expect(schema?.required ?? []).not.toContain(INTENT_ARG);
}

const stubSandbox: CloudflareSandboxRuntime = {
  exec: () => Promise.reject(new Error('schema-only stub')),
  readFile: () => Promise.reject(new Error('schema-only stub')),
  writeFile: () => Promise.reject(new Error('schema-only stub')),
  mkdir: () => Promise.reject(new Error('schema-only stub')),
  listFiles: () => Promise.reject(new Error('schema-only stub')),
  deleteFile: () => Promise.reject(new Error('schema-only stub')),
};

/**
 * Coverage pin for the tool-intent capability: every native coding tool, on
 * every engine, must emit `intent` as the FIRST schema property (and never
 * require it). Mirrors the `LOCAL_CODING_BUNDLE_NAMES` pin so "did we get
 * them all" is a failing test rather than a review question.
 */
describe('intent coverage', () => {
  it('every local bundle tool emits intent as its first schema property', () => {
    const tools = createLocalCodingTools();
    expect(tools.map((tool) => tool.name).sort()).toEqual(
      [...LOCAL_CODING_BUNDLE_NAMES].sort()
    );
    for (const tool of tools) {
      expectIntentFirst(tool.name, tool.schema as SchemaLike);
    }
  });

  it('every local registry definition emits intent first', () => {
    for (const def of createLocalCodingToolDefinitions()) {
      expectIntentFirst(def.name, def.parameters);
    }
  });

  it('every cloudflare bundle tool emits intent first', () => {
    const tools = createCloudflareCodingTools({ sandbox: stubSandbox });
    expect(tools.map((tool) => tool.name).sort()).toEqual(
      [...CLOUDFLARE_CODING_TOOL_NAMES].sort()
    );
    for (const tool of tools) {
      expectIntentFirst(tool.name, tool.schema as SchemaLike);
    }
  });

  it('shared execution schemas emit intent first (all engines)', () => {
    expectIntentFirst('bash_tool', BashExecutionToolSchema);
    expectIntentFirst('execute_code', CodeExecutionToolSchema);
    expectIntentFirst(
      'bash_tool:stateful',
      buildBashExecutionToolSchema({ statefulSessions: true })
    );
    expectIntentFirst(
      'execute_code:stateful',
      buildCodeExecutionToolSchema({ statefulSessions: true })
    );
    expectIntentFirst(
      'run_tools_with_code',
      ProgrammaticToolCallingSchema
    );
    expectIntentFirst(
      'run_tools_with_bash',
      BashProgrammaticToolCallingSchema
    );
  });

  it('read_file, skill, subagent, tool_search, and web_search emit intent first', () => {
    expectIntentFirst('read_file', ReadFileToolDefinition.parameters);
    expectIntentFirst('skill', SkillToolDefinition.parameters);
    expectIntentFirst('subagent', SubagentToolSchema);
    expectIntentFirst(
      'subagent:runtime',
      createSubagentToolDefinition([
        { type: 'researcher', name: 'Researcher', description: 'Researches' },
      ]).parameters
    );
    expectIntentFirst('tool_search', ToolSearchToolSchema);
    expectIntentFirst('web_search', WebSearchToolSchema);
  });
});
