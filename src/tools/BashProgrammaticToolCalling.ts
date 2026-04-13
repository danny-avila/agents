import { createServer } from 'http';
import { spawn } from 'child_process';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { Server } from 'http';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { unwrapToolResponse } from './ProgrammaticToolCalling';
import { Constants } from '@/common';

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_TIMEOUT = 60000;
const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

/** Bash reserved words that get `_tool` suffix when used as function names */
const BASH_RESERVED = new Set([
  'if',
  'then',
  'else',
  'elif',
  'fi',
  'case',
  'esac',
  'for',
  'while',
  'until',
  'do',
  'done',
  'in',
  'function',
  'select',
  'time',
  'coproc',
  'declare',
  'typeset',
  'local',
  'readonly',
  'export',
  'unset',
]);

// ============================================================================
// Description Components
// ============================================================================

const STATELESS_WARNING = `CRITICAL - STATELESS EXECUTION:
Each call is a fresh bash shell. Variables and state do NOT persist between calls.
You MUST complete your entire workflow in ONE code block.
DO NOT split work across multiple calls expecting to reuse variables.`;

const CORE_RULES = `Rules:
- EVERYTHING in one call—no state persists between executions
- Tools are pre-defined as bash functions—DO NOT redefine them
- Each tool function accepts a JSON string argument
- Only stdout output returns to the model—use echo/printf for all output
- Use jq for JSON parsing if available`;

const ADDITIONAL_RULES = `- Tool names normalized: hyphens→underscores, reserved words get \`_tool\` suffix
- Requires curl for tool function calls`;

const EXAMPLES = `Example (Complete workflow in one call):
  # Query data and process
  data=$(query_database '{"sql": "SELECT * FROM users"}')
  echo "$data" | jq '.[] | .name'

Example (Parallel calls):
  web_search '{"query": "SF weather"}' > /tmp/sf.txt &
  web_search '{"query": "NY weather"}' > /tmp/ny.txt &
  wait
  echo "SF: $(cat /tmp/sf.txt)"
  echo "NY: $(cat /tmp/ny.txt)"`;

const CODE_PARAM_DESCRIPTION = `Bash code that calls tools programmatically. Tools are available as bash functions.

${STATELESS_WARNING}

Each tool function accepts a JSON string as its argument.
Example: tool_name '{"key": "value"}'

${EXAMPLES}

${CORE_RULES}`;

// ============================================================================
// Schema
// ============================================================================

export const BashProgrammaticToolCallingSchema = {
  type: 'object',
  properties: {
    code: {
      type: 'string',
      minLength: 1,
      description: CODE_PARAM_DESCRIPTION,
    },
    timeout: {
      type: 'integer',
      minimum: 1000,
      maximum: 300000,
      default: DEFAULT_TIMEOUT,
      description:
        'Maximum execution time in milliseconds. Default: 60 seconds. Max: 5 minutes.',
    },
  },
  required: ['code'],
} as const;

export const BashProgrammaticToolCallingName =
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING;

export const BashProgrammaticToolCallingDescription = `
Run tools via bash code. Tools are available as bash functions that accept JSON string arguments.

${STATELESS_WARNING}

${CORE_RULES}
${ADDITIONAL_RULES}

When to use: shell pipelines, parallel execution (& and wait), file processing, text manipulation.

${EXAMPLES}
`.trim();

export const BashProgrammaticToolCallingDefinition = {
  name: BashProgrammaticToolCallingName,
  description: BashProgrammaticToolCallingDescription,
  schema: BashProgrammaticToolCallingSchema,
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Normalizes a tool name to a valid bash function identifier.
 * 1. Replace hyphens, spaces, dots with underscores
 * 2. Remove any other invalid characters
 * 3. Prefix with underscore if starts with number
 * 4. Append `_tool` if it's a bash reserved word
 */
export function normalizeToBashIdentifier(name: string): string {
  let normalized = name.replace(/[-\s.]/g, '_');
  normalized = normalized.replace(/[^a-zA-Z0-9_]/g, '');

  if (/^[0-9]/.test(normalized)) {
    normalized = '_' + normalized;
  }

  if (BASH_RESERVED.has(normalized)) {
    normalized = normalized + '_tool';
  }

  return normalized;
}

/**
 * Generates a bash function for a tool that calls the local HTTP tool server.
 * The function accepts a JSON string argument and returns the tool output.
 */
function generateToolFunction(toolDef: t.LCTool, port: number): string {
  const bashName = normalizeToBashIdentifier(toolDef.name);
  const escapedName = toolDef.name.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `${bashName}() {
  local __input="\${1:-{}}"
  printf '{"name":"${escapedName}","input":%s}' "$__input" | \\
    curl -s -X POST "http://127.0.0.1:${port}/tool_call" \\
      -H "Content-Type: application/json" -d @-
}`;
}

/**
 * Generates the full bash script with tool functions and user code.
 */
function generateBashScript(
  toolDefs: t.LCTool[],
  port: number,
  code: string
): string {
  const header = '#!/usr/bin/env bash';
  const curlCheck = `if ! command -v curl &>/dev/null; then
  echo "Error: curl is required for tool function calls but was not found." >&2
  exit 1
fi`;

  const portExport = `export __TOOL_PORT=${port}`;
  const functions = toolDefs
    .map((td) => generateToolFunction(td, port))
    .join('\n\n');

  return [
    header,
    '',
    curlCheck,
    '',
    portExport,
    '',
    '# Tool functions',
    functions,
    '',
    '# User code',
    code,
  ].join('\n');
}

/**
 * Starts a local HTTP server that handles tool invocation requests.
 * Binds to 127.0.0.1 on a random available port.
 */
async function startToolServer(
  toolMap: t.ToolMap
): Promise<{ server: Server; port: number }> {
  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      if (req.method !== 'POST' || req.url !== '/tool_call') {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      let body = '';
      req.on('data', (chunk: Buffer) => {
        body += chunk.toString();
      });

      req.on('end', () => {
        void (async (): Promise<void> => {
          try {
            const parsed = JSON.parse(body) as {
              name: string;
              input: Record<string, unknown>;
            };
            const matchedTool = toolMap.get(parsed.name);

            if (!matchedTool) {
              res.writeHead(404);
              res.end(
                `Tool '${parsed.name}' not found. Available: ${Array.from(toolMap.keys()).join(', ')}`
              );
              return;
            }

            const result = await matchedTool.invoke(parsed.input, {
              metadata: { [Constants.BASH_PROGRAMMATIC_TOOL_CALLING]: true },
            });

            const isMCPTool = matchedTool.mcp === true;
            const unwrapped = unwrapToolResponse(result, isMCPTool);
            const output =
              typeof unwrapped === 'string'
                ? unwrapped
                : JSON.stringify(unwrapped);

            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end(output);
          } catch (error) {
            res.writeHead(500);
            res.end(`Tool execution error: ${(error as Error).message}`);
          }
        })();
      });
    });

    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (address == null || typeof address === 'string') {
        reject(new Error('Failed to get tool server port'));
        return;
      }
      resolve({ server, port: address.port });
    });
  });
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve) => {
    server.close(() => resolve());
    setTimeout(resolve, 5000);
  });
}

function executeBashScript(
  script: string,
  options: { bashPath: string; timeout: number; workDir?: string }
): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(options.bashPath, ['-c', script], {
      timeout: options.timeout,
      cwd: options.workDir,
      env: { ...process.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('error', (err: Error) => {
      reject(new Error(`Failed to start bash process: ${err.message}`));
    });

    proc.on('close', (code: number | null, signal: string | null) => {
      if (signal != null) {
        stderr += `\nProcess terminated by signal: ${signal}\n`;
      }
      resolve({ stdout, stderr, exitCode: code });
    });
  });
}

// ============================================================================
// Tool Factory
// ============================================================================

/**
 * Creates a Bash Programmatic Tool Calling tool for multi-tool orchestration.
 *
 * This tool enables AI agents to write bash scripts that call multiple tools
 * as bash functions. A local HTTP server handles tool invocations via curl,
 * supporting loops, conditionals, pipelines, and parallel execution.
 *
 * The tool map must be provided at runtime via config.toolCall (injected by ToolNode).
 *
 * @param initParams - Configuration parameters (bashPath, timeout, workDir, debug)
 * @returns A LangChain DynamicStructuredTool for bash programmatic tool calling
 */
export function createBashProgrammaticToolCallingTool(
  initParams: t.BashProgrammaticToolCallingParams = {}
): DynamicStructuredTool {
  const bashPath = initParams.bashPath ?? 'bash';
  const defaultTimeout = initParams.defaultTimeout ?? DEFAULT_TIMEOUT;
  const workDir = initParams.workDir;
  const debug = initParams.debug ?? process.env.BASH_PTC_DEBUG === 'true';

  return tool(
    async (rawParams, config) => {
      const params = rawParams as { code: string; timeout?: number };
      const { code, timeout = defaultTimeout } = params;

      const { toolMap, toolDefs } = (config.toolCall ?? {}) as ToolCall &
        Partial<t.ProgrammaticCache>;

      if (toolMap == null || toolMap.size === 0) {
        throw new Error(
          'No toolMap provided. ' +
            'ToolNode should inject this from AgentContext when invoked through the graph.'
        );
      }

      if (toolDefs == null || toolDefs.length === 0) {
        throw new Error(
          'No tool definitions provided. ' + 'Ensure ToolNode injects toolDefs.'
        );
      }

      const { server, port } = await startToolServer(toolMap);

      try {
        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[BashPTC Debug] Started tool server on port ${port} with ${toolMap.size} tools`
          );
        }

        const script = generateBashScript(toolDefs, port, code);

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(`[BashPTC Debug] Generated script:\n${script}`);
        }

        const { stdout, stderr, exitCode } = await executeBashScript(script, {
          bashPath,
          timeout,
          workDir,
        });

        let formatted = '';
        if (stdout) {
          formatted += `stdout:\n${stdout}\n`;
        } else {
          formatted += emptyOutputMessage;
        }
        if (stderr) {
          formatted += `stderr:\n${stderr}\n`;
        }
        if (exitCode !== 0 && exitCode !== null) {
          formatted += `exit code: ${exitCode}\n`;
        }

        return formatted.trim();
      } catch (error) {
        throw new Error(
          `Bash programmatic execution failed: ${(error as Error).message}`
        );
      } finally {
        await closeServer(server);
        if (debug) {
          // eslint-disable-next-line no-console
          console.log('[BashPTC Debug] Tool server stopped');
        }
      }
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: BashProgrammaticToolCallingDescription,
      schema: BashProgrammaticToolCallingSchema,
    }
  );
}
