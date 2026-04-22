import { config } from 'dotenv';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import {
  makeRequest,
  executeTools,
  fetchSessionFiles,
  formatCompletedResponse,
} from './ProgrammaticToolCalling';
import { getCodeBaseURL } from './CodeExecutor';
import { Constants } from '@/common';

config();

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MAX_ROUND_TRIPS = 20;
const DEFAULT_TIMEOUT = 60000;

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
- Only echo/printf output returns to the model
- Generated files are automatically available in /mnt/data/ for subsequent executions`;

const ADDITIONAL_RULES =
  '- Tool names normalized: hyphens→underscores, reserved words get `_tool` suffix';

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
 * Extracts tool names that are actually called in the bash code.
 * Bash functions are invoked as commands (no parentheses), so we match
 * the normalized name as a word boundary.
 */
export function extractUsedBashToolNames(
  code: string,
  toolNameMap: Map<string, string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const [bashName, originalName] of toolNameMap) {
    const escapedName = bashName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\b`, 'g');

    if (pattern.test(code)) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

/**
 * Filters tool definitions to only include tools actually used in the bash code.
 */
export function filterBashToolsByUsage(
  toolDefs: t.LCTool[],
  code: string,
  debug = false
): t.LCTool[] {
  const toolNameMap = new Map<string, string>();
  for (const def of toolDefs) {
    const bashName = normalizeToBashIdentifier(def.name);
    toolNameMap.set(bashName, def.name);
  }

  const usedToolNames = extractUsedBashToolNames(code, toolNameMap);

  if (debug) {
    // eslint-disable-next-line no-console
    console.log(
      `[BashPTC Debug] Tool filtering: found ${usedToolNames.size}/${toolDefs.length} tools in code`
    );
    if (usedToolNames.size > 0) {
      // eslint-disable-next-line no-console
      console.log(
        `[BashPTC Debug] Matched tools: ${Array.from(usedToolNames).join(', ')}`
      );
    }
  }

  if (usedToolNames.size === 0) {
    if (debug) {
      // eslint-disable-next-line no-console
      console.log(
        '[BashPTC Debug] No tools detected in code - sending all tools as fallback'
      );
    }
    return toolDefs;
  }

  return toolDefs.filter((def) => usedToolNames.has(def.name));
}

// ============================================================================
// Tool Factory
// ============================================================================

/**
 * Creates a Bash Programmatic Tool Calling tool for multi-tool orchestration.
 *
 * This tool enables AI agents to write bash scripts that orchestrate multiple
 * tool calls programmatically via the remote Code API, reducing LLM round-trips.
 *
 * The tool map must be provided at runtime via config.toolCall (injected by ToolNode).
 */
export function createBashProgrammaticToolCallingTool(
  initParams: t.BashProgrammaticToolCallingParams = {}
): DynamicStructuredTool {
  const baseUrl = initParams.baseUrl ?? getCodeBaseURL();
  const maxRoundTrips = initParams.maxRoundTrips ?? DEFAULT_MAX_ROUND_TRIPS;
  const proxy = initParams.proxy ?? process.env.PROXY;
  const debug = initParams.debug ?? process.env.BASH_PTC_DEBUG === 'true';
  const EXEC_ENDPOINT = `${baseUrl}/exec/programmatic`;

  return tool(
    async (rawParams, config) => {
      const params = rawParams as { code: string; timeout?: number };
      const { code, timeout = DEFAULT_TIMEOUT } = params;

      const { toolMap, toolDefs, session_id, _injected_files } =
        (config.toolCall ?? {}) as ToolCall &
          Partial<t.ProgrammaticCache> & {
            session_id?: string;
            _injected_files?: t.CodeEnvFile[];
          };

      if (toolMap == null || toolMap.size === 0) {
        throw new Error(
          'No toolMap provided. ' +
            'ToolNode should inject this from AgentContext when invoked through the graph.'
        );
      }

      if (toolDefs == null || toolDefs.length === 0) {
        throw new Error(
          'No tool definitions provided. ' +
            'Either pass tools in the input or ensure ToolNode injects toolDefs.'
        );
      }

      let roundTrip = 0;

      try {
        // ====================================================================
        // Phase 1: Filter tools and make initial request
        // ====================================================================

        const effectiveTools = filterBashToolsByUsage(toolDefs, code, debug);

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[BashPTC Debug] Sending ${effectiveTools.length} tools to API ` +
              `(filtered from ${toolDefs.length})`
          );
        }

        let files: t.CodeEnvFile[] | undefined;
        if (_injected_files && _injected_files.length > 0) {
          files = _injected_files;
        } else if (session_id != null && session_id.length > 0) {
          files = await fetchSessionFiles(baseUrl, session_id, proxy);
        }

        let response = await makeRequest(
          EXEC_ENDPOINT,
          {
            lang: 'bash',
            code,
            tools: effectiveTools,
            session_id,
            timeout,
            ...(files && files.length > 0 ? { files } : {}),
          },
          proxy
        );

        // ====================================================================
        // Phase 2: Handle response loop
        // ====================================================================

        while (response.status === 'tool_call_required') {
          roundTrip++;

          if (roundTrip > maxRoundTrips) {
            throw new Error(
              `Exceeded maximum round trips (${maxRoundTrips}). ` +
                'This may indicate an infinite loop, excessive tool calls, ' +
                'or a logic error in your code.'
            );
          }

          if (debug) {
            // eslint-disable-next-line no-console
            console.log(
              `[BashPTC Debug] Round trip ${roundTrip}: ${response.tool_calls?.length ?? 0} tool(s) to execute`
            );
          }

          const toolResults = await executeTools(
            response.tool_calls ?? [],
            toolMap
          );

          response = await makeRequest(
            EXEC_ENDPOINT,
            {
              continuation_token: response.continuation_token,
              tool_results: toolResults,
            },
            proxy
          );
        }

        // ====================================================================
        // Phase 3: Handle final state
        // ====================================================================

        if (response.status === 'completed') {
          return formatCompletedResponse(response);
        }

        if (response.status === 'error') {
          throw new Error(
            `Execution error: ${response.error}` +
              (response.stderr != null && response.stderr !== ''
                ? `\n\nStderr:\n${response.stderr}`
                : '')
          );
        }

        throw new Error(`Unexpected response status: ${response.status}`);
      } catch (error) {
        throw new Error(
          `Bash programmatic execution failed: ${(error as Error).message}`
        );
      }
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: BashProgrammaticToolCallingDescription,
      schema: BashProgrammaticToolCallingSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
