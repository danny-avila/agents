import { config } from 'dotenv';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ProgrammaticToolCallingJsonSchema } from './ptcTimeout';
import type * as t from '@/types';
import {
  BASH_SHELL_GUIDANCE,
  CODE_ARTIFACT_PATH_GUIDANCE,
  appendFailedExecutionFileReminder,
  buildCodeApiExecutionErrorMessage,
  buildCodeApiEndpoint,
  CodeApiRequestError,
  getCodeBaseURL,
  selectRuntimeSessionHint,
} from './CodeExecutor';
import {
  assertUnambiguousIdentifiers,
  projectProgrammaticToolMap,
  resolveProgrammaticToolDefinitions,
  selectProgrammaticTools,
  type ProgrammaticInvocationParams,
} from './ProgrammaticCallerPolicy';
import {
  clampCodeApiRunTimeoutMs,
  createCodeApiRunTimeoutSchema,
  resolveCodeApiRunTimeoutMs,
} from './ptcTimeout';
import {
  makeRequest,
  executeTools,
  runPlainExecution,
  formatCompletedResponse,
} from './ProgrammaticToolCalling';
import { logCodeApiDiagnostic } from '@/tools/diagnostics';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { Constants } from '@/common';

config();

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MAX_ROUND_TRIPS = 20;
const DEFAULT_RUN_TIMEOUT_MS = resolveCodeApiRunTimeoutMs();
const BASH_LAST_BACKGROUND_PID_GUARD = ': &\nwait "$!"';

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
- One call: state does not persist
- Tools are pre-defined as bash functions—DO NOT redefine them
- Each tool function accepts a JSON string argument
- Save tool output with raw=$(tool '{}'); printf '%s\n' "$raw" > /mnt/data/file.json; direct tool > file may be empty
- Tool stdout is normalized to one compact JSON value when possible; parse saved stdout once, then use fromjson? // . only for JSON-string fields
- Only echo/printf output returns to the model
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- timeout caps one sandbox run/replay iteration, not the total multi-round-trip workflow`;

const ADDITIONAL_RULES =
  '- Tool names normalized: hyphens→underscores, reserved words get `_tool` suffix';

const EXAMPLES = `Example (Complete workflow in one call):
  # Query data and process
  data=$(query_database '{"sql": "SELECT * FROM users"}')
  echo "$data" | jq '.[] | .name'

Example (Parallel calls):
  { sf=$(web_search '{"query": "SF weather"}'); printf '%s\n' "$sf" > /mnt/data/sf.json; } &
  { ny=$(web_search '{"query": "NY weather"}'); printf '%s\n' "$ny" > /mnt/data/ny.json; } &
  wait
  echo "SF: $(jq -r . /mnt/data/sf.json)"
  echo "NY: $(jq -r . /mnt/data/ny.json)"`;

const CODE_PARAM_DESCRIPTION = `Bash code that calls tools programmatically. Tools are available as bash functions.

${STATELESS_WARNING}

Each tool function accepts a JSON string as its argument.
Example: tool_name '{"key": "value"}'

${EXAMPLES}

${CORE_RULES}`;

const TOOL_MANIFEST_DESCRIPTION =
  'Exact registered tool names used by the code. Required when direct-only tools are configured; ' +
  'validated before execution starts. Pass [] when the code calls no tools at all.';

// ============================================================================
// Schema
// ============================================================================

export function createBashProgrammaticToolCallingSchema(
  maxRunTimeoutMs = DEFAULT_RUN_TIMEOUT_MS
): ProgrammaticToolCallingJsonSchema {
  return {
    type: 'object',
    properties: {
      intent: { ...INTENT_PROPERTY },
      code: {
        type: 'string',
        minLength: 1,
        description: CODE_PARAM_DESCRIPTION,
      },
      tool_manifest: {
        type: 'array',
        items: { type: 'string' },
        uniqueItems: true,
        description: TOOL_MANIFEST_DESCRIPTION,
      },
      timeout: createCodeApiRunTimeoutSchema(maxRunTimeoutMs),
    },
    required: ['code'],
  } as const;
}

export const BashProgrammaticToolCallingSchema =
  createBashProgrammaticToolCallingSchema();

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

function prepareBashProgrammaticCode(code: string): string {
  /* The Code API's generated Bash wrapper reads `$!` after user code. A user
   * `set -u` makes that expansion fail when no background process has run.
   * Seed and reap a no-op job before user code so strict mode remains active
   * for the payload while the wrapper can safely read its special parameter. */
  return `${BASH_LAST_BACKGROUND_PID_GUARD}\n${code}`;
}

function maybeParseJsonResultString(result: unknown): unknown {
  if (typeof result !== 'string') {
    return result;
  }

  const trimmed = result.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return result;
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return result;
  }
}

export function normalizeBashToolResultsForReplay(
  toolResults: t.PTCToolResult[]
): t.PTCToolResult[] {
  return toolResults.map((toolResult) => {
    if (toolResult.is_error) {
      return toolResult;
    }

    return {
      ...toolResult,
      result: maybeParseJsonResultString(toolResult.result),
    };
  });
}

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
  const maxRunTimeoutMs = resolveCodeApiRunTimeoutMs(initParams.runTimeoutMs);
  const proxy = initParams.proxy ?? process.env.PROXY;
  const debug = initParams.debug ?? process.env.BASH_PTC_DEBUG === 'true';
  const EXEC_ENDPOINT = buildCodeApiEndpoint(baseUrl, 'exec/programmatic');

  return tool(
    async (rawParams, config) => {
      const params = rawParams as ProgrammaticInvocationParams;
      const { code } = params;
      const preparedCode = prepareBashProgrammaticCode(code);
      const timeout = clampCodeApiRunTimeoutMs(params.timeout, maxRunTimeoutMs);

      const toolCall = (config.toolCall ?? {}) as ToolCall &
        Partial<t.ProgrammaticCache> & {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };
      const {
        toolMap,
        disallowedToolDefs,
        session_id,
        _injected_files,
        _runtime_session_hint,
      } = toolCall;
      const toolDefs = resolveProgrammaticToolDefinitions(
        toolCall as typeof toolCall & { tools?: t.LCTool[] }
      );

      const programmaticToolName =
        toolCall.programmaticToolName ??
        (typeof toolCall.name === 'string' && toolCall.name !== ''
          ? toolCall.name
          : Constants.BASH_PROGRAMMATIC_TOOL_CALLING);
      const effectiveTools = selectProgrammaticTools({
        requestedToolNames: params.tool_manifest,
        allowedToolDefs: toolDefs,
        disallowedToolDefs,
        programmaticToolName,
      });

      /* These guard the replay path only. A call that selected no tools never
       * replays, and event-driven ToolNode configurations legitimately inject
       * an empty toolMap/toolDefs. Injected context is still required to
       * conclude that — with nothing injected at all, an empty selection means
       * the host wired the tool up wrong. */
      const needsNoTools =
        effectiveTools.length === 0 &&
        (params.tool_manifest?.length === 0 ||
          toolDefs != null ||
          disallowedToolDefs != null);

      if (!needsNoTools) {
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
      }

      assertUnambiguousIdentifiers(
        effectiveTools,
        normalizeToBashIdentifier,
        programmaticToolName
      );

      const effectiveToolMap = projectProgrammaticToolMap(
        toolMap ?? new Map(),
        effectiveTools
      );

      let roundTrip = 0;

      try {
        // ====================================================================
        // Phase 1: Send the validated tool manifest with the initial request
        // ====================================================================

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[BashPTC Debug] Sending ${effectiveTools.length} tools to API ` +
              `(selected from ${toolDefs?.length ?? 0})`
          );
        }

        /* `/files/<session_id>` HTTP fallback removed — codeapi's
         * sessionAuth requires kind/id query params unavailable at
         * this point. See `CodeExecutor.ts` for full rationale. */
        let files: t.CodeEnvFile[] | undefined;
        if (_injected_files && _injected_files.length > 0) {
          files = _injected_files;
        } else if (session_id != null && session_id.length > 0) {
          logCodeApiDiagnostic(
            'BashProgrammaticToolCalling',
            'debug',
            'session carried no injected files; exec will run without input files',
            { files: 'none' }
          );
        }

        /* The hint rides the INITIAL request only; continuation_token binds
         * later round-trips. Prefer trusted per-agent factory context over
         * legacy ToolNode injection. Explicit default profiles always drop it.
         * BashPTC keeps its stateless runtime prompt in v1. */
        const selectedRuntimeSessionHint = selectRuntimeSessionHint(
          initParams.runtimeSessionHint,
          _runtime_session_hint
        );
        const runtimeSessionHint =
          initParams.executionProfile !== 'default' &&
          typeof selectedRuntimeSessionHint === 'string' &&
          selectedRuntimeSessionHint !== ''
            ? selectedRuntimeSessionHint
            : undefined;

        /* Raw `code`, not `preparedCode`: the `$!` guard exists for the
         * programmatic replay wrapper, which plain `/exec` never applies. */
        if (needsNoTools) {
          return await runPlainExecution({
            baseUrl,
            lang: 'bash',
            code,
            timeout,
            sessionId: session_id,
            files,
            runtimeSessionHint,
            proxy,
            authHeaders: initParams.authHeaders,
            executionProfile: initParams.executionProfile,
          });
        }

        let response = await makeRequest(
          EXEC_ENDPOINT,
          {
            lang: 'bash',
            code: preparedCode,
            tools: effectiveTools,
            session_id,
            timeout,
            ...(files && files.length > 0 ? { files } : {}),
            ...(runtimeSessionHint != null
              ? { runtime_session_hint: runtimeSessionHint }
              : {}),
          },
          proxy,
          initParams.authHeaders,
          initParams.executionProfile
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

          const toolResults = normalizeBashToolResultsForReplay(
            await executeTools(
              response.tool_calls ?? [],
              effectiveToolMap,
              Constants.BASH_PROGRAMMATIC_TOOL_CALLING
            )
          );

          response = await makeRequest(
            EXEC_ENDPOINT,
            {
              continuation_token: response.continuation_token,
              tool_results: toolResults,
            },
            proxy,
            initParams.authHeaders,
            initParams.executionProfile
          );
        }

        // ====================================================================
        // Phase 3: Handle final state
        // ====================================================================

        if (response.status === 'completed') {
          return formatCompletedResponse(response, code);
        }

        if (response.status === 'error') {
          throw new Error(buildCodeApiExecutionErrorMessage(response));
        }

        throw new CodeApiRequestError();
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          (error as Error).message,
          code
        );
        throw new Error(
          `Bash programmatic execution failed: ${messageWithReminder}`
        );
      }
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: BashProgrammaticToolCallingDescription,
      schema: createBashProgrammaticToolCallingSchema(maxRunTimeoutMs),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
