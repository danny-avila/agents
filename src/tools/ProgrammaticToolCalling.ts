// src/tools/ProgrammaticToolCalling.ts
import { z } from 'zod';
import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { imageExtRegex, getCodeBaseURL } from './CodeExecutor';
import { EnvVar, Constants } from '@/common';

config();

// ============================================================================
// Constants
// ============================================================================

const imageMessage = 'Image is already displayed to the user';
const otherMessage = 'File is already downloaded by the user';
const accessMessage =
  'Note: Files are READ-ONLY. Save changes to NEW filenames. To access these files in future executions, provide the `session_id` as a parameter (not in your code).';
const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

/** Default max round-trips to prevent infinite loops */
const DEFAULT_MAX_ROUND_TRIPS = 20;

/** Default execution timeout in milliseconds */
const DEFAULT_TIMEOUT = 60000;

// ============================================================================
// Schema
// ============================================================================

const ProgrammaticToolCallingSchema = z.object({
  code: z
    .string()
    .min(1)
    .describe(
      `Python code that calls tools programmatically. Tools are automatically available as async Python functions - DO NOT define them yourself.

The Code API generates async function stubs from the tool definitions. Just call them directly:

Example (Simple call):
  result = await get_weather(city="San Francisco")
  print(result)

Example (Parallel - Fastest):
  results = await asyncio.gather(
      get_weather(city="SF"),
      get_weather(city="NYC"),
      get_weather(city="London")
  )
  for city, weather in zip(["SF", "NYC", "London"], results):
      print(f"{city}: {weather['temperature']}°F")

Example (Loop with processing):
  team = await get_team_members()
  for member in team:
      expenses = await get_expenses(user_id=member['id'])
      total = sum(e['amount'] for e in expenses)
      print(f"{member['name']}: \${total:.2f}")

Example (Conditional logic):
  data = await fetch_data(source="primary")
  if not data:
      data = await fetch_data(source="backup")
  print(f"Got {len(data)} records")

Requirements:
- Tools are pre-defined as async functions - DO NOT write function definitions
- Use await for all tool calls
- Use asyncio.gather() for parallel execution of independent calls
- Only print() output flows back to the context window
- Tool results from programmatic calls do NOT consume context tokens`
    ),
  session_id: z
    .string()
    .optional()
    .describe(
      'Session ID for file access (same as regular code execution). Files load into /mnt/data/ and are READ-ONLY.'
    ),
  timeout: z
    .number()
    .int()
    .min(1000)
    .max(300000)
    .optional()
    .default(DEFAULT_TIMEOUT)
    .describe(
      'Maximum execution time in milliseconds. Default: 60 seconds. Max: 5 minutes.'
    ),
});

// ============================================================================
// Helper Functions
// ============================================================================

/** Python reserved keywords that get `_tool` suffix in Code API */
const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
]);

/**
 * Normalizes a tool name to Python identifier format.
 * Must match the Code API's `normalizePythonFunctionName` exactly:
 * 1. Replace hyphens and spaces with underscores
 * 2. Remove any other invalid characters
 * 3. Prefix with underscore if starts with number
 * 4. Append `_tool` if it's a Python keyword
 * @param name - The tool name to normalize
 * @returns Normalized Python-safe identifier
 */
export function normalizeToPythonIdentifier(name: string): string {
  let normalized = name.replace(/[-\s]/g, '_');

  normalized = normalized.replace(/[^a-zA-Z0-9_]/g, '');

  if (/^[0-9]/.test(normalized)) {
    normalized = '_' + normalized;
  }

  if (PYTHON_KEYWORDS.has(normalized)) {
    normalized = normalized + '_tool';
  }

  return normalized;
}

/**
 * Extracts tool names that are actually called in the Python code.
 * Handles hyphen/underscore conversion since Python identifiers use underscores.
 * @param code - The Python code to analyze
 * @param toolNameMap - Map from normalized Python name to original tool name
 * @returns Set of original tool names found in the code
 */
export function extractUsedToolNames(
  code: string,
  toolNameMap: Map<string, string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const [pythonName, originalName] of toolNameMap) {
    const escapedName = pythonName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\s*\\(`, 'g');

    if (pattern.test(code)) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

/**
 * Filters tool definitions to only include tools actually used in the code.
 * Handles the hyphen-to-underscore conversion for Python compatibility.
 * @param toolDefs - All available tool definitions
 * @param code - The Python code to analyze
 * @param debug - Enable debug logging
 * @returns Filtered array of tool definitions
 */
export function filterToolsByUsage(
  toolDefs: t.LCTool[],
  code: string,
  debug = false
): t.LCTool[] {
  const toolNameMap = new Map<string, string>();
  for (const tool of toolDefs) {
    const pythonName = normalizeToPythonIdentifier(tool.name);
    toolNameMap.set(pythonName, tool.name);
  }

  const usedToolNames = extractUsedToolNames(code, toolNameMap);

  if (debug) {
    // eslint-disable-next-line no-console
    console.log(
      `[PTC Debug] Tool filtering: found ${usedToolNames.size}/${toolDefs.length} tools in code`
    );
    if (usedToolNames.size > 0) {
      // eslint-disable-next-line no-console
      console.log(
        `[PTC Debug] Matched tools: ${Array.from(usedToolNames).join(', ')}`
      );
    }
  }

  if (usedToolNames.size === 0) {
    if (debug) {
      // eslint-disable-next-line no-console
      console.log(
        '[PTC Debug] No tools detected in code - sending all tools as fallback'
      );
    }
    return toolDefs;
  }

  return toolDefs.filter((tool) => usedToolNames.has(tool.name));
}

/**
 * Fetches files from a previous session to make them available for the current execution.
 * Files are returned as CodeEnvFile references to be included in the request.
 * @param baseUrl - The base URL for the Code API
 * @param apiKey - The API key for authentication
 * @param sessionId - The session ID to fetch files from
 * @param proxy - Optional HTTP proxy URL
 * @returns Array of CodeEnvFile references, or empty array if fetch fails
 */
export async function fetchSessionFiles(
  baseUrl: string,
  apiKey: string,
  sessionId: string,
  proxy?: string
): Promise<t.CodeEnvFile[]> {
  try {
    const filesEndpoint = `${baseUrl}/files/${sessionId}?detail=full`;
    const fetchOptions: RequestInit = {
      method: 'GET',
      headers: {
        'User-Agent': 'LibreChat/1.0',
        'X-API-Key': apiKey,
      },
    };

    if (proxy != null && proxy !== '') {
      fetchOptions.agent = new HttpsProxyAgent(proxy);
    }

    const response = await fetch(filesEndpoint, fetchOptions);
    if (!response.ok) {
      throw new Error(`Failed to fetch files for session: ${response.status}`);
    }

    const files = await response.json();
    if (!Array.isArray(files) || files.length === 0) {
      return [];
    }

    return files.map((file: Record<string, unknown>) => {
      // Extract the ID from the file name (part after session ID prefix and before extension)
      const nameParts = (file.name as string).split('/');
      const id = nameParts.length > 1 ? nameParts[1].split('.')[0] : '';

      return {
        session_id: sessionId,
        id,
        name: (file.metadata as Record<string, unknown>)[
          'original-filename'
        ] as string,
      };
    });
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(
      `Failed to fetch files for session: ${sessionId}, ${(error as Error).message}`
    );
    return [];
  }
}

/**
 * Makes an HTTP request to the Code API.
 * @param endpoint - The API endpoint URL
 * @param apiKey - The API key for authentication
 * @param body - The request body
 * @param proxy - Optional HTTP proxy URL
 * @returns The parsed API response
 */
export async function makeRequest(
  endpoint: string,
  apiKey: string,
  body: Record<string, unknown>,
  proxy?: string
): Promise<t.ProgrammaticExecutionResponse> {
  const fetchOptions: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'LibreChat/1.0',
      'X-API-Key': apiKey,
    },
    body: JSON.stringify(body),
  };

  if (proxy != null && proxy !== '') {
    fetchOptions.agent = new HttpsProxyAgent(proxy);
  }

  const response = await fetch(endpoint, fetchOptions);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `HTTP error! status: ${response.status}, body: ${errorText}`
    );
  }

  return (await response.json()) as t.ProgrammaticExecutionResponse;
}

/**
 * Unwraps tool responses that may be formatted as tuples.
 * MCP tools return [content, artifacts], we need to extract the raw data.
 * @param result - The raw result from tool.invoke()
 * @param isMCPTool - Whether this is an MCP tool (has mcp property)
 * @returns Unwrapped raw data (string, object, or parsed JSON)
 */
export function unwrapToolResponse(
  result: unknown,
  isMCPTool: boolean
): unknown {
  // Only unwrap if this is an MCP tool and result is a tuple
  if (!isMCPTool) {
    return result;
  }

  // Check if result is a tuple/array with [content, artifacts]
  if (Array.isArray(result) && result.length >= 1) {
    const [content] = result;

    // If first element is a string, return it
    if (typeof content === 'string') {
      // Try to parse as JSON if it looks like JSON
      if (typeof content === 'string' && content.trim().startsWith('{')) {
        try {
          return JSON.parse(content);
        } catch {
          return content;
        }
      }
      return content;
    }

    // If first element is an array (content blocks), extract text/data
    if (Array.isArray(content)) {
      // If it's an array of content blocks (like [{ type: 'text', text: '...' }])
      if (
        content.length > 0 &&
        typeof content[0] === 'object' &&
        'type' in content[0]
      ) {
        // Extract text from content blocks
        const texts = content
          .filter((block: unknown) => {
            if (typeof block !== 'object' || block === null) return false;
            const b = block as Record<string, unknown>;
            return b.type === 'text' && typeof b.text === 'string';
          })
          .map((block: unknown) => {
            const b = block as Record<string, unknown>;
            return b.text as string;
          });

        if (texts.length > 0) {
          const combined = texts.join('\n');
          // Try to parse as JSON if it looks like JSON (objects or arrays)
          if (
            combined.trim().startsWith('{') ||
            combined.trim().startsWith('[')
          ) {
            try {
              return JSON.parse(combined);
            } catch {
              return combined;
            }
          }
          return combined;
        }
      }
      // Otherwise return the content array as-is
      return content;
    }

    // If first element is an object, return it
    if (typeof content === 'object' && content !== null) {
      return content;
    }
  }

  // Not a formatted response, return as-is
  return result;
}

/**
 * Executes tools in parallel when requested by the API.
 * Uses Promise.all for parallel execution, catching individual errors.
 * Unwraps formatted responses (e.g., MCP tool tuples) to raw data.
 * @param toolCalls - Array of tool calls from the API
 * @param toolMap - Map of tool names to executable tools
 * @returns Array of tool results
 */
export async function executeTools(
  toolCalls: t.PTCToolCall[],
  toolMap: t.ToolMap
): Promise<t.PTCToolResult[]> {
  const executions = toolCalls.map(async (call): Promise<t.PTCToolResult> => {
    const tool = toolMap.get(call.name);

    if (!tool) {
      return {
        call_id: call.id,
        result: null,
        is_error: true,
        error_message: `Tool '${call.name}' not found. Available tools: ${Array.from(toolMap.keys()).join(', ')}`,
      };
    }

    try {
      const result = await tool.invoke(call.input, {
        metadata: { [Constants.PROGRAMMATIC_TOOL_CALLING]: true },
      });

      const isMCPTool = tool.mcp === true;
      const unwrappedResult = unwrapToolResponse(result, isMCPTool);

      return {
        call_id: call.id,
        result: unwrappedResult,
        is_error: false,
      };
    } catch (error) {
      return {
        call_id: call.id,
        result: null,
        is_error: true,
        error_message: (error as Error).message || 'Tool execution failed',
      };
    }
  });

  return await Promise.all(executions);
}

/**
 * Formats the completed response for the agent.
 * @param response - The completed API response
 * @returns Tuple of [formatted string, artifact]
 */
export function formatCompletedResponse(
  response: t.ProgrammaticExecutionResponse
): [string, t.ProgrammaticExecutionArtifact] {
  let formatted = '';

  if (response.stdout != null && response.stdout !== '') {
    formatted += `stdout:\n${response.stdout}\n`;
  } else {
    formatted += emptyOutputMessage;
  }

  if (response.stderr != null && response.stderr !== '') {
    formatted += `stderr:\n${response.stderr}\n`;
  }

  if (response.files && response.files.length > 0) {
    formatted += 'Generated files:\n';

    const fileCount = response.files.length;
    for (let i = 0; i < fileCount; i++) {
      const file = response.files[i];
      const isImage = imageExtRegex.test(file.name);
      formatted += `- /mnt/data/${file.name} | ${isImage ? imageMessage : otherMessage}`;

      if (i < fileCount - 1) {
        formatted += fileCount <= 3 ? ', ' : ',\n';
      }
    }

    formatted += `\nsession_id: ${response.session_id}\n\n${accessMessage}`;
  }

  return [
    formatted.trim(),
    {
      session_id: response.session_id,
      files: response.files,
    },
  ];
}

// ============================================================================
// Tool Factory
// ============================================================================

/**
 * Creates a Programmatic Tool Calling tool for complex multi-tool workflows.
 *
 * This tool enables AI agents to write Python code that orchestrates multiple
 * tool calls programmatically, reducing LLM round-trips and token usage.
 *
 * The tool map must be provided at runtime via config.configurable.toolMap.
 *
 * @param params - Configuration parameters (apiKey, baseUrl, maxRoundTrips, proxy)
 * @returns A LangChain DynamicStructuredTool for programmatic tool calling
 *
 * @example
 * const ptcTool = createProgrammaticToolCallingTool({
 *   apiKey: process.env.CODE_API_KEY,
 *   maxRoundTrips: 20
 * });
 *
 * const [output, artifact] = await ptcTool.invoke(
 *   { code, tools },
 *   { configurable: { toolMap } }
 * );
 */
export function createProgrammaticToolCallingTool(
  initParams: t.ProgrammaticToolCallingParams = {}
): DynamicStructuredTool<typeof ProgrammaticToolCallingSchema> {
  const apiKey =
    (initParams[EnvVar.CODE_API_KEY] as string | undefined) ??
    initParams.apiKey ??
    getEnvironmentVariable(EnvVar.CODE_API_KEY) ??
    '';

  if (!apiKey) {
    throw new Error(
      'No API key provided for programmatic tool calling. ' +
        'Set CODE_API_KEY environment variable or pass apiKey in initParams.'
    );
  }

  const baseUrl = initParams.baseUrl ?? getCodeBaseURL();
  const maxRoundTrips = initParams.maxRoundTrips ?? DEFAULT_MAX_ROUND_TRIPS;
  const proxy = initParams.proxy ?? process.env.PROXY;
  const debug = initParams.debug ?? process.env.PTC_DEBUG === 'true';
  const EXEC_ENDPOINT = `${baseUrl}/exec/programmatic`;

  const description = `
Run tools via Python code. Tools are injected as async functions—call with \`await\`.

Rules:
- Tools are pre-defined—DO NOT define them yourself
- Use \`await\` for calls, \`asyncio.gather()\` for parallel; NEVER call \`asyncio.run()\`
- Only \`print()\` output returns to the model; tool results are raw dicts/lists/strings
- Stateless: variables/imports don't persist; use \`session_id\` param for file access
- Files mount at \`/mnt/data/\` (READ-ONLY); write changes to NEW filenames
- Tool names normalized: hyphens→underscores, keywords get \`_tool\` suffix

When to use (vs. direct tool calls): loops, conditionals, parallel execution, aggregation.

Examples:
  result = await get_weather(city="NYC"); print(result)
  sf, ny = await asyncio.gather(get_weather(city="SF"), get_weather(city="NY"))
`.trim();

  return tool<typeof ProgrammaticToolCallingSchema>(
    async (params, config) => {
      const { code, session_id, timeout = DEFAULT_TIMEOUT } = params;

      // Extra params injected by ToolNode (follows web_search pattern)
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
          'No tool definitions provided. ' +
            'Either pass tools in the input or ensure ToolNode injects toolDefs.'
        );
      }

      let roundTrip = 0;

      try {
        // ====================================================================
        // Phase 1: Filter tools and make initial request
        // ====================================================================

        const effectiveTools = filterToolsByUsage(toolDefs, code, debug);

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[PTC Debug] Sending ${effectiveTools.length} tools to API ` +
              `(filtered from ${toolDefs.length})`
          );
        }

        // Fetch files from previous session if session_id is provided
        let files: t.CodeEnvFile[] | undefined;
        if (session_id != null && session_id.length > 0) {
          files = await fetchSessionFiles(baseUrl, apiKey, session_id, proxy);
        }

        let response = await makeRequest(
          EXEC_ENDPOINT,
          apiKey,
          {
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
              `[PTC Debug] Round trip ${roundTrip}: ${response.tool_calls?.length ?? 0} tool(s) to execute`
            );
          }

          const toolResults = await executeTools(
            response.tool_calls ?? [],
            toolMap
          );

          response = await makeRequest(
            EXEC_ENDPOINT,
            apiKey,
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
          `Programmatic execution failed: ${(error as Error).message}`
        );
      }
    },
    {
      name: Constants.PROGRAMMATIC_TOOL_CALLING,
      description,
      schema: ProgrammaticToolCallingSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
