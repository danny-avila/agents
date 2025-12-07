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

/**
 * Extracts tool names that are actually called in the Python code.
 * Matches patterns like `await tool_name(`, `tool_name(`, and asyncio.gather calls.
 * @param code - The Python code to analyze
 * @param availableToolNames - Set of available tool names to match against
 * @returns Set of tool names found in the code
 */
export function extractUsedToolNames(
  code: string,
  availableToolNames: Set<string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const toolName of availableToolNames) {
    const escapedName = toolName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\s*\\(`, 'g');

    if (pattern.test(code)) {
      usedTools.add(toolName);
    }
  }

  return usedTools;
}

/**
 * Filters tool definitions to only include tools actually used in the code.
 * @param toolDefs - All available tool definitions
 * @param code - The Python code to analyze
 * @returns Filtered array of tool definitions
 */
export function filterToolsByUsage(
  toolDefs: t.LCTool[],
  code: string
): t.LCTool[] {
  const availableToolNames = new Set(toolDefs.map((tool) => tool.name));
  const usedToolNames = extractUsedToolNames(code, availableToolNames);

  if (usedToolNames.size === 0) {
    return toolDefs;
  }

  return toolDefs.filter((tool) => usedToolNames.has(tool.name));
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
 * Executes tools in parallel when requested by the API.
 * Uses Promise.all for parallel execution, catching individual errors.
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
      return {
        call_id: call.id,
        result,
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
  const EXEC_ENDPOINT = `${baseUrl}/exec/programmatic`;

  const description = `
Run tools by writing Python code. Tools are available as async functions - just call them with await.

This is different from execute_code: here you can call your tools (like get_weather, get_expenses, etc.) directly in Python code.

Usage:
- Tools are pre-defined as async functions - call them with await
- Use asyncio.gather() to run multiple tools in parallel
- Only print() output is returned - tool results stay in Python

Examples:
- Simple: result = await get_weather(city="NYC")
- Loop: for user in users: data = await get_expenses(user_id=user['id'])
- Parallel: sf, ny = await asyncio.gather(get_weather(city="SF"), get_weather(city="NY"))

When to use this instead of calling tools directly:
- You need to call tools in a loop (process many items)
- You want parallel execution (asyncio.gather)
- You need conditionals based on tool results
- You want to aggregate/filter data before returning
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

        const effectiveTools = filterToolsByUsage(toolDefs, code);

        let response = await makeRequest(
          EXEC_ENDPOINT,
          apiKey,
          {
            code,
            tools: effectiveTools,
            session_id,
            timeout,
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

          // eslint-disable-next-line no-console
          console.log(
            `[PTC] Round trip ${roundTrip}: ${response.tool_calls?.length ?? 0} tool(s) to execute`
          );

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
