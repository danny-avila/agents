// src/tools/ToolSearch.ts
import { z } from 'zod';
import * as okapibm25Module from 'okapibm25';
import { config } from 'dotenv';

type BM25Fn = (
  documents: string[],
  keywords: string[],
  constants?: { k1?: number; b?: number }
) => number[];

function getBM25Function(): BM25Fn {
  const mod = okapibm25Module as unknown as {
    default: BM25Fn | { default: BM25Fn } | undefined;
  };
  if (typeof mod === 'function') return mod;
  if (typeof mod.default === 'function') return mod.default;
  if (mod.default != null && typeof mod.default.default === 'function')
    return mod.default.default;
  throw new Error('Could not resolve BM25 function from okapibm25 module');
}

const BM25 = getBM25Function();
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import { getCodeBaseURL } from './CodeExecutor';
import { EnvVar, Constants } from '@/common';

config();

/** Maximum allowed regex pattern length */
const MAX_PATTERN_LENGTH = 200;

/** Maximum allowed regex nesting depth */
const MAX_REGEX_COMPLEXITY = 5;

/** Default search timeout in milliseconds */
const SEARCH_TIMEOUT = 5000;

/** Zod schema type for tool search parameters */
type ToolSearchSchema = z.ZodObject<{
  query: z.ZodDefault<z.ZodOptional<z.ZodString>>;
  fields: z.ZodDefault<
    z.ZodOptional<z.ZodArray<z.ZodEnum<['name', 'description', 'parameters']>>>
  >;
  max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
  mcp_server: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodArray<z.ZodString>]>>;
}>;

/**
 * Creates the Zod schema with dynamic query description based on mode.
 * @param mode - The search mode determining query interpretation
 * @returns Zod schema for tool search parameters
 */
function createToolSearchSchema(mode: t.ToolSearchMode): ToolSearchSchema {
  const queryDescription =
    mode === 'local'
      ? 'Search term to find in tool names and descriptions. Case-insensitive substring matching. Optional if mcp_server is provided.'
      : 'Regex pattern to search tool names and descriptions. Optional if mcp_server is provided.';

  return z.object({
    query: z
      .string()
      .max(MAX_PATTERN_LENGTH)
      .optional()
      .default('')
      .describe(queryDescription),
    fields: z
      .array(z.enum(['name', 'description', 'parameters']))
      .optional()
      .default(['name', 'description'])
      .describe('Which fields to search. Default: name and description'),
    max_results: z
      .number()
      .int()
      .min(1)
      .max(50)
      .optional()
      .default(10)
      .describe('Maximum number of matching tools to return'),
    mcp_server: z
      .union([z.string(), z.array(z.string())])
      .optional()
      .describe(
        'Filter to tools from specific MCP server(s). Can be a single server name or array of names. If provided without a query, lists all tools from those servers.'
      ),
  });
}

/**
 * Extracts the MCP server name from a tool name.
 * MCP tools follow the pattern: toolName_mcp_serverName
 * @param toolName - The full tool name
 * @returns The server name if it's an MCP tool, undefined otherwise
 */
function extractMcpServerName(toolName: string): string | undefined {
  const delimiterIndex = toolName.indexOf(Constants.MCP_DELIMITER);
  if (delimiterIndex === -1) {
    return undefined;
  }
  return toolName.substring(delimiterIndex + Constants.MCP_DELIMITER.length);
}

/**
 * Checks if a tool belongs to a specific MCP server.
 * @param toolName - The full tool name
 * @param serverName - The server name to match
 * @returns True if the tool belongs to the specified server
 */
function isFromMcpServer(toolName: string, serverName: string): boolean {
  const toolServer = extractMcpServerName(toolName);
  return toolServer === serverName;
}

/**
 * Checks if a tool belongs to any of the specified MCP servers.
 * @param toolName - The full tool name
 * @param serverNames - Array of server names to match
 * @returns True if the tool belongs to any of the specified servers
 */
function isFromAnyMcpServer(toolName: string, serverNames: string[]): boolean {
  const toolServer = extractMcpServerName(toolName);
  if (toolServer === undefined) {
    return false;
  }
  return serverNames.includes(toolServer);
}

/**
 * Normalizes server filter input to always be an array.
 * @param serverFilter - String, array of strings, or undefined
 * @returns Array of server names (empty if none specified)
 */
function normalizeServerFilter(
  serverFilter: string | string[] | undefined
): string[] {
  if (serverFilter === undefined) {
    return [];
  }
  if (typeof serverFilter === 'string') {
    return serverFilter === '' ? [] : [serverFilter];
  }
  return serverFilter.filter((s) => s !== '');
}

/**
 * Extracts all unique MCP server names from a tool registry.
 * @param toolRegistry - The tool registry to scan
 * @param onlyDeferred - If true, only considers deferred tools
 * @returns Array of unique server names, sorted alphabetically
 */
function getAvailableMcpServers(
  toolRegistry: t.LCToolRegistry | undefined,
  onlyDeferred: boolean = true
): string[] {
  if (!toolRegistry) {
    return [];
  }

  const servers = new Set<string>();
  for (const [, toolDef] of toolRegistry) {
    if (onlyDeferred && toolDef.defer_loading !== true) {
      continue;
    }
    const server = extractMcpServerName(toolDef.name);
    if (server !== undefined && server !== '') {
      servers.add(server);
    }
  }

  return Array.from(servers).sort();
}

/**
 * Escapes special regex characters in a string to use as a literal pattern.
 * @param pattern - The string to escape
 * @returns The escaped string safe for use in a RegExp
 */
function escapeRegexSpecialChars(pattern: string): string {
  return pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Counts the maximum nesting depth of groups in a regex pattern.
 * @param pattern - The regex pattern to analyze
 * @returns The maximum nesting depth
 */
function countNestedGroups(pattern: string): number {
  let maxDepth = 0;
  let currentDepth = 0;

  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === '(' && (i === 0 || pattern[i - 1] !== '\\')) {
      currentDepth++;
      maxDepth = Math.max(maxDepth, currentDepth);
    } else if (pattern[i] === ')' && (i === 0 || pattern[i - 1] !== '\\')) {
      currentDepth = Math.max(0, currentDepth - 1);
    }
  }

  return maxDepth;
}

/**
 * Detects nested quantifiers that can cause catastrophic backtracking.
 * Patterns like (a+)+, (a*)*, (a+)*, etc.
 * @param pattern - The regex pattern to check
 * @returns True if nested quantifiers are detected
 */
function hasNestedQuantifiers(pattern: string): boolean {
  const nestedQuantifierPattern = /\([^)]*[+*][^)]*\)[+*?]/;
  return nestedQuantifierPattern.test(pattern);
}

/**
 * Checks if a regex pattern contains potentially dangerous constructs.
 * @param pattern - The regex pattern to validate
 * @returns True if the pattern is dangerous
 */
function isDangerousPattern(pattern: string): boolean {
  if (hasNestedQuantifiers(pattern)) {
    return true;
  }

  if (countNestedGroups(pattern) > MAX_REGEX_COMPLEXITY) {
    return true;
  }

  const dangerousPatterns = [
    /\.\{1000,\}/, // Excessive wildcards
    /\(\?=\.\{100,\}\)/, // Runaway lookaheads
    /\([^)]*\|\s*\){20,}/, // Excessive alternation (rough check)
    /\(\.\*\)\+/, // (.*)+
    /\(\.\+\)\+/, // (.+)+
    /\(\.\*\)\*/, // (.*)*
    /\(\.\+\)\*/, // (.+)*
  ];

  for (const dangerous of dangerousPatterns) {
    if (dangerous.test(pattern)) {
      return true;
    }
  }

  return false;
}

/**
 * Sanitizes a regex pattern for safe execution.
 * If the pattern is dangerous, it will be escaped to a literal string search.
 * @param pattern - The regex pattern to sanitize
 * @returns Object containing the safe pattern and whether it was escaped
 */
function sanitizeRegex(pattern: string): { safe: string; wasEscaped: boolean } {
  if (isDangerousPattern(pattern)) {
    return {
      safe: escapeRegexSpecialChars(pattern),
      wasEscaped: true,
    };
  }

  try {
    new RegExp(pattern);
    return { safe: pattern, wasEscaped: false };
  } catch {
    return {
      safe: escapeRegexSpecialChars(pattern),
      wasEscaped: true,
    };
  }
}

/**
 * Simplifies tool parameters for search purposes.
 * Extracts only the essential structure needed for parameter name searching.
 * @param parameters - The tool's JSON schema parameters
 * @returns Simplified parameters object
 */
function simplifyParametersForSearch(
  parameters?: t.JsonSchemaType
): t.JsonSchemaType | undefined {
  if (!parameters) {
    return undefined;
  }

  if (parameters.properties) {
    return {
      type: parameters.type,
      properties: Object.fromEntries(
        Object.entries(parameters.properties).map(([key, value]) => [
          key,
          { type: (value as t.JsonSchemaType).type },
        ])
      ),
    } as t.JsonSchemaType;
  }

  return { type: parameters.type };
}

/**
 * Tokenizes a string into lowercase words for BM25.
 * Splits on underscores and non-alphanumeric characters for consistent matching.
 * @param text - The text to tokenize
 * @returns Array of lowercase tokens
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]/g, ' ')
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

/**
 * Creates a searchable document string from tool metadata.
 * @param tool - The tool metadata
 * @param fields - Which fields to include
 * @returns Combined document string for BM25
 */
function createToolDocument(tool: t.ToolMetadata, fields: string[]): string {
  const parts: string[] = [];

  if (fields.includes('name')) {
    const baseName = tool.name.replace(/_/g, ' ');
    parts.push(baseName, baseName);
  }

  if (fields.includes('description') && tool.description) {
    parts.push(tool.description);
  }

  if (fields.includes('parameters') && tool.parameters?.properties) {
    const paramNames = Object.keys(tool.parameters.properties).join(' ');
    parts.push(paramNames);
  }

  return parts.join(' ');
}

/**
 * Determines which field had the best match for a query.
 * @param tool - The tool to check
 * @param queryTokens - Tokenized query
 * @param fields - Fields to check
 * @returns The matched field and a snippet
 */
function findMatchedField(
  tool: t.ToolMetadata,
  queryTokens: string[],
  fields: string[]
): { field: string; snippet: string } {
  if (fields.includes('name')) {
    const nameLower = tool.name.toLowerCase();
    for (const token of queryTokens) {
      if (nameLower.includes(token)) {
        return { field: 'name', snippet: tool.name };
      }
    }
  }

  if (fields.includes('description') && tool.description) {
    const descLower = tool.description.toLowerCase();
    for (const token of queryTokens) {
      if (descLower.includes(token)) {
        return {
          field: 'description',
          snippet: tool.description.substring(0, 100),
        };
      }
    }
  }

  if (fields.includes('parameters') && tool.parameters?.properties) {
    const paramNames = Object.keys(tool.parameters.properties);
    const paramLower = paramNames.join(' ').toLowerCase();
    for (const token of queryTokens) {
      if (paramLower.includes(token)) {
        return { field: 'parameters', snippet: paramNames.join(', ') };
      }
    }
  }

  const fallbackSnippet = tool.description
    ? tool.description.substring(0, 100)
    : tool.name;
  return { field: 'unknown', snippet: fallbackSnippet };
}

/**
 * Performs BM25-based search for better relevance ranking.
 * Uses Okapi BM25 algorithm for term frequency and document length normalization.
 * @param tools - Array of tool metadata to search
 * @param query - The search query
 * @param fields - Which fields to search
 * @param maxResults - Maximum results to return
 * @returns Search response with matching tools ranked by BM25 score
 */
function performLocalSearch(
  tools: t.ToolMetadata[],
  query: string,
  fields: string[],
  maxResults: number
): t.ToolSearchResponse {
  if (tools.length === 0 || !query.trim()) {
    return {
      tool_references: [],
      total_tools_searched: tools.length,
      pattern_used: query,
    };
  }

  const documents = tools.map((tool) => createToolDocument(tool, fields));
  const queryTokens = tokenize(query);

  if (queryTokens.length === 0) {
    return {
      tool_references: [],
      total_tools_searched: tools.length,
      pattern_used: query,
    };
  }

  const scores = BM25(documents, queryTokens, { k1: 1.5, b: 0.75 }) as number[];

  const maxScore = Math.max(...scores.filter((s) => s > 0), 1);
  const queryLower = query.toLowerCase().trim();

  const results: t.ToolSearchResult[] = [];
  for (let i = 0; i < tools.length; i++) {
    if (scores[i] > 0) {
      const { field, snippet } = findMatchedField(
        tools[i],
        queryTokens,
        fields
      );
      let normalizedScore = Math.min(scores[i] / maxScore, 1.0);

      // Boost score for exact base name match
      const baseName = getBaseToolName(tools[i].name).toLowerCase();
      if (baseName === queryLower) {
        normalizedScore = 1.0;
      } else if (baseName.startsWith(queryLower)) {
        normalizedScore = Math.max(normalizedScore, 0.95);
      }

      results.push({
        tool_name: tools[i].name,
        match_score: normalizedScore,
        matched_field: field,
        snippet,
      });
    }
  }

  results.sort((a, b) => b.match_score - a.match_score);
  const topResults = results.slice(0, maxResults);

  return {
    tool_references: topResults,
    total_tools_searched: tools.length,
    pattern_used: query,
  };
}

/**
 * Generates the JavaScript search script to be executed in the sandbox.
 * Uses plain JavaScript for maximum compatibility with the Code API.
 * @param deferredTools - Array of tool metadata to search through
 * @param fields - Which fields to search
 * @param maxResults - Maximum number of results to return
 * @param sanitizedPattern - The sanitized regex pattern
 * @returns The JavaScript code string
 */
function generateSearchScript(
  deferredTools: t.ToolMetadata[],
  fields: string[],
  maxResults: number,
  sanitizedPattern: string
): string {
  const lines = [
    '// Tool definitions (injected)',
    'var tools = ' + JSON.stringify(deferredTools) + ';',
    'var searchFields = ' + JSON.stringify(fields) + ';',
    'var maxResults = ' + maxResults + ';',
    'var pattern = ' + JSON.stringify(sanitizedPattern) + ';',
    '',
    '// Compile regex (pattern is sanitized client-side)',
    'var regex;',
    'try {',
    '  regex = new RegExp(pattern, \'i\');',
    '} catch (e) {',
    '  regex = new RegExp(pattern.replace(/[.*+?^${}()[\\]\\\\|]/g, "\\\\$&"), "i");',
    '}',
    '',
    '// Search logic',
    'var results = [];',
    '',
    'for (var j = 0; j < tools.length; j++) {',
    '  var tool = tools[j];',
    '  var bestScore = 0;',
    '  var matchedField = \'\';',
    '  var snippet = \'\';',
    '',
    '  // Search name (highest priority)',
    '  if (searchFields.indexOf(\'name\') >= 0 && regex.test(tool.name)) {',
    '    bestScore = 0.95;',
    '    matchedField = \'name\';',
    '    snippet = tool.name;',
    '  }',
    '',
    '  // Search description (medium priority)',
    '  if (searchFields.indexOf(\'description\') >= 0 && tool.description && regex.test(tool.description)) {',
    '    if (bestScore === 0) {',
    '      bestScore = 0.75;',
    '      matchedField = \'description\';',
    '      snippet = tool.description.substring(0, 100);',
    '    }',
    '  }',
    '',
    '  // Search parameter names (lower priority)',
    '  if (searchFields.indexOf(\'parameters\') >= 0 && tool.parameters && tool.parameters.properties) {',
    '    var paramNames = Object.keys(tool.parameters.properties).join(\' \');',
    '    if (regex.test(paramNames)) {',
    '      if (bestScore === 0) {',
    '        bestScore = 0.60;',
    '        matchedField = \'parameters\';',
    '        snippet = paramNames;',
    '      }',
    '    }',
    '  }',
    '',
    '  if (bestScore > 0) {',
    '    results.push({',
    '      tool_name: tool.name,',
    '      match_score: bestScore,',
    '      matched_field: matchedField,',
    '      snippet: snippet',
    '    });',
    '  }',
    '}',
    '',
    '// Sort by score (descending) and limit results',
    'results.sort(function(a, b) { return b.match_score - a.match_score; });',
    'var topResults = results.slice(0, maxResults);',
    '',
    '// Output as JSON',
    'console.log(JSON.stringify({',
    '  tool_references: topResults.map(function(r) {',
    '    return {',
    '      tool_name: r.tool_name,',
    '      match_score: r.match_score,',
    '      matched_field: r.matched_field,',
    '      snippet: r.snippet',
    '    };',
    '  }),',
    '  total_tools_searched: tools.length,',
    '  pattern_used: pattern',
    '}));',
  ];
  return lines.join('\n');
}

/**
 * Parses the search results from stdout JSON.
 * @param stdout - The stdout string containing JSON results
 * @returns Parsed search response
 */
function parseSearchResults(stdout: string): t.ToolSearchResponse {
  const jsonMatch = stdout.trim();
  const parsed = JSON.parse(jsonMatch) as t.ToolSearchResponse;
  return parsed;
}

/**
 * Formats search results as structured JSON for efficient parsing.
 * @param searchResponse - The parsed search response
 * @returns JSON string with search results
 */
function formatSearchResults(searchResponse: t.ToolSearchResponse): string {
  const { tool_references, total_tools_searched, pattern_used } =
    searchResponse;

  const output = {
    found: tool_references.length,
    tools: tool_references.map((ref) => ({
      name: ref.tool_name,
      score: Number(ref.match_score.toFixed(2)),
      matched_in: ref.matched_field,
      snippet: ref.snippet,
    })),
    total_searched: total_tools_searched,
    query: pattern_used,
  };

  return JSON.stringify(output, null, 2);
}

/**
 * Extracts the base tool name (without MCP server suffix) from a full tool name.
 * @param toolName - The full tool name
 * @returns The base tool name without server suffix
 */
function getBaseToolName(toolName: string): string {
  const delimiterIndex = toolName.indexOf(Constants.MCP_DELIMITER);
  if (delimiterIndex === -1) {
    return toolName;
  }
  return toolName.substring(0, delimiterIndex);
}

/**
 * Generates a compact listing of deferred tools grouped by server.
 * Format: "server: tool1, tool2, tool3"
 * Non-MCP tools are grouped under "other".
 * @param toolRegistry - The tool registry
 * @param onlyDeferred - Whether to only include deferred tools
 * @returns Formatted string with tools grouped by server
 */
function getDeferredToolsListing(
  toolRegistry: t.LCToolRegistry | undefined,
  onlyDeferred: boolean
): string {
  if (!toolRegistry) {
    return '';
  }

  const toolsByServer: Record<string, string[]> = {};

  for (const lcTool of toolRegistry.values()) {
    if (onlyDeferred && lcTool.defer_loading !== true) {
      continue;
    }

    const toolName = lcTool.name;
    const serverName = extractMcpServerName(toolName) ?? 'other';
    const baseName = getBaseToolName(toolName);

    if (!(serverName in toolsByServer)) {
      toolsByServer[serverName] = [];
    }
    toolsByServer[serverName].push(baseName);
  }

  const serverNames = Object.keys(toolsByServer).sort((a, b) => {
    if (a === 'other') return 1;
    if (b === 'other') return -1;
    return a.localeCompare(b);
  });

  if (serverNames.length === 0) {
    return '';
  }

  const lines = serverNames.map(
    (server) => `${server}: ${toolsByServer[server].join(', ')}`
  );

  return lines.join('\n');
}

/**
 * Formats a server listing response as structured JSON.
 * NOTE: This is a PREVIEW only - tools are NOT discovered/loaded.
 * @param tools - Array of tool metadata from the server(s)
 * @param serverNames - The MCP server name(s)
 * @returns JSON string showing all tools grouped by server
 */
function formatServerListing(
  tools: t.ToolMetadata[],
  serverNames: string | string[]
): string {
  const servers = Array.isArray(serverNames) ? serverNames : [serverNames];

  if (tools.length === 0) {
    return JSON.stringify(
      {
        listing_mode: true,
        servers,
        total_tools: 0,
        tools_by_server: {},
        hint: 'No tools found from the specified MCP server(s).',
      },
      null,
      2
    );
  }

  const toolsByServer: Record<
    string,
    Array<{ name: string; description: string }>
  > = {};
  for (const tool of tools) {
    const server = extractMcpServerName(tool.name) ?? 'unknown';
    if (!(server in toolsByServer)) {
      toolsByServer[server] = [];
    }
    toolsByServer[server].push({
      name: getBaseToolName(tool.name),
      description:
        tool.description.length > 100
          ? tool.description.substring(0, 97) + '...'
          : tool.description,
    });
  }

  const output = {
    listing_mode: true,
    servers,
    total_tools: tools.length,
    tools_by_server: toolsByServer,
    hint: `To use a tool, search for it by name (e.g., query: "${getBaseToolName(tools[0]?.name ?? 'tool_name')}") to load it.`,
  };

  return JSON.stringify(output, null, 2);
}

/**
 * Creates a Tool Search tool for discovering tools from a large registry.
 *
 * This tool enables AI agents to dynamically discover tools from a large library
 * without loading all tool definitions into the LLM context window. The agent
 * can search for relevant tools on-demand.
 *
 * **Modes:**
 * - `code_interpreter` (default): Uses external sandbox for regex search. Safer for complex patterns.
 * - `local`: Uses safe substring matching locally. No network call, faster, completely safe from ReDoS.
 *
 * The tool registry can be provided either:
 * 1. At initialization time via params.toolRegistry
 * 2. At runtime via config.configurable.toolRegistry when invoking
 *
 * @param params - Configuration parameters for the tool (toolRegistry is optional)
 * @returns A LangChain DynamicStructuredTool for tool searching
 *
 * @example
 * // Option 1: Code interpreter mode (regex via sandbox)
 * const tool = createToolSearch({ apiKey, toolRegistry });
 * await tool.invoke({ query: 'expense.*report' });
 *
 * @example
 * // Option 2: Local mode (safe substring search, no API key needed)
 * const tool = createToolSearch({ mode: 'local', toolRegistry });
 * await tool.invoke({ query: 'expense' });
 */
function createToolSearch(
  initParams: t.ToolSearchParams = {}
): DynamicStructuredTool<ReturnType<typeof createToolSearchSchema>> {
  const mode: t.ToolSearchMode = initParams.mode ?? 'code_interpreter';
  const defaultOnlyDeferred = initParams.onlyDeferred ?? true;
  const schema = createToolSearchSchema(mode);

  const apiKey: string =
    mode === 'code_interpreter'
      ? ((initParams[EnvVar.CODE_API_KEY] as string | undefined) ??
        initParams.apiKey ??
        getEnvironmentVariable(EnvVar.CODE_API_KEY) ??
        '')
      : '';

  if (mode === 'code_interpreter' && !apiKey) {
    throw new Error(
      'No API key provided for tool search in code_interpreter mode. Use mode: "local" to search without an API key.'
    );
  }

  const baseEndpoint = initParams.baseUrl ?? getCodeBaseURL();
  const EXEC_ENDPOINT = `${baseEndpoint}/exec`;

  const deferredToolsListing = getDeferredToolsListing(
    initParams.toolRegistry,
    defaultOnlyDeferred
  );

  const toolsListSection =
    deferredToolsListing.length > 0
      ? `

Deferred tools (search to load):
${deferredToolsListing}`
      : '';

  const mcpNote =
    deferredToolsListing.includes(Constants.MCP_DELIMITER) ||
    deferredToolsListing.split('\n').some((line) => !line.startsWith('other:'))
      ? `
- MCP tools use format: toolName${Constants.MCP_DELIMITER}serverName
- Use mcp_server param to filter by server`
      : '';

  const description =
    mode === 'local'
      ? `
Searches deferred tools using BM25 ranking. Multi-word queries supported.
${mcpNote}${toolsListSection}
`.trim()
      : `
Searches deferred tools by regex pattern.
${mcpNote}${toolsListSection}
`.trim();

  return tool<typeof schema>(
    async (params, config) => {
      const {
        query,
        fields = ['name', 'description'],
        max_results = 10,
        mcp_server,
      } = params;

      const {
        toolRegistry: paramToolRegistry,
        onlyDeferred: paramOnlyDeferred,
        mcpServer: paramMcpServer,
      } = config.toolCall ?? {};

      const toolRegistry = paramToolRegistry ?? initParams.toolRegistry;
      const onlyDeferred =
        paramOnlyDeferred !== undefined
          ? paramOnlyDeferred
          : defaultOnlyDeferred;
      const rawServerFilter =
        mcp_server ?? paramMcpServer ?? initParams.mcpServer;
      const serverFilters = normalizeServerFilter(rawServerFilter);
      const hasServerFilter = serverFilters.length > 0;

      if (toolRegistry == null) {
        return [
          'Error: No tool registry provided. Configure toolRegistry at agent level or initialization.',
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: query,
              error: 'No tool registry provided',
            },
          },
        ];
      }

      const toolsArray: t.LCTool[] = Array.from(toolRegistry.values());
      const deferredTools: t.ToolMetadata[] = toolsArray
        .filter((lcTool) => {
          if (onlyDeferred === true && lcTool.defer_loading !== true) {
            return false;
          }
          if (
            hasServerFilter &&
            !isFromAnyMcpServer(lcTool.name, serverFilters)
          ) {
            return false;
          }
          return true;
        })
        .map((lcTool) => ({
          name: lcTool.name,
          description: lcTool.description ?? '',
          parameters: simplifyParametersForSearch(lcTool.parameters),
        }));

      if (deferredTools.length === 0) {
        const serverMsg = hasServerFilter
          ? ` from MCP server(s): ${serverFilters.join(', ')}`
          : '';
        return [
          `No tools available to search${serverMsg}. The tool registry is empty or no matching deferred tools are registered.`,
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: query,
              mcp_server: serverFilters,
            },
          },
        ];
      }

      const isServerListing = hasServerFilter && query === '';

      if (isServerListing) {
        const formattedOutput = formatServerListing(
          deferredTools,
          serverFilters
        );

        return [
          formattedOutput,
          {
            tool_references: [],
            metadata: {
              total_available: deferredTools.length,
              mcp_server: serverFilters,
              listing_mode: true,
            },
          },
        ];
      }

      if (mode === 'local') {
        const searchResponse = performLocalSearch(
          deferredTools,
          query,
          fields,
          max_results
        );
        const formattedOutput = formatSearchResults(searchResponse);

        return [
          formattedOutput,
          {
            tool_references: searchResponse.tool_references,
            metadata: {
              total_searched: searchResponse.total_tools_searched,
              pattern: searchResponse.pattern_used,
              mcp_server: serverFilters.length > 0 ? serverFilters : undefined,
            },
          },
        ];
      }

      const { safe: sanitizedPattern, wasEscaped } = sanitizeRegex(query);
      let warningMessage = '';
      if (wasEscaped) {
        warningMessage =
          'Note: The provided pattern was converted to a literal search for safety.\n\n';
      }

      const searchScript = generateSearchScript(
        deferredTools,
        fields,
        max_results,
        sanitizedPattern
      );

      const postData = {
        lang: 'js',
        code: searchScript,
        timeout: SEARCH_TIMEOUT,
      };

      try {
        const fetchOptions: RequestInit = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'LibreChat/1.0',
            'X-API-Key': apiKey,
          },
          body: JSON.stringify(postData),
        };

        if (process.env.PROXY != null && process.env.PROXY !== '') {
          fetchOptions.agent = new HttpsProxyAgent(process.env.PROXY);
        }

        const response = await fetch(EXEC_ENDPOINT, fetchOptions);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: t.ExecuteResult = await response.json();

        if (result.stderr && result.stderr.trim()) {
          // eslint-disable-next-line no-console
          console.warn('[ToolSearch] stderr:', result.stderr);
        }

        if (!result.stdout || !result.stdout.trim()) {
          return [
            `${warningMessage}No tools matched the pattern "${sanitizedPattern}".\nTotal tools searched: ${deferredTools.length}`,
            {
              tool_references: [],
              metadata: {
                total_searched: deferredTools.length,
                pattern: sanitizedPattern,
              },
            },
          ];
        }

        const searchResponse = parseSearchResults(result.stdout);
        const formattedOutput = `${warningMessage}${formatSearchResults(searchResponse)}`;

        return [
          formattedOutput,
          {
            tool_references: searchResponse.tool_references,
            metadata: {
              total_searched: searchResponse.total_tools_searched,
              pattern: searchResponse.pattern_used,
            },
          },
        ];
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error('[ToolSearch] Error:', error);

        const errorMessage =
          error instanceof Error ? error.message : String(error);
        return [
          `Tool search failed: ${errorMessage}\n\nSuggestion: Try a simpler search pattern or search for specific tool names.`,
          {
            tool_references: [],
            metadata: {
              total_searched: 0,
              pattern: sanitizedPattern,
              error: errorMessage,
            },
          },
        ];
      }
    },
    {
      name: Constants.TOOL_SEARCH,
      description,
      schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export {
  createToolSearch,
  performLocalSearch,
  extractMcpServerName,
  isFromMcpServer,
  isFromAnyMcpServer,
  normalizeServerFilter,
  getAvailableMcpServers,
  getDeferredToolsListing,
  getBaseToolName,
  formatServerListing,
  sanitizeRegex,
  escapeRegexSpecialChars,
  isDangerousPattern,
  countNestedGroups,
  hasNestedQuantifiers,
};
