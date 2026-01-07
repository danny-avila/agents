// src/tools/ToolSearchRegex.ts
import { z } from 'zod';
import { config } from 'dotenv';
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
  query: z.ZodString;
  fields: z.ZodDefault<
    z.ZodOptional<z.ZodArray<z.ZodEnum<['name', 'description', 'parameters']>>>
  >;
  max_results: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
}>;

/**
 * Creates the Zod schema with dynamic query description based on mode.
 * @param mode - The search mode determining query interpretation
 * @returns Zod schema for tool search parameters
 */
function createToolSearchSchema(mode: t.ToolSearchMode): ToolSearchSchema {
  const queryDescription =
    mode === 'local'
      ? 'Search term to find in tool names and descriptions. Case-insensitive substring matching.'
      : 'Regex pattern to search tool names and descriptions. Special regex characters will be sanitized for safety.';

  return z.object({
    query: z.string().min(1).max(MAX_PATTERN_LENGTH).describe(queryDescription),
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
  });
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
 * Performs safe local substring search without regex.
 * Uses case-insensitive String.includes() for complete safety against ReDoS.
 * @param tools - Array of tool metadata to search
 * @param query - The search term (treated as literal substring)
 * @param fields - Which fields to search
 * @param maxResults - Maximum results to return
 * @returns Search response with matching tools
 */
function performLocalSearch(
  tools: t.ToolMetadata[],
  query: string,
  fields: string[],
  maxResults: number
): t.ToolSearchResponse {
  const lowerQuery = query.toLowerCase();
  const results: t.ToolSearchResult[] = [];

  for (const tool of tools) {
    let bestScore = 0;
    let matchedField = '';
    let snippet = '';

    if (fields.includes('name')) {
      const lowerName = tool.name.toLowerCase();
      if (lowerName.includes(lowerQuery)) {
        const isExactMatch = lowerName === lowerQuery;
        const startsWithQuery = lowerName.startsWith(lowerQuery);
        bestScore = isExactMatch ? 1.0 : startsWithQuery ? 0.95 : 0.85;
        matchedField = 'name';
        snippet = tool.name;
      }
    }

    if (fields.includes('description') && tool.description) {
      const lowerDesc = tool.description.toLowerCase();
      if (lowerDesc.includes(lowerQuery) && bestScore === 0) {
        bestScore = 0.7;
        matchedField = 'description';
        snippet = tool.description.substring(0, 100);
      }
    }

    if (fields.includes('parameters') && tool.parameters?.properties) {
      const paramNames = Object.keys(tool.parameters.properties)
        .join(' ')
        .toLowerCase();
      if (paramNames.includes(lowerQuery) && bestScore === 0) {
        bestScore = 0.55;
        matchedField = 'parameters';
        snippet = Object.keys(tool.parameters.properties).join(' ');
      }
    }

    if (bestScore > 0) {
      results.push({
        tool_name: tool.name,
        match_score: bestScore,
        matched_field: matchedField,
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
 * Formats search results into a human-readable string.
 * @param searchResponse - The parsed search response
 * @returns Formatted string for LLM consumption
 */
function formatSearchResults(searchResponse: t.ToolSearchResponse): string {
  const { tool_references, total_tools_searched, pattern_used } =
    searchResponse;

  if (tool_references.length === 0) {
    return `No tools matched the pattern "${pattern_used}".\nTotal tools searched: ${total_tools_searched}`;
  }

  let response = `Found ${tool_references.length} matching tools:\n\n`;

  for (const ref of tool_references) {
    response += `- ${ref.tool_name} (score: ${ref.match_score.toFixed(2)})\n`;
    response += `  Matched in: ${ref.matched_field}\n`;
    response += `  Snippet: ${ref.snippet}\n\n`;
  }

  response += `Total tools searched: ${total_tools_searched}\n`;
  response += `Pattern used: ${pattern_used}`;

  return response;
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
 * const tool = createToolSearchRegexTool({ apiKey, toolRegistry });
 * await tool.invoke({ query: 'expense.*report' });
 *
 * @example
 * // Option 2: Local mode (safe substring search, no API key needed)
 * const tool = createToolSearchRegexTool({ mode: 'local', toolRegistry });
 * await tool.invoke({ query: 'expense' });
 */
function createToolSearchRegexTool(
  initParams: t.ToolSearchRegexParams = {}
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

  const description =
    mode === 'local'
      ? `
Searches through available tools to find ones matching your search term.

Usage:
- Provide a search term to find in tool names and descriptions.
- Uses case-insensitive substring matching (fast and safe).
- Use this when you need to discover tools for a specific task.
- Results include tool names, match quality scores, and snippets showing where the match occurred.
- Higher scores (0.95+) indicate name matches, medium scores (0.70+) indicate description matches.
`.trim()
      : `
Searches through available tools to find ones matching your query pattern.

Usage:
- Provide a regex pattern to search tool names and descriptions.
- Use this when you need to discover tools for a specific task.
- Results include tool names, match quality scores, and snippets showing where the match occurred.
- Higher scores (0.9+) indicate name matches, medium scores (0.7+) indicate description matches.
`.trim();

  return tool<typeof schema>(
    async (params, config) => {
      const {
        query,
        fields = ['name', 'description'],
        max_results = 10,
      } = params;

      const {
        toolRegistry: paramToolRegistry,
        onlyDeferred: paramOnlyDeferred,
      } = config.toolCall ?? {};

      const toolRegistry = paramToolRegistry ?? initParams.toolRegistry;
      const onlyDeferred =
        paramOnlyDeferred !== undefined
          ? paramOnlyDeferred
          : defaultOnlyDeferred;

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
        .filter((lcTool) =>
          onlyDeferred === true ? lcTool.defer_loading === true : true
        )
        .map((lcTool) => ({
          name: lcTool.name,
          description: lcTool.description ?? '',
          parameters: simplifyParametersForSearch(lcTool.parameters),
        }));

      if (deferredTools.length === 0) {
        return [
          'No tools available to search. The tool registry is empty or no deferred tools are registered.',
          {
            tool_references: [],
            metadata: { total_searched: 0, pattern: query },
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
          console.warn('[ToolSearchRegex] stderr:', result.stderr);
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
        console.error('[ToolSearchRegex] Error:', error);

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
      name: Constants.TOOL_SEARCH_REGEX,
      description,
      schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export {
  createToolSearchRegexTool,
  performLocalSearch,
  sanitizeRegex,
  escapeRegexSpecialChars,
  isDangerousPattern,
  countNestedGroups,
  hasNestedQuantifiers,
};
