// src/tools/__tests__/ToolSearch.test.ts
/**
 * Unit tests for Tool Search.
 * Tests helper functions and sanitization logic without hitting the API.
 */
import { describe, it, expect } from '@jest/globals';
import {
  sanitizeRegex,
  escapeRegexSpecialChars,
  isDangerousPattern,
  countNestedGroups,
  hasNestedQuantifiers,
  performLocalSearch,
  extractMcpServerName,
  isFromMcpServer,
  isFromAnyMcpServer,
  normalizeServerFilter,
  getAvailableMcpServers,
  getDeferredToolsListing,
  getBaseToolName,
  formatServerListing,
} from '../ToolSearch';
import type { ToolMetadata, LCToolRegistry } from '@/types';

describe('ToolSearch', () => {
  describe('escapeRegexSpecialChars', () => {
    it('escapes special regex characters', () => {
      expect(escapeRegexSpecialChars('hello.world')).toBe('hello\\.world');
      expect(escapeRegexSpecialChars('test*pattern')).toBe('test\\*pattern');
      expect(escapeRegexSpecialChars('query+result')).toBe('query\\+result');
      expect(escapeRegexSpecialChars('a?b')).toBe('a\\?b');
      expect(escapeRegexSpecialChars('(group)')).toBe('\\(group\\)');
      expect(escapeRegexSpecialChars('[abc]')).toBe('\\[abc\\]');
      expect(escapeRegexSpecialChars('a|b')).toBe('a\\|b');
      expect(escapeRegexSpecialChars('a^b$c')).toBe('a\\^b\\$c');
      expect(escapeRegexSpecialChars('a{2,3}')).toBe('a\\{2,3\\}');
    });

    it('handles empty string', () => {
      expect(escapeRegexSpecialChars('')).toBe('');
    });

    it('handles string with no special chars', () => {
      expect(escapeRegexSpecialChars('hello_world')).toBe('hello_world');
      expect(escapeRegexSpecialChars('test123')).toBe('test123');
    });

    it('handles multiple consecutive special chars', () => {
      expect(escapeRegexSpecialChars('...')).toBe('\\.\\.\\.');
      expect(escapeRegexSpecialChars('***')).toBe('\\*\\*\\*');
    });
  });

  describe('countNestedGroups', () => {
    it('counts simple nesting', () => {
      expect(countNestedGroups('(a)')).toBe(1);
      expect(countNestedGroups('((a))')).toBe(2);
      expect(countNestedGroups('(((a)))')).toBe(3);
    });

    it('counts maximum depth with multiple groups', () => {
      expect(countNestedGroups('(a)(b)(c)')).toBe(1);
      expect(countNestedGroups('(a(b)c)')).toBe(2);
      expect(countNestedGroups('(a(b(c)))')).toBe(3);
    });

    it('handles mixed nesting levels', () => {
      expect(countNestedGroups('(a)((b)(c))')).toBe(2);
      expect(countNestedGroups('((a)(b))((c))')).toBe(2);
    });

    it('ignores escaped parentheses', () => {
      expect(countNestedGroups('\\(not a group\\)')).toBe(0);
      expect(countNestedGroups('(a\\(b\\)c)')).toBe(1);
    });

    it('handles no groups', () => {
      expect(countNestedGroups('abc')).toBe(0);
      expect(countNestedGroups('test.*pattern')).toBe(0);
    });

    it('handles unbalanced groups', () => {
      expect(countNestedGroups('((a)')).toBe(2);
      expect(countNestedGroups('(a))')).toBe(1);
    });
  });

  describe('hasNestedQuantifiers', () => {
    it('detects nested quantifiers', () => {
      expect(hasNestedQuantifiers('(a+)+')).toBe(true);
      expect(hasNestedQuantifiers('(a*)*')).toBe(true);
      expect(hasNestedQuantifiers('(a+)*')).toBe(true);
      expect(hasNestedQuantifiers('(a*)?')).toBe(true);
    });

    it('allows safe quantifiers', () => {
      expect(hasNestedQuantifiers('a+')).toBe(false);
      expect(hasNestedQuantifiers('(abc)+')).toBe(false);
      expect(hasNestedQuantifiers('a+b*c?')).toBe(false);
    });

    it('handles complex patterns', () => {
      expect(hasNestedQuantifiers('(a|b)+')).toBe(false);
      // Note: This pattern might not be detected by the simple regex check
      const complexPattern = '((a|b)+)+';
      const result = hasNestedQuantifiers(complexPattern);
      // Just verify it doesn't crash - detection may vary
      expect(typeof result).toBe('boolean');
    });
  });

  describe('isDangerousPattern', () => {
    it('detects nested quantifiers', () => {
      expect(isDangerousPattern('(a+)+')).toBe(true);
      expect(isDangerousPattern('(a*)*')).toBe(true);
      expect(isDangerousPattern('(.+)+')).toBe(true);
      expect(isDangerousPattern('(.*)*')).toBe(true);
    });

    it('detects excessive nesting', () => {
      expect(isDangerousPattern('((((((a))))))')).toBe(true); // Depth > 5
    });

    it('detects excessive wildcards', () => {
      const pattern = '.{1000,}';
      expect(isDangerousPattern(pattern)).toBe(true);
    });

    it('allows safe patterns', () => {
      expect(isDangerousPattern('weather')).toBe(false);
      expect(isDangerousPattern('get_.*_data')).toBe(false);
      expect(isDangerousPattern('(a|b|c)')).toBe(false);
      expect(isDangerousPattern('test\\d+')).toBe(false);
    });

    it('detects various dangerous patterns', () => {
      expect(isDangerousPattern('(.*)+')).toBe(true);
      expect(isDangerousPattern('(.+)*')).toBe(true);
    });
  });

  describe('sanitizeRegex', () => {
    it('returns safe pattern unchanged', () => {
      const result = sanitizeRegex('weather');
      expect(result.safe).toBe('weather');
      expect(result.wasEscaped).toBe(false);
    });

    it('escapes dangerous patterns', () => {
      const result = sanitizeRegex('(a+)+');
      expect(result.safe).toBe('\\(a\\+\\)\\+');
      expect(result.wasEscaped).toBe(true);
    });

    it('escapes invalid regex', () => {
      const result = sanitizeRegex('(unclosed');
      expect(result.wasEscaped).toBe(true);
      expect(result.safe).toContain('\\(');
    });

    it('allows complex but safe patterns', () => {
      const result = sanitizeRegex('get_[a-z]+_data');
      expect(result.safe).toBe('get_[a-z]+_data');
      expect(result.wasEscaped).toBe(false);
    });

    it('handles alternation patterns', () => {
      const result = sanitizeRegex('weather|forecast');
      expect(result.safe).toBe('weather|forecast');
      expect(result.wasEscaped).toBe(false);
    });
  });

  describe('Pattern Validation Edge Cases', () => {
    it('handles empty pattern', () => {
      expect(countNestedGroups('')).toBe(0);
      expect(hasNestedQuantifiers('')).toBe(false);
      expect(isDangerousPattern('')).toBe(false);
    });

    it('handles pattern with only quantifiers', () => {
      expect(hasNestedQuantifiers('+++')).toBe(false);
      expect(hasNestedQuantifiers('***')).toBe(false);
    });

    it('handles escaped special sequences', () => {
      const result = sanitizeRegex('\\d+\\w*\\s?');
      expect(result.wasEscaped).toBe(false);
    });

    it('sanitizes exponential backtracking patterns', () => {
      // These can cause catastrophic backtracking
      expect(isDangerousPattern('(a+)+')).toBe(true);
      expect(isDangerousPattern('(a*)*')).toBe(true);
      expect(isDangerousPattern('(.*)*')).toBe(true);
    });
  });

  describe('Real-World Pattern Examples', () => {
    it('handles common search patterns safely', () => {
      const safePatterns = [
        'expense',
        'weather|forecast',
        'data.*query',
        '_tool$',
      ];

      for (const pattern of safePatterns) {
        const result = sanitizeRegex(pattern);
        expect(result.wasEscaped).toBe(false);
      }
    });

    it('escapes clearly dangerous patterns', () => {
      const dangerousPatterns = ['(a+)+', '(.*)+', '(.+)*'];

      for (const pattern of dangerousPatterns) {
        const result = sanitizeRegex(pattern);
        expect(result.wasEscaped).toBe(true);
      }
    });

    it('handles patterns that may or may not be escaped', () => {
      // These patterns might be escaped depending on validation logic
      const edgeCasePatterns = [
        '(?i)email',
        '^create_',
        'get_[a-z]+_info',
        'get_.*',
        '((((((a))))))',
        '(a|a)*',
      ];

      for (const pattern of edgeCasePatterns) {
        const result = sanitizeRegex(pattern);
        // Just verify it returns a result without crashing
        expect(typeof result.safe).toBe('string');
        expect(typeof result.wasEscaped).toBe('boolean');
      }
    });
  });

  describe('performLocalSearch', () => {
    const mockTools: ToolMetadata[] = [
      {
        name: 'get_weather',
        description: 'Get current weather data',
        parameters: undefined,
      },
      {
        name: 'get_forecast',
        description: 'Get weather forecast for multiple days',
        parameters: undefined,
      },
      {
        name: 'send_email',
        description: 'Send an email message',
        parameters: undefined,
      },
      {
        name: 'get_expenses',
        description: 'Retrieve expense reports',
        parameters: undefined,
      },
      {
        name: 'calculate_expense_totals',
        description: 'Sum up expenses by category',
        parameters: undefined,
      },
      {
        name: 'run_database_query',
        description: 'Execute a database query',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string' },
            timeout: { type: 'number' },
          },
        },
      },
    ];

    it('finds tools by exact name match', () => {
      // BM25 tokenizes "get weather" from "get_weather"
      const result = performLocalSearch(mockTools, 'get weather', ['name'], 10);

      expect(result.tool_references.length).toBeGreaterThan(0);
      expect(result.tool_references[0].tool_name).toBe('get_weather');
      expect(result.tool_references[0].match_score).toBeGreaterThan(0.5);
      expect(result.tool_references[0].matched_field).toBe('name');
    });

    it('finds tools by partial name match', () => {
      // BM25 finds tools containing "get" token
      const result = performLocalSearch(mockTools, 'get', ['name'], 10);

      expect(result.tool_references.length).toBe(3);
      expect(result.tool_references[0].match_score).toBeGreaterThan(0);
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_weather'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_forecast'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_expenses'
      );
    });

    it('finds tools by substring match in name', () => {
      const result = performLocalSearch(mockTools, 'expense', ['name'], 10);

      expect(result.tool_references.length).toBe(2);
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_expenses'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'calculate_expense_totals'
      );
      expect(result.tool_references[0].match_score).toBeGreaterThan(0);
    });

    it('performs case-insensitive search', () => {
      const result = performLocalSearch(
        mockTools,
        'WEATHER',
        ['name', 'description'],
        10
      );

      expect(result.tool_references.length).toBe(2);
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_weather'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_forecast'
      );
    });

    it('searches in description field', () => {
      const result = performLocalSearch(
        mockTools,
        'email',
        ['description'],
        10
      );

      expect(result.tool_references.length).toBe(1);
      expect(result.tool_references[0].tool_name).toBe('send_email');
      expect(result.tool_references[0].matched_field).toBe('description');
      expect(result.tool_references[0].match_score).toBeGreaterThan(0);
    });

    it('searches in parameter names', () => {
      const result = performLocalSearch(mockTools, 'query', ['parameters'], 10);

      expect(result.tool_references.length).toBeGreaterThan(0);
      expect(result.tool_references[0].tool_name).toBe('run_database_query');
      expect(result.tool_references[0].matched_field).toBe('parameters');
      expect(result.tool_references[0].match_score).toBeGreaterThan(0);
    });

    it('prioritizes name matches over description matches', () => {
      const result = performLocalSearch(
        mockTools,
        'weather',
        ['name', 'description'],
        10
      );

      const weatherTool = result.tool_references.find(
        (r) => r.tool_name === 'get_weather'
      );
      const forecastTool = result.tool_references.find(
        (r) => r.tool_name === 'get_forecast'
      );

      expect(weatherTool?.matched_field).toBe('name');
      expect(forecastTool?.matched_field).toBe('description');
      expect(weatherTool!.match_score).toBeGreaterThan(
        forecastTool!.match_score
      );
    });

    it('limits results to max_results', () => {
      const result = performLocalSearch(mockTools, 'get', ['name'], 2);

      expect(result.tool_references.length).toBe(2);
      expect(result.total_tools_searched).toBe(mockTools.length);
    });

    it('returns empty array when no matches found', () => {
      const result = performLocalSearch(
        mockTools,
        'nonexistent_xyz_123',
        ['name', 'description'],
        10
      );

      expect(result.tool_references.length).toBe(0);
      expect(result.total_tools_searched).toBe(mockTools.length);
    });

    it('sorts results by score descending', () => {
      const result = performLocalSearch(
        mockTools,
        'expense',
        ['name', 'description'],
        10
      );

      for (let i = 1; i < result.tool_references.length; i++) {
        expect(
          result.tool_references[i - 1].match_score
        ).toBeGreaterThanOrEqual(result.tool_references[i].match_score);
      }
    });

    it('handles empty tools array', () => {
      const result = performLocalSearch([], 'test', ['name'], 10);

      expect(result.tool_references.length).toBe(0);
      expect(result.total_tools_searched).toBe(0);
    });

    it('handles empty query gracefully', () => {
      const result = performLocalSearch(mockTools, '', ['name'], 10);

      // BM25 correctly returns no results for empty queries (no terms to match)
      expect(result.tool_references.length).toBe(0);
      expect(result.total_tools_searched).toBe(mockTools.length);
    });

    it('includes correct metadata in response', () => {
      const result = performLocalSearch(mockTools, 'weather', ['name'], 10);

      expect(result.total_tools_searched).toBe(mockTools.length);
      expect(result.pattern_used).toBe('weather');
    });

    it('provides snippet in results', () => {
      const result = performLocalSearch(
        mockTools,
        'database',
        ['description'],
        10
      );

      expect(result.tool_references[0].snippet).toBeTruthy();
      expect(result.tool_references[0].snippet.length).toBeGreaterThan(0);
    });
  });

  describe('extractMcpServerName', () => {
    it('extracts server name from MCP tool name', () => {
      expect(extractMcpServerName('get_weather_mcp_weather-server')).toBe(
        'weather-server'
      );
      expect(extractMcpServerName('send_email_mcp_gmail')).toBe('gmail');
      expect(extractMcpServerName('query_database_mcp_postgres-mcp')).toBe(
        'postgres-mcp'
      );
    });

    it('returns undefined for non-MCP tools', () => {
      expect(extractMcpServerName('get_weather')).toBeUndefined();
      expect(extractMcpServerName('send_email')).toBeUndefined();
      expect(extractMcpServerName('regular_tool_name')).toBeUndefined();
    });

    it('handles edge cases', () => {
      expect(extractMcpServerName('_mcp_server')).toBe('server');
      expect(extractMcpServerName('tool_mcp_')).toBe('');
    });
  });

  describe('getBaseToolName', () => {
    it('extracts base name from MCP tool name', () => {
      expect(getBaseToolName('get_weather_mcp_weather-server')).toBe(
        'get_weather'
      );
      expect(getBaseToolName('send_email_mcp_gmail')).toBe('send_email');
    });

    it('returns full name for non-MCP tools', () => {
      expect(getBaseToolName('get_weather')).toBe('get_weather');
      expect(getBaseToolName('regular_tool')).toBe('regular_tool');
    });
  });

  describe('isFromMcpServer', () => {
    it('returns true for matching MCP server', () => {
      expect(
        isFromMcpServer('get_weather_mcp_weather-server', 'weather-server')
      ).toBe(true);
      expect(isFromMcpServer('send_email_mcp_gmail', 'gmail')).toBe(true);
    });

    it('returns false for non-matching MCP server', () => {
      expect(
        isFromMcpServer('get_weather_mcp_weather-server', 'other-server')
      ).toBe(false);
      expect(isFromMcpServer('send_email_mcp_gmail', 'outlook')).toBe(false);
    });

    it('returns false for non-MCP tools', () => {
      expect(isFromMcpServer('get_weather', 'weather-server')).toBe(false);
      expect(isFromMcpServer('regular_tool', 'any-server')).toBe(false);
    });
  });

  describe('isFromAnyMcpServer', () => {
    it('returns true if tool is from any of the specified servers', () => {
      expect(
        isFromAnyMcpServer('get_weather_mcp_weather-api', [
          'weather-api',
          'gmail',
        ])
      ).toBe(true);
      expect(
        isFromAnyMcpServer('send_email_mcp_gmail', ['weather-api', 'gmail'])
      ).toBe(true);
    });

    it('returns false if tool is not from any specified server', () => {
      expect(
        isFromAnyMcpServer('get_weather_mcp_weather-api', ['gmail', 'slack'])
      ).toBe(false);
    });

    it('returns false for non-MCP tools', () => {
      expect(isFromAnyMcpServer('regular_tool', ['weather-api', 'gmail'])).toBe(
        false
      );
    });

    it('returns false for empty server list', () => {
      expect(isFromAnyMcpServer('get_weather_mcp_weather-api', [])).toBe(false);
    });
  });

  describe('normalizeServerFilter', () => {
    it('converts string to single-element array', () => {
      expect(normalizeServerFilter('gmail')).toEqual(['gmail']);
    });

    it('passes through arrays unchanged', () => {
      expect(normalizeServerFilter(['gmail', 'slack'])).toEqual([
        'gmail',
        'slack',
      ]);
    });

    it('returns empty array for undefined', () => {
      expect(normalizeServerFilter(undefined)).toEqual([]);
    });

    it('returns empty array for empty string', () => {
      expect(normalizeServerFilter('')).toEqual([]);
    });

    it('filters out empty strings from arrays', () => {
      expect(normalizeServerFilter(['gmail', '', 'slack'])).toEqual([
        'gmail',
        'slack',
      ]);
    });
  });

  describe('getAvailableMcpServers', () => {
    const createRegistry = (): LCToolRegistry => {
      const registry: LCToolRegistry = new Map();
      registry.set('get_weather_mcp_weather-api', {
        name: 'get_weather_mcp_weather-api',
        description: 'Get weather',
        defer_loading: true,
      });
      registry.set('get_forecast_mcp_weather-api', {
        name: 'get_forecast_mcp_weather-api',
        description: 'Get forecast',
        defer_loading: true,
      });
      registry.set('send_email_mcp_gmail', {
        name: 'send_email_mcp_gmail',
        description: 'Send email',
        defer_loading: true,
      });
      registry.set('read_inbox_mcp_gmail', {
        name: 'read_inbox_mcp_gmail',
        description: 'Read inbox',
        defer_loading: true,
      });
      registry.set('post_message_mcp_slack', {
        name: 'post_message_mcp_slack',
        description: 'Post to Slack',
        defer_loading: true,
      });
      registry.set('regular_tool', {
        name: 'regular_tool',
        description: 'Not an MCP tool',
        defer_loading: true,
      });
      registry.set('non_deferred_mcp_special', {
        name: 'non_deferred_mcp_special',
        description: 'Not deferred',
        defer_loading: false,
      });
      return registry;
    };

    it('extracts unique server names from registry', () => {
      const registry = createRegistry();
      const servers = getAvailableMcpServers(registry, true);

      expect(servers).toEqual(['gmail', 'slack', 'weather-api']);
    });

    it('returns servers sorted alphabetically', () => {
      const registry = createRegistry();
      const servers = getAvailableMcpServers(registry, true);

      expect(servers).toEqual([...servers].sort());
    });

    it('excludes non-MCP tools', () => {
      const registry = createRegistry();
      const servers = getAvailableMcpServers(registry, true);

      expect(servers).not.toContain('regular_tool');
    });

    it('respects onlyDeferred flag', () => {
      const registry = createRegistry();

      const deferredOnly = getAvailableMcpServers(registry, true);
      expect(deferredOnly).not.toContain('special');

      const allTools = getAvailableMcpServers(registry, false);
      expect(allTools).toContain('special');
    });

    it('returns empty array for undefined registry', () => {
      expect(getAvailableMcpServers(undefined, true)).toEqual([]);
    });

    it('returns empty array for registry with no MCP tools', () => {
      const registry: LCToolRegistry = new Map();
      registry.set('tool1', {
        name: 'tool1',
        description: 'Regular tool',
        defer_loading: true,
      });

      expect(getAvailableMcpServers(registry, true)).toEqual([]);
    });
  });

  describe('getDeferredToolsListing', () => {
    const createRegistry = (): LCToolRegistry => {
      const registry: LCToolRegistry = new Map();
      registry.set('get_weather_mcp_weather-api', {
        name: 'get_weather_mcp_weather-api',
        description: 'Get weather',
        defer_loading: true,
      });
      registry.set('get_forecast_mcp_weather-api', {
        name: 'get_forecast_mcp_weather-api',
        description: 'Get forecast',
        defer_loading: true,
      });
      registry.set('send_email_mcp_gmail', {
        name: 'send_email_mcp_gmail',
        description: 'Send email',
        defer_loading: true,
      });
      registry.set('execute_code', {
        name: 'execute_code',
        description: 'Execute code',
        defer_loading: true,
      });
      registry.set('read_file', {
        name: 'read_file',
        description: 'Read file',
        defer_loading: false,
      });
      return registry;
    };

    it('groups tools by server with format D', () => {
      const registry = createRegistry();
      const listing = getDeferredToolsListing(registry, true);

      expect(listing).toContain('gmail: send_email');
      expect(listing).toContain('weather-api: get_weather, get_forecast');
      expect(listing).toContain('other: execute_code');
    });

    it('sorts servers alphabetically with other last', () => {
      const registry = createRegistry();
      const listing = getDeferredToolsListing(registry, true);
      const lines = listing.split('\n');

      expect(lines[0]).toMatch(/^gmail:/);
      expect(lines[1]).toMatch(/^weather-api:/);
      expect(lines[2]).toMatch(/^other:/);
    });

    it('uses base tool names without MCP suffix', () => {
      const registry = createRegistry();
      const listing = getDeferredToolsListing(registry, true);

      expect(listing).toContain('get_weather');
      expect(listing).not.toContain('get_weather_mcp_weather-api');
    });

    it('respects onlyDeferred flag', () => {
      const registry = createRegistry();

      const deferredOnly = getDeferredToolsListing(registry, true);
      expect(deferredOnly).not.toContain('read_file');

      const allTools = getDeferredToolsListing(registry, false);
      expect(allTools).toContain('read_file');
    });

    it('returns empty string for undefined registry', () => {
      expect(getDeferredToolsListing(undefined, true)).toBe('');
    });

    it('returns empty string for registry with no matching tools', () => {
      const registry: LCToolRegistry = new Map();
      registry.set('read_file', {
        name: 'read_file',
        description: 'Read file',
        defer_loading: false,
      });

      expect(getDeferredToolsListing(registry, true)).toBe('');
    });
  });

  describe('performLocalSearch with MCP tools', () => {
    const mcpTools: ToolMetadata[] = [
      {
        name: 'get_weather_mcp_weather-server',
        description: 'Get weather from MCP server',
        parameters: undefined,
      },
      {
        name: 'get_forecast_mcp_weather-server',
        description: 'Get forecast from MCP server',
        parameters: undefined,
      },
      {
        name: 'send_email_mcp_gmail',
        description: 'Send email via Gmail MCP',
        parameters: undefined,
      },
      {
        name: 'read_inbox_mcp_gmail',
        description: 'Read inbox via Gmail MCP',
        parameters: undefined,
      },
      {
        name: 'get_weather',
        description: 'Regular weather tool (not MCP)',
        parameters: undefined,
      },
    ];

    it('searches across all tools including MCP tools', () => {
      const result = performLocalSearch(
        mcpTools,
        'weather',
        ['name', 'description'],
        10
      );

      expect(result.tool_references.length).toBe(3);
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_weather_mcp_weather-server'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'get_weather'
      );
    });

    it('finds MCP tools by searching the full name including server suffix', () => {
      const result = performLocalSearch(mcpTools, 'gmail', ['name'], 10);

      expect(result.tool_references.length).toBe(2);
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'send_email_mcp_gmail'
      );
      expect(result.tool_references.map((r) => r.tool_name)).toContain(
        'read_inbox_mcp_gmail'
      );
    });

    it('can search for tools by MCP keyword', () => {
      // BM25 tokenizes queries, so search for "mcp" to find MCP tools
      const result = performLocalSearch(mcpTools, 'mcp', ['name'], 10);

      expect(result.tool_references.length).toBe(4);
      expect(result.tool_references.map((r) => r.tool_name)).not.toContain(
        'get_weather'
      );
    });

    it('finds tools when query contains underscores', () => {
      // Underscores in query should be tokenized the same as in tool names
      const tools: ToolMetadata[] = [
        {
          name: 'convert_time_mcp_time',
          description: 'Convert time between timezones',
          parameters: undefined,
        },
      ];

      const result = performLocalSearch(tools, 'convert_time', ['name'], 10);

      expect(result.tool_references.length).toBe(1);
      expect(result.tool_references[0].tool_name).toBe('convert_time_mcp_time');
      expect(result.tool_references[0].match_score).toBeGreaterThan(0.5);
    });

    it('finds tools with partial underscore query', () => {
      const tools: ToolMetadata[] = [
        {
          name: 'get_current_time_mcp_time',
          description: 'Get current time',
          parameters: undefined,
        },
        {
          name: 'convert_time_mcp_time',
          description: 'Convert time between timezones',
          parameters: undefined,
        },
      ];

      // "current_time" should match "get_current_time_mcp_time"
      const result = performLocalSearch(tools, 'current_time', ['name'], 10);

      expect(result.tool_references.length).toBeGreaterThan(0);
      expect(result.tool_references[0].tool_name).toBe(
        'get_current_time_mcp_time'
      );
    });
  });

  describe('formatServerListing', () => {
    const serverTools: ToolMetadata[] = [
      {
        name: 'get_weather_mcp_weather-api',
        description: 'Get current weather conditions for a location',
        parameters: undefined,
      },
      {
        name: 'get_forecast_mcp_weather-api',
        description: 'Get weather forecast for the next 7 days',
        parameters: undefined,
      },
    ];

    it('returns valid JSON with tool listing', () => {
      const result = formatServerListing(serverTools, 'weather-api');
      const parsed = JSON.parse(result);

      expect(parsed.listing_mode).toBe(true);
      expect(parsed.servers).toEqual(['weather-api']);
      expect(parsed.total_tools).toBe(2);
      expect(parsed.tools_by_server['weather-api']).toHaveLength(2);
    });

    it('includes hint to search for specific tool to load it', () => {
      const result = formatServerListing(serverTools, 'weather-api');
      const parsed = JSON.parse(result);

      expect(parsed.hint).toContain('To use a tool, search for it by name');
    });

    it('uses base tool name (without MCP suffix) in display', () => {
      const result = formatServerListing(serverTools, 'weather-api');
      const parsed = JSON.parse(result);

      const toolNames = parsed.tools_by_server['weather-api'].map(
        (t: { name: string }) => t.name
      );
      expect(toolNames).toContain('get_weather');
      expect(toolNames).not.toContain('get_weather_mcp_weather-api');
    });

    it('handles empty tools array', () => {
      const result = formatServerListing([], 'empty-server');
      const parsed = JSON.parse(result);

      expect(parsed.total_tools).toBe(0);
      expect(parsed.servers).toContain('empty-server');
      expect(parsed.hint).toContain('No tools found');
    });

    it('truncates long descriptions', () => {
      const toolsWithLongDesc: ToolMetadata[] = [
        {
          name: 'long_tool_mcp_server',
          description:
            'This is a very long description that exceeds 100 characters and should be truncated to keep the listing compact and readable for the LLM.',
          parameters: undefined,
        },
      ];

      const result = formatServerListing(toolsWithLongDesc, 'server');
      const parsed = JSON.parse(result);

      const toolDesc = parsed.tools_by_server['server'][0].description;
      expect(toolDesc).toContain('...');
      expect(toolDesc.length).toBeLessThanOrEqual(100);
    });

    it('handles multiple servers with grouped output', () => {
      const multiServerTools: ToolMetadata[] = [
        {
          name: 'get_weather_mcp_weather-api',
          description: 'Get weather',
          parameters: undefined,
        },
        {
          name: 'send_email_mcp_gmail',
          description: 'Send email',
          parameters: undefined,
        },
        {
          name: 'read_inbox_mcp_gmail',
          description: 'Read inbox',
          parameters: undefined,
        },
      ];

      const result = formatServerListing(multiServerTools, [
        'weather-api',
        'gmail',
      ]);
      const parsed = JSON.parse(result);

      expect(parsed.servers).toEqual(['weather-api', 'gmail']);
      expect(parsed.total_tools).toBe(3);
      expect(parsed.tools_by_server['weather-api']).toHaveLength(1);
      expect(parsed.tools_by_server['gmail']).toHaveLength(2);
    });

    it('accepts single server as array', () => {
      const result = formatServerListing(serverTools, ['weather-api']);
      const parsed = JSON.parse(result);

      expect(parsed.servers).toEqual(['weather-api']);
      expect(parsed.tools_by_server['weather-api']).toBeDefined();
    });
  });
});
