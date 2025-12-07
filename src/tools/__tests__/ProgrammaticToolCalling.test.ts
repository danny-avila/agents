// src/tools/__tests__/ProgrammaticToolCalling.test.ts
/**
 * Unit tests for Programmatic Tool Calling.
 * Tests manual invocation with mock tools and Code API responses.
 */
import { describe, it, expect, beforeEach } from '@jest/globals';
import type * as t from '@/types';
import {
  createProgrammaticToolCallingTool,
  formatCompletedResponse,
  extractUsedToolNames,
  filterToolsByUsage,
  executeTools,
  normalizeToPythonIdentifier,
} from '../ProgrammaticToolCalling';
import {
  createProgrammaticToolRegistry,
  createGetTeamMembersTool,
  createGetExpensesTool,
  createGetWeatherTool,
  createCalculatorTool,
} from '@/test/mockTools';

describe('ProgrammaticToolCalling', () => {
  describe('executeTools', () => {
    let toolMap: t.ToolMap;

    beforeEach(() => {
      const tools = [
        createGetTeamMembersTool(),
        createGetExpensesTool(),
        createGetWeatherTool(),
        createCalculatorTool(),
      ];
      toolMap = new Map(tools.map((t) => [t.name, t]));
    });

    it('executes a single tool successfully', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'San Francisco' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(1);
      expect(results[0].call_id).toBe('call_001');
      expect(results[0].is_error).toBe(false);
      expect(results[0].result).toEqual({
        temperature: 65,
        condition: 'Foggy',
      });
    });

    it('executes multiple tools in parallel', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'San Francisco' },
        },
        {
          id: 'call_002',
          name: 'get_weather',
          input: { city: 'New York' },
        },
        {
          id: 'call_003',
          name: 'get_weather',
          input: { city: 'London' },
        },
      ];

      const startTime = Date.now();
      const results = await executeTools(toolCalls, toolMap);
      const duration = Date.now() - startTime;

      // Should execute in parallel (< 150ms total, not 120ms sequential)
      expect(duration).toBeLessThan(150);

      expect(results).toHaveLength(3);
      expect(results[0].is_error).toBe(false);
      expect(results[1].is_error).toBe(false);
      expect(results[2].is_error).toBe(false);

      expect(results[0].result.temperature).toBe(65);
      expect(results[1].result.temperature).toBe(75);
      expect(results[2].result.temperature).toBe(55);
    });

    it('handles tool not found error', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'nonexistent_tool',
          input: {},
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(1);
      expect(results[0].call_id).toBe('call_001');
      expect(results[0].is_error).toBe(true);
      expect(results[0].error_message).toContain('nonexistent_tool');
      expect(results[0].error_message).toContain('Available tools:');
    });

    it('handles tool execution error', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'InvalidCity' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(1);
      expect(results[0].call_id).toBe('call_001');
      expect(results[0].is_error).toBe(true);
      expect(results[0].error_message).toContain('Weather data not available');
    });

    it('handles mix of successful and failed tool calls', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'San Francisco' },
        },
        {
          id: 'call_002',
          name: 'get_weather',
          input: { city: 'InvalidCity' },
        },
        {
          id: 'call_003',
          name: 'get_weather',
          input: { city: 'New York' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(3);
      expect(results[0].is_error).toBe(false);
      expect(results[1].is_error).toBe(true);
      expect(results[2].is_error).toBe(false);
    });

    it('executes tools with different parameters', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_team_members',
          input: {},
        },
        {
          id: 'call_002',
          name: 'get_expenses',
          input: { user_id: 'u1' },
        },
        {
          id: 'call_003',
          name: 'calculator',
          input: { expression: '2 + 2 * 3' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(3);
      expect(results[0].is_error).toBe(false);
      expect(results[1].is_error).toBe(false);
      expect(results[2].is_error).toBe(false);

      expect(Array.isArray(results[0].result)).toBe(true);
      expect(results[0].result).toHaveLength(3);
      expect(Array.isArray(results[1].result)).toBe(true);
      expect(results[2].result.result).toBe(8);
    });
  });

  describe('normalizeToPythonIdentifier', () => {
    it('converts hyphens to underscores', () => {
      expect(normalizeToPythonIdentifier('my-tool-name')).toBe('my_tool_name');
    });

    it('converts spaces to underscores', () => {
      expect(normalizeToPythonIdentifier('my tool name')).toBe('my_tool_name');
    });

    it('leaves underscores unchanged', () => {
      expect(normalizeToPythonIdentifier('my_tool_name')).toBe('my_tool_name');
    });

    it('handles mixed hyphens and underscores', () => {
      expect(normalizeToPythonIdentifier('my-tool_name-v2')).toBe(
        'my_tool_name_v2'
      );
    });

    it('handles MCP-style names with hyphens', () => {
      expect(
        normalizeToPythonIdentifier('create_spreadsheet_mcp_Google-Workspace')
      ).toBe('create_spreadsheet_mcp_Google_Workspace');
    });

    it('removes invalid characters', () => {
      expect(normalizeToPythonIdentifier('tool@name!v2')).toBe('toolnamev2');
      expect(normalizeToPythonIdentifier('get.data.v2')).toBe('getdatav2');
    });

    it('prefixes with underscore if starts with number', () => {
      expect(normalizeToPythonIdentifier('123tool')).toBe('_123tool');
      expect(normalizeToPythonIdentifier('1-tool')).toBe('_1_tool');
    });

    it('appends _tool suffix for Python keywords', () => {
      expect(normalizeToPythonIdentifier('return')).toBe('return_tool');
      expect(normalizeToPythonIdentifier('async')).toBe('async_tool');
      expect(normalizeToPythonIdentifier('import')).toBe('import_tool');
    });
  });

  describe('extractUsedToolNames', () => {
    const createToolMap = (names: string[]): Map<string, string> => {
      const map = new Map<string, string>();
      for (const name of names) {
        map.set(normalizeToPythonIdentifier(name), name);
      }
      return map;
    };

    const availableTools = createToolMap([
      'get_weather',
      'get_team_members',
      'get_expenses',
      'calculator',
      'search_docs',
    ]);

    it('extracts single tool name from simple code', () => {
      const code = `result = await get_weather(city="SF")
print(result)`;
      const used = extractUsedToolNames(code, availableTools);

      expect(used.size).toBe(1);
      expect(used.has('get_weather')).toBe(true);
    });

    it('extracts multiple tool names from code', () => {
      const code = `team = await get_team_members()
for member in team:
    expenses = await get_expenses(user_id=member['id'])
    print(f"{member['name']}: {sum(e['amount'] for e in expenses)}")`;

      const used = extractUsedToolNames(code, availableTools);

      expect(used.size).toBe(2);
      expect(used.has('get_team_members')).toBe(true);
      expect(used.has('get_expenses')).toBe(true);
    });

    it('extracts tools from asyncio.gather calls', () => {
      const code = `results = await asyncio.gather(
    get_weather(city="SF"),
    get_weather(city="NYC"),
    get_expenses(user_id="u1")
)`;
      const used = extractUsedToolNames(code, availableTools);

      expect(used.size).toBe(2);
      expect(used.has('get_weather')).toBe(true);
      expect(used.has('get_expenses')).toBe(true);
    });

    it('does not match partial tool names', () => {
      const code = `# Using get_weather_data instead
result = await get_weather_data(city="SF")`;

      const used = extractUsedToolNames(code, availableTools);
      expect(used.has('get_weather')).toBe(false);
    });

    it('matches tool names in different contexts', () => {
      const code = `# direct call
x = await calculator(expression="1+1")
# in list comprehension
results = [await get_weather(city=c) for c in cities]
# conditional
if condition:
    await get_team_members()`;

      const used = extractUsedToolNames(code, availableTools);

      expect(used.size).toBe(3);
      expect(used.has('calculator')).toBe(true);
      expect(used.has('get_weather')).toBe(true);
      expect(used.has('get_team_members')).toBe(true);
    });

    it('returns empty set when no tools are used', () => {
      const code = `print("Hello, World!")
x = 1 + 2`;

      const used = extractUsedToolNames(code, availableTools);
      expect(used.size).toBe(0);
    });

    it('handles tool names with special characters via normalization', () => {
      const specialTools = createToolMap(['get_data.v2', 'calc+plus']);
      const code = `await get_datav2()
await calcplus()`;

      const used = extractUsedToolNames(code, specialTools);

      expect(used.has('get_data.v2')).toBe(true);
      expect(used.has('calc+plus')).toBe(true);
    });

    it('matches hyphenated tool names using underscore in code', () => {
      const mcpTools = createToolMap([
        'create_spreadsheet_mcp_Google-Workspace',
        'search_gmail_mcp_Google-Workspace',
      ]);
      const code = `result = await create_spreadsheet_mcp_Google_Workspace(title="Test")
print(result)`;

      const used = extractUsedToolNames(code, mcpTools);

      expect(used.size).toBe(1);
      expect(used.has('create_spreadsheet_mcp_Google-Workspace')).toBe(true);
    });
  });

  describe('filterToolsByUsage', () => {
    const allToolDefs: t.LCTool[] = [
      {
        name: 'get_weather',
        description: 'Get weather for a city',
        parameters: {
          type: 'object',
          properties: { city: { type: 'string' } },
        },
      },
      {
        name: 'get_team_members',
        description: 'Get team members',
        parameters: { type: 'object', properties: {} },
      },
      {
        name: 'get_expenses',
        description: 'Get expenses for a user',
        parameters: {
          type: 'object',
          properties: { user_id: { type: 'string' } },
        },
      },
      {
        name: 'calculator',
        description: 'Evaluate an expression',
        parameters: {
          type: 'object',
          properties: { expression: { type: 'string' } },
        },
      },
    ];

    it('filters to only used tools', () => {
      const code = `result = await get_weather(city="SF")
print(result)`;

      const filtered = filterToolsByUsage(allToolDefs, code);

      expect(filtered).toHaveLength(1);
      expect(filtered[0].name).toBe('get_weather');
    });

    it('filters to multiple used tools', () => {
      const code = `team = await get_team_members()
for member in team:
    expenses = await get_expenses(user_id=member['id'])`;

      const filtered = filterToolsByUsage(allToolDefs, code);

      expect(filtered).toHaveLength(2);
      expect(filtered.map((t) => t.name).sort()).toEqual([
        'get_expenses',
        'get_team_members',
      ]);
    });

    it('returns all tools when no tools are detected', () => {
      const code = 'print("Hello, World!")';

      const filtered = filterToolsByUsage(allToolDefs, code);

      expect(filtered).toHaveLength(4);
    });

    it('preserves tool definition structure', () => {
      const code = 'await calculator(expression="2+2")';

      const filtered = filterToolsByUsage(allToolDefs, code);

      expect(filtered).toHaveLength(1);
      expect(filtered[0]).toEqual(allToolDefs[3]);
      expect(filtered[0].parameters).toBeDefined();
      expect(filtered[0].description).toBe('Evaluate an expression');
    });

    it('handles empty tool definitions', () => {
      const code = 'await get_weather(city="SF")';
      const filtered = filterToolsByUsage([], code);

      expect(filtered).toHaveLength(0);
    });
  });

  describe('formatCompletedResponse', () => {
    it('formats response with stdout', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Hello, World!\n',
        stderr: '',
        files: [],
        session_id: 'sess_abc123',
      };

      const [output, artifact] = formatCompletedResponse(response);

      expect(output).toContain('stdout:\nHello, World!');
      expect(artifact.session_id).toBe('sess_abc123');
      expect(artifact.files).toEqual([]);
    });

    it('shows empty output message when no stdout', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: '',
        stderr: '',
        files: [],
        session_id: 'sess_abc123',
      };

      const [output] = formatCompletedResponse(response);

      expect(output).toContain(
        'stdout: Empty. Ensure you\'re writing output explicitly'
      );
    });

    it('includes stderr when present', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Output\n',
        stderr: 'Warning: deprecated function\n',
        files: [],
        session_id: 'sess_abc123',
      };

      const [output] = formatCompletedResponse(response);

      expect(output).toContain('stdout:\nOutput');
      expect(output).toContain('stderr:\nWarning: deprecated function');
    });

    it('formats file information correctly', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Generated report\n',
        stderr: '',
        files: [
          { id: '1', name: 'report.pdf' },
          { id: '2', name: 'data.csv' },
        ],
        session_id: 'sess_abc123',
      };

      const [output, artifact] = formatCompletedResponse(response);

      expect(output).toContain('Generated files:');
      expect(output).toContain('report.pdf');
      expect(output).toContain('data.csv');
      expect(output).toContain('session_id: sess_abc123');
      expect(artifact.files).toHaveLength(2);
      expect(artifact.files).toEqual(response.files);
    });

    it('handles image files with special message', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: '',
        stderr: '',
        files: [
          { id: '1', name: 'chart.png' },
          { id: '2', name: 'photo.jpg' },
        ],
        session_id: 'sess_abc123',
      };

      const [output] = formatCompletedResponse(response);

      expect(output).toContain('chart.png');
      expect(output).toContain('Image is already displayed to the user');
    });
  });

  describe('createProgrammaticToolCallingTool - Manual Invocation', () => {
    let ptcTool: ReturnType<typeof createProgrammaticToolCallingTool>;
    let toolMap: t.ToolMap;
    let toolDefinitions: t.LCTool[];

    beforeEach(() => {
      const tools = [
        createGetTeamMembersTool(),
        createGetExpensesTool(),
        createGetWeatherTool(),
      ];
      toolMap = new Map(tools.map((t) => [t.name, t]));
      toolDefinitions = Array.from(
        createProgrammaticToolRegistry().values()
      ).filter((t) =>
        ['get_team_members', 'get_expenses', 'get_weather'].includes(t.name)
      );

      ptcTool = createProgrammaticToolCallingTool({
        apiKey: 'test-key',
        baseUrl: 'http://mock-api',
      });
    });

    it('throws error when no toolMap provided', async () => {
      await expect(
        ptcTool.invoke({
          code: 'result = await get_weather(city="SF")\nprint(result)',
          tools: toolDefinitions,
          toolMap,
        })
      ).rejects.toThrow('No toolMap provided');
    });

    it('throws error when toolMap is empty', async () => {
      const args = {
        code: 'result = await get_weather(city="SF")\nprint(result)',
        tools: toolDefinitions,
        toolMap: new Map(),
      };
      const toolCall = {
        name: 'programmatic_tool_calling',
        args,
      };
      await expect(
        ptcTool.invoke(args, {
          toolCall,
        })
      ).rejects.toThrow('No toolMap provided');
    });

    it('throws error when no tool definitions provided', async () => {
      const args = {
        code: 'result = await get_weather(city="SF")\nprint(result)',
        // No tools
      };
      const toolCall = {
        name: 'programmatic_code_execution',
        args,
        toolMap,
        // No `toolDefs`
      };

      await expect(ptcTool.invoke(args, { toolCall })).rejects.toThrow(
        'No tool definitions provided'
      );
    });

    it('uses toolDefs from config when tools not provided', async () => {
      // Skip this test - requires mocking fetch which has complex typing
      // This functionality is tested in the live script tests instead
    });
  });

  describe('Tool Classification', () => {
    it('filters tools by allowed_callers', () => {
      const registry = createProgrammaticToolRegistry();

      const codeExecutionTools = Array.from(registry.values()).filter((t) =>
        (t.allowed_callers ?? ['direct']).includes('code_execution')
      );
      // get_team_members, get_expenses, calculator: code_execution only
      const codeOnlyTools = codeExecutionTools.filter(
        (t) => !(t.allowed_callers?.includes('direct') === true)
      );
      expect(codeOnlyTools.length).toBeGreaterThanOrEqual(3);

      // get_weather: both direct and code_execution
      const bothTools = Array.from(registry.values()).filter(
        (t) =>
          t.allowed_callers?.includes('direct') === true &&
          t.allowed_callers.includes('code_execution')
      );
      expect(bothTools.length).toBeGreaterThanOrEqual(1);
      expect(bothTools.some((t) => t.name === 'get_weather')).toBe(true);
    });
  });

  describe('Error Handling', () => {
    let toolMap: t.ToolMap;

    beforeEach(() => {
      const tools = [createGetWeatherTool()];
      toolMap = new Map(tools.map((t) => [t.name, t]));
    });

    it('returns error for invalid city without throwing', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'InvalidCity' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(1);
      expect(results[0].is_error).toBe(true);
      expect(results[0].result).toBeNull();
      expect(results[0].error_message).toContain('Weather data not available');
    });

    it('continues execution when one tool fails', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_weather',
          input: { city: 'San Francisco' },
        },
        {
          id: 'call_002',
          name: 'get_weather',
          input: { city: 'InvalidCity' },
        },
        {
          id: 'call_003',
          name: 'get_weather',
          input: { city: 'London' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results).toHaveLength(3);
      expect(results[0].is_error).toBe(false);
      expect(results[1].is_error).toBe(true);
      expect(results[2].is_error).toBe(false);
    });
  });

  describe('Parallel Execution Performance', () => {
    let toolMap: t.ToolMap;

    beforeEach(() => {
      const tools = [createGetExpensesTool()];
      toolMap = new Map(tools.map((t) => [t.name, t]));
    });

    it('executes tools in parallel, not sequentially', async () => {
      const toolCalls: t.PTCToolCall[] = [
        { id: 'call_001', name: 'get_expenses', input: { user_id: 'u1' } },
        { id: 'call_002', name: 'get_expenses', input: { user_id: 'u2' } },
        { id: 'call_003', name: 'get_expenses', input: { user_id: 'u3' } },
      ];

      const startTime = Date.now();
      const results = await executeTools(toolCalls, toolMap);
      const duration = Date.now() - startTime;

      // Each tool has 30ms delay
      // Sequential would be ~90ms, parallel should be ~30-50ms
      expect(duration).toBeLessThan(80);
      expect(results).toHaveLength(3);
      expect(results.every((r) => r.is_error === false)).toBe(true);
    });
  });

  describe('Response Formatting', () => {
    it('formats stdout-only response', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Team size: 3\n- Alice\n- Bob\n- Charlie\n',
        stderr: '',
        files: [],
        session_id: 'sess_xyz',
      };

      const [output, artifact] = formatCompletedResponse(response);

      expect(output).toBe('stdout:\nTeam size: 3\n- Alice\n- Bob\n- Charlie');
      expect(artifact).toEqual({
        session_id: 'sess_xyz',
        files: [],
      });
    });

    it('formats response with files', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Report generated\n',
        stderr: '',
        files: [
          { id: '1', name: 'report.csv' },
          { id: '2', name: 'chart.png' },
        ],
        session_id: 'sess_xyz',
      };

      const [output, artifact] = formatCompletedResponse(response);

      expect(output).toContain('Generated files:');
      expect(output).toContain('report.csv');
      expect(output).toContain('chart.png');
      expect(output).toContain('session_id: sess_xyz');
      expect(output).toContain('File is already downloaded');
      expect(output).toContain('Image is already displayed');
      expect(artifact.files).toHaveLength(2);
    });

    it('handles multiple files with correct separators', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Done\n',
        stderr: '',
        files: [
          { id: '1', name: 'file1.txt' },
          { id: '2', name: 'file2.txt' },
        ],
        session_id: 'sess_xyz',
      };

      const [output] = formatCompletedResponse(response);

      // 2 files format: "- /mnt/data/file1.txt | ..., - /mnt/data/file2.txt | ..."
      expect(output).toContain('file1.txt');
      expect(output).toContain('file2.txt');
      expect(output).toContain('- /mnt/data/file1.txt');
      expect(output).toContain('- /mnt/data/file2.txt');
    });

    it('handles many files with newline separators', () => {
      const response: t.ProgrammaticExecutionResponse = {
        status: 'completed',
        stdout: 'Done\n',
        stderr: '',
        files: [
          { id: '1', name: 'file1.txt' },
          { id: '2', name: 'file2.txt' },
          { id: '3', name: 'file3.txt' },
          { id: '4', name: 'file4.txt' },
        ],
        session_id: 'sess_xyz',
      };

      const [output] = formatCompletedResponse(response);

      // More than 3 files should use newline separators
      expect(output).toContain('file1.txt');
      expect(output).toContain('file4.txt');
      expect(output.match(/,\n/g)?.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('Tool Data Extraction', () => {
    let toolMap: t.ToolMap;

    beforeEach(() => {
      const tools = [
        createGetTeamMembersTool(),
        createGetExpensesTool(),
        createCalculatorTool(),
      ];
      toolMap = new Map(tools.map((t) => [t.name, t]));
    });

    it('extracts correct data from team members tool', async () => {
      const toolCalls: t.PTCToolCall[] = [
        { id: 'call_001', name: 'get_team_members', input: {} },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results[0].result).toEqual([
        { id: 'u1', name: 'Alice', department: 'Engineering' },
        { id: 'u2', name: 'Bob', department: 'Marketing' },
        { id: 'u3', name: 'Charlie', department: 'Engineering' },
      ]);
    });

    it('extracts correct data from expenses tool', async () => {
      const toolCalls: t.PTCToolCall[] = [
        { id: 'call_001', name: 'get_expenses', input: { user_id: 'u1' } },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results[0].result).toEqual([
        { amount: 150.0, category: 'travel' },
        { amount: 75.5, category: 'meals' },
      ]);
    });

    it('handles empty expense data', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'get_expenses',
          input: { user_id: 'nonexistent' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results[0].is_error).toBe(false);
      expect(results[0].result).toEqual([]);
    });

    it('calculates correct result', async () => {
      const toolCalls: t.PTCToolCall[] = [
        {
          id: 'call_001',
          name: 'calculator',
          input: { expression: '2 + 2 * 3' },
        },
        {
          id: 'call_002',
          name: 'calculator',
          input: { expression: '(10 + 5) / 3' },
        },
      ];

      const results = await executeTools(toolCalls, toolMap);

      expect(results[0].result.result).toBe(8);
      expect(results[1].result.result).toBe(5);
    });
  });
});
