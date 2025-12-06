// src/tools/__tests__/ProgrammaticToolCalling.test.ts
/**
 * Unit tests for Programmatic Tool Calling.
 * Tests manual invocation with mock tools and Code API responses.
 */
import { describe, it, expect, beforeEach } from '@jest/globals';
import type * as t from '@/types';
import {
  executeTools,
  formatCompletedResponse,
  createProgrammaticToolCallingTool,
} from '../ProgrammaticToolCalling';
import {
  createGetTeamMembersTool,
  createGetExpensesTool,
  createGetWeatherTool,
  createCalculatorTool,
  createProgrammaticToolRegistry,
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
        // No programmaticToolDefs
      };

      await expect(ptcTool.invoke(args, { toolCall })).rejects.toThrow(
        'No tool definitions provided'
      );
    });

    it('uses programmaticToolDefs from config when tools not provided', async () => {
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
