// src/test/mockTools.ts
/**
 * Shared mock tools for testing across all test scripts.
 * Centralizes tool definitions to follow DRY principles.
 */
import { tool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { LCTool, LCToolRegistry } from '@/types';

// ============================================================================
// Mock Tool Implementations
// ============================================================================

/**
 * Mock get_team_members tool - returns list of team members
 */
export function createGetTeamMembersTool(): StructuredToolInterface {
  return tool(
    async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
      return [
        { id: 'u1', name: 'Alice', department: 'Engineering' },
        { id: 'u2', name: 'Bob', department: 'Marketing' },
        { id: 'u3', name: 'Charlie', department: 'Engineering' },
      ];
    },
    {
      name: 'get_team_members',
      description:
        'Get list of team members. Returns array of objects with id, name, and department fields.',
      schema: { type: 'object', properties: {}, required: [] },
    }
  );
}

/**
 * Mock get_expenses tool - returns expense records for a user
 */
export function createGetExpensesTool(): StructuredToolInterface {
  const expenseData: Record<
    string,
    Array<{ amount: number; category: string }>
  > = {
    u1: [
      { amount: 150.0, category: 'travel' },
      { amount: 75.5, category: 'meals' },
    ],
    u2: [
      { amount: 200.0, category: 'marketing' },
      { amount: 50.0, category: 'meals' },
      { amount: 300.0, category: 'events' },
    ],
    u3: [
      { amount: 500.0, category: 'equipment' },
      { amount: 120.0, category: 'travel' },
      { amount: 80.0, category: 'meals' },
    ],
  };

  return tool(
    async (input) => {
      const { user_id } = input as { user_id: string };
      await new Promise((resolve) => setTimeout(resolve, 30));
      return expenseData[user_id] ?? [];
    },
    {
      name: 'get_expenses',
      description:
        'Get expense records for a user. Returns array of objects with amount and category fields.',
      schema: {
        type: 'object',
        properties: {
          user_id: {
            type: 'string',
            description: 'The user ID to fetch expenses for',
          },
        },
        required: ['user_id'],
      },
    }
  );
}

/**
 * Mock get_weather tool - returns weather data for a city
 */
export function createGetWeatherTool(): StructuredToolInterface {
  const weatherData: Record<
    string,
    { temperature: number; condition: string } | undefined
  > = {
    'San Francisco': { temperature: 65, condition: 'Foggy' },
    'New York': { temperature: 75, condition: 'Sunny' },
    London: { temperature: 55, condition: 'Rainy' },
    Tokyo: { temperature: 80, condition: 'Humid' },
    SF: { temperature: 65, condition: 'Foggy' },
    NYC: { temperature: 75, condition: 'Sunny' },
  };

  return tool(
    async (input) => {
      const { city } = input as { city: string };
      await new Promise((resolve) => setTimeout(resolve, 40));
      const weather = weatherData[city];
      if (!weather) {
        throw new Error(`Weather data not available for city: ${city}`);
      }
      return weather;
    },
    {
      name: 'get_weather',
      description:
        'Get current weather for a city. Returns object with temperature (number) and condition (string) fields.',
      schema: {
        type: 'object',
        properties: {
          city: { type: 'string', description: 'City name' },
        },
        required: ['city'],
      },
    }
  );
}

/**
 * Mock calculator tool - evaluates mathematical expressions
 */
export function createCalculatorTool(): StructuredToolInterface {
  return tool(
    async (input) => {
      const { expression } = input as { expression: string };
      await new Promise((resolve) => setTimeout(resolve, 10));
      // Simple eval for demo (in production, use a proper math parser)

      const result = eval(expression);
      return { expression, result };
    },
    {
      name: 'calculator',
      description: 'Evaluate a mathematical expression',
      schema: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'Mathematical expression to evaluate',
          },
        },
        required: ['expression'],
      },
    }
  );
}

// ============================================================================
// Tool Registry Definitions
// ============================================================================

/**
 * Creates a tool registry for programmatic tool calling tests.
 * Tools are configured with allowed_callers to demonstrate classification.
 */
export function createProgrammaticToolRegistry(): LCToolRegistry {
  const toolDefs: LCTool[] = [
    {
      name: 'get_team_members',
      description:
        'Get list of team members. Returns array of objects with id, name, and department fields.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
      allowed_callers: ['code_execution'], // Programmatic-only
      defer_loading: false,
    },
    {
      name: 'get_expenses',
      description:
        'Get expense records for a user. Returns array of objects with amount and category fields.',
      parameters: {
        type: 'object',
        properties: {
          user_id: {
            type: 'string',
            description: 'The user ID to fetch expenses for',
          },
        },
        required: ['user_id'],
      },
      allowed_callers: ['code_execution'], // Programmatic-only
      defer_loading: false,
    },
    {
      name: 'get_weather',
      description:
        'Get current weather for a city. Returns object with temperature (number) and condition (string) fields.',
      parameters: {
        type: 'object',
        properties: {
          city: {
            type: 'string',
            description: 'City name',
          },
        },
        required: ['city'],
      },
      allowed_callers: ['direct', 'code_execution'], // Both contexts
      defer_loading: false,
    },
    {
      name: 'calculator',
      description: 'Evaluate a mathematical expression',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'Mathematical expression to evaluate',
          },
        },
        required: ['expression'],
      },
      allowed_callers: ['code_execution'], // Programmatic-only
      defer_loading: false,
    },
  ];

  return new Map(toolDefs.map((t) => [t.name, t]));
}

/**
 * Creates a sample tool registry for tool search tests.
 * Includes mix of deferred and non-deferred tools.
 */
export function createToolSearchToolRegistry(): LCToolRegistry {
  const tools: LCTool[] = [
    {
      name: 'get_expenses',
      description:
        'Retrieve expense records from the database. Supports filtering by date range, category, and amount.',
      parameters: {
        type: 'object',
        properties: {
          start_date: {
            type: 'string',
            description: 'Start date for filtering',
          },
          end_date: { type: 'string', description: 'End date for filtering' },
          category: { type: 'string', description: 'Expense category' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'calculate_expense_totals',
      description:
        'Calculate total expenses by category or time period. Returns aggregated financial data.',
      parameters: {
        type: 'object',
        properties: {
          group_by: {
            type: 'string',
            description: 'Group by category or month',
          },
        },
      },
      defer_loading: true,
    },
    {
      name: 'create_budget',
      description:
        'Create a new budget plan with spending limits for different categories.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Budget name' },
          limits: { type: 'object', description: 'Category spending limits' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'get_weather',
      description: 'Get current weather conditions for a specified location.',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City or coordinates' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'get_forecast',
      description: 'Get weather forecast for the next 7 days for a location.',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City or coordinates' },
          days: { type: 'number', description: 'Number of days to forecast' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'send_email',
      description:
        'Send an email to one or more recipients with attachments support.',
      parameters: {
        type: 'object',
        properties: {
          to: {
            type: 'array',
            items: { type: 'string' },
            description: 'Recipients',
          },
          subject: { type: 'string', description: 'Email subject' },
          body: { type: 'string', description: 'Email body' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'search_files',
      description: 'Search for files in the file system by name or content.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' },
          path: { type: 'string', description: 'Directory to search in' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'run_database_query',
      description:
        'Execute a SQL query against the database and return results.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'SQL query to execute' },
          database: { type: 'string', description: 'Target database' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'generate_report',
      description:
        'Generate a PDF or Excel report from data with customizable templates.',
      parameters: {
        type: 'object',
        properties: {
          template: { type: 'string', description: 'Report template name' },
          format: { type: 'string', description: 'Output format: pdf or xlsx' },
          data: { type: 'object', description: 'Data to include in report' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'translate_text',
      description:
        'Translate text between languages using machine translation.',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Text to translate' },
          source_lang: { type: 'string', description: 'Source language code' },
          target_lang: { type: 'string', description: 'Target language code' },
        },
      },
      defer_loading: true,
    },
    {
      name: 'calculator',
      description:
        'Perform mathematical calculations. Supports basic arithmetic and scientific functions.',
      defer_loading: false, // Not deferred - should be excluded by default
    },
  ];

  return new Map(tools.map((t) => [t.name, t]));
}
