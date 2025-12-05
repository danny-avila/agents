// src/scripts/tool_search_regex.ts
/**
 * Test script for the Tool Search Regex tool.
 * Run with: npm run tool_search_regex
 *
 * Demonstrates runtime registry injection - the tool registry is passed
 * at invocation time, not at initialization time.
 */
import { config } from 'dotenv';
config();

import {
  createToolSearchRegexTool,
  ToolSearchRuntimeConfig,
} from '@/tools/ToolSearchRegex';
import type { LCTool, LCToolRegistry } from '@/types';

/**
 * Creates a sample tool registry with various mock tools for testing.
 * These simulate the deferred tools that would exist in a real system.
 */
function createSampleToolRegistry(): LCToolRegistry {
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
      defer_loading: false, // This one is NOT deferred - should be excluded by default
    },
  ];

  return new Map(tools.map((t) => [t.name, t]));
}

interface RunTestOptions {
  fields?: ('name' | 'description' | 'parameters')[];
  max_results?: number;
  showArtifact?: boolean;
  toolRegistry: LCToolRegistry;
  onlyDeferred?: boolean;
}

async function runTest(
  searchTool: ReturnType<typeof createToolSearchRegexTool>,
  testName: string,
  query: string,
  options: RunTestOptions
): Promise<void> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`TEST: ${testName}`);
  console.log(`Query: "${query}"`);
  if (options.fields) console.log(`Fields: ${options.fields.join(', ')}`);
  if (options.max_results) console.log(`Max Results: ${options.max_results}`);
  console.log('='.repeat(60));

  try {
    const startTime = Date.now();

    const runtimeConfig: ToolSearchRuntimeConfig = {
      toolRegistry: options.toolRegistry,
      onlyDeferred: options.onlyDeferred ?? true,
    };

    const result = await searchTool.invoke(
      {
        query,
        fields: options.fields,
        max_results: options.max_results,
      },
      { configurable: runtimeConfig }
    );
    const duration = Date.now() - startTime;

    console.log(`\nResult (${duration}ms):`);
    if (Array.isArray(result)) {
      console.log(result[0]);
      if (options.showArtifact) {
        console.log('\n--- Artifact ---');
        console.dir(result[1], { depth: null });
      }
    } else {
      console.log(result);
    }
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
  }
}

async function main(): Promise<void> {
  console.log('Tool Search Regex - Test Script');
  console.log('================================');
  console.log('Demonstrating runtime tool registry injection\n');

  const apiKey = process.env.LIBRECHAT_CODE_API_KEY;
  if (!apiKey) {
    console.error(
      'Error: LIBRECHAT_CODE_API_KEY environment variable is not set.'
    );
    console.log('Please set it in your .env file or environment.');
    process.exit(1);
  }

  console.log('Creating sample tool registry...');
  const toolRegistry = createSampleToolRegistry();
  console.log(
    `Registry contains ${toolRegistry.size} tools (${Array.from(toolRegistry.values()).filter((t) => t.defer_loading).length} deferred)`
  );

  console.log('\nCreating Tool Search Regex tool (without registry)...');
  const searchTool = createToolSearchRegexTool({ apiKey });
  console.log('Tool created successfully!');
  console.log(
    'Note: Registry will be passed at runtime with each invocation.\n'
  );

  const baseOptions = { toolRegistry, onlyDeferred: true };

  // Test 1: Simple keyword search (with artifact display)
  await runTest(searchTool, 'Simple keyword search', 'expense', {
    ...baseOptions,
    showArtifact: true,
  });

  // Test 2: Search for weather-related tools
  await runTest(searchTool, 'Weather tools', 'weather|forecast', baseOptions);

  // Test 3: Search with case variations
  await runTest(searchTool, 'Case insensitive search', 'EMAIL', baseOptions);

  // Test 4: Search in description only
  await runTest(searchTool, 'Description-only search', 'database', {
    ...baseOptions,
    fields: ['description'],
  });

  // Test 5: Search with parameters field
  await runTest(searchTool, 'Parameters search', 'query', {
    ...baseOptions,
    fields: ['parameters'],
  });

  // Test 6: Limited results
  await runTest(searchTool, 'Limited to 2 results', 'get', {
    ...baseOptions,
    max_results: 2,
  });

  // Test 7: Pattern that matches nothing
  await runTest(searchTool, 'No matches', 'xyznonexistent123', baseOptions);

  // Test 8: Regex pattern with character class
  await runTest(
    searchTool,
    'Regex with character class',
    'get_[a-z]+',
    baseOptions
  );

  // Test 9: Dangerous pattern (should be sanitized)
  await runTest(
    searchTool,
    'Dangerous pattern (sanitized)',
    '(a+)+',
    baseOptions
  );

  // Test 10: Search all fields
  await runTest(searchTool, 'All fields search', 'text', {
    ...baseOptions,
    fields: ['name', 'description', 'parameters'],
  });

  // Test 11: Search ALL tools (not just deferred)
  await runTest(searchTool, 'Search ALL tools (incl. non-deferred)', 'calc', {
    toolRegistry,
    onlyDeferred: false, // Include non-deferred tools
  });

  console.log('\n' + '='.repeat(60));
  console.log('All tests completed!');
  console.log('='.repeat(60) + '\n');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
