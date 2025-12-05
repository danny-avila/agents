// src/scripts/programmatic_exec.ts
/**
 * Test script for Programmatic Tool Calling (PTC).
 * Run with: npm run programmatic_exec
 *
 * Demonstrates runtime toolMap injection - the tool map is passed
 * at invocation time, not at initialization time.
 *
 * This tests the PTC tool in isolation with mock tools.
 *
 * IMPORTANT: The Python code passed to PTC should NOT define the tool functions.
 * The Code API automatically generates async function stubs from the tool definitions.
 * The code should just CALL the tools as if they're already available:
 *   - result = await get_weather(city="SF")
 *   - results = await asyncio.gather(tool1(), tool2())
 */
import { config } from 'dotenv';
config();

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { LCTool, ToolMap } from '@/types';
import {
  createProgrammaticToolCallingTool,
  ProgrammaticRuntimeConfig,
} from '@/tools/ProgrammaticToolCalling';

// ============================================================================
// Mock Tool Factories
// ============================================================================

/**
 * Creates a mock get_team_members tool that returns a list of team members.
 */
function createGetTeamMembersTool(): StructuredToolInterface {
  return tool(
    async () => {
      // Simulate API delay
      await new Promise((resolve) => setTimeout(resolve, 50));
      return [
        { id: 'u1', name: 'Alice', department: 'Engineering' },
        { id: 'u2', name: 'Bob', department: 'Marketing' },
        { id: 'u3', name: 'Charlie', department: 'Engineering' },
      ];
    },
    {
      name: 'get_team_members',
      description: 'Get list of team members with their IDs and departments',
      schema: z.object({}),
    }
  );
}

/**
 * Creates a mock get_expenses tool that returns expenses for a user.
 */
function createGetExpensesTool(): StructuredToolInterface {
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
    async ({ user_id }: { user_id: string }) => {
      await new Promise((resolve) => setTimeout(resolve, 30));
      return expenseData[user_id] ?? [];
    },
    {
      name: 'get_expenses',
      description: 'Get expense records for a specific user',
      schema: z.object({
        user_id: z.string().describe('The user ID to fetch expenses for'),
      }),
    }
  );
}

/**
 * Creates a mock get_weather tool that returns weather data.
 */
function createGetWeatherTool(): StructuredToolInterface {
  const weatherData: Record<
    string,
    { temperature: number; condition: string }
  > = {
    'San Francisco': { temperature: 65, condition: 'Foggy' },
    'New York': { temperature: 75, condition: 'Sunny' },
    London: { temperature: 55, condition: 'Rainy' },
    Tokyo: { temperature: 80, condition: 'Humid' },
    SF: { temperature: 65, condition: 'Foggy' },
    NYC: { temperature: 75, condition: 'Sunny' },
  };

  return tool(
    async ({ city }: { city: string }) => {
      await new Promise((resolve) => setTimeout(resolve, 40));
      const weather = weatherData[city];
      if (!weather) {
        throw new Error(`Weather data not available for city: ${city}`);
      }
      return weather;
    },
    {
      name: 'get_weather',
      description: 'Get current weather for a city',
      schema: z.object({
        city: z.string().describe('City name'),
      }),
    }
  );
}

/**
 * Creates a mock calculator tool.
 */
function createCalculatorTool(): StructuredToolInterface {
  return tool(
    async ({ expression }: { expression: string }) => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      // Simple eval for demo (in production, use a proper math parser)
      // eslint-disable-next-line no-eval
      const result = eval(expression);
      return { expression, result };
    },
    {
      name: 'calculator',
      description: 'Evaluate a mathematical expression',
      schema: z.object({
        expression: z.string().describe('Mathematical expression to evaluate'),
      }),
    }
  );
}

// ============================================================================
// Tool Definitions (Schemas for Code API)
// ============================================================================

const toolDefinitions: LCTool[] = [
  {
    name: 'get_team_members',
    description: 'Get list of team members with their IDs and departments',
    parameters: {
      type: 'object',
      properties: {},
      required: [],
    },
  },
  {
    name: 'get_expenses',
    description: 'Get expense records for a specific user',
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
  },
  {
    name: 'get_weather',
    description: 'Get current weather for a city',
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
  },
];

// ============================================================================
// Test Runner
// ============================================================================

interface RunTestOptions {
  toolMap: ToolMap;
  tools?: LCTool[];
  session_id?: string;
  timeout?: number;
  showArtifact?: boolean;
}

async function runTest(
  ptcTool: ReturnType<typeof createProgrammaticToolCallingTool>,
  testName: string,
  code: string,
  options: RunTestOptions
): Promise<void> {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`TEST: ${testName}`);
  console.log('='.repeat(70));
  console.log('\nCode:');
  console.log('```python');
  console.log(code.trim());
  console.log('```\n');

  try {
    const startTime = Date.now();

    const runtimeConfig: ProgrammaticRuntimeConfig = {
      toolMap: options.toolMap,
    };

    const result = await ptcTool.invoke(
      {
        code,
        tools: options.tools ?? toolDefinitions,
        session_id: options.session_id,
        timeout: options.timeout,
      },
      { configurable: runtimeConfig }
    );

    const duration = Date.now() - startTime;

    console.log(`Result (${duration}ms):`);
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

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  console.log('Programmatic Tool Calling (PTC) - Test Script');
  console.log('==============================================');
  console.log('Demonstrating runtime toolMap injection\n');

  const apiKey = process.env.LIBRECHAT_CODE_API_KEY;
  if (!apiKey) {
    console.error(
      'Error: LIBRECHAT_CODE_API_KEY environment variable is not set.'
    );
    console.log('Please set it in your .env file or environment.');
    process.exit(1);
  }

  console.log('Creating mock tools...');
  const mockTools: StructuredToolInterface[] = [
    createGetTeamMembersTool(),
    createGetExpensesTool(),
    createGetWeatherTool(),
    createCalculatorTool(),
  ];

  const toolMap: ToolMap = new Map(mockTools.map((t) => [t.name, t]));
  console.log(
    `ToolMap contains ${toolMap.size} tools: ${Array.from(toolMap.keys()).join(', ')}`
  );

  console.log('\nCreating PTC tool (without toolMap)...');
  const ptcTool = createProgrammaticToolCallingTool({ apiKey });
  console.log('PTC tool created successfully!');
  console.log(
    'Note: toolMap will be passed at runtime with each invocation.\n'
  );

  const baseOptions = { toolMap };

  // =========================================================================
  // Test 1: Simple async tool call
  // =========================================================================
  await runTest(
    ptcTool,
    'Simple async tool call',
    `
# Tools are auto-generated as async functions - just await them
result = await get_weather(city="San Francisco")
print(f"Weather in SF: {result['temperature']}°F, {result['condition']}")
    `,
    { ...baseOptions, showArtifact: true }
  );

  // =========================================================================
  // Test 2: Sequential loop with await
  // =========================================================================
  await runTest(
    ptcTool,
    'Sequential loop - Process team expenses',
    `
# Each tool call uses await
team = await get_team_members()
print("Team expense report:")
print("-" * 30)
total = 0
for member in team:
    expenses = await get_expenses(user_id=member['id'])
    member_total = sum(e['amount'] for e in expenses)
    total += member_total
    print(f"{member['name']}: \${member_total:.2f}")
print("-" * 30)
print(f"Total: \${total:.2f}")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 3: Parallel execution with asyncio.gather
  // =========================================================================
  await runTest(
    ptcTool,
    'Parallel execution - Weather for multiple cities',
    `
# Use asyncio.gather for parallel tool calls - single round-trip!
import asyncio

cities = ["San Francisco", "New York", "London"]
results = await asyncio.gather(*[
    get_weather(city=city)
    for city in cities
])

print("Weather report:")
for city, weather in zip(cities, results):
    print(f"  {city}: {weather['temperature']}°F, {weather['condition']}")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 4: Chained dependencies
  // =========================================================================
  await runTest(
    ptcTool,
    'Chained dependencies - Get team then process each',
    `
# Get team first, then fetch expenses for each
team = await get_team_members()
engineering = [m for m in team if m['department'] == 'Engineering']

print(f"Engineering team ({len(engineering)} members):")
for member in engineering:
    expenses = await get_expenses(user_id=member['id'])
    equipment = sum(e['amount'] for e in expenses if e['category'] == 'equipment')
    print(f"  {member['name']}: \${equipment:.2f} on equipment")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 5: Conditional logic
  // =========================================================================
  await runTest(
    ptcTool,
    'Conditional logic - Find high spenders',
    `
team = await get_team_members()
high_spenders = []

for member in team:
    expenses = await get_expenses(user_id=member['id'])
    total = sum(e['amount'] for e in expenses)
    if total > 300:
        high_spenders.append((member['name'], total))

if high_spenders:
    print("High spenders (over $300):")
    for name, amount in sorted(high_spenders, key=lambda x: x[1], reverse=True):
        print(f"  {name}: \${amount:.2f}")
else:
    print("No high spenders found.")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 6: Mixed parallel and sequential
  // =========================================================================
  await runTest(
    ptcTool,
    'Mixed - Parallel expense fetch after sequential team fetch',
    `
import asyncio

# Step 1: Get team (one tool call)
team = await get_team_members()
print(f"Fetched {len(team)} team members")

# Step 2: Get all expenses in parallel (single round-trip for all!)
all_expenses = await asyncio.gather(*[
    get_expenses(user_id=member['id'])
    for member in team
])

# Step 3: Process and output
print("\\nExpense summary:")
for member, expenses in zip(team, all_expenses):
    total = sum(e['amount'] for e in expenses)
    print(f"  {member['name']}: \${total:.2f} ({len(expenses)} items)")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 7: Calculator usage
  // =========================================================================
  await runTest(
    ptcTool,
    'Calculator tool usage',
    `
# All tools are async - use await
result1 = await calculator(expression="2 + 2 * 3")
result2 = await calculator(expression="(10 + 5) / 3")

print(f"2 + 2 * 3 = {result1['result']}")
print(f"(10 + 5) / 3 = {result2['result']:.2f}")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 8: Error handling in code
  // =========================================================================
  await runTest(
    ptcTool,
    'Error handling - Invalid city',
    `
# Tool errors become Python exceptions - handle with try/except
cities = ["San Francisco", "InvalidCity", "New York"]

for city in cities:
    try:
        weather = await get_weather(city=city)
        print(f"{city}: {weather['temperature']}°F")
    except Exception as e:
        print(f"{city}: Error - {e}")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 9: Early termination
  // =========================================================================
  await runTest(
    ptcTool,
    'Early termination - Stop when condition met',
    `
# Stop as soon as we find what we need - no wasted tool calls
team = await get_team_members()

for member in team:
    expenses = await get_expenses(user_id=member['id'])
    if any(e['category'] == 'equipment' for e in expenses):
        print(f"First team member with equipment expense: {member['name']}")
        equipment_total = sum(e['amount'] for e in expenses if e['category'] == 'equipment')
        print(f"Equipment total: \${equipment_total:.2f}")
        break
else:
    print("No team member has equipment expenses")
    `,
    baseOptions
  );

  // =========================================================================
  // Test 10: Subset of tools
  // =========================================================================
  await runTest(
    ptcTool,
    'Subset of tools - Only weather',
    `
# Only the weather tool is available in this execution
import asyncio

sf, nyc = await asyncio.gather(
    get_weather(city="San Francisco"),
    get_weather(city="New York")
)
print(f"SF: {sf['temperature']}°F vs NYC: {nyc['temperature']}°F")
difference = abs(sf['temperature'] - nyc['temperature'])
print(f"Temperature difference: {difference}°F")
    `,
    {
      ...baseOptions,
      tools: [toolDefinitions.find((t) => t.name === 'get_weather')!],
    }
  );

  console.log('\n' + '='.repeat(70));
  console.log('All tests completed!');
  console.log('='.repeat(70) + '\n');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
