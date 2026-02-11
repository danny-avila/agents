// src/tools/__tests__/ProgrammaticToolCalling.integration.test.ts
/**
 * Integration tests for Programmatic Tool Calling.
 * These tests hit the LIVE Code API and verify end-to-end functionality.
 *
 * Run with: npm test -- ProgrammaticToolCalling.integration.test.ts
 *
 * Requires LIBRECHAT_CODE_API_KEY environment variable.
 * Tests are skipped when the API key is not available.
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { describe, it, expect, beforeAll } from '@jest/globals';
import type * as t from '@/types';
import { createProgrammaticToolCallingTool } from '../ProgrammaticToolCalling';
import {
  createGetTeamMembersTool,
  createGetExpensesTool,
  createGetWeatherTool,
  createCalculatorTool,
  createProgrammaticToolRegistry,
} from '@/test/mockTools';

const apiKey = process.env.LIBRECHAT_CODE_API_KEY;
const shouldSkip = apiKey == null || apiKey === '';

const describeIfApiKey = shouldSkip ? describe.skip : describe;

describeIfApiKey('ProgrammaticToolCalling - Live API Integration', () => {
  let ptcTool: ReturnType<typeof createProgrammaticToolCallingTool>;
  let toolMap: t.ToolMap;
  let toolDefinitions: t.LCTool[];

  beforeAll(() => {
    const tools = [
      createGetTeamMembersTool(),
      createGetExpensesTool(),
      createGetWeatherTool(),
      createCalculatorTool(),
    ];

    toolMap = new Map(tools.map((t) => [t.name, t]));
    toolDefinitions = Array.from(createProgrammaticToolRegistry().values());

    ptcTool = createProgrammaticToolCallingTool({ apiKey: apiKey! });
  });

  it('executes simple single tool call', async () => {
    const args = {
      code: `
result = await get_weather(city="San Francisco")
print(f"Temperature: {result['temperature']}°F")
print(f"Condition: {result['condition']}")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    // When responseFormat is 'content_and_artifact', invoke() returns just the content
    // The artifact is available via other methods or when used in a graph
    expect(typeof output).toBe('string');
    expect(output).toContain('stdout:');
    expect(output).toContain('Temperature: 65°F');
    expect(output).toContain('Condition: Foggy');
  }, 10000);

  it('executes sequential tool calls in a loop', async () => {
    const args = {
      code: `
team = await get_team_members()
print(f"Team size: {len(team)}")

total = 0
for member in team:
    expenses = await get_expenses(user_id=member['id'])
    member_total = sum(e['amount'] for e in expenses)
    total += member_total
    print(f"{member['name']}: \${member_total:.2f}")

print(f"Grand total: \${total:.2f}")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('Team size: 3');
    expect(output).toContain('Alice:');
    expect(output).toContain('Bob:');
    expect(output).toContain('Charlie:');
    expect(output).toContain('Grand total:');
  }, 15000);

  it('executes parallel tool calls with asyncio.gather', async () => {
    const args = {
      code: `
import asyncio

cities = ["San Francisco", "New York", "London"]
results = await asyncio.gather(*[
    get_weather(city=city)
    for city in cities
])

for city, weather in zip(cities, results):
    print(f"{city}: {weather['temperature']}°F, {weather['condition']}")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('San Francisco: 65°F, Foggy');
    expect(output).toContain('New York: 75°F, Sunny');
    expect(output).toContain('London: 55°F, Rainy');
  }, 10000);

  it('handles conditional logic', async () => {
    const args = {
      code: `
team = await get_team_members()
high_spenders = []

for member in team:
    expenses = await get_expenses(user_id=member['id'])
    total = sum(e['amount'] for e in expenses)
    if total > 300:
        high_spenders.append((member['name'], total))

if high_spenders:
    print("High spenders (over $300):")
    for name, amount in high_spenders:
        print(f"  {name}: \${amount:.2f}")
else:
    print("No high spenders found.")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('High spenders');
    expect(output).toContain('Bob:');
    expect(output).toContain('Charlie:');
  }, 15000);

  it('handles early termination with break', async () => {
    const args = {
      code: `
team = await get_team_members()

for member in team:
    expenses = await get_expenses(user_id=member['id'])
    if any(e['category'] == 'equipment' for e in expenses):
        print(f"First member with equipment: {member['name']}")
        equipment_total = sum(e['amount'] for e in expenses if e['category'] == 'equipment')
        print(f"Equipment total: \${equipment_total:.2f}")
        break
else:
    print("No equipment expenses found")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('First member with equipment: Charlie');
    expect(output).toContain('Equipment total: $500.00');
  }, 15000);

  it('handles tool execution errors gracefully', async () => {
    const args = {
      code: `
cities = ["San Francisco", "InvalidCity", "New York"]

for city in cities:
    try:
        weather = await get_weather(city=city)
        print(f"{city}: {weather['temperature']}°F")
    except Exception as e:
        print(f"{city}: Error - {e}")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('San Francisco: 65°F');
    expect(output).toContain('InvalidCity: Error');
    expect(output).toContain('New York: 75°F');
  }, 15000);

  it('uses calculator tool', async () => {
    const args = {
      code: `
result1 = await calculator(expression="2 + 2 * 3")
result2 = await calculator(expression="(10 + 5) / 3")

print(f"2 + 2 * 3 = {result1['result']}")
print(f"(10 + 5) / 3 = {result2['result']:.2f}")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('2 + 2 * 3 = 8');
    expect(output).toContain('(10 + 5) / 3 = 5.00');
  }, 10000);

  it('mixes parallel and sequential execution', async () => {
    const args = {
      code: `
import asyncio

# Step 1: Get team (sequential)
team = await get_team_members()
print(f"Fetched {len(team)} team members")

# Step 2: Get all expenses in parallel
all_expenses = await asyncio.gather(*[
    get_expenses(user_id=member['id'])
    for member in team
])

# Step 3: Process
print("\\nExpense summary:")
for member, expenses in zip(team, all_expenses):
    total = sum(e['amount'] for e in expenses)
    print(f"  {member['name']}: \${total:.2f} ({len(expenses)} items)")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: toolDefinitions,
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('Fetched 3 team members');
    expect(output).toContain('Expense summary:');
    expect(output).toContain('Alice:');
    expect(output).toContain('Bob:');
    expect(output).toContain('Charlie:');
    expect(output).toContain('(2 items)');
    expect(output).toContain('(3 items)');
  }, 15000);

  it('uses only provided tool definitions (subset)', async () => {
    const weatherToolDef = toolDefinitions.find(
      (t) => t.name === 'get_weather'
    );

    const args = {
      code: `
import asyncio

sf, nyc = await asyncio.gather(
    get_weather(city="San Francisco"),
    get_weather(city="New York")
)

print(f"SF: {sf['temperature']}°F vs NYC: {nyc['temperature']}°F")
difference = abs(sf['temperature'] - nyc['temperature'])
print(f"Temperature difference: {difference}°F")
      `,
    };
    const toolCall = {
      name: 'programmatic_code_execution',
      args,
      toolMap,
      toolDefs: [weatherToolDef!],
    };

    const output = await ptcTool.invoke(args, { toolCall });

    expect(output).toContain('SF: 65°F vs NYC: 75°F');
    expect(output).toContain('Temperature difference: 10°F');
  }, 10000);
});
