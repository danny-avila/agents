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

import { createToolSearchRegexTool } from '@/tools/ToolSearchRegex';
import type { LCToolRegistry } from '@/types';
import { createToolSearchToolRegistry } from '@/test/mockTools';

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

    // Manual testing uses schema params directly
    // (ToolNode uses different param structure when injecting)
    const result = await searchTool.invoke({
      query,
      fields: options.fields,
      max_results: options.max_results,
    });
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
  const toolRegistry = createToolSearchToolRegistry();
  console.log(
    `Registry contains ${toolRegistry.size} tools (${Array.from(toolRegistry.values()).filter((t) => t.defer_loading).length} deferred)`
  );

  console.log('\nCreating Tool Search Regex tool WITH registry for testing...');
  const searchTool = createToolSearchRegexTool({ apiKey, toolRegistry });
  console.log('Tool created successfully!');
  console.log(
    'Note: In production, ToolNode injects toolRegistry via params when invoked through the graph.\n'
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
