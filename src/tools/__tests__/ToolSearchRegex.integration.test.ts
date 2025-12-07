// src/tools/__tests__/ToolSearchRegex.integration.test.ts
/**
 * Integration tests for Tool Search Regex.
 * These tests hit the LIVE Code API and verify end-to-end search functionality.
 *
 * Run with: npm test -- ToolSearchRegex.integration.test.ts
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { describe, it, expect, beforeAll } from '@jest/globals';
import { createToolSearchRegexTool } from '../ToolSearchRegex';
import { createToolSearchToolRegistry } from '@/test/mockTools';

describe('ToolSearchRegex - Live API Integration', () => {
  let searchTool: ReturnType<typeof createToolSearchRegexTool>;
  const toolRegistry = createToolSearchToolRegistry();

  beforeAll(() => {
    const apiKey = process.env.LIBRECHAT_CODE_API_KEY;
    if (apiKey == null || apiKey === '') {
      throw new Error(
        'LIBRECHAT_CODE_API_KEY not set. Required for integration tests.'
      );
    }

    searchTool = createToolSearchRegexTool({ apiKey, toolRegistry });
  });

  it('searches for expense-related tools', async () => {
    const output = await searchTool.invoke({ query: 'expense' });

    expect(typeof output).toBe('string');
    expect(output).toContain('Found');
    expect(output).toContain('matching tools');
    expect(output).toContain('get_expenses');
    expect(output).toContain('calculate_expense_totals');
    expect(output).toContain('score: 0.95');
  }, 10000);

  it('searches for weather tools with OR pattern', async () => {
    const output = await searchTool.invoke({ query: 'weather|forecast' });

    expect(output).toContain('Found');
    expect(output).toContain('get_weather');
    expect(output).toContain('get_forecast');
  }, 10000);

  it('performs case-insensitive search', async () => {
    const output = await searchTool.invoke({ query: 'EMAIL' });

    expect(output).toContain('send_email');
  }, 10000);

  it('searches descriptions only', async () => {
    const output = await searchTool.invoke({
      query: 'database',
      fields: ['description'],
    });

    expect(output).toContain('Found');
    expect(output).toContain('Matched in: description');
    expect(output).toContain('run_database_query');
  }, 10000);

  it('searches parameter names', async () => {
    const output = await searchTool.invoke({
      query: 'query',
      fields: ['parameters'],
    });

    expect(output).toContain('Found');
    expect(output).toContain('Matched in: parameters');
  }, 10000);

  it('limits results to specified count', async () => {
    const output = await searchTool.invoke({
      query: 'get',
      max_results: 2,
    });

    expect(output).toContain('Found 2 matching tools');
    const toolMatches = output.match(/- \w+ \(score:/g);
    expect(toolMatches?.length).toBe(2);
  }, 10000);

  it('returns no matches for nonsense pattern', async () => {
    const output = await searchTool.invoke({
      query: 'xyznonexistent999',
    });

    expect(output).toContain('No tools matched');
    expect(output).toContain('Total tools searched:');
  }, 10000);

  it('uses regex character classes', async () => {
    const output = await searchTool.invoke({ query: 'get_[a-z]+' });

    expect(output).toContain('Found');
    expect(output).toContain('matching tools');
  }, 10000);

  it('sanitizes dangerous patterns with warning', async () => {
    const output = await searchTool.invoke({ query: '(a+)+' });

    expect(output).toContain(
      'Note: The provided pattern was converted to a literal search for safety'
    );
    // After sanitization, pattern is shown in the output
    expect(output).toContain('Total tools searched:');
    expect(output).toContain('No tools matched');
  }, 10000);

  it('searches across all fields', async () => {
    const output = await searchTool.invoke({
      query: 'text',
      fields: ['name', 'description', 'parameters'],
    });

    expect(output).toContain('Found');
    expect(output).toContain('translate_text');
  }, 10000);

  it('handles complex regex patterns', async () => {
    const output = await searchTool.invoke({
      query: '^(get|create)_.*',
    });

    expect(output).toContain('Found');
    expect(output).toContain('matching tools');
  }, 10000);

  it('prioritizes name matches over description matches', async () => {
    const output = await searchTool.invoke({ query: 'generate' });

    expect(output).toContain('generate_report');
    expect(output).toContain('score:');

    const lines = output.split('\n');
    const generateLine = lines.find((l: string) =>
      l.includes('generate_report')
    );
    expect(generateLine).toContain('score: 0.95'); // Name match = 0.95
  }, 10000);

  it('finds tools by partial name match', async () => {
    const output = await searchTool.invoke({ query: 'budget' });

    expect(output).toContain('create_budget');
  }, 10000);

  it('handles very specific patterns', async () => {
    const output = await searchTool.invoke({
      query: 'send_email',
    });

    expect(output).toContain('Found');
    expect(output).toContain('send_email');
    expect(output).toContain('score: 0.95');
  }, 10000);
});
