import { tool } from '@langchain/core/tools';
import type { GraphTools } from '@/types';
import { partitionAndMarkOpenRouterToolCache } from './toolCache';

type OpenRouterTool = {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: object;
  };
  cache_control?: { type: 'ephemeral' };
  defer_loading?: boolean;
};

function createOpenAITool(name: string): OpenRouterTool {
  return {
    type: 'function',
    function: {
      name,
      description: `${name} description`,
      parameters: {
        type: 'object',
        properties: {},
      },
    },
  };
}

describe('partitionAndMarkOpenRouterToolCache', () => {
  it('marks the last static OpenRouter tool before deferred tools', () => {
    const tools = [
      createOpenAITool('static_one'),
      createOpenAITool('static_two'),
      createOpenAITool('dynamic_one'),
    ] as GraphTools;

    const result = partitionAndMarkOpenRouterToolCache(
      tools,
      (name) => name === 'dynamic_one'
    ) as OpenRouterTool[];

    expect(result.map((entry) => entry.function.name)).toEqual([
      'static_one',
      'static_two',
      'dynamic_one',
    ]);
    expect(result[0]).not.toHaveProperty('cache_control');
    expect(result[1].cache_control).toEqual({ type: 'ephemeral' });
    expect(result[2]).not.toHaveProperty('cache_control');
  });

  it('converts LangChain tools to OpenAI tools before adding cache control', () => {
    const staticTool = tool(async () => 'static', {
      name: 'static_tool',
      description: 'Static tool',
      schema: {
        type: 'object',
        properties: {},
      },
    });
    const dynamicTool = tool(async () => 'dynamic', {
      name: 'dynamic_tool',
      description: 'Dynamic tool',
      schema: {
        type: 'object',
        properties: {},
      },
    });

    const result = partitionAndMarkOpenRouterToolCache(
      [dynamicTool, staticTool] as GraphTools,
      (name) => name === 'dynamic_tool'
    ) as OpenRouterTool[];

    expect(result.map((entry) => entry.function.name)).toEqual([
      'static_tool',
      'dynamic_tool',
    ]);
    expect(result[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(result[1]).not.toHaveProperty('cache_control');
  });
});
