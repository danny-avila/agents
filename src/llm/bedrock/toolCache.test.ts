import { tool } from '@langchain/core/tools';
import type { Tool } from '@aws-sdk/client-bedrock-runtime';
import type { GraphTools } from '@/types';
import {
  insertBedrockToolCachePoint,
  partitionAndMarkBedrockToolCache,
} from './toolCache';

type OpenAITool = {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: object;
  };
};

function createOpenAITool(name: string): OpenAITool {
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

function toolName(entry: Tool): string {
  if ('cachePoint' in entry) {
    return 'cachePoint';
  }
  return entry.toolSpec?.name ?? 'missing';
}

function toolNames(tools: Tool[] | undefined): string[] {
  return (tools ?? []).map(toolName);
}

describe('partitionAndMarkBedrockToolCache', () => {
  it('inserts the Bedrock cache point after the last static tool', () => {
    const tools = [
      createOpenAITool('static_one'),
      createOpenAITool('static_two'),
      createOpenAITool('dynamic_one'),
    ] as GraphTools;

    const marked = partitionAndMarkBedrockToolCache(
      tools,
      (name) => name === 'dynamic_one'
    ) as Tool[];
    const result = insertBedrockToolCachePoint({ tools: marked }, false);

    expect(toolNames(result?.tools)).toEqual([
      'static_one',
      'static_two',
      'cachePoint',
      'dynamic_one',
    ]);
    expect(JSON.stringify(result?.tools)).not.toContain(
      '__lc_bedrock_cache_point_after'
    );
  });

  it('converts LangChain tools to Bedrock tool specs before marking', () => {
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

    const marked = partitionAndMarkBedrockToolCache(
      [dynamicTool, staticTool] as GraphTools,
      (name) => name === 'dynamic_tool'
    ) as Tool[];
    const result = insertBedrockToolCachePoint({ tools: marked }, false);

    expect(toolNames(result?.tools)).toEqual([
      'static_tool',
      'cachePoint',
      'dynamic_tool',
    ]);
  });

  it('does not add a cache point when every tool is deferred', () => {
    const tools = [createOpenAITool('dynamic_one')] as GraphTools;
    const marked = partitionAndMarkBedrockToolCache(
      tools,
      () => true
    ) as Tool[];
    const result = insertBedrockToolCachePoint({ tools: marked }, false);

    expect(toolNames(result?.tools)).toEqual(['dynamic_one']);
    expect(JSON.stringify(result?.tools)).not.toContain(
      '__lc_bedrock_skip_tool_cache'
    );
  });

  it('can fall back to caching all directly-bound tools', () => {
    const result = insertBedrockToolCachePoint(
      {
        tools: [
          {
            toolSpec: {
              name: 'direct_tool',
              description: 'Direct tool',
              inputSchema: { json: { type: 'object', properties: {} } },
            },
          },
        ],
      },
      true
    );

    expect(toolNames(result?.tools)).toEqual(['direct_tool', 'cachePoint']);
  });
});
