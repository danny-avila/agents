import type { GraphTools } from '@/types';
import { prepareToolsForPromptCache } from '@/llm/promptCacheTools';
import { Providers } from '@/common';

function createTool(name: string): {
  name: string;
  description: string;
  schema: { type: 'object'; properties: Record<string, never> };
} {
  return {
    name,
    description: `${name} description`,
    schema: { type: 'object', properties: {} },
  };
}

describe('prepareToolsForPromptCache', () => {
  const tools = [createTool('stable'), createTool('deferred')] as GraphTools;
  const isDeferred = (name: string): boolean => name === 'deferred';

  it('marks the same Anthropic static prefix for every caller', () => {
    const prepared = prepareToolsForPromptCache({
      provider: Providers.ANTHROPIC,
      clientOptions: { promptCache: true },
      tools,
      isDeferred,
    }) as Array<{ name: string; extras?: Record<string, unknown> }>;

    expect(prepared.map((tool) => tool.name)).toEqual(['stable', 'deferred']);
    expect(prepared[0].extras?.cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(prepared[1].extras?.cache_control).toBeUndefined();
  });

  it('marks the same OpenRouter static prefix for every caller', () => {
    const prepared = prepareToolsForPromptCache({
      provider: Providers.OPENROUTER,
      clientOptions: { promptCache: true },
      tools,
      isDeferred,
    }) as Array<{
      function: { name: string };
      cache_control?: { type: string; ttl?: string };
    }>;

    expect(prepared.map((tool) => tool.function.name)).toEqual([
      'stable',
      'deferred',
    ]);
    expect(prepared[0].cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(prepared[1].cache_control).toBeUndefined();
  });

  it('marks Claude Bedrock tools and leaves Nova tools unchanged', () => {
    const claude = prepareToolsForPromptCache({
      provider: Providers.BEDROCK,
      clientOptions: {
        promptCache: true,
        model: 'anthropic.claude-sonnet',
      },
      tools,
      isDeferred,
    });
    const nova = prepareToolsForPromptCache({
      provider: Providers.BEDROCK,
      clientOptions: { promptCache: true, model: 'amazon.nova-pro-v1:0' },
      tools,
      isDeferred,
    });

    expect(claude).not.toBe(tools);
    expect(nova).toBe(tools);
  });

  it('returns the original tools when prompt caching is disabled', () => {
    expect(
      prepareToolsForPromptCache({
        provider: Providers.ANTHROPIC,
        clientOptions: { promptCache: false },
        tools,
        isDeferred,
      })
    ).toBe(tools);
  });
});
