// src/agents/__tests__/AgentContext.test.ts
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { addBedrockCacheControl } from '@/messages/cache';
import { Constants, Providers } from '@/common';
import { AgentContext } from '../AgentContext';

describe('AgentContext', () => {
  type TestSystemContentBlock =
    | {
        type: 'text';
        text: string;
        cache_control?: { type: 'ephemeral'; ttl?: '1h' };
      }
    | { cachePoint: { type: 'default'; ttl?: '1h' } };

  type ContextOptions = {
    agentConfig?: Partial<t.AgentInputs>;
    tokenCounter?: t.TokenCounter;
    indexTokenCountMap?: Record<string, number>;
  };

  const createBasicContext = (options: ContextOptions = {}): AgentContext => {
    const { agentConfig = {}, tokenCounter, indexTokenCountMap } = options;
    return AgentContext.fromConfig(
      {
        agentId: 'test-agent',
        provider: Providers.OPENAI,
        instructions: 'Test instructions',
        ...agentConfig,
      },
      tokenCounter,
      indexTokenCountMap
    );
  };

  const createMockTool = (name: string): t.GenericTool =>
    ({
      name,
      description: `Mock ${name} tool`,
      invoke: jest.fn(),
      schema: { type: 'object', properties: {} },
    }) as unknown as t.GenericTool;

  describe('fromConfig — host-supplied graphTools', () => {
    it('copies graphTools onto the context without aliasing the host array', () => {
      const askTool = createMockTool('ask_user_question');
      const hostGraphTools = [askTool];
      const ctx = createBasicContext({
        agentConfig: { graphTools: hostGraphTools },
      });

      expect(ctx.graphTools).toEqual([askTool]);
      expect(ctx.graphTools).not.toBe(hostGraphTools);

      /** The SDK pushes graph-managed tools (handoff/subagent) into this
       *  array later — that must never mutate the host's input. */
      (ctx.graphTools as t.GenericTool[]).push(createMockTool('subagent'));
      expect(hostGraphTools).toHaveLength(1);
    });

    it('leaves graphTools undefined when the host supplies none (or an empty list)', () => {
      expect(createBasicContext().graphTools).toBeUndefined();
      expect(
        createBasicContext({ agentConfig: { graphTools: [] } }).graphTools
      ).toBeUndefined();
    });

    it('binds host graphTools to the model in event-driven mode', () => {
      const askTool = createMockTool('ask_user_question');
      const ctx = createBasicContext({
        agentConfig: {
          graphTools: [askTool],
          toolDefinitions: [{ name: 'echo', description: 'host event tool' }],
        },
      });

      const bound = ctx.getToolsForBinding() ?? [];
      const names = (bound as Array<{ name?: string }>).map((t) => t.name);
      expect(names).toContain('ask_user_question');
      expect(names).toContain('echo');
    });
  });

  describe('System Runnable - Lazy Creation', () => {
    it('creates system runnable on first access', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Hello world' },
      });
      expect(ctx.systemRunnable).toBeDefined();
    });

    it('returns cached system runnable on subsequent access', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Hello world' },
      });
      const first = ctx.systemRunnable;
      const second = ctx.systemRunnable;
      expect(first).toBe(second);
    });

    it('returns undefined when no instructions provided', () => {
      const ctx = createBasicContext({
        agentConfig: {
          instructions: undefined,
          additional_instructions: undefined,
        },
      });
      expect(ctx.systemRunnable).toBeUndefined();
    });

    it('keeps additional_instructions after stable instructions', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Base instructions',
          additional_instructions: 'Additional instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      expect(result[0].content).toBe(
        'Base instructions\n\nAdditional instructions'
      );
    });

    it('moves Anthropic dynamic instructions behind stable history', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: { model: 'claude-3-5-sonnet', promptCache: true },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Hello'),
        new HumanMessage('Second'),
      ]);
      const content = result[0].content as TestSystemContentBlock[];
      expect(content).toEqual([
        {
          type: 'text',
          text: 'Stable instructions',
          cache_control: { type: 'ephemeral', ttl: '1h' },
        },
      ]);
      expect(result[1].content).toBe('Hello');
      expect(result[2].content).toBe('Dynamic instructions');
      expect(result[3].content).toBe('Second');
    });

    it('places Anthropic dynamic instructions before a single latest user prompt', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: { model: 'claude-3-5-sonnet', promptCache: true },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Latest'),
      ]);

      expect(result[1].content).toBe('Dynamic instructions');
      expect(result[2].content).toBe('Latest');
    });

    it('omits Anthropic cache control when only dynamic system text exists', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: { model: 'claude-3-5-sonnet', promptCache: true },
          instructions: undefined,
          additional_instructions: 'Dynamic only',
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      const content = result[0].content as TestSystemContentBlock[];
      expect(content).toEqual([{ type: 'text', text: 'Dynamic only' }]);
      expect(content[0]).not.toHaveProperty('cache_control');
    });

    it('keeps cross-run summaries in the dynamic Anthropic tail', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: { model: 'claude-3-5-sonnet', promptCache: true },
          instructions: 'Stable instructions',
        },
      });
      ctx.setInitialSummary('Prior summary', 13);

      const result = await ctx.systemRunnable!.invoke([]);
      const content = result[0].content as TestSystemContentBlock[];
      expect(content).toHaveLength(1);
      expect(content[0]).toHaveProperty('cache_control');
      expect(result[1].content).toBe(
        '## Conversation Summary\n\nPrior summary'
      );
    });

    it('places the Bedrock cache point before dynamic system text', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.BEDROCK,
          clientOptions: {
            model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      const content = result[0].content as TestSystemContentBlock[];
      expect(content).toEqual([
        { type: 'text', text: 'Stable instructions' },
        { cachePoint: { type: 'default', ttl: '1h' } },
        { type: 'text', text: 'Dynamic instructions' },
      ]);
    });

    it('uses plain Bedrock system text when only dynamic system text exists', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.BEDROCK,
          clientOptions: {
            model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
            promptCache: true,
          },
          instructions: undefined,
          additional_instructions: 'Dynamic only',
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      expect(result[0].content).toBe('Dynamic only');
    });

    it('keeps non-cache providers as plain system text with promptCache-like options', async () => {
      const clientOptions: t.OpenAIClientOptions & { promptCache: true } = {
        modelName: 'gpt-4o-mini',
        promptCache: true,
      };
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENAI,
          clientOptions,
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      expect(result[0].content).toBe(
        'Stable instructions\n\nDynamic instructions'
      );
    });

    it('moves OpenRouter dynamic instructions behind stable history', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Hello'),
        new HumanMessage('Second'),
      ]);
      const content = result[0].content as TestSystemContentBlock[];
      expect(content).toEqual([
        {
          type: 'text',
          text: 'Stable instructions',
          cache_control: { type: 'ephemeral', ttl: '1h' },
        },
      ]);
      expect(result[1].content).toBe('Hello');
      expect(result[2].content).toBe('Dynamic instructions');
      expect(result[3].content).toBe('Second');
    });

    it('keeps dynamic-only OpenRouter instructions as system text', async () => {
      const tokenCounter = (msg: { content: unknown }): number => {
        const content =
          typeof msg.content === 'string'
            ? msg.content
            : JSON.stringify(msg.content);
        return content.length;
      };
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: undefined,
          additional_instructions: 'Dynamic only',
        },
        tokenCounter,
      });

      ctx.initializeSystemRunnable();
      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new HumanMessage('Second'),
      ]);
      const secondContent = result[2].content as TestSystemContentBlock[];

      expect(result).toHaveLength(3);
      expect(result[0].content).toBe('Dynamic only');
      expect(result[1].content).toBe('First');
      expect(secondContent[0]).toMatchObject({
        type: 'text',
        text: 'Second',
        cache_control: { type: 'ephemeral' },
      });
      expect(ctx.systemMessageTokens).toBeGreaterThan(0);
      expect(ctx.dynamicInstructionTokens).toBe(0);
      expect(ctx.instructionTokens).toBe(ctx.systemMessageTokens);
    });

    it('does not cache OpenRouter body messages after dynamic instructions', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'google/gemini-2.5-flash',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new HumanMessage('Second'),
      ]);

      expect(result[1].content).toBe('First');
      expect(result[2].content).toBe('Dynamic instructions');
      expect(result[3].content).toBe('Second');
    });

    it('keeps the first OpenRouter user message before single-turn dynamic instructions', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Latest'),
      ]);

      expect(result[1].content).toBe('Latest');
      expect(result[2].content).toBe('Dynamic instructions');
    });

    it('caches stable Anthropic history before dynamic instructions', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: {
            model: 'claude-3-5-sonnet',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new AIMessage('Stable assistant history'),
        new HumanMessage('Latest'),
      ]);
      const stableHistory = result[2].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('First');
      expect(stableHistory[0]).toMatchObject({
        type: 'text',
        text: 'Stable assistant history',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Dynamic instructions');
      expect(result[4].content).toBe('Latest');
    });

    it('keeps Anthropic dynamic instructions attached to the latest user turn during tool follow-up', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: {
            model: 'claude-3-5-sonnet',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
      ]);

      expect(result[1].content).toBe('Dynamic instructions');
      expect(result[2].content).toBe('Use the tool');
      expect((result[3] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[4].getType()).toBe('tool');
    });

    it('keeps Anthropic stable history cacheable before dynamic tool-follow-up context', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: {
            model: 'claude-3-5-sonnet',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Earlier'),
        new AIMessage('Earlier assistant response'),
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
      ]);
      const stableAssistant = result[2].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('Earlier');
      expect(stableAssistant[0]).toMatchObject({
        type: 'text',
        text: 'Earlier assistant response',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Dynamic instructions');
      expect(result[4].content).toBe('Use the tool');
      expect((result[5] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[6].getType()).toBe('tool');
    });

    it('keeps Anthropic dynamic context on latest no-tool turn after mixed prior turns', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: {
            model: 'claude-3-5-sonnet',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First turn, no tools'),
        new AIMessage('First assistant response'),
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
        new AIMessage('4'),
        new HumanMessage('Now answer without tools'),
      ]);
      const firstAssistant = result[2].content as TestSystemContentBlock[];
      const toolAnswer = result[6].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('First turn, no tools');
      expect(firstAssistant[0]).toMatchObject({
        type: 'text',
        text: 'First assistant response',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Use the tool');
      expect((result[4] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[5].getType()).toBe('tool');
      expect(toolAnswer[0]).toMatchObject({
        type: 'text',
        text: '4',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[7].content).toBe('Dynamic instructions');
      expect(result[8].content).toBe('Now answer without tools');
    });

    it('caches stable OpenRouter history before dynamic instructions', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new AIMessage('Stable assistant history'),
        new HumanMessage('Latest'),
      ]);
      const stableHistory = result[2].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('First');
      expect(stableHistory[0]).toMatchObject({
        type: 'text',
        text: 'Stable assistant history',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Dynamic instructions');
      expect(result[4].content).toBe('Latest');
    });

    it('keeps OpenRouter opening user message before dynamic tool-follow-up context', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
      ]);

      expect(result[1].content).toBe('Use the tool');
      expect(result[2].content).toBe('Dynamic instructions');
      expect((result[3] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[4].getType()).toBe('tool');
    });

    it('keeps OpenRouter stable history cacheable before dynamic tool-follow-up context', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Earlier'),
        new AIMessage('Earlier assistant response'),
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
      ]);
      const stableAssistant = result[2].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('Earlier');
      expect(stableAssistant[0]).toMatchObject({
        type: 'text',
        text: 'Earlier assistant response',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Dynamic instructions');
      expect(result[4].content).toBe('Use the tool');
      expect((result[5] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[6].getType()).toBe('tool');
    });

    it('keeps OpenRouter dynamic context on latest no-tool turn after mixed prior turns', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'anthropic/claude-haiku-4.5',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First turn, no tools'),
        new AIMessage('First assistant response'),
        new HumanMessage('Use the tool'),
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '2+2' },
              type: 'tool_call',
            },
          ],
        }),
        new ToolMessage({
          content: '4',
          name: 'calculator',
          tool_call_id: 'call_1',
        }),
        new AIMessage('4'),
        new HumanMessage('Now answer without tools'),
      ]);
      const firstAssistant = result[2].content as TestSystemContentBlock[];
      const toolAnswer = result[6].content as TestSystemContentBlock[];

      expect(result[1].content).toBe('First turn, no tools');
      expect(firstAssistant[0]).toMatchObject({
        type: 'text',
        text: 'First assistant response',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[3].content).toBe('Use the tool');
      expect((result[4] as AIMessage).tool_calls?.[0]?.id).toBe('call_1');
      expect(result[5].getType()).toBe('tool');
      expect(toolAnswer[0]).toMatchObject({
        type: 'text',
        text: '4',
        cache_control: { type: 'ephemeral' },
      });
      expect(result[7].content).toBe('Dynamic instructions');
      expect(result[8].content).toBe('Now answer without tools');
    });

    it('adds a single OpenRouter body cache point on the tail when there is no dynamic tail', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'google/gemini-3.1-pro-preview',
            promptCache: true,
          },
          instructions: 'Stable instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new HumanMessage('Second'),
      ]);
      const secondContent = result[2].content as TestSystemContentBlock[];
      expect(result[1].content).toBe('First');
      expect(secondContent[0]).toHaveProperty('cache_control');
    });

    it('places OpenRouter user-message summaries after the first stable message', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'google/gemini-3.1-pro-preview',
            promptCache: true,
          },
          instructions: 'Stable instructions',
        },
      });
      ctx.setSummary('Rotating summary', 7);

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('First'),
        new HumanMessage('Second'),
      ]);

      expect(result[1].content).toBe('First');
      expect(result[2].content).toContain('Rotating summary');
      expect(result[3].content).toBe('Second');
    });

    it('preserves the Bedrock system cache point through message cache-control pass', async () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.BEDROCK,
          clientOptions: {
            model: 'anthropic.claude-sonnet-4-5-20250929-v1:0',
            promptCache: true,
          },
          instructions: 'Stable instructions',
          additional_instructions: 'Dynamic instructions',
        },
      });

      const result = await ctx.systemRunnable!.invoke([
        new HumanMessage('Hello'),
      ]);
      // The graph applies the same resolved TTL it stamped on the system
      // checkpoint, so the 1h system cachePoint is preserved (normalized to 1h).
      const finalMessages = addBedrockCacheControl(result, '1h');
      expect(finalMessages[0].content).toEqual([
        { type: 'text', text: 'Stable instructions' },
        { cachePoint: { type: 'default', ttl: '1h' } },
        { type: 'text', text: 'Dynamic instructions' },
      ]);
    });
  });

  describe('System Runnable - Stale Flag', () => {
    it('rebuilds when marked stale via markToolsAsDiscovered', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'deferred_tool',
          {
            name: 'deferred_tool',
            description: 'A deferred code-only tool',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Test', toolRegistry },
      });

      const firstRunnable = ctx.systemRunnable;
      const hasNew = ctx.markToolsAsDiscovered(['deferred_tool']);
      expect(hasNew).toBe(true);

      const secondRunnable = ctx.systemRunnable;
      expect(secondRunnable).not.toBe(firstRunnable);
    });

    it('does not rebuild when discovering already-known tools', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'tool1',
          {
            name: 'tool1',
            description: 'Tool 1',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Test', toolRegistry },
      });

      ctx.markToolsAsDiscovered(['tool1']);
      const firstRunnable = ctx.systemRunnable;

      const hasNew = ctx.markToolsAsDiscovered(['tool1']);
      expect(hasNew).toBe(false);

      const secondRunnable = ctx.systemRunnable;
      expect(secondRunnable).toBe(firstRunnable);
    });
  });

  describe('markToolsAsDiscovered', () => {
    it('returns true when new tools are discovered', () => {
      const ctx = createBasicContext();
      const result = ctx.markToolsAsDiscovered(['tool1', 'tool2']);
      expect(result).toBe(true);
      expect(ctx.discoveredToolNames.has('tool1')).toBe(true);
      expect(ctx.discoveredToolNames.has('tool2')).toBe(true);
    });

    it('returns false when all tools already discovered', () => {
      const ctx = createBasicContext();
      ctx.markToolsAsDiscovered(['tool1']);
      const result = ctx.markToolsAsDiscovered(['tool1']);
      expect(result).toBe(false);
    });

    it('returns true if at least one tool is new', () => {
      const ctx = createBasicContext();
      ctx.markToolsAsDiscovered(['tool1']);
      const result = ctx.markToolsAsDiscovered(['tool1', 'tool2']);
      expect(result).toBe(true);
      expect(ctx.discoveredToolNames.size).toBe(2);
    });

    it('handles empty array gracefully', () => {
      const ctx = createBasicContext();
      const result = ctx.markToolsAsDiscovered([]);
      expect(result).toBe(false);
    });
  });

  describe('buildProgrammaticOnlyToolsInstructions', () => {
    it('includes code_execution-only tools in system message', async () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'programmatic_tool',
          {
            name: 'programmatic_tool',
            description: 'Only callable via code execution',
            allowed_callers: ['code_execution'],
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Base',
          toolDefinitions: [{ name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING }],
          toolRegistry,
        },
      });

      const runnable = ctx.systemRunnable;
      expect(runnable).toBeDefined();
      const result = await runnable!.invoke([]);
      expect(result[0].content).toContain('run_tools_with_bash');
      expect(result[0].content).not.toContain('run_tools_with_code');
    });

    it('uses Python PTC guidance when only run_tools_with_code is available', async () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'programmatic_tool',
          {
            name: 'programmatic_tool',
            description: 'Only callable via code execution',
            allowed_callers: ['code_execution'],
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Base',
          toolDefinitions: [{ name: Constants.PROGRAMMATIC_TOOL_CALLING }],
          toolRegistry,
        },
      });

      const result = await ctx.systemRunnable!.invoke([]);
      expect(result[0].content).toContain('run_tools_with_code');
      expect(result[0].content).toContain('Python code');
      expect(result[0].content).not.toContain('run_tools_with_bash');
    });

    it('excludes direct-callable tools from programmatic section', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'direct_tool',
          {
            name: 'direct_tool',
            description: 'Direct callable',
            allowed_callers: ['direct'],
          },
        ],
        [
          'both_tool',
          {
            name: 'both_tool',
            description: 'Both direct and code',
            allowed_callers: ['direct', 'code_execution'],
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Base', toolRegistry },
      });

      expect(ctx.systemRunnable).toBeDefined();
    });

    it('excludes deferred code_execution-only tools until discovered', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'deferred_code_tool',
          {
            name: 'deferred_code_tool',
            description: 'Deferred and code-only',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
        [
          'immediate_code_tool',
          {
            name: 'immediate_code_tool',
            description: 'Immediate and code-only',
            allowed_callers: ['code_execution'],
            defer_loading: false,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Base', toolRegistry },
      });

      const firstRunnable = ctx.systemRunnable;
      expect(firstRunnable).toBeDefined();

      ctx.markToolsAsDiscovered(['deferred_code_tool']);

      const secondRunnable = ctx.systemRunnable;
      expect(secondRunnable).not.toBe(firstRunnable);
    });
  });

  describe('getToolsForBinding', () => {
    it('returns all tools when no toolRegistry', () => {
      const tools = [createMockTool('tool1'), createMockTool('tool2')];
      const ctx = createBasicContext({ agentConfig: { tools } });
      const result = ctx.getToolsForBinding();
      expect(result).toEqual(tools);
    });

    it('dedupes duplicate named tools before binding and logs a warning', () => {
      const warnSpy = jest
        .spyOn(console, 'warn')
        .mockImplementation(() => undefined);
      const nativeWebSearch = {
        name: 'web_search',
        type: 'web_search_20250305',
      } as unknown as t.GenericTool;

      try {
        const ctx = createBasicContext({
          agentConfig: {
            provider: Providers.ANTHROPIC,
            toolDefinitions: [
              {
                name: 'web_search',
                description: 'LibreChat search',
                parameters: { type: 'object', properties: {} },
              },
            ],
            tools: [nativeWebSearch],
          },
        });

        const result = ctx.getToolsForBinding() as t.GenericTool[];

        expect(result).toHaveLength(1);
        expect(result.map((tool) => tool.name)).toEqual(['web_search']);
        expect(result[0]).not.toBe(nativeWebSearch);
        expect(warnSpy).toHaveBeenCalledTimes(1);
        expect(warnSpy).toHaveBeenCalledWith(
          expect.stringContaining('Removed duplicate tool binding(s)')
        );

        ctx.getToolsForBinding();
        expect(warnSpy).toHaveBeenCalledTimes(1);
      } finally {
        warnSpy.mockRestore();
      }
    });

    it('excludes code_execution-only tools', () => {
      const tools = [
        createMockTool('direct_tool'),
        createMockTool('code_only_tool'),
      ];
      const toolRegistry: t.LCToolRegistry = new Map([
        ['direct_tool', { name: 'direct_tool', allowed_callers: ['direct'] }],
        [
          'code_only_tool',
          { name: 'code_only_tool', allowed_callers: ['code_execution'] },
        ],
      ]);

      const ctx = createBasicContext({ agentConfig: { tools, toolRegistry } });
      const result = ctx.getToolsForBinding();
      expect(result?.length).toBe(1);
      expect((result?.[0] as t.GenericTool).name).toBe('direct_tool');
    });

    it('excludes deferred tools until discovered', () => {
      const tools = [
        createMockTool('immediate_tool'),
        createMockTool('deferred_tool'),
      ];
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'immediate_tool',
          {
            name: 'immediate_tool',
            allowed_callers: ['direct'],
            defer_loading: false,
          },
        ],
        [
          'deferred_tool',
          {
            name: 'deferred_tool',
            allowed_callers: ['direct'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({ agentConfig: { tools, toolRegistry } });

      let result = ctx.getToolsForBinding();
      expect(result?.length).toBe(1);
      expect((result?.[0] as t.GenericTool).name).toBe('immediate_tool');

      ctx.markToolsAsDiscovered(['deferred_tool']);
      result = ctx.getToolsForBinding();
      expect(result?.length).toBe(2);
    });

    it('includes tools with both direct and code_execution callers', () => {
      const tools = [createMockTool('hybrid_tool')];
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'hybrid_tool',
          {
            name: 'hybrid_tool',
            allowed_callers: ['direct', 'code_execution'],
          },
        ],
      ]);

      const ctx = createBasicContext({ agentConfig: { tools, toolRegistry } });
      const result = ctx.getToolsForBinding();
      expect(result?.length).toBe(1);
    });

    it('defaults to direct when allowed_callers not specified', () => {
      const tools = [createMockTool('default_tool')];
      const toolRegistry: t.LCToolRegistry = new Map([
        ['default_tool', { name: 'default_tool' }],
      ]);

      const ctx = createBasicContext({ agentConfig: { tools, toolRegistry } });
      const result = ctx.getToolsForBinding();
      expect(result?.length).toBe(1);
    });
  });

  describe('Token Accounting', () => {
    const mockTokenCounter = (msg: { content: unknown }): number => {
      const content =
        typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content);
      return content.length;
    };

    it('counts system message tokens on first access', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Hello' },
        tokenCounter: mockTokenCounter,
      });

      ctx.initializeSystemRunnable();
      expect(ctx.instructionTokens).toBeGreaterThan(0);
    });

    it('updates token count when system message changes', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'code_tool',
          {
            name: 'code_tool',
            description: 'A tool with a long description that adds tokens',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Short', toolRegistry },
        tokenCounter: mockTokenCounter,
      });

      ctx.initializeSystemRunnable();
      const initialTokens = ctx.instructionTokens;

      ctx.markToolsAsDiscovered(['code_tool']);
      void ctx.systemRunnable;

      expect(ctx.instructionTokens).toBeGreaterThan(initialTokens);
    });

    it('excludes deferred-undiscovered toolDefinitions from toolSchemaTokens', async () => {
      const activeDef: t.LCTool = {
        name: 'active_tool',
        description: 'Always loaded',
        parameters: { type: 'object', properties: {} },
      };
      const deferredDef: t.LCTool = {
        name: 'deferred_tool',
        description: 'Loaded via tool search',
        parameters: { type: 'object', properties: {} },
        defer_loading: true,
      };

      const ctxBase = createBasicContext({
        agentConfig: { toolDefinitions: [activeDef] },
        tokenCounter: mockTokenCounter,
      });
      const ctxWithDeferred = createBasicContext({
        agentConfig: { toolDefinitions: [activeDef, deferredDef] },
        tokenCounter: mockTokenCounter,
      });

      await ctxBase.tokenCalculationPromise;
      await ctxWithDeferred.tokenCalculationPromise;

      expect(ctxWithDeferred.toolSchemaTokens).toBe(ctxBase.toolSchemaTokens);
    });

    it('counts OpenRouter dynamic instructions outside the system message', () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'google/gemini-3.1-pro-preview',
            promptCache: true,
          },
          instructions: 'Stable',
          additional_instructions: 'Dynamic tail',
        },
        tokenCounter: mockTokenCounter,
      });

      ctx.initializeSystemRunnable();

      expect(ctx.systemMessageTokens).toBeGreaterThan(0);
      expect(ctx.dynamicInstructionTokens).toBeGreaterThan(0);
      expect(ctx.instructionTokens).toBe(
        ctx.systemMessageTokens + ctx.dynamicInstructionTokens
      );
      expect(ctx.getTokenBudgetBreakdown().dynamicInstructionTokens).toBe(
        ctx.dynamicInstructionTokens
      );
    });

    it('clears OpenRouter dynamic instruction tokens when no prompt remains', () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.OPENROUTER,
          clientOptions: {
            model: 'google/gemini-3.1-pro-preview',
            promptCache: true,
          },
          instructions: 'Stable instructions',
        },
        tokenCounter: mockTokenCounter,
      });

      ctx.setInitialSummary('Volatile summary', 8);
      ctx.initializeSystemRunnable();
      expect(ctx.dynamicInstructionTokens).toBeGreaterThan(0);

      ctx.instructions = undefined;
      ctx.clearSummary();
      ctx.initializeSystemRunnable();

      expect(ctx.systemRunnable).toBeUndefined();
      expect(ctx.systemMessageTokens).toBe(0);
      expect(ctx.dynamicInstructionTokens).toBe(0);
      expect(ctx.instructionTokens).toBe(0);
    });

    it('excludes programmatic-only toolDefinitions from toolSchemaTokens', async () => {
      // getEventDrivenToolsForBinding excludes definitions whose
      // allowed_callers omit 'direct'. Accounting must mirror that — a
      // programmatic-only definition is never bound to the model and
      // shouldn't inflate toolSchemaTokens.
      const activeDef: t.LCTool = {
        name: 'active_tool',
        description: 'Always loaded',
        parameters: { type: 'object', properties: {} },
      };
      const programmaticDef: t.LCTool = {
        name: 'programmatic_tool',
        description: 'Only callable via code execution',
        parameters: { type: 'object', properties: {} },
        allowed_callers: ['code_execution'],
      };

      const ctxBase = createBasicContext({
        agentConfig: { toolDefinitions: [activeDef] },
        tokenCounter: mockTokenCounter,
      });
      const ctxWithProgrammatic = createBasicContext({
        agentConfig: { toolDefinitions: [activeDef, programmaticDef] },
        tokenCounter: mockTokenCounter,
      });

      await ctxBase.tokenCalculationPromise;
      await ctxWithProgrammatic.tokenCalculationPromise;

      expect(ctxWithProgrammatic.toolSchemaTokens).toBe(
        ctxBase.toolSchemaTokens
      );
    });

    it('excludes deferred-undiscovered instance tools from toolSchemaTokens', async () => {
      const activeTool = createMockTool('active_tool');
      const deferredTool = createMockTool('deferred_tool');
      const programmaticTool = createMockTool('programmatic_tool');
      const toolRegistry: t.LCToolRegistry = new Map([
        ['active_tool', { name: 'active_tool' }],
        ['deferred_tool', { name: 'deferred_tool', defer_loading: true }],
        [
          'programmatic_tool',
          {
            name: 'programmatic_tool',
            allowed_callers: ['code_execution'],
          },
        ],
      ]);

      const ctxBase = createBasicContext({
        agentConfig: { tools: [activeTool], toolRegistry },
        tokenCounter: mockTokenCounter,
      });
      const ctxWithExcluded = createBasicContext({
        agentConfig: {
          tools: [activeTool, deferredTool, programmaticTool],
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });

      await ctxBase.tokenCalculationPromise;
      await ctxWithExcluded.tokenCalculationPromise;

      expect(ctxWithExcluded.toolSchemaTokens).toBe(ctxBase.toolSchemaTokens);
    });

    it('includes deferred instance tools once discovered via discoveredTools input', async () => {
      const tools = [createMockTool('deferred_tool')];
      const toolRegistry: t.LCToolRegistry = new Map([
        ['deferred_tool', { name: 'deferred_tool', defer_loading: true }],
      ]);

      const ctxUndiscovered = createBasicContext({
        agentConfig: { tools, toolRegistry },
        tokenCounter: mockTokenCounter,
      });
      const ctxDiscovered = createBasicContext({
        agentConfig: {
          tools,
          toolRegistry,
          discoveredTools: ['deferred_tool'],
        },
        tokenCounter: mockTokenCounter,
      });

      await ctxUndiscovered.tokenCalculationPromise;
      await ctxDiscovered.tokenCalculationPromise;

      expect(ctxUndiscovered.toolSchemaTokens).toBe(0);
      expect(ctxDiscovered.toolSchemaTokens).toBeGreaterThan(0);
    });

    it('does not filter instance tools in event-driven mode (matches getEventDrivenToolsForBinding)', async () => {
      // In event-driven mode, getEventDrivenToolsForBinding appends
      // `this.tools` UNFILTERED. Accounting must do the same — otherwise we
      // under-count and risk exceeding the model's context budget.
      const activeDef: t.LCTool = {
        name: 'active_def',
        description: 'Always loaded',
        parameters: { type: 'object', properties: {} },
      };
      const nativeTool = createMockTool('native_tool');
      // Registry marks the native tool as deferred-undiscovered. In the
      // non-event-driven path this would exclude it; in event-driven mode
      // it is still bound and must still be counted.
      const toolRegistry: t.LCToolRegistry = new Map([
        ['native_tool', { name: 'native_tool', defer_loading: true }],
      ]);

      const ctxWithoutNative = createBasicContext({
        agentConfig: {
          toolDefinitions: [activeDef],
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });
      const ctxWithNative = createBasicContext({
        agentConfig: {
          toolDefinitions: [activeDef],
          tools: [nativeTool],
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });

      await ctxWithoutNative.tokenCalculationPromise;
      await ctxWithNative.tokenCalculationPromise;

      expect(ctxWithNative.toolSchemaTokens).toBeGreaterThan(
        ctxWithoutNative.toolSchemaTokens
      );
    });

    it('includes deferred toolDefinitions once discovered via discoveredTools input', async () => {
      const toolDefinitions: t.LCTool[] = [
        {
          name: 'deferred_tool',
          description: 'Loaded via tool search',
          parameters: { type: 'object', properties: {} },
          defer_loading: true,
        },
      ];

      const ctxUndiscovered = createBasicContext({
        agentConfig: { toolDefinitions },
        tokenCounter: mockTokenCounter,
      });
      const ctxDiscovered = createBasicContext({
        agentConfig: { toolDefinitions, discoveredTools: ['deferred_tool'] },
        tokenCounter: mockTokenCounter,
      });

      await ctxUndiscovered.tokenCalculationPromise;
      await ctxDiscovered.tokenCalculationPromise;

      expect(ctxUndiscovered.toolSchemaTokens).toBe(0);
      expect(ctxDiscovered.toolSchemaTokens).toBeGreaterThan(0);
    });

    it('getTokenBudgetBreakdown toolCount excludes deferred-undiscovered toolDefinitions', () => {
      const toolDefinitions: t.LCTool[] = [
        {
          name: 'active',
          parameters: { type: 'object', properties: {} },
        },
        {
          name: 'deferred',
          defer_loading: true,
          parameters: { type: 'object', properties: {} },
        },
      ];

      const ctx = createBasicContext({ agentConfig: { toolDefinitions } });

      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(1);
    });

    it('getTokenBudgetBreakdown toolCount excludes deferred-undiscovered instance tools', () => {
      // Mirrors the toolDefinitions test for the instance-tools path so
      // toolCount stays aligned with toolSchemaTokens (and with what
      // getToolsForBinding actually emits) for non-event-driven runs.
      const tools = [
        createMockTool('active_tool'),
        createMockTool('deferred_tool'),
        createMockTool('programmatic_tool'),
      ];
      const toolRegistry: t.LCToolRegistry = new Map([
        ['active_tool', { name: 'active_tool' }],
        ['deferred_tool', { name: 'deferred_tool', defer_loading: true }],
        [
          'programmatic_tool',
          {
            name: 'programmatic_tool',
            allowed_callers: ['code_execution'],
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { tools, toolRegistry },
      });

      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(1);
      ctx.markToolsAsDiscovered(['deferred_tool']);
      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(2);
    });

    it('getTokenBudgetBreakdown toolCount reflects newly discovered deferred tools', () => {
      const toolDefinitions: t.LCTool[] = [
        {
          name: 'deferred',
          defer_loading: true,
          parameters: { type: 'object', properties: {} },
        },
      ];

      const ctx = createBasicContext({ agentConfig: { toolDefinitions } });

      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(0);
      ctx.markToolsAsDiscovered(['deferred']);
      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(1);
    });

    it('getTokenBudgetBreakdown toolCount includes graphTools', () => {
      // graphTools (handoff/subagent) are bound to the model alongside
      // instance tools. Now that toolCount derives from getToolsForBinding(),
      // graphTools are reflected in the diagnostic just like they're
      // counted in toolSchemaTokens. Locks in that alignment.
      const ctx = createBasicContext({
        agentConfig: { tools: [createMockTool('direct_tool')] },
      });
      ctx.graphTools = [createMockTool('handoff_tool')];

      expect(ctx.getTokenBudgetBreakdown().toolCount).toBe(2);
    });

    it('refreshes toolSchemaTokens and per-tool counts after markToolsAsDiscovered', async () => {
      const toolDefinitions: t.LCTool[] = [
        {
          name: 'deferred',
          description: 'Loaded via tool search',
          parameters: { type: 'object', properties: {} },
          defer_loading: true,
        },
      ];

      const ctx = createBasicContext({
        agentConfig: { toolDefinitions },
        tokenCounter: mockTokenCounter,
      });

      await ctx.tokenCalculationPromise;
      expect(ctx.toolSchemaTokens).toBe(0);
      expect(ctx.toolTokenCounts).toEqual({});

      ctx.markToolsAsDiscovered(['deferred']);
      await ctx.tokenCalculationPromise;
      expect(ctx.toolSchemaTokens).toBeGreaterThan(0);
      expect(ctx.toolTokenCounts?.deferred).toBeGreaterThan(0);
      expect(ctx.deferredToolNames).toContain('deferred');
    });
  });

  describe('reset()', () => {
    it('clears all cached state', () => {
      const ctx = createBasicContext({ agentConfig: { instructions: 'Test' } });

      ctx.markToolsAsDiscovered(['tool1']);
      void ctx.systemRunnable;
      ctx.systemMessageTokens = 100;
      ctx.indexTokenCountMap = { '0': 50 };
      ctx.currentUsage = { input_tokens: 100 };

      ctx.reset();

      expect(ctx.discoveredToolNames.size).toBe(0);
      expect(ctx.instructionTokens).toBe(0);
      expect(ctx.indexTokenCountMap).toEqual({});
      expect(ctx.currentUsage).toBeUndefined();
    });

    it('preserves summarization settings across resets', () => {
      const ctx = createBasicContext({
        agentConfig: {
          summarizationEnabled: true,
          summarizationConfig: {
            provider: Providers.ANTHROPIC,
            model: 'claude-sonnet-4-5',
            prompt: 'Keep decisions and next steps concise.',
            trigger: {
              type: 'token_ratio',
              value: 0.8,
            },
          },
        },
      });

      ctx.reset();

      expect(ctx.summarizationEnabled).toBe(true);
      expect(ctx.summarizationConfig).toEqual({
        provider: 'anthropic',
        model: 'claude-sonnet-4-5',
        prompt: 'Keep decisions and next steps concise.',
        trigger: {
          type: 'token_ratio',
          value: 0.8,
        },
      });
    });

    it('shouldSkipSummarization returns true when message count unchanged', () => {
      const ctx = createBasicContext({
        agentConfig: { summarizationEnabled: true },
      });
      ctx.markSummarizationTriggered(10);
      expect(ctx.shouldSkipSummarization(10)).toBe(true);
    });

    it('shouldSkipSummarization returns false with any new messages', () => {
      const ctx = createBasicContext({
        agentConfig: { summarizationEnabled: true },
      });
      ctx.markSummarizationTriggered(10);
      expect(ctx.shouldSkipSummarization(11)).toBe(false);
    });

    it('shouldSkipSummarization returns false when no prior summarization', () => {
      const ctx = createBasicContext({
        agentConfig: { summarizationEnabled: true },
      });
      expect(ctx.shouldSkipSummarization(5)).toBe(false);
    });

    it('shouldSkipSummarization allows unlimited summarizations per run', () => {
      const ctx = createBasicContext({
        agentConfig: { summarizationEnabled: true },
      });
      for (let i = 0; i < 10; i++) {
        ctx.markSummarizationTriggered(i * 5);
      }
      // Even after 10 summarizations, new messages allow another
      expect(ctx.shouldSkipSummarization(50)).toBe(false);
    });

    it('rebuilds indexTokenCountMap from base map after reset', async () => {
      const tokenCounter = jest.fn(() => 5);
      const ctx = createBasicContext({
        tokenCounter,
        indexTokenCountMap: { '0': 10, '1': 20 },
      });

      await ctx.tokenCalculationPromise;
      ctx.indexTokenCountMap = {};

      ctx.reset();
      await ctx.tokenCalculationPromise;

      expect(ctx.indexTokenCountMap['1']).toBe(20);
      expect(ctx.indexTokenCountMap['0'] ?? 0).toBeGreaterThanOrEqual(10);
    });

    it('forces rebuild on next systemRunnable access', () => {
      const ctx = createBasicContext({ agentConfig: { instructions: 'Test' } });

      const firstRunnable = ctx.systemRunnable;
      ctx.reset();

      ctx.instructions = 'Test';
      const secondRunnable = ctx.systemRunnable;

      expect(secondRunnable).not.toBe(firstRunnable);
    });
  });

  describe('initializeSystemRunnable()', () => {
    it('explicitly initializes system runnable', () => {
      const ctx = createBasicContext({ agentConfig: { instructions: 'Test' } });

      expect(ctx['cachedSystemRunnable']).toBeUndefined();
      ctx.initializeSystemRunnable();
      expect(ctx['cachedSystemRunnable']).toBeDefined();
    });

    it('is idempotent when not stale', () => {
      const ctx = createBasicContext({ agentConfig: { instructions: 'Test' } });

      ctx.initializeSystemRunnable();
      const first = ctx['cachedSystemRunnable'];

      ctx.initializeSystemRunnable();
      const second = ctx['cachedSystemRunnable'];

      expect(first).toBe(second);
    });
  });

  describe('Edge Cases', () => {
    it('handles empty toolRegistry gracefully', () => {
      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Test',
          toolRegistry: new Map(),
          tools: [],
        },
      });

      expect(ctx.systemRunnable).toBeDefined();
      expect(ctx.getToolsForBinding()).toEqual([]);
    });

    it('handles undefined tools array', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Test', tools: undefined },
      });

      expect(ctx.getToolsForBinding()).toBeUndefined();
    });

    it('handles tool not in registry', () => {
      const tools = [createMockTool('unknown_tool')];
      const toolRegistry: t.LCToolRegistry = new Map();

      const ctx = createBasicContext({ agentConfig: { tools, toolRegistry } });
      const result = ctx.getToolsForBinding();

      expect(result?.length).toBe(1);
    });

    it('handles tool without name property', () => {
      const toolWithoutName = { invoke: jest.fn() } as unknown as t.GenericTool;
      const toolRegistry: t.LCToolRegistry = new Map();

      const ctx = createBasicContext({
        agentConfig: { tools: [toolWithoutName], toolRegistry },
      });

      const result = ctx.getToolsForBinding();
      expect(result?.length).toBe(1);
    });

    it('handles discovery of non-existent tool', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'real_tool',
          { name: 'real_tool', allowed_callers: ['code_execution'] },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: { instructions: 'Test', toolRegistry },
      });

      const result = ctx.markToolsAsDiscovered(['fake_tool']);
      expect(result).toBe(true);
      expect(ctx.discoveredToolNames.has('fake_tool')).toBe(true);
    });
  });

  describe('Multi-Step Run Flow (emulating createCallModel)', () => {
    /**
     * This test emulates the flow in Graph.createCallModel across multiple turns:
     *
     * Turn 1: User asks a question
     *   - No tool search results yet
     *   - System runnable built with initial instructions
     *   - Token map initialized
     *
     * Turn 2: Tool results come back (including tool search)
     *   - extractToolDiscoveries() finds new tools
     *   - markToolsAsDiscovered() called → sets stale flag
     *   - systemRunnable getter rebuilds with discovered tools
     *   - Token counts updated
     *
     * Turn 3: Another turn with same discovered tools
     *   - No new discoveries
     *   - systemRunnable returns cached (not rebuilt)
     *   - Token counts unchanged
     */

    const mockTokenCounter = (msg: { content: unknown }): number => {
      const content =
        typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content);
      return Math.ceil(content.length / 4); // ~4 chars per token (realistic)
    };

    it('handles complete multi-step run with tool discovery', () => {
      // Setup: Tools with different configurations
      const tools = [
        createMockTool('always_available'),
        createMockTool('deferred_direct_tool'),
        createMockTool('deferred_code_tool'),
      ];

      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'always_available',
          {
            name: 'always_available',
            description: 'Always available tool',
            allowed_callers: ['direct'],
            defer_loading: false,
          },
        ],
        [
          'deferred_direct_tool',
          {
            name: 'deferred_direct_tool',
            description: 'Deferred but direct-callable',
            allowed_callers: ['direct'],
            defer_loading: true,
          },
        ],
        [
          'deferred_code_tool',
          {
            name: 'deferred_code_tool',
            description:
              'Deferred and code-execution only - hidden until discovered',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'You are a helpful assistant.',
          tools,
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });

      // ========== TURN 1: Initial call (like first createCallModel) ==========

      // In createCallModel, we first check for tool discoveries (none yet)
      const turn1Discoveries: string[] = []; // No tool search results
      if (turn1Discoveries.length > 0) {
        ctx.markToolsAsDiscovered(turn1Discoveries);
      }

      // Get tools for binding
      const turn1Tools = ctx.getToolsForBinding();
      expect(turn1Tools?.length).toBe(1); // Only 'always_available'
      expect(turn1Tools?.map((t) => (t as t.GenericTool).name)).toEqual([
        'always_available',
      ]);

      // Access system runnable (triggers lazy build)
      const turn1Runnable = ctx.systemRunnable;
      expect(turn1Runnable).toBeDefined();

      // Store initial token count
      const turn1Tokens = ctx.instructionTokens;
      expect(turn1Tokens).toBeGreaterThan(0);

      // Simulate token map update (as done in fromConfig flow)
      ctx.updateTokenMapWithInstructions({ '0': 10, '1': 20 });
      expect(ctx.indexTokenCountMap['0']).toBe(10);
      expect(ctx.indexTokenCountMap['1']).toBe(20);

      // ========== TURN 2: Tool search results come back ==========

      // Simulate tool search discovering tools
      const turn2Discoveries = ['deferred_direct_tool', 'deferred_code_tool'];
      const hasNewDiscoveries = ctx.markToolsAsDiscovered(turn2Discoveries);
      expect(hasNewDiscoveries).toBe(true);

      // Get tools for binding - now includes discovered direct tool
      const turn2Tools = ctx.getToolsForBinding();
      expect(turn2Tools?.length).toBe(2); // 'always_available' + 'deferred_direct_tool'
      expect(turn2Tools?.map((t) => (t as t.GenericTool).name)).toContain(
        'always_available'
      );
      expect(turn2Tools?.map((t) => (t as t.GenericTool).name)).toContain(
        'deferred_direct_tool'
      );
      // Note: 'deferred_code_tool' is NOT in binding (code_execution only)

      // Access system runnable - should rebuild due to stale flag
      const turn2Runnable = ctx.systemRunnable;
      expect(turn2Runnable).not.toBe(turn1Runnable); // Different instance = rebuilt

      // Token count should increase (now includes deferred_code_tool in system message)
      const turn2Tokens = ctx.instructionTokens;
      expect(turn2Tokens).toBeGreaterThan(turn1Tokens);

      // ========== TURN 3: Another turn, same discoveries ==========

      // Same discoveries (duplicates)
      const turn3Discoveries = ['deferred_direct_tool'];
      const hasNewDiscoveriesTurn3 =
        ctx.markToolsAsDiscovered(turn3Discoveries);
      expect(hasNewDiscoveriesTurn3).toBe(false); // No NEW discoveries

      // Tools should be same as turn 2
      const turn3Tools = ctx.getToolsForBinding();
      expect(turn3Tools?.length).toBe(2);

      // System runnable should be CACHED (same reference)
      const turn3Runnable = ctx.systemRunnable;
      expect(turn3Runnable).toBe(turn2Runnable); // Same instance = cached

      // Token count unchanged
      expect(ctx.instructionTokens).toBe(turn2Tokens);
    });

    it('maintains consistent indexTokenCountMap across turns', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Base instructions' },
        tokenCounter: mockTokenCounter,
      });

      // Initial setup (simulating fromConfig flow)
      ctx.initializeSystemRunnable();
      const initialSystemTokens = ctx.instructionTokens;

      // Simulate message token counts from conversation
      const messageTokenCounts = { '0': 50, '1': 100, '2': 75 };
      ctx.updateTokenMapWithInstructions(messageTokenCounts);

      // Verify token map: first message keeps its real token count (no inflation)
      expect(ctx.indexTokenCountMap['0']).toBe(50);
      expect(ctx.indexTokenCountMap['1']).toBe(100);
      expect(ctx.indexTokenCountMap['2']).toBe(75);

      // Simulate turn where system message changes
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'new_code_tool',
          {
            name: 'new_code_tool',
            description:
              'A newly discovered code-only tool with detailed documentation',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);
      ctx.toolRegistry = toolRegistry;

      // Discover the tool
      ctx.markToolsAsDiscovered(['new_code_tool']);

      // Access system runnable to trigger rebuild
      void ctx.systemRunnable;

      // Token count should have increased
      const newSystemTokens = ctx.instructionTokens;
      expect(newSystemTokens).toBeGreaterThan(initialSystemTokens);

      // If we update token map again, it should use NEW instruction tokens
      const newMessageTokenCounts = { '0': 60, '1': 110 };
      ctx.updateTokenMapWithInstructions(newMessageTokenCounts);

      expect(ctx.indexTokenCountMap['0']).toBe(60);
      expect(ctx.indexTokenCountMap['1']).toBe(110);
    });

    it('correctly tracks token delta when system message content changes', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'tool_a',
          {
            name: 'tool_a',
            description: 'Short description',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
        [
          'tool_b',
          {
            name: 'tool_b',
            description: 'Another tool that adds more content',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'You are helpful.',
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });

      ctx.initializeSystemRunnable();
      const baseTokens = ctx.instructionTokens;

      // Discover tool_a
      ctx.markToolsAsDiscovered(['tool_a']);
      void ctx.systemRunnable;
      const tokensAfterA = ctx.instructionTokens;
      expect(tokensAfterA).toBeGreaterThan(baseTokens);

      // Discover tool_b - adds more content
      ctx.markToolsAsDiscovered(['tool_b']);
      void ctx.systemRunnable;
      const tokensAfterB = ctx.instructionTokens;
      expect(tokensAfterB).toBeGreaterThan(tokensAfterA);

      // Both deltas should be positive (each discovery adds tokens)
      const deltaBaseToA = tokensAfterA - baseTokens;
      const deltaAToB = tokensAfterB - tokensAfterA;
      expect(deltaBaseToA).toBeGreaterThan(0);
      expect(deltaAToB).toBeGreaterThan(0);
    });

    it('handles reset between runs correctly', () => {
      const toolRegistry: t.LCToolRegistry = new Map([
        [
          'discovered_tool',
          {
            name: 'discovered_tool',
            description: 'Will be discovered',
            allowed_callers: ['code_execution'],
            defer_loading: true,
          },
        ],
      ]);

      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Assistant instructions',
          toolRegistry,
        },
        tokenCounter: mockTokenCounter,
      });

      // ========== RUN 1 ==========
      ctx.initializeSystemRunnable();
      ctx.markToolsAsDiscovered(['discovered_tool']);
      void ctx.systemRunnable;

      expect(ctx.discoveredToolNames.has('discovered_tool')).toBe(true);
      const run1Tokens = ctx.instructionTokens;
      expect(run1Tokens).toBeGreaterThan(0);

      // ========== RESET (new run) ==========
      ctx.reset();

      // Verify state is cleared
      expect(ctx.discoveredToolNames.size).toBe(0);
      const resetTokens = ctx.instructionTokens;
      expect(resetTokens).toBeGreaterThan(0);
      expect(resetTokens).toBeLessThan(run1Tokens);

      // ========== RUN 2 ==========
      // Re-initialize (as fromConfig would do)
      ctx.initializeSystemRunnable();

      // System runnable should NOT include the previously discovered tool
      // (because discoveredToolNames was cleared)
      const run2Tokens = ctx.instructionTokens;
      expect(run2Tokens).toBe(resetTokens);

      // Token count should be lower than run 1 (no discovered tool in system message)
      expect(run2Tokens).toBeLessThan(run1Tokens);

      // Discover again
      ctx.markToolsAsDiscovered(['discovered_tool']);
      void ctx.systemRunnable;

      // Now should match run 1
      expect(ctx.instructionTokens).toBe(run1Tokens);
    });
  });

  describe('Summary Token Accounting', () => {
    const charTokenCounter: t.TokenCounter = (msg) => {
      const raw = msg.content;
      if (typeof raw === 'string') return raw.length;
      if (Array.isArray(raw)) {
        let total = 0;
        for (let i = 0; i < raw.length; i++) {
          const item = raw[i] as unknown;
          if (typeof item === 'string') {
            total += item.length;
          } else if (
            typeof item === 'object' &&
            item != null &&
            'text' in item
          ) {
            const text = (item as Record<string, unknown>).text;
            if (typeof text === 'string') total += text.length;
          }
        }
        return total;
      }
      return 0;
    };

    it('mid-run setSummary increases instructionTokens by the summary token count', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      const baseInstructionTokens = ctx.instructionTokens;
      expect(baseInstructionTokens).toBeGreaterThan(0);

      // Mid-run summary is injected as HumanMessage but still counts as
      // instruction overhead so the pruner reserves budget for it.
      ctx.setSummary('User asked about math. Key results: 2+2=4, 3*5=15.', 50);
      expect(ctx.hasSummary()).toBe(true);

      void ctx.systemRunnable;
      expect(ctx.instructionTokens).toBe(baseInstructionTokens + 50);
    });

    it('summary text appears in rebuilt system message', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      ctx.setSummary('Prior context: user computed factorials.', 40);

      const runnable = ctx.systemRunnable;
      expect(runnable).toBeDefined();
    });

    it('clearSummary removes summary overhead from instructionTokens', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      const baseTokens = ctx.instructionTokens;

      ctx.setSummary('Summary of the conversation so far.', 35);
      void ctx.systemRunnable;
      expect(ctx.instructionTokens).toBe(baseTokens + 35);

      ctx.clearSummary();
      void ctx.systemRunnable;
      expect(ctx.instructionTokens).toBe(baseTokens);
    });

    it('reset preserves durable summary and maintains token counts', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
        indexTokenCountMap: { '0': 10, '1': 20 },
      });

      void ctx.systemRunnable;
      ctx.setSummary('Summary text.', 15);
      void ctx.systemRunnable;
      expect(ctx.hasSummary()).toBe(true);
      const tokensWithSummary = ctx.instructionTokens;

      ctx.reset();
      // Summary should survive reset (durable cross-run state)
      expect(ctx.hasSummary()).toBe(true);
      expect(ctx.getSummaryText()).toBe('Summary text.');

      void ctx.systemRunnable;
      const postResetTokens = ctx.instructionTokens;
      expect(postResetTokens).toBeGreaterThan(0);
      // Token count should be the same since summary is preserved
      expect(postResetTokens).toBe(tokensWithSummary);
    });

    it('updateTokenMapWithInstructions copies base map without inflating index 0', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      ctx.setSummary('Summary of prior context with key facts.', 40);
      void ctx.systemRunnable;

      const instructionTokens = ctx.instructionTokens;
      expect(instructionTokens).toBeGreaterThan(0);

      const baseMap: Record<string, number> = { '0': 5, '1': 10 };
      ctx.updateTokenMapWithInstructions(baseMap);

      // Index 0 should contain the real message token count, NOT inflated
      // with instruction tokens.  Instruction overhead is now handled by
      // getInstructionTokens() in the pruning factory.
      expect(ctx.indexTokenCountMap['0']).toBe(5);
      expect(ctx.indexTokenCountMap['1']).toBe(10);
    });

    it('hasSummary returns false before setSummary and true after', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
      });

      expect(ctx.hasSummary()).toBe(false);
      ctx.setSummary('Some summary.', 10);
      expect(ctx.hasSummary()).toBe(true);
      ctx.clearSummary();
      expect(ctx.hasSummary()).toBe(false);
    });
  });

  describe('shouldSkipSummarization — re-trigger after summary', () => {
    it('allows re-summarization after rebuildTokenMapAfterSummarization resets baseline', () => {
      const ctx = createBasicContext();

      expect(ctx.shouldSkipSummarization(25)).toBe(false);
      ctx.markSummarizationTriggered(25);

      // Same count — skip
      expect(ctx.shouldSkipSummarization(25)).toBe(true);

      ctx.setSummary('Summary of conversation', 100);
      // Full compaction: empty state, baseline resets to 0
      ctx.rebuildTokenMapAfterSummarization({});

      // Baseline is 0 after full compaction. Guard `_lastSummarizationMsgCount > 0`
      // is false, so all counts are allowed.
      expect(ctx.shouldSkipSummarization(0)).toBe(false);
      expect(ctx.shouldSkipSummarization(1)).toBe(false);
    });

    it('allows summarization after full compaction resets to empty state', () => {
      const ctx = createBasicContext();

      ctx.markSummarizationTriggered(20);
      ctx.setSummary('Summary', 50);
      ctx.rebuildTokenMapAfterSummarization({});

      // Baseline is 0 after full compaction. The guard `_lastSummarizationMsgCount > 0`
      // is false, so summarization is always allowed — the model starts fresh.
      expect(ctx.shouldSkipSummarization(0)).toBe(false);
      expect(ctx.shouldSkipSummarization(1)).toBe(false);
    });
  });

  describe('updateLastCallUsage', () => {
    it('records usage without modifying toolSchemaTokens', () => {
      const ctx = createBasicContext();
      ctx.toolSchemaTokens = 200;

      ctx.updateLastCallUsage({ input_tokens: 500, output_tokens: 30 });

      expect(ctx.lastCallUsage).toBeDefined();
      expect(ctx.lastCallUsage!.inputTokens).toBe(500);
      expect(ctx.lastCallUsage!.outputTokens).toBe(30);
      expect(ctx.toolSchemaTokens).toBe(200);
    });

    it('handles additive cache tokens', () => {
      const ctx = createBasicContext();

      ctx.updateLastCallUsage({
        input_tokens: 5,
        output_tokens: 100,
        input_token_details: { cache_creation: 8000, cache_read: 0 },
      });

      // cache_creation (8000) > input_tokens (5) → additive
      expect(ctx.lastCallUsage!.inputTokens).toBe(8005);
    });
  });

  describe('projectContextUsage', () => {
    const countByChars = (msg: { content: unknown }): number => {
      const content =
        typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content);
      return content.length;
    };

    const buildBranch = (
      maxContextTokens: number,
      perMessageTokens: number,
      count: number
    ): { ctx: AgentContext; messages: AIMessage[] } => {
      const ctx = createBasicContext({ tokenCounter: countByChars });
      ctx.maxContextTokens = maxContextTokens;
      const messages: AIMessage[] = [];
      for (let i = 0; i < count; i++) {
        // countByChars counts content length, and projectContextUsage recounts
        // the supplied messages — so size content to the intended per-msg tokens.
        const content = 'x'.repeat(perMessageTokens);
        messages.push(
          i % 2 === 0
            ? (new HumanMessage(content) as unknown as AIMessage)
            : new AIMessage(content)
        );
      }
      return { ctx, messages };
    };

    it('returns null without a tokenizer or a window', () => {
      const noCounter = createBasicContext({});
      noCounter.maxContextTokens = 1000;
      expect(
        noCounter.projectContextUsage([new HumanMessage('hi')])
      ).toBeNull();

      const noWindow = createBasicContext({ tokenCounter: countByChars });
      noWindow.maxContextTokens = undefined;
      expect(noWindow.projectContextUsage([new HumanMessage('hi')])).toBeNull();
    });

    it('keeps the whole branch and reports headroom when it fits', () => {
      const { ctx, messages } = buildBranch(100_000, 1_000, 4);
      const usage = ctx.projectContextUsage(messages);

      expect(usage).not.toBeNull();
      expect(usage!.breakdown.messageCount).toBe(4);
      expect(usage!.breakdown.maxContextTokens).toBe(100_000);
      expect(usage!.remainingContextTokens).toBeGreaterThan(0);
      expect(usage!.breakdown.messageTokens).toBeGreaterThan(0);

      const max = usage!.contextBudget ?? usage!.breakdown.maxContextTokens;
      const used = max - (usage!.remainingContextTokens ?? 0);
      expect(used).toBeLessThanOrEqual(max);
    });

    it('prunes older messages when the branch exceeds the window', () => {
      const { ctx, messages } = buildBranch(3_000, 1_000, 6);
      const usage = ctx.projectContextUsage(messages);

      expect(usage).not.toBeNull();
      expect(usage!.breakdown.messageCount).toBeGreaterThan(0);
      expect(usage!.breakdown.messageCount).toBeLessThan(6);
      expect(usage!.remainingContextTokens).toBeGreaterThanOrEqual(0);

      const max = usage!.contextBudget ?? usage!.breakdown.maxContextTokens;
      expect(max - (usage!.remainingContextTokens ?? 0)).toBeLessThanOrEqual(
        max
      );
    });

    it('does not mutate the context (local pruner, no field writes)', () => {
      const { ctx, messages } = buildBranch(3_000, 1_000, 6);
      const mapBefore = { ...ctx.indexTokenCountMap };

      expect(ctx.pruneMessages).toBeUndefined();
      ctx.projectContextUsage(messages);

      expect(ctx.pruneMessages).toBeUndefined();
      expect(ctx.indexTokenCountMap).toEqual(mapBefore);
    });

    it('does not mutate the caller messages under context pressure', () => {
      const ctx = createBasicContext({ tokenCounter: countByChars });
      ctx.maxContextTokens = 400;
      const consumed = new ToolMessage({
        content: 'x'.repeat(20_000),
        tool_call_id: 't1',
        name: 'tool',
      });
      const messages: AIMessage[] = [
        new HumanMessage('question') as unknown as AIMessage,
        new AIMessage({
          content: '',
          tool_calls: [{ id: 't1', name: 'tool', args: {} }],
        }),
        consumed as unknown as AIMessage,
        new AIMessage('final answer'),
      ];
      const originalRef = messages[2];
      const originalContent = (messages[2] as unknown as ToolMessage).content;

      ctx.projectContextUsage(messages);

      expect(messages[2]).toBe(originalRef);
      expect((messages[2] as unknown as ToolMessage).content).toBe(
        originalContent
      );
    });

    it('recounts the supplied branch, ignoring a stale context token map', () => {
      const ctx = createBasicContext({ tokenCounter: countByChars });
      ctx.maxContextTokens = 3_000;
      // Empty/stale map — if it were reused, every message would count as 0 and
      // nothing would prune. The fresh recount must drive pruning instead.
      ctx.indexTokenCountMap = {};
      const messages: AIMessage[] = [];
      for (let i = 0; i < 6; i++) {
        messages.push(
          new HumanMessage('x'.repeat(1_000)) as unknown as AIMessage
        );
      }

      const usage = ctx.projectContextUsage(messages);

      expect(usage).not.toBeNull();
      expect(usage!.breakdown.messageCount).toBeLessThan(6);
    });

    it('uses a caller-supplied token map when provided', () => {
      const { ctx, messages } = buildBranch(3_000, 1, 6);
      // Each message is ~1 char, so a recount would fit all 6. The supplied map
      // claims 1000 each, forcing a prune — proving the map is honored.
      const indexTokenCountMap: Record<string, number> = {};
      for (let i = 0; i < messages.length; i++) {
        indexTokenCountMap[String(i)] = 1_000;
      }

      const usage = ctx.projectContextUsage(messages, { indexTokenCountMap });

      expect(usage!.breakdown.messageCount).toBeLessThan(6);
    });

    it('ignores this context live usage so projections are not recalibrated', () => {
      const build = (): { ctx: AgentContext; messages: AIMessage[] } => {
        const ctx = createBasicContext({ tokenCounter: countByChars });
        ctx.maxContextTokens = 5_000;
        const messages: AIMessage[] = [0, 1, 2].map(
          () => new HumanMessage('x'.repeat(1_000)) as unknown as AIMessage
        );
        return { ctx, messages };
      };

      const clean = build();
      const cleanUsage = clean.ctx.projectContextUsage(clean.messages);

      const dirty = build();
      dirty.ctx.currentUsage = {
        input_tokens: 4_000,
        output_tokens: 50,
        total_tokens: 4_050,
      };
      dirty.ctx.updateLastCallUsage({ input_tokens: 4_000, output_tokens: 50 });
      const dirtyUsage = dirty.ctx.projectContextUsage(dirty.messages);

      expect(dirtyUsage!.remainingContextTokens).toBe(
        cleanUsage!.remainingContextTokens
      );
      expect(dirtyUsage!.calibrationRatio).toBe(cleanUsage!.calibrationRatio);
    });

    it('does not mutate AI message content arrays during projection', () => {
      const ctx = createBasicContext({
        agentConfig: {
          provider: Providers.ANTHROPIC,
          clientOptions: {
            model: 'claude-x',
            thinking: { type: 'enabled', budget_tokens: 1024 },
          } as never,
        },
        tokenCounter: countByChars,
      });
      ctx.maxContextTokens = 2_000;
      const aiContent = [
        { type: 'thinking', thinking: 'step by step', signature: 'sig' },
        { type: 'text', text: 'the answer' },
      ];
      const ai = new AIMessage({ content: aiContent as never });
      const messages: AIMessage[] = [
        new HumanMessage('question') as unknown as AIMessage,
        ai,
        new HumanMessage('another') as unknown as AIMessage,
      ];
      const contentRef = ai.content;
      const lenBefore = (ai.content as unknown[]).length;

      ctx.projectContextUsage(messages);

      expect(messages[1].content).toBe(contentRef);
      expect((messages[1].content as unknown[]).length).toBe(lenBefore);
    });

    it('honors an explicit calibrationRatio seed', () => {
      const base = buildBranch(100_000, 1_000, 4);
      const baseUsage = base.ctx.projectContextUsage(base.messages);

      const scaled = buildBranch(100_000, 1_000, 4);
      const scaledUsage = scaled.ctx.projectContextUsage(scaled.messages, {
        calibrationRatio: 3,
      });

      expect(scaledUsage!.calibrationRatio).toBe(3);
      expect(scaledUsage!.remainingContextTokens).not.toBe(
        baseUsage!.remainingContextTokens
      );
    });

    it('refreshes a stale system runnable before projecting', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'system prompt' },
        tokenCounter: countByChars,
      });
      ctx.maxContextTokens = 5_000;
      ctx.initializeSystemRunnable();
      const systemBefore = ctx.systemMessageTokens;

      // Adds a handoff preamble + marks stale, but defers the token recount.
      ctx.setHandoffContext('PriorAgent', ['SiblingA', 'SiblingB']);
      ctx.projectContextUsage([new HumanMessage('hi') as unknown as AIMessage]);

      expect(ctx.systemMessageTokens).toBeGreaterThan(systemBefore);
    });
  });
});
