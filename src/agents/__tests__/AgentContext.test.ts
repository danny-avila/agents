// src/agents/__tests__/AgentContext.test.ts
import { AgentContext } from '../AgentContext';
import { Providers } from '@/common';
import type * as t from '@/types';

describe('AgentContext', () => {
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

    it('includes additional_instructions in system message', () => {
      const ctx = createBasicContext({
        agentConfig: {
          instructions: 'Base instructions',
          additional_instructions: 'Additional instructions',
        },
      });
      expect(ctx.systemRunnable).toBeDefined();
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
    it('includes code_execution-only tools in system message', () => {
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
        agentConfig: { instructions: 'Base', toolRegistry },
      });

      const runnable = ctx.systemRunnable;
      expect(runnable).toBeDefined();
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
  });

  describe('reset()', () => {
    it('clears all cached state', () => {
      const ctx = createBasicContext({ agentConfig: { instructions: 'Test' } });

      ctx.markToolsAsDiscovered(['tool1']);
      void ctx.systemRunnable;
      ctx.instructionTokens = 100;
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

    it('setSummary increases instructionTokens and marks systemRunnable stale', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      const baseInstructionTokens = ctx.instructionTokens;
      expect(baseInstructionTokens).toBeGreaterThan(0);

      ctx.setSummary('User asked about math. Key results: 2+2=4, 3*5=15.', 50);
      expect(ctx.hasSummary()).toBe(true);

      void ctx.systemRunnable;
      const postSummaryTokens = ctx.instructionTokens;
      expect(postSummaryTokens).toBeGreaterThan(baseInstructionTokens);
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

    it('clearSummary restores instructionTokens to pre-summary level', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
      });

      void ctx.systemRunnable;
      const baseTokens = ctx.instructionTokens;

      ctx.setSummary('Summary of the conversation so far.', 35);
      void ctx.systemRunnable;
      const withSummaryTokens = ctx.instructionTokens;
      expect(withSummaryTokens).toBeGreaterThan(baseTokens);

      ctx.clearSummary();
      void ctx.systemRunnable;
      const afterClearTokens = ctx.instructionTokens;
      expect(afterClearTokens).toBe(baseTokens);
    });

    it('reset clears summary and restores base token counts', () => {
      const ctx = createBasicContext({
        agentConfig: { instructions: 'Be helpful.' },
        tokenCounter: charTokenCounter,
        indexTokenCountMap: { '0': 10, '1': 20 },
      });

      void ctx.systemRunnable;
      ctx.setSummary('Summary text.', 15);
      void ctx.systemRunnable;
      expect(ctx.hasSummary()).toBe(true);

      ctx.reset();
      expect(ctx.hasSummary()).toBe(false);

      void ctx.systemRunnable;
      const postResetTokens = ctx.instructionTokens;
      expect(postResetTokens).toBeGreaterThan(0);
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
});
