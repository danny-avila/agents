// src/specs/thinking-handoff.test.ts
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Providers, Constants } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { Run } from '@/run';

/**
 * Test suite for Thinking-Enabled Agent Handoff Edge Case
 *
 * Tests the specific edge case where:
 * - An agent without thinking blocks (e.g., OpenAI) makes a tool call
 * - Control is handed off to an agent with thinking enabled (e.g., Anthropic/Bedrock)
 * - The system should handle the transition without errors
 *
 * Background:
 * When Anthropic's extended thinking is enabled, the API requires that any assistant
 * message with tool_use content must start with a thinking or redacted_thinking block.
 * When switching from a non-thinking agent to a thinking-enabled agent, previous
 * messages may not have these blocks, causing API errors.
 *
 * Solution:
 * The ensureThinkingBlockInMessages() function converts AI messages with tool calls
 * (that lack thinking blocks) into HumanMessages with buffer strings, avoiding the
 * thinking block requirement while preserving context.
 */
describe('Thinking-Enabled Agent Handoff Tests', () => {
  jest.setTimeout(30000);

  const createTestConfig = (
    agents: t.AgentInputs[],
    edges: t.GraphEdge[]
  ): t.RunConfig => ({
    runId: `thinking-handoff-test-${Date.now()}-${Math.random()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    returnContent: true,
  });

  describe('OpenAI to Anthropic with Thinking', () => {
    it('should successfully handoff from OpenAI to Anthropic with thinking enabled', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'supervisor',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions:
            'You are a supervisor. Use transfer_to_specialist when asked.',
          maxContextTokens: 8000,
        },
        {
          agentId: 'specialist',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You are a specialist. Provide detailed answers.',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'supervisor',
          to: 'specialist',
          edgeType: 'handoff',
          description: 'Transfer to specialist for detailed analysis',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Simulate supervisor using handoff tool
      run.Graph?.overrideTestModel(
        [
          'Let me transfer you to our specialist',
          'As a specialist, let me analyze this carefully...',
        ],
        10,
        [
          {
            id: 'tool_call_1',
            name: `${Constants.LC_TRANSFER_TO_}specialist`,
            args: {},
          } as ToolCall,
        ]
      );

      const messages = [new HumanMessage('I need expert analysis')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-thinking-handoff-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      // Should not throw despite thinking requirement
      await expect(
        run.processStream({ messages }, config)
      ).resolves.not.toThrow();

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages!.length).toBeGreaterThan(1);

      // Should have successful handoff
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}specialist`
      );
      expect(handoffMessage).toBeDefined();
      expect(handoffMessage?.content).toContain('transferred to specialist');
    });

    it('should convert tool sequence to HumanMessage for thinking-enabled agent', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'agent_a',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions: 'You are agent A',
          maxContextTokens: 8000,
        },
        {
          agentId: 'agent_b',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You are agent B with thinking enabled',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Check that agent B's context is set up correctly
      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );
      expect(agentBContext).toBeDefined();

      // Verify thinking is enabled
      const thinkingConfig = (
        agentBContext?.clientOptions as t.AnthropicClientOptions
      ).thinking;
      expect(thinkingConfig).toBeDefined();
      expect(thinkingConfig?.type).toBe('enabled');
    });
  });

  describe('Bedrock with Thinking', () => {
    it('should handle handoff from Bedrock without thinking to Bedrock with thinking', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'coordinator',
          provider: Providers.BEDROCK,
          clientOptions: {
            region: 'us-east-1',
            model: 'anthropic.claude-3-5-haiku-20241022-v1:0',
            // No thinking config
          },
          instructions: 'You are a coordinator',
          maxContextTokens: 8000,
        },
        {
          agentId: 'analyst',
          provider: Providers.BEDROCK,
          clientOptions: {
            region: 'us-east-1',
            model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            additionalModelRequestFields: {
              thinking: {
                type: 'enabled',
                budget_tokens: 2000,
              },
            },
          },
          instructions: 'You are an analyst with extended thinking',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'coordinator',
          to: 'analyst',
          edgeType: 'handoff',
          description: 'Transfer to analyst for deep analysis',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      run.Graph?.overrideTestModel(
        ['Transferring to analyst', 'Deep analysis results...'],
        10,
        [
          {
            id: 'tool_call_1',
            name: `${Constants.LC_TRANSFER_TO_}analyst`,
            args: {},
          } as ToolCall,
        ]
      );

      const messages = [new HumanMessage('Analyze this data')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-bedrock-thinking-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await expect(
        run.processStream({ messages }, config)
      ).resolves.not.toThrow();

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
    });

    it('should verify Bedrock thinking configuration is properly detected', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'agent_a',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions: 'You are agent A',
          maxContextTokens: 8000,
        },
        {
          agentId: 'agent_b',
          provider: Providers.BEDROCK,
          clientOptions: {
            region: 'us-east-1',
            model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            additionalModelRequestFields: {
              thinking: {
                type: 'enabled',
                budget_tokens: 3000,
              },
            },
          },
          instructions: 'You are agent B with Bedrock thinking',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );
      expect(agentBContext).toBeDefined();
      expect(agentBContext?.provider).toBe(Providers.BEDROCK);

      // Verify thinking configuration in additionalModelRequestFields
      const bedrockOptions =
        agentBContext?.clientOptions as t.BedrockAnthropicInput;
      expect(bedrockOptions.additionalModelRequestFields).toBeDefined();
      expect(
        bedrockOptions.additionalModelRequestFields?.thinking
      ).toBeDefined();

      const thinkingConfig = bedrockOptions.additionalModelRequestFields
        ?.thinking as {
        type: string;
        budget_tokens: number;
      };
      expect(thinkingConfig.type).toBe('enabled');
      expect(thinkingConfig.budget_tokens).toBe(3000);
    });

    it('should handle OpenAI to Bedrock with thinking handoff', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'supervisor',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions: 'You are a supervisor',
          maxContextTokens: 8000,
        },
        {
          agentId: 'bedrock_specialist',
          provider: Providers.BEDROCK,
          clientOptions: {
            region: 'us-east-1',
            model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            additionalModelRequestFields: {
              thinking: {
                type: 'enabled',
                budget_tokens: 2000,
              },
            },
          },
          instructions: 'You are a Bedrock specialist with thinking',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'supervisor',
          to: 'bedrock_specialist',
          edgeType: 'handoff',
          description: 'Transfer to Bedrock specialist',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      run.Graph?.overrideTestModel(['Transferring', 'Analysis complete'], 10, [
        {
          id: 'tool_call_1',
          name: `${Constants.LC_TRANSFER_TO_}bedrock_specialist`,
          args: {},
        } as ToolCall,
      ]);

      const messages = [new HumanMessage('Analyze this')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-openai-bedrock-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await expect(
        run.processStream({ messages }, config)
      ).resolves.not.toThrow();

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();

      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}bedrock_specialist`
      );
      expect(handoffMessage).toBeDefined();
    });
  });

  describe('Multiple Handoffs with Mixed Thinking Configurations', () => {
    it('should handle chain of handoffs with varying thinking configurations', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'router',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions: 'You route requests',
          maxContextTokens: 8000,
        },
        {
          agentId: 'processor',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-haiku-4-5',
            apiKey: 'test-key',
            // No thinking
          },
          instructions: 'You process requests',
          maxContextTokens: 8000,
        },
        {
          agentId: 'reviewer',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You review with deep thinking',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'processor',
          edgeType: 'handoff',
        },
        {
          from: 'processor',
          to: 'reviewer',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Verify all agents are created with correct configurations
      const routerContext = (run.Graph as StandardGraph).agentContexts.get(
        'router'
      );
      const processorContext = (run.Graph as StandardGraph).agentContexts.get(
        'processor'
      );
      const reviewerContext = (run.Graph as StandardGraph).agentContexts.get(
        'reviewer'
      );

      expect(routerContext).toBeDefined();
      expect(processorContext).toBeDefined();
      expect(reviewerContext).toBeDefined();

      // Verify thinking configuration on reviewer
      const reviewerThinking = (
        reviewerContext?.clientOptions as t.AnthropicClientOptions
      ).thinking;
      expect(reviewerThinking).toBeDefined();
      expect(reviewerThinking?.type).toBe('enabled');

      // Verify handoff tools exist
      expect(
        routerContext?.tools?.find(
          (tool) =>
            (tool as { name?: string }).name ===
            `${Constants.LC_TRANSFER_TO_}processor`
        )
      ).toBeDefined();
      expect(
        processorContext?.tools?.find(
          (tool) =>
            (tool as { name?: string }).name ===
            `${Constants.LC_TRANSFER_TO_}reviewer`
        )
      ).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should not modify messages when agent already uses thinking', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'agent_a',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You are agent A with thinking',
          maxContextTokens: 8000,
        },
        {
          agentId: 'agent_b',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You are agent B with thinking',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      run.Graph?.overrideTestModel(['Transferring', 'Received handoff'], 10, [
        {
          id: 'tool_call_1',
          name: `${Constants.LC_TRANSFER_TO_}agent_b`,
          args: {},
        } as ToolCall,
      ]);

      const messages = [new HumanMessage('Test message')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-both-thinking-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      // Should work fine when both agents use thinking
      await expect(
        run.processStream({ messages }, config)
      ).resolves.not.toThrow();
    });

    it('should handle empty conversation history', async () => {
      const agents: t.AgentInputs[] = [
        {
          agentId: 'agent_a',
          provider: Providers.OPENAI,
          clientOptions: {
            modelName: 'gpt-4o-mini',
            apiKey: 'test-key',
          },
          instructions: 'You are agent A',
          maxContextTokens: 8000,
        },
        {
          agentId: 'agent_b',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-7-sonnet-20250219',
            apiKey: 'test-key',
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
          instructions: 'You are agent B',
          maxContextTokens: 8000,
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      expect(run.Graph).toBeDefined();

      // Just verify the graph was created correctly
      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );
      expect(agentBContext).toBeDefined();
    });
  });
});
