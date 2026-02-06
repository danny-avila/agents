// src/specs/agent-handoffs.test.ts
import { DynamicStructuredTool } from '@langchain/core/tools';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Providers, Constants } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { Run } from '@/run';

/**
 * Helper to safely get tool name from tool object
 */
const getToolName = (tool: t.GraphTools[0]): string | undefined => {
  return (tool as { name?: string }).name;
};

/**
 * Helper to safely get tool description from tool object
 */
const getToolDescription = (tool: t.GraphTools[0]): string | undefined => {
  return (tool as { description?: string }).description;
};

/**
 * Helper to safely get tool schema from tool object
 */
const getToolSchema = (tool: t.GraphTools[0]): unknown => {
  return (tool as { schema?: unknown }).schema;
};

/**
 * Helper to find tool by name
 */
const findToolByName = (
  tools: t.GraphTools | undefined,
  name: string
): t.GraphTools[0] | undefined => {
  return tools?.find((tool) => getToolName(tool) === name);
};

/**
 * Test suite for Agent Handoffs feature
 *
 * Tests cover:
 * - Basic handoff between two agents
 * - Handoffs with custom descriptions
 * - Handoffs with prompts and prompt keys
 * - Sequential handoffs (A -> B -> C)
 * - Bidirectional handoffs (A <-> B)
 * - Multiple handoff options from single agent
 * - Handoff tool creation and execution
 * - Error cases and edge conditions
 */
describe('Agent Handoffs Tests', () => {
  jest.setTimeout(30000);

  const createTestConfig = (
    agents: t.AgentInputs[],
    edges: t.GraphEdge[]
  ): t.RunConfig => ({
    runId: `handoff-test-${Date.now()}-${Math.random()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    returnContent: true,
  });

  const createBasicAgent = (
    agentId: string,
    instructions: string
  ): t.AgentInputs => ({
    agentId,
    provider: Providers.ANTHROPIC,
    clientOptions: {
      modelName: 'claude-haiku-4-5',
      apiKey: 'test-key',
    },
    instructions,
    maxContextTokens: 28000,
  });

  describe('Basic Handoff Tests', () => {
    it('should create handoff tool for agent with outgoing handoff edge', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      expect(run.Graph).toBeDefined();

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      expect(agentAContext).toBeDefined();
      expect(agentAContext?.graphTools).toBeDefined();

      // Check that handoff tool was created
      const handoffTool = findToolByName(
        agentAContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(handoffTool).toBeDefined();
      expect(getToolDescription(handoffTool!)).toBe('Transfer to agent B');
    });

    it('should successfully handoff from agent A to agent B', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A. Transfer to agent B.'),
        createBasicAgent('agent_b', 'You are agent B. Respond to the user.'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B when needed',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Override models to simulate handoff behavior
      run.Graph?.overrideTestModel(
        [
          'Transferring to agent B', // Agent A response
          'Hello from agent B', // Agent B response
        ],
        10,
        [
          {
            id: 'tool_call_1',
            name: `${Constants.LC_TRANSFER_TO_}agent_b`,
            args: {},
          } as ToolCall,
        ]
      );

      const messages = [new HumanMessage('Hello')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-handoff-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages!.length).toBeGreaterThan(1);

      // Check for tool message indicating handoff
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffToolMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(handoffToolMessage).toBeDefined();
      expect(handoffToolMessage?.content).toContain('transferred to agent_b');
    });

    it('should not create handoff tool for agent without outgoing edges', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
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

      // Agent B should not have handoff tools (no outgoing edges)
      const handoffTools = agentBContext?.graphTools?.filter((tool) => {
        const name = getToolName(tool);
        return name?.startsWith(Constants.LC_TRANSFER_TO_) ?? false;
      });
      expect(handoffTools?.length ?? 0).toBe(0);
    });
  });

  describe('Bidirectional Handoffs', () => {
    it('should create handoff tools for both agents in bidirectional setup', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
        },
        {
          from: 'agent_b',
          to: 'agent_a',
          edgeType: 'handoff',
          description: 'Transfer to agent A',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );

      // Agent A should have tool to transfer to B
      const agentAHandoffTool = findToolByName(
        agentAContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(agentAHandoffTool).toBeDefined();

      // Agent B should have tool to transfer to A
      const agentBHandoffTool = findToolByName(
        agentBContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}agent_a`
      );
      expect(agentBHandoffTool).toBeDefined();
    });

    it('should handle handoff from A to B in bidirectional setup', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
        {
          from: 'agent_b',
          to: 'agent_a',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Simulate single handoff from A to B
      run.Graph?.overrideTestModel(
        ['Transferring to B', 'Response from B'],
        10,
        [
          {
            id: 'tool_call_1',
            name: `${Constants.LC_TRANSFER_TO_}agent_b`,
            args: {},
          } as ToolCall,
        ]
      );

      const messages = [new HumanMessage('Start conversation')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-bidirectional-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();

      // Should have a handoff tool message
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(handoffMessage).toBeDefined();
    });
  });

  describe('Sequential Handoffs (Chain)', () => {
    it('should create handoff tools for chain of agents A -> B -> C', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
        createBasicAgent('agent_c', 'You are agent C'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
        },
        {
          from: 'agent_b',
          to: 'agent_c',
          edgeType: 'handoff',
          description: 'Transfer to agent C',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );
      const agentCContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_c'
      );

      // Agent A should have tool to transfer to B
      expect(
        findToolByName(
          agentAContext?.graphTools,
          `${Constants.LC_TRANSFER_TO_}agent_b`
        )
      ).toBeDefined();

      // Agent B should have tool to transfer to C
      expect(
        findToolByName(
          agentBContext?.graphTools,
          `${Constants.LC_TRANSFER_TO_}agent_c`
        )
      ).toBeDefined();

      // Agent C should have no handoff tools
      const agentCHandoffTools = agentCContext?.graphTools?.filter((tool) => {
        const name = getToolName(tool);
        return name?.startsWith(Constants.LC_TRANSFER_TO_) ?? false;
      });
      expect(agentCHandoffTools?.length ?? 0).toBe(0);
    });
  });

  describe('Multiple Handoff Options', () => {
    it('should create multiple handoff tools when agent has multiple outgoing edges', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router agent'),
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
        createBasicAgent('agent_c', 'You are agent C'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'agent_a',
          edgeType: 'handoff',
          description: 'Transfer to agent A for task A',
        },
        {
          from: 'router',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B for task B',
        },
        {
          from: 'router',
          to: 'agent_c',
          edgeType: 'handoff',
          description: 'Transfer to agent C for task C',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const routerContext = (run.Graph as StandardGraph).agentContexts.get(
        'router'
      );
      expect(routerContext).toBeDefined();

      // Router should have 3 handoff tools
      const handoffTools = routerContext?.graphTools?.filter((tool) => {
        const name = getToolName(tool);
        return name?.startsWith(Constants.LC_TRANSFER_TO_) ?? false;
      });
      expect(handoffTools?.length).toBe(3);

      // Verify each tool exists
      expect(
        findToolByName(handoffTools, `${Constants.LC_TRANSFER_TO_}agent_a`)
      ).toBeDefined();
      expect(
        findToolByName(handoffTools, `${Constants.LC_TRANSFER_TO_}agent_b`)
      ).toBeDefined();
      expect(
        findToolByName(handoffTools, `${Constants.LC_TRANSFER_TO_}agent_c`)
      ).toBeDefined();
    });

    it('should route to correct agent based on handoff tool used', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'agent_a',
          edgeType: 'handoff',
          description: 'Transfer to agent A',
        },
        {
          from: 'router',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // Router chooses agent_b
      run.Graph?.overrideTestModel(
        ['Routing to agent B', 'Hello from agent B'],
        10,
        [
          {
            id: 'tool_call_1',
            name: `${Constants.LC_TRANSFER_TO_}agent_b`,
            args: {},
          } as ToolCall,
        ]
      );

      const messages = [new HumanMessage('Route this message')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-routing-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      // Should have handoff to agent_b, not agent_a
      const handoffToB = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(handoffToB).toBeDefined();

      const handoffToA = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}agent_a`
      );
      expect(handoffToA).toBeUndefined();
    });
  });

  describe('Handoffs with Prompts', () => {
    it('should create handoff tool with prompt parameter when prompt is specified', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B with instructions',
          prompt: 'Provide specific instructions for agent B',
          promptKey: 'instructions',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      const handoffTool = findToolByName(
        agentAContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}agent_b`
      );

      expect(handoffTool).toBeDefined();
      // Tool should accept parameters (schema should be defined)
      expect(getToolSchema(handoffTool!)).toBeDefined();
    });

    it('should use default promptKey when not specified', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          prompt: 'Instructions for handoff',
          // promptKey not specified, should default to 'instructions'
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      const handoffTool = findToolByName(
        agentAContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}agent_b`
      );

      expect(handoffTool).toBeDefined();
      expect(getToolSchema(handoffTool!)).toBeDefined();
    });

    it('should include prompt content in handoff tool message', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
          prompt: 'Additional context for agent B',
          promptKey: 'context',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      run.Graph?.overrideTestModel(['Transferring with context'], 10, [
        {
          id: 'tool_call_1',
          name: `${Constants.LC_TRANSFER_TO_}agent_b`,
          args: { context: 'User needs help with booking' },
        } as ToolCall,
      ]);

      const messages = [new HumanMessage('Help me')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-prompt-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}agent_b`
      );

      expect(handoffMessage).toBeDefined();
      // Tool message should contain the prompt key and value
      expect(handoffMessage?.content).toContain('Context:');
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle self-referential edge gracefully', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_a',
          edgeType: 'handoff',
          description: 'Self-handoff (should be allowed but unusual)',
        },
      ];

      // Should not throw during creation
      expect(async () => {
        await Run.create(createTestConfig(agents, edges));
      }).not.toThrow();
    });

    it('should handle empty edges array', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [];

      const run = await Run.create(createTestConfig(agents, edges));

      expect(run.Graph).toBeDefined();

      // Agents should have no handoff tools
      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );
      const handoffTools = agentAContext?.graphTools?.filter((tool) => {
        const name = getToolName(tool);
        return name?.startsWith(Constants.LC_TRANSFER_TO_) ?? false;
      });
      expect(handoffTools?.length ?? 0).toBe(0);
    });

    it('should start from first agent when no edges are defined', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [];

      const run = await Run.create(createTestConfig(agents, edges));

      run.Graph?.overrideTestModel(['Response from first agent'], 10);

      const messages = [new HumanMessage('Hello')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-no-edges-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages!.length).toBeGreaterThan(0);
    });

    it('should handle agents with existing tools alongside handoff tools', async () => {
      const customTool = new DynamicStructuredTool({
        name: 'custom_tool',
        description: 'A custom tool',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> => 'Tool result',
      });

      const agents: t.AgentInputs[] = [
        {
          ...createBasicAgent('agent_a', 'You are agent A'),
          tools: [customTool],
        },
        createBasicAgent('agent_b', 'You are agent B'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          description: 'Transfer to agent B',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentAContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_a'
      );

      // Agent A should have custom tool in tools and handoff tool in graphTools
      expect(findToolByName(agentAContext?.tools, 'custom_tool')).toBeDefined();

      expect(
        findToolByName(
          agentAContext?.graphTools,
          `${Constants.LC_TRANSFER_TO_}agent_b`
        )
      ).toBeDefined();
    });
  });

  describe('Graph Structure Analysis', () => {
    it('should correctly identify starting nodes with no incoming edges', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'Starting agent'),
        createBasicAgent('agent_b', 'Middle agent'),
        createBasicAgent('agent_c', 'End agent'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
        {
          from: 'agent_b',
          to: 'agent_c',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      // agent_a should be the starting node (no incoming edges)
      expect(run.Graph).toBeDefined();
      // This is internal behavior, but we can test via execution
      run.Graph?.overrideTestModel(['Response from agent A'], 10);

      const messages = [new HumanMessage('Start')];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-starting-node-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      // Should start from agent_a
      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
    });

    it('should handle multiple starting nodes (parallel entry points)', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'Starting agent A'),
        createBasicAgent('agent_b', 'Starting agent B'),
        createBasicAgent('agent_c', 'Shared destination'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_c',
          edgeType: 'handoff',
        },
        {
          from: 'agent_b',
          to: 'agent_c',
          edgeType: 'handoff',
        },
      ];

      // Both agent_a and agent_b have no incoming edges, so both are starting nodes
      const run = await Run.create(createTestConfig(agents, edges));

      expect(run.Graph).toBeDefined();
    });
  });

  describe('Handoff Tool Naming', () => {
    it('should use correct naming convention for handoff tools', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('flight_assistant', 'You handle flights'),
        createBasicAgent('hotel_assistant', 'You handle hotels'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'flight_assistant',
          to: 'hotel_assistant',
          edgeType: 'handoff',
          description: 'Transfer to hotel booking',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const flightContext = (run.Graph as StandardGraph).agentContexts.get(
        'flight_assistant'
      );
      const handoffTool = findToolByName(
        flightContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}hotel_assistant`
      );

      expect(handoffTool).toBeDefined();
      expect(getToolName(handoffTool!)).toBe(
        `${Constants.LC_TRANSFER_TO_}hotel_assistant`
      );
    });

    it('should preserve agent ID format in tool names', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_with_underscores', 'Agent with underscores'),
        createBasicAgent('AgentWithCamelCase', 'Agent with camel case'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_with_underscores',
          to: 'AgentWithCamelCase',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      const agentContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_with_underscores'
      );
      const handoffTool = findToolByName(
        agentContext?.graphTools,
        `${Constants.LC_TRANSFER_TO_}AgentWithCamelCase`
      );

      expect(handoffTool).toBeDefined();
      expect(getToolName(handoffTool!)).toBe(
        `${Constants.LC_TRANSFER_TO_}AgentWithCamelCase`
      );
    });
  });
});
