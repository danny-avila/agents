// src/specs/agent-handoffs.test.ts
import { MemorySaver } from '@langchain/langgraph';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  getBufferString,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  RunnableLambda,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { Providers, GraphEvents, Constants } from '@/common';
import { createHandlers } from '@/utils/handlers';
import { StandardGraph } from '@/graphs/Graph';
import { ToolNode } from '@/tools/ToolNode';
import { FakeChatModel } from '@/llm/fake';
import * as events from '@/utils/events';
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

const createGraphTool = (name: string, output: string): t.GenericTool =>
  new RunnableToolLike({
    name,
    description: `Test ${name} tool`,
    schema: z.object({}),
    bound: RunnableLambda.from(async (): Promise<string> => output),
  });

class MetadataCapturingToolNode extends ToolNode<t.BaseGraphState> {
  observedConfig?: RunnableConfig;

  protected override async run(
    input: t.BaseGraphState,
    config: RunnableConfig
  ): Promise<t.BaseGraphState> {
    this.observedConfig = config;
    return input;
  }
}

type HandoffModelScript = {
  promptMarker: string;
  response: string;
  toolCalls?: ToolCall[];
};

function validateToolHistory(messages: t.BaseGraphState['messages']): void {
  const toolCallIds = new Set<string>();
  const toolResultIds = new Set<string>();
  for (const message of messages) {
    if (message.getType() === 'ai') {
      for (const toolCall of (message as AIMessage).tool_calls ?? []) {
        if (toolCall.id != null) {
          toolCallIds.add(toolCall.id);
        }
      }
    } else if (message.getType() === 'tool') {
      toolResultIds.add((message as ToolMessage).tool_call_id);
    }
  }
  for (const toolCallId of toolCallIds) {
    if (!toolResultIds.has(toolCallId)) {
      throw new Error(`Tool call ${toolCallId} has no matching result`);
    }
  }
  for (const toolResultId of toolResultIds) {
    if (!toolCallIds.has(toolResultId)) {
      throw new Error(`Tool result ${toolResultId} has no matching call`);
    }
  }
}

class ScriptedHandoffModel extends FakeChatModel {
  private scripts: HandoffModelScript[];

  constructor(scripts: HandoffModelScript[]) {
    super({ responses: [''] });
    this.scripts = scripts;
  }

  override async *_streamResponseChunks(
    messages: t.BaseGraphState['messages'],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    validateToolHistory(messages);
    const prompt = getBufferString(messages);
    let script: HandoffModelScript | undefined;
    let latestMarkerIndex = -1;
    for (const candidate of this.scripts) {
      const markerIndex = prompt.lastIndexOf(candidate.promptMarker);
      if (markerIndex > latestMarkerIndex) {
        script = candidate;
        latestMarkerIndex = markerIndex;
      }
    }
    if (script == null) {
      throw new Error(`No handoff model script matched prompt: ${prompt}`);
    }
    const model = new FakeChatModel({
      responses: [script.response],
      toolCalls: script.toolCalls,
    });
    yield* model._streamResponseChunks(messages, options, runManager);
  }
}

type HandoffReceptionProbe = {
  processHandoffReception(
    messages: t.BaseGraphState['messages'],
    agentId: string
  ): {
    instructions: string | null;
    parallelGroupId?: number;
    filteredMessages?: t.BaseGraphState['messages'];
  } | null;
};

type PendingSend = {
  node: string;
  args: { messages?: t.BaseGraphState['messages'] };
};

const isPendingSend = (value: unknown): value is PendingSend => {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const candidate = value as { node?: unknown; args?: unknown };
  return (
    typeof candidate.node === 'string' &&
    candidate.args != null &&
    typeof candidate.args === 'object'
  );
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
    skipCleanup: true,
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

    it('should not expose an inbound handoff tool to its recipient', async () => {
      const leakedTransferTool = createGraphTool(
        `${Constants.LC_TRANSFER_TO_}agent_b`,
        'transferred'
      );
      const retainedGraphTool = createGraphTool(
        'retained_graph_tool',
        'retained'
      );
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        {
          ...createBasicAgent('agent_b', 'You are agent B'),
          graphTools: [leakedTransferTool, retainedGraphTool],
        },
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
        },
      ];

      const config = createTestConfig(agents, edges);
      config.tokenCounter = () => 1;
      const run = await Run.create(config);

      const agentBContext = (run.Graph as StandardGraph).agentContexts.get(
        'agent_b'
      );
      expect(agentBContext).toBeDefined();

      // Agent B has no outgoing edge, so any inbound transfer is removed.
      const handoffTools = agentBContext?.graphTools?.filter((tool) => {
        const name = getToolName(tool);
        return name?.startsWith(Constants.LC_TRANSFER_TO_) ?? false;
      });
      expect(handoffTools?.length ?? 0).toBe(0);
      expect(
        findToolByName(agentBContext?.graphTools, 'retained_graph_tool')
      ).toBeDefined();
      await agentBContext?.tokenCalculationPromise;
      expect(agentBContext?.toolTokenCounts).not.toHaveProperty(
        `${Constants.LC_TRANSFER_TO_}agent_b`
      );
      expect(agentBContext?.toolTokenCounts).toHaveProperty(
        'retained_graph_tool'
      );
    });

    it('stamps the owning agent on tool observation metadata', async () => {
      const node = new MetadataCapturingToolNode({
        tools: [],
        executingAgentId: 'mateo',
        executingAgentName: 'Mateo Serrano',
        rootAgentId: 'leila',
        rootAgentName: 'Leila Mensah',
      });

      await node.invoke(
        { messages: [] },
        { metadata: { requestId: 'request-1' } }
      );

      expect(node.observedConfig?.metadata).toMatchObject({
        requestId: 'request-1',
        agentId: 'mateo',
        activeAgentId: 'mateo',
        activeAgentName: 'Mateo Serrano',
        rootAgentId: 'leila',
        rootAgentName: 'Leila Mensah',
      });
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

  describe('Parallel Handoffs', () => {
    it('should expose one runtime group on simultaneous destination content', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('left', 'You are the left specialist'),
        createBasicAgent('right', 'You are the right specialist'),
      ];
      const edges: t.GraphEdge[] = [
        { from: 'router', to: 'left', edgeType: 'handoff' },
        { from: 'router', to: 'right', edgeType: 'handoff' },
      ];
      const { contentParts, handlers } = createHandlers();
      const runConfig = createTestConfig(agents, edges);
      runConfig.customHandlers = handlers;
      const run = await Run.create(runConfig);

      run.Graph?.overrideTestModel(
        ['Routing', 'Left complete', 'Right complete'],
        10,
        [
          {
            id: 'tool_call_left',
            name: `${Constants.LC_TRANSFER_TO_}left`,
            args: {},
          } as ToolCall,
          {
            id: 'tool_call_right',
            name: `${Constants.LC_TRANSFER_TO_}right`,
            args: {},
          } as ToolCall,
        ]
      );

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-parallel-handoff-group-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('Run both specialists')] },
        config
      );

      const definedParts = contentParts.filter(
        (part): part is t.MessageContentComplex => part != null
      );
      const routerParts = definedParts.filter(
        (part) => part.agentId === 'router'
      );
      const leftParts = definedParts.filter((part) => part.agentId === 'left');
      const rightParts = definedParts.filter(
        (part) => part.agentId === 'right'
      );
      const leftGroupId = leftParts[0]?.groupId;

      expect(routerParts.length).toBeGreaterThan(0);
      expect(routerParts.every((part) => part.groupId == null)).toBe(true);
      expect(leftParts.length).toBeGreaterThan(0);
      expect(rightParts.length).toBeGreaterThan(0);
      expect(leftGroupId).toEqual(expect.any(Number));
      expect(leftParts.every((part) => part.groupId === leftGroupId)).toBe(
        true
      );
      expect(rightParts.every((part) => part.groupId === leftGroupId)).toBe(
        true
      );
    });

    it('should checkpoint one durable group ID before parallel recipients run', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('left', 'You are the left specialist'),
        createBasicAgent('right', 'You are the right specialist'),
      ];
      const edges: t.GraphEdge[] = [
        { from: 'router', to: 'left', edgeType: 'handoff' },
        { from: 'router', to: 'right', edgeType: 'handoff' },
      ];
      const checkpointer = new MemorySaver();
      const runConfig = createTestConfig(agents, edges);
      runConfig.graphConfig.compileOptions = { checkpointer };
      const run = await Run.create(runConfig);

      run.Graph?.overrideTestModel(
        ['Routing', 'Left complete', 'Right complete'],
        10,
        [
          {
            id: 'tool_call_checkpoint_left',
            name: `${Constants.LC_TRANSFER_TO_}left`,
            args: {},
          } as ToolCall,
          {
            id: 'tool_call_checkpoint_right',
            name: `${Constants.LC_TRANSFER_TO_}right`,
            args: {},
          } as ToolCall,
        ]
      );

      const threadId = 'test-parallel-handoff-checkpoint-thread';
      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
        durability: 'sync';
      } = {
        configurable: { thread_id: threadId },
        streamMode: 'values',
        version: 'v2',
        durability: 'sync',
      };
      const historicalToolCall = {
        id: 'tool_call_before_parallel_handoff',
        name: 'historical_lookup',
        args: {},
      };
      await run.processStream(
        {
          messages: [
            new HumanMessage('Look this up first'),
            new AIMessage({ content: '', tool_calls: [historicalToolCall] }),
            new ToolMessage({
              content: 'Historical result',
              name: historicalToolCall.name,
              tool_call_id: historicalToolCall.id,
            }),
            new HumanMessage('Run both specialists'),
          ],
        },
        config
      );

      let persistedHandoffs: ToolMessage[] = [];
      let persistedSends: PendingSend[] = [];
      for await (const tuple of checkpointer.list({
        configurable: { thread_id: threadId },
      })) {
        const sends = (tuple.pendingWrites ?? [])
          .filter(([, channel]) => channel === '__pregel_tasks')
          .map(([, , value]) => value)
          .filter(isPendingSend)
          .filter(({ node }) => node === 'left' || node === 'right');
        const handoffs = sends.flatMap((send) =>
          (send.args.messages ?? []).filter(
            (message): message is ToolMessage =>
              message.getType() === 'tool' &&
              message.name === `${Constants.LC_TRANSFER_TO_}${send.node}`
          )
        );
        if (handoffs.length === 2) {
          persistedHandoffs = handoffs;
          persistedSends = sends;
          break;
        }
      }

      expect(persistedHandoffs).toHaveLength(2);
      const persistedGroupIds = persistedHandoffs.map(
        (message) => message.additional_kwargs[Constants.HANDOFF_GROUP_ID]
      );
      const persistedBatchIds = persistedHandoffs.map(
        (message) => message.additional_kwargs[Constants.HANDOFF_PARALLEL_BATCH]
      );
      const persistedGroupId = persistedGroupIds[0];
      if (typeof persistedGroupId !== 'number') {
        throw new Error('Expected a persisted numeric handoff group ID');
      }

      expect(Number.isSafeInteger(persistedGroupId)).toBe(true);
      expect(persistedGroupId).toBeGreaterThanOrEqual(2 ** 48);
      expect(persistedGroupIds[1]).toBe(persistedGroupId);
      expect(persistedBatchIds[0]).toEqual(expect.any(String));
      expect(persistedBatchIds[1]).toBe(persistedBatchIds[0]);

      const historicalResults = persistedSends.flatMap((send) =>
        (send.args.messages ?? []).filter(
          (message): message is ToolMessage =>
            message.getType() === 'tool' &&
            (message as ToolMessage).tool_call_id === historicalToolCall.id
        )
      );
      expect(historicalResults).toHaveLength(2);
      for (const result of historicalResults) {
        expect(
          result.additional_kwargs[Constants.HANDOFF_PARALLEL_BATCH]
        ).toBeUndefined();
        expect(
          result.additional_kwargs[Constants.HANDOFF_GROUP_ID]
        ).toBeUndefined();
        expect(
          result.additional_kwargs.handoff_parallel_siblings
        ).toBeUndefined();
      }
    });

    it('should suppress a static group when a hybrid router selects one handoff', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('left', 'You are the left specialist'),
        createBasicAgent('right', 'You are the right specialist'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: ['left', 'right'],
          edgeType: 'direct',
        },
        { from: 'router', to: 'left', edgeType: 'handoff' },
        { from: 'router', to: 'right', edgeType: 'handoff' },
      ];
      const { contentParts, handlers } = createHandlers();
      const runConfig = createTestConfig(agents, edges);
      runConfig.customHandlers = handlers;
      const run = await Run.create(runConfig);

      run.Graph?.overrideTestModel(['Routing', 'Left complete'], 10, [
        {
          id: 'tool_call_left_only',
          name: `${Constants.LC_TRANSFER_TO_}left`,
          args: {},
        } as ToolCall,
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-hybrid-handoff-group-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('Run only the left specialist')] },
        config
      );

      const definedParts = contentParts.filter(
        (part): part is t.MessageContentComplex => part != null
      );
      const leftParts = definedParts.filter((part) => part.agentId === 'left');
      const rightParts = definedParts.filter(
        (part) => part.agentId === 'right'
      );

      expect(leftParts.length).toBeGreaterThan(0);
      expect(leftParts.every((part) => part.groupId == null)).toBe(true);
      expect(rightParts).toHaveLength(0);
    });

    it('should not reuse a historical handoff when a target is later reached directly', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'historical-router-marker'),
        createBasicAgent('recipient', 'historical-recipient-agent-marker'),
        createBasicAgent('sibling', 'historical-sibling-agent-marker'),
      ];
      const edges: t.GraphEdge[] = [
        { from: 'router', to: 'recipient', edgeType: 'direct' },
        {
          from: 'router',
          to: 'recipient',
          edgeType: 'handoff',
          prompt: 'Instructions for the recipient',
        },
        {
          from: 'router',
          to: 'sibling',
          edgeType: 'handoff',
          prompt: 'Instructions for the sibling',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      run.Graph.overrideModel = new ScriptedHandoffModel([
        {
          promptMarker: 'stale-recipient-instructions-marker',
          response: 'Historical recipient complete',
        },
        {
          promptMarker: 'second-direct-request-marker',
          response: 'Direct pass complete',
        },
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-historical-handoff-direct-thread',
        },
        streamMode: 'values',
        version: 'v2',
      };
      const historicalToolCall = {
        id: 'tool_call_historical_recipient',
        name: `${Constants.LC_TRANSFER_TO_}recipient`,
        args: { instructions: 'stale-recipient-instructions-marker' },
      };
      await run.processStream(
        {
          messages: [
            new HumanMessage('first-historical-request-marker'),
            new AIMessage({ content: '', tool_calls: [historicalToolCall] }),
            new ToolMessage({
              content:
                'Successfully transferred to recipient\n\nInstructions: stale-recipient-instructions-marker',
              name: historicalToolCall.name,
              tool_call_id: historicalToolCall.id,
              additional_kwargs: {
                handoff_source_name: 'Router',
                handoff_instructions: 'stale-recipient-instructions-marker',
                handoff_parallel_siblings: ['sibling'],
                [Constants.HANDOFF_GROUP_ID]: 2 ** 48 + 1,
              },
            }),
            new HumanMessage('second-direct-request-marker'),
          ],
        },
        config
      );

      const recipientSteps = run.Graph.getRunSteps('recipient');
      expect(recipientSteps.length).toBeGreaterThan(0);
      expect(recipientSteps.every((step) => step.groupId == null)).toBe(true);
      const directPassMessages = run
        .getRunMessages()
        ?.filter(
          (message) =>
            message.getType() === 'ai' &&
            message.content === 'Direct pass complete'
        );
      expect(directPassMessages).toHaveLength(2);
    });

    it('should clear the handoff identity before a direct re-entry', async () => {
      const agents: t.AgentInputs[] = [
        {
          ...createBasicAgent('router', 'stale-context-router-marker'),
          name: 'Router',
        },
        {
          ...createBasicAgent('recipient', 'stale-context-recipient-marker'),
          name: 'Recipient',
        },
        {
          ...createBasicAgent('sibling', 'stale-context-sibling-marker'),
          name: 'Sibling',
        },
        {
          ...createBasicAgent('relay', 'stale-context-relay-marker'),
          name: 'Relay',
        },
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'recipient',
          edgeType: 'handoff',
          prompt: 'Instructions for the recipient',
        },
        {
          from: 'router',
          to: 'sibling',
          edgeType: 'handoff',
          prompt: 'Instructions for the sibling',
        },
        {
          from: 'recipient',
          to: 'relay',
          edgeType: 'handoff',
          prompt: 'Instructions for the relay',
        },
        { from: 'relay', to: 'recipient', edgeType: 'direct' },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      const model = new ScriptedHandoffModel([
        {
          promptMarker: 'stale-context-router-marker',
          response: 'Routing in parallel',
          toolCalls: [
            {
              id: 'tool_call_stale_context_recipient',
              name: `${Constants.LC_TRANSFER_TO_}recipient`,
              args: { instructions: 'recipient-first-handoff-marker' },
            },
            {
              id: 'tool_call_stale_context_sibling',
              name: `${Constants.LC_TRANSFER_TO_}sibling`,
              args: { instructions: 'sibling-first-handoff-marker' },
            },
          ],
        },
        {
          promptMarker: 'recipient-first-handoff-marker',
          response: 'Sending through relay',
          toolCalls: [
            {
              id: 'tool_call_stale_context_relay',
              name: `${Constants.LC_TRANSFER_TO_}relay`,
              args: { instructions: 'relay-handoff-marker' },
            },
          ],
        },
        {
          promptMarker: 'sibling-first-handoff-marker',
          response: 'Sibling complete',
        },
        {
          promptMarker: 'relay-handoff-marker',
          response: 'relay-direct-reentry-marker',
        },
        {
          promptMarker: 'relay-direct-reentry-marker',
          response: 'Recipient direct complete',
        },
      ]);
      run.Graph.overrideModel = model;

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-clear-handoff-context-direct-reentry-thread',
        },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('stale-context-router-marker')] },
        config
      );

      const recipientContext = run.Graph.agentContexts.get('recipient');
      const recipientSystemRunnable = recipientContext?.systemRunnable;
      if (recipientSystemRunnable == null) {
        throw new Error('Expected recipient system instructions');
      }
      const recipientSystemPrompt = getBufferString(
        await recipientSystemRunnable.invoke([])
      );
      expect(recipientSystemPrompt).toContain('stale-context-recipient-marker');
      expect(recipientSystemPrompt).not.toContain('## Multi-Agent Workflow');
      expect(recipientSystemPrompt).not.toContain('transferred from "Router"');
      expect(recipientSystemPrompt).not.toContain('Running in parallel with:');
    });

    it('should clear the runtime group after a parallel target hands off sequentially', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'router-script-marker'),
        createBasicAgent('left', 'left-script-marker'),
        createBasicAgent('right', 'right-script-marker'),
        createBasicAgent('final', 'final-script-marker'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'left',
          edgeType: 'handoff',
          prompt: 'Instructions for the left specialist',
        },
        {
          from: 'router',
          to: 'right',
          edgeType: 'handoff',
          prompt: 'Instructions for the right specialist',
        },
        {
          from: 'left',
          to: 'final',
          edgeType: 'handoff',
          prompt: 'Instructions for the final specialist',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      run.Graph.overrideModel = new ScriptedHandoffModel([
        {
          promptMarker: 'router-script-marker',
          response: 'Routing',
          toolCalls: [
            {
              id: 'tool_call_nested_left',
              name: `${Constants.LC_TRANSFER_TO_}left`,
              args: { instructions: 'left-script-marker' },
            },
            {
              id: 'tool_call_nested_right',
              name: `${Constants.LC_TRANSFER_TO_}right`,
              args: { instructions: 'right-script-marker' },
            },
          ],
        },
        {
          promptMarker: 'left-script-marker',
          response: 'Forwarding',
          toolCalls: [
            {
              id: 'tool_call_nested_final',
              name: `${Constants.LC_TRANSFER_TO_}final`,
              args: { instructions: 'final-script-marker' },
            },
          ],
        },
        {
          promptMarker: 'right-script-marker',
          response: 'Right complete',
        },
        {
          promptMarker: 'final-script-marker',
          response: 'Final complete',
        },
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-nested-handoff-group-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('router-script-marker')] },
        config
      );

      const leftSteps = run.Graph.getRunSteps('left');
      const rightSteps = run.Graph.getRunSteps('right');
      const finalSteps = run.Graph.getRunSteps('final');
      const parallelGroupId = leftSteps[0]?.groupId;

      expect(parallelGroupId).toEqual(expect.any(Number));
      expect(rightSteps.length).toBeGreaterThan(0);
      expect(rightSteps.every((step) => step.groupId === parallelGroupId)).toBe(
        true
      );
      expect(finalSteps.length).toBeGreaterThan(0);
      expect(finalSteps.every((step) => step.groupId == null)).toBe(true);
    });

    it('should remove every transfer call from mixed conditional recipient history', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'mixed-router-marker'),
        createBasicAgent('left', 'mixed-left-marker'),
        createBasicAgent('right', 'mixed-right-marker'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'left',
          edgeType: 'handoff',
          prompt: 'Instructions for the left specialist',
        },
        {
          from: 'router',
          to: 'right',
          condition: () => 'right',
          prompt: 'Instructions for the right specialist',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      run.Graph.overrideModel = new ScriptedHandoffModel([
        {
          promptMarker: 'mixed-router-marker',
          response: 'Routing',
          toolCalls: [
            {
              id: 'tool_call_mixed_left',
              name: `${Constants.LC_TRANSFER_TO_}left`,
              args: { instructions: 'mixed-left-marker' },
            },
            {
              id: 'tool_call_mixed_right',
              name: 'conditional_transfer',
              args: { instructions: 'mixed-right-marker' },
            },
          ],
        },
        {
          promptMarker: 'mixed-left-marker',
          response: 'Left complete',
        },
        {
          promptMarker: 'mixed-right-marker',
          response: 'Right complete',
        },
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-mixed-parallel-handoff-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await expect(
        run.processStream(
          { messages: [new HumanMessage('mixed-router-marker')] },
          config
        )
      ).resolves.toBeDefined();

      const leftSteps = run.Graph.getRunSteps('left');
      const rightSteps = run.Graph.getRunSteps('right');
      const groupId = leftSteps[0]?.groupId;
      expect(groupId).toEqual(expect.any(Number));
      expect(rightSteps.length).toBeGreaterThan(0);
      expect(rightSteps.every((step) => step.groupId === groupId)).toBe(true);
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

    it('should deliver custom prompt key content to the receiving agent', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          prompt: 'Brief for agent B',
          promptKey: 'brief',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      const brief = 'Investigate the cache invalidation path';
      const rawBrief = `  ${brief}\n`;

      run.Graph?.overrideTestModel(
        ['Transferring', 'Investigation complete'],
        10,
        [
          {
            id: 'tool_call_custom_prompt',
            name: `${Constants.LC_TRANSFER_TO_}agent_b`,
            args: { brief: rawBrief },
          } as ToolCall,
        ]
      );

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-custom-prompt-key-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('Delegate this investigation')] },
        config
      );

      const receivedBrief = run
        .getRunMessages()
        ?.some(
          (message) =>
            message.getType() === 'human' && message.content === brief
        );
      expect(receivedBrief).toBe(true);
      expect(
        run
          .getRunMessages()
          ?.some(
            (message) =>
              message.getType() === 'human' && message.content === rawBrief
          )
      ).toBe(false);
    });

    it('should deliver custom prompt key content through a conditional handoff', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          condition: () => 'agent_b',
          prompt: 'Brief for agent B',
          promptKey: 'brief',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      const brief = 'Review the conditional routing result';

      run.Graph?.overrideTestModel(['Transferring', 'Review complete'], 10, [
        {
          id: 'tool_call_conditional_prompt',
          name: 'conditional_transfer',
          args: { brief },
        } as ToolCall,
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-conditional-custom-prompt-key-thread',
        },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('Delegate conditionally')] },
        config
      );

      const receivedBrief = run
        .getRunMessages()
        ?.some(
          (message) =>
            message.getType() === 'human' && message.content === brief
        );
      expect(receivedBrief).toBe(true);
    });

    it('should recover a complete custom-key payload from a legacy checkpoint', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('agent_a', 'You are agent A'),
        createBasicAgent('agent_b', 'You are agent B'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'agent_a',
          to: 'agent_b',
          edgeType: 'handoff',
          prompt: 'Brief for agent B',
          promptKey: 'brief',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      const graph = run.Graph as unknown as HandoffReceptionProbe;
      const toolCall = {
        id: 'tool_call_legacy_prompt',
        name: `${Constants.LC_TRANSFER_TO_}agent_b`,
        args: {},
      };
      const payload =
        'Inspect the routing decision.\nContext: retain this entire line.';
      const context = graph.processHandoffReception(
        [
          new AIMessage({ content: '', tool_calls: [toolCall] }),
          new ToolMessage({
            content: `Successfully transferred to agent_b\n\nBrief: ${payload}`,
            name: toolCall.name,
            tool_call_id: toolCall.id,
          }),
        ],
        'agent_b'
      );

      expect(context?.instructions).toBe(payload);

      const whitespaceContext = graph.processHandoffReception(
        [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id: 'tool_call_whitespace_prompt',
                name: `${Constants.LC_TRANSFER_TO_}agent_b`,
                args: {},
              },
            ],
          }),
          new ToolMessage({
            content: 'Successfully transferred to agent_b\n\nBrief:   ',
            name: `${Constants.LC_TRANSFER_TO_}agent_b`,
            tool_call_id: 'tool_call_whitespace_prompt',
            additional_kwargs: {
              handoff_instructions: '  \n ',
            },
          }),
        ],
        'agent_b'
      );
      expect(whitespaceContext?.instructions).toBe('');
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

  describe('Tool Call Before Handoff (Issue #54)', () => {
    it('should complete handoff when router calls a non-handoff tool in the same turn', async () => {
      /**
       * Reproduces the bug from issue #54:
       * When a router calls a regular tool AND a handoff tool in the same turn,
       * the filtered messages for the receiving agent end with a ToolMessage.
       * Previously, instructions were appended as a HumanMessage (tool → user),
       * which many APIs reject. The fix injects instructions into the last
       * ToolMessage instead.
       */
      const customTool = new DynamicStructuredTool({
        name: 'list_upload_sessions',
        description: 'List available upload sessions',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> =>
          JSON.stringify({ sessions: [{ id: 'sess_1', status: 'ready' }] }),
      });

      const agents: t.AgentInputs[] = [
        {
          ...createBasicAgent('router', 'You are a router'),
          tools: [customTool],
          toolMap: new Map([['list_upload_sessions', customTool]]) as t.ToolMap,
        },
        createBasicAgent('data_analyst', 'You are a data analyst'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'data_analyst',
          edgeType: 'handoff',
          description: 'Transfer to data analyst',
          prompt: 'Instructions for the analyst about what to analyze',
          promptKey: 'instructions',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));

      /**
       * Simulate router calling list_upload_sessions AND handoff in the same turn.
       * The first model response includes both tool calls.
       * The second model response is the data_analyst's reply.
       */
      run.Graph?.overrideTestModel(
        [
          'Checking available sessions and transferring to analyst',
          'Here is my analysis of the available sessions',
        ],
        10,
        [
          {
            id: 'tool_call_1',
            name: 'list_upload_sessions',
            args: {},
          } as ToolCall,
          {
            id: 'tool_call_2',
            name: `${Constants.LC_TRANSFER_TO_}data_analyst`,
            args: { instructions: 'Analyze the upload session data' },
          } as ToolCall,
        ]
      );

      const messages = [
        new HumanMessage('Check my upload sessions and analyze them'),
      ];

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-tool-before-handoff-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      /**
       * This should complete without error. Before the fix, the receiving
       * agent would get an invalid tool → user message sequence.
       */
      await run.processStream({ messages }, config);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages!.length).toBeGreaterThan(1);

      /** Verify that the handoff occurred */
      const toolMessages = finalMessages!.filter(
        (msg) => msg.getType() === 'tool'
      ) as ToolMessage[];

      const handoffMessage = toolMessages.find(
        (msg) => msg.name === `${Constants.LC_TRANSFER_TO_}data_analyst`
      );
      expect(handoffMessage).toBeDefined();

      /** Verify the flow completed (agent B responded) */
      const aiMessages = finalMessages!.filter((msg) => msg.getType() === 'ai');
      expect(aiMessages.length).toBeGreaterThanOrEqual(1);
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

    it('should return exact-name guidance for handoff names with extra suffixes', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('data_analyst', 'You are a data analyst'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'data_analyst',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create(createTestConfig(agents, edges));
      const correctName = `${Constants.LC_TRANSFER_TO_}data_analyst`;
      const wrongName = `${correctName}_analyst`;

      run.Graph?.overrideTestModel(
        ['Trying to transfer', 'Stopping after invalid tool name'],
        10,
        [
          {
            id: 'tool_call_wrong_handoff',
            name: wrongName,
            args: { instructions: 'Analyze the uploaded data' },
          } as ToolCall,
        ]
      );

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-wrong-handoff-name-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream(
        { messages: [new HumanMessage('Please analyze my data')] },
        config
      );

      const toolMessages = run
        .getRunMessages()!
        .filter((msg) => msg.getType() === 'tool') as ToolMessage[];
      const wrongNameMessage = toolMessages.find(
        (msg) => msg.name === wrongName
      );

      expect(wrongNameMessage).toBeDefined();
      expect(wrongNameMessage?.status).toBe('error');
      expect(wrongNameMessage?.content).toContain(
        `Did you mean "${correctName}"`
      );
    });

    it('should include toolMap handoffs when direct tool names are present', async () => {
      const correctName = `${Constants.LC_TRANSFER_TO_}data_analyst`;
      const wrongName = `${correctName}_analyst`;
      const handoffTool = new DynamicStructuredTool({
        name: correctName,
        description: 'Transfer to data analyst',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> => 'transferred',
      }) as t.GenericTool;
      const node = new ToolNode({
        tools: [handoffTool],
        directToolNames: new Set(['execute_code']),
      });
      const result = (await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id: 'wrong_handoff', name: wrongName, args: {} }],
          }),
        ],
      })) as { messages: ToolMessage[] };
      const wrongNameMessage = result.messages.find(
        (msg) => msg.tool_call_id === 'wrong_handoff'
      );

      expect(wrongNameMessage).toBeDefined();
      expect(wrongNameMessage?.status).toBe('error');
      expect(wrongNameMessage?.content).toContain(
        `Did you mean "${correctName}"`
      );
    });

    it('should keep event-driven unknown handoffs local without direct tool names', async () => {
      const executedToolNames: string[] = [];
      const correctName = `${Constants.LC_TRANSFER_TO_}data_analyst`;
      const wrongName = `${correctName}_analyst`;
      const lookupName = 'lookup_sessions';
      const handoffTool = new DynamicStructuredTool({
        name: correctName,
        description: 'Transfer to data analyst',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> => 'transferred',
      }) as t.GenericTool;
      const lookupTool = new DynamicStructuredTool({
        name: lookupName,
        description: 'List upload sessions',
        schema: { type: 'object', properties: {}, required: [] },
        func: async (): Promise<string> => 'sessions',
      }) as t.GenericTool;
      const dispatchSpy = jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (event, data): Promise<void> => {
          if (event !== GraphEvents.ON_TOOL_EXECUTE) {
            return;
          }
          const batch = data as t.ToolExecuteBatchRequest;
          executedToolNames.push(
            ...batch.toolCalls.map((toolCall) => toolCall.name)
          );
          batch.resolve(
            batch.toolCalls.map((toolCall) => ({
              toolCallId: toolCall.id,
              status: 'success' as const,
              content: `host result for ${toolCall.name}`,
            }))
          );
        });
      const node = new ToolNode({
        tools: [handoffTool, lookupTool],
        eventDrivenMode: true,
        toolCallStepIds: new Map([
          ['lookup_call', 'step_lookup'],
          ['wrong_handoff', 'step_wrong_handoff'],
        ]),
      });

      try {
        const result = (await node.invoke({
          messages: [
            new AIMessage({
              content: '',
              tool_calls: [
                { id: 'lookup_call', name: lookupName, args: {} },
                { id: 'wrong_handoff', name: wrongName, args: {} },
              ],
            }),
          ],
        })) as { messages: ToolMessage[] };
        const wrongNameMessage = result.messages.find(
          (msg) => msg.tool_call_id === 'wrong_handoff'
        );

        expect(executedToolNames).toEqual([lookupName]);
        expect(wrongNameMessage).toBeDefined();
        expect(wrongNameMessage?.status).toBe('error');
        expect(wrongNameMessage?.content).toContain(
          `Did you mean "${correctName}"`
        );
      } finally {
        dispatchSpy.mockRestore();
      }
    });

    it('should not dispatch mistyped graph handoffs to event-driven tool hosts', async () => {
      const executedToolNames: string[] = [];
      const agents: t.AgentInputs[] = [
        {
          ...createBasicAgent('router', 'You are a router'),
          toolDefinitions: [
            {
              name: 'lookup_sessions',
              description: 'List upload sessions',
              parameters: {
                type: 'object',
                properties: {},
                required: [],
              },
            },
          ],
        },
        createBasicAgent('data_analyst', 'You are a data analyst'),
      ];

      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'data_analyst',
          edgeType: 'handoff',
        },
      ];

      const run = await Run.create({
        ...createTestConfig(agents, edges),
        customHandlers: {
          [GraphEvents.ON_TOOL_EXECUTE]: {
            handle: (_event: string, data: t.StreamEventData): void => {
              const batch = data as t.ToolExecuteBatchRequest;
              executedToolNames.push(
                ...batch.toolCalls.map((toolCall) => toolCall.name)
              );
              batch.resolve(
                batch.toolCalls.map((toolCall) => ({
                  toolCallId: toolCall.id,
                  status: 'success' as const,
                  content: `host result for ${toolCall.name}`,
                }))
              );
            },
          },
        },
      });
      const correctName = `${Constants.LC_TRANSFER_TO_}data_analyst`;
      const wrongName = `${correctName}_analyst`;

      run.Graph?.overrideTestModel(
        ['Checking sessions and transferring', 'Handled invalid transfer'],
        10,
        [
          {
            id: 'tool_call_lookup',
            name: 'lookup_sessions',
            args: {},
          } as ToolCall,
          {
            id: 'tool_call_wrong_handoff',
            name: wrongName,
            args: { instructions: 'Analyze the upload session data' },
          } as ToolCall,
        ]
      );

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: {
          thread_id: 'test-event-wrong-handoff-name-thread',
        },
        streamMode: 'values',
        version: 'v2' as const,
      };

      await run.processStream(
        { messages: [new HumanMessage('Check my sessions and analyze them')] },
        config
      );

      const toolMessages = run
        .getRunMessages()!
        .filter((msg) => msg.getType() === 'tool') as ToolMessage[];
      const wrongNameMessage = toolMessages.find(
        (msg) => msg.name === wrongName
      );

      expect(executedToolNames).toEqual(['lookup_sessions']);
      expect(wrongNameMessage).toBeDefined();
      expect(wrongNameMessage?.status).toBe('error');
      expect(wrongNameMessage?.content).toContain(
        `Did you mean "${correctName}"`
      );
    });
  });

  describe('Handoffs with a malformed sibling call (invalid_tool_calls regression)', () => {
    /**
     * Regression for the invalid-call promotion crossing handoff boundaries:
     * a handoff tool snapshots `update.messages` from the PRE-promotion state
     * with a filtered same-id copy of the AI message, and commands apply
     * after sibling reducer updates. Un-patched, the stale copy overwrote the
     * promoted replacement and a parallel Send child's state omitted the
     * synthesized result — the child agent's provider request then carried an
     * invalid call/result pairing. `validateToolHistory` inside the scripted
     * model asserts the pairing on EVERY model invocation, including the
     * children's.
     */
    const MALFORMED_CALL_ID = 'tool_call_malformed_sibling';

    /** Corrupts the malformed sibling's streamed args into a non-object JSON
     *  string so `collapseToolCallChunks` files it under `invalid_tool_calls`
     *  — the shape a malformed provider stream produces. */
    class MalformedSiblingHandoffModel extends ScriptedHandoffModel {
      override async *_streamResponseChunks(
        messages: t.BaseGraphState['messages'],
        options: this['ParsedCallOptions'],
        runManager?: CallbackManagerForLLMRun
      ): AsyncGenerator<ChatGenerationChunk> {
        for await (const chunk of super._streamResponseChunks(
          messages,
          options,
          runManager
        )) {
          const chunkMessage = chunk.message as unknown as {
            tool_call_chunks?: Array<{ id?: string; args?: string }>;
          };
          for (const toolCallChunk of chunkMessage.tool_call_chunks ?? []) {
            if (toolCallChunk.id === MALFORMED_CALL_ID) {
              toolCallChunk.args = '"malformed';
            }
          }
          yield chunk;
        }
      }
    }

    const expectPromotedPairing = (
      finalMessages: t.BaseGraphState['messages']
    ): void => {
      const promoted = finalMessages.find(
        (msg): msg is AIMessage =>
          msg.getType() === 'ai' &&
          ((msg as AIMessage).tool_calls ?? []).some(
            (call) => call.id === MALFORMED_CALL_ID
          )
      );
      expect(promoted).toBeDefined();
      expect(promoted!.invalid_tool_calls ?? []).toHaveLength(0);
      const synthesized = finalMessages.find(
        (msg): msg is ToolMessage =>
          msg.getType() === 'tool' &&
          (msg as ToolMessage).tool_call_id === MALFORMED_CALL_ID
      );
      expect(synthesized).toBeDefined();
      expect(String(synthesized!.content)).toContain('Malformed');
    };

    it('single handoff: the child state keeps the promoted call and its synthesized result', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('specialist', 'You are the specialist'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'specialist',
          edgeType: 'handoff',
          prompt: 'Work the specialist task',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      run.Graph.overrideModel = new MalformedSiblingHandoffModel([
        {
          promptMarker: 'single-malformed-request-marker',
          response: 'Routing with a malformed sibling',
          toolCalls: [
            {
              id: 'tool_call_transfer_specialist',
              name: `${Constants.LC_TRANSFER_TO_}specialist`,
              args: { instructions: 'single-specialist-instructions-marker' },
            } as ToolCall,
            {
              id: MALFORMED_CALL_ID,
              name: 'broken_tool',
              args: {},
            } as ToolCall,
          ],
        },
        {
          promptMarker: 'single-specialist-instructions-marker',
          response: 'Specialist complete',
        },
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-handoff-malformed-single-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('single-malformed-request-marker')] },
        config
      );

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      /** The child agent actually ran — its scripted response is present —
       *  and every model call it made passed validateToolHistory. */
      const specialistReply = finalMessages!.find(
        (msg) =>
          msg.getType() === 'ai' &&
          String(msg.content).includes('Specialist complete')
      );
      expect(specialistReply).toBeDefined();
      expectPromotedPairing(finalMessages!);
    });

    it('parallel Send handoffs: each child state keeps the promoted call and its synthesized result', async () => {
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('left', 'You are the left specialist'),
        createBasicAgent('right', 'You are the right specialist'),
      ];
      const edges: t.GraphEdge[] = [
        {
          from: 'router',
          to: 'left',
          edgeType: 'handoff',
          prompt: 'Work the left task',
        },
        {
          from: 'router',
          to: 'right',
          edgeType: 'handoff',
          prompt: 'Work the right task',
        },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      if (run.Graph == null) {
        throw new Error('Expected a multi-agent graph');
      }
      run.Graph.overrideModel = new MalformedSiblingHandoffModel([
        {
          promptMarker: 'parallel-malformed-request-marker',
          response: 'Routing both with a malformed sibling',
          toolCalls: [
            {
              id: 'tool_call_transfer_left',
              name: `${Constants.LC_TRANSFER_TO_}left`,
              args: { instructions: 'parallel-left-instructions-marker' },
            } as ToolCall,
            {
              id: 'tool_call_transfer_right',
              name: `${Constants.LC_TRANSFER_TO_}right`,
              args: { instructions: 'parallel-right-instructions-marker' },
            } as ToolCall,
            {
              id: MALFORMED_CALL_ID,
              name: 'broken_tool',
              args: {},
            } as ToolCall,
          ],
        },
        {
          promptMarker: 'parallel-left-instructions-marker',
          response: 'Left complete',
        },
        {
          promptMarker: 'parallel-right-instructions-marker',
          response: 'Right complete',
        },
      ]);

      const config: Partial<RunnableConfig> & {
        version: 'v1' | 'v2';
        streamMode: string;
      } = {
        configurable: { thread_id: 'test-handoff-malformed-parallel-thread' },
        streamMode: 'values',
        version: 'v2',
      };
      await run.processStream(
        { messages: [new HumanMessage('parallel-malformed-request-marker')] },
        config
      );

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      /** Both Send children ran off their patched child states, and every
       *  child model call passed validateToolHistory — the un-patched Send
       *  state omitted the synthesized result and carried the stale same-id
       *  copy instead. */
      const leftReply = finalMessages!.find(
        (msg) =>
          msg.getType() === 'ai' &&
          String(msg.content).includes('Left complete')
      );
      const rightReply = finalMessages!.find(
        (msg) =>
          msg.getType() === 'ai' &&
          String(msg.content).includes('Right complete')
      );
      expect(leftReply).toBeDefined();
      expect(rightReply).toBeDefined();
      expectPromotedPairing(finalMessages!);
    });

    it('reception strips transfer tool_use content blocks alongside the calls (array-content providers)', async () => {
      /**
       * The retained filtered AI message used to keep `content` verbatim:
       * with the promoted sibling holding the message in state, an
       * Anthropic child would replay the stripped transfer's `tool_use`
       * block (and a parallel sibling's block, whose result never reaches
       * this recipient) as unmatched calls. Both must be filtered with the
       * tool-call filtering; non-transfer blocks stay.
       */
      const agents: t.AgentInputs[] = [
        createBasicAgent('router', 'You are a router'),
        createBasicAgent('left', 'You are the left specialist'),
        createBasicAgent('right', 'You are the right specialist'),
      ];
      const edges: t.GraphEdge[] = [
        { from: 'router', to: 'left', edgeType: 'handoff' },
        { from: 'router', to: 'right', edgeType: 'handoff' },
      ];
      const run = await Run.create(createTestConfig(agents, edges));
      const graph = run.Graph as unknown as HandoffReceptionProbe;

      const transferLeft = {
        id: 'tc_transfer_left',
        name: `${Constants.LC_TRANSFER_TO_}left`,
        args: {},
      };
      const context = graph.processHandoffReception(
        [
          new AIMessage({
            id: 'ai_reception_blocks',
            content: [
              { type: 'text', text: 'Routing.' },
              {
                type: 'tool_use',
                id: 'tc_transfer_left',
                name: `${Constants.LC_TRANSFER_TO_}left`,
                input: {},
              },
              {
                type: 'tool_use',
                id: 'tc_transfer_right_sibling',
                name: `${Constants.LC_TRANSFER_TO_}right`,
                input: {},
              },
              {
                type: 'tool_use',
                id: 'tc_promoted_sibling',
                name: 'unknown',
                input: {},
              },
            ],
            tool_calls: [
              transferLeft,
              { id: 'tc_promoted_sibling', name: 'unknown', args: {} },
            ],
          }),
          new ToolMessage({
            content: 'Successfully transferred to left',
            name: transferLeft.name,
            tool_call_id: transferLeft.id,
          }),
          new ToolMessage({
            content: 'Error: Malformed args.',
            name: 'unknown',
            tool_call_id: 'tc_promoted_sibling',
          }),
        ],
        'left'
      );

      const retained = context?.filteredMessages?.find(
        (msg): msg is AIMessage => msg.getType() === 'ai'
      );
      expect(retained).toBeDefined();
      expect(retained!.tool_calls?.map((call) => call.id)).toEqual([
        'tc_promoted_sibling',
      ]);
      const blocks = retained!.content as Array<{ type?: string; id?: string }>;
      /** Both this recipient's transfer block AND the parallel sibling's
       *  are gone; the text and the promoted sibling's block remain. */
      expect(blocks.map((block) => block.id ?? block.type)).toEqual([
        'text',
        'tc_promoted_sibling',
      ]);
    });
  });
});
