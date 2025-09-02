import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

/**
 * Example of parallel multi-agent system with fan-in/fan-out pattern
 *
 * Graph structure:
 * START -> researcher
 * researcher -> [analyst1, analyst2, analyst3] (fan-out)
 * [analyst1, analyst2, analyst3] -> summarizer (fan-in)
 * summarizer -> END
 */
async function testParallelMultiAgent() {
  console.log('Testing Parallel Multi-Agent System (Fan-in/Fan-out)...\n');

  // Note: You may see "Run ID not found in run map" errors during parallel execution.
  // This is a known issue with LangGraph's event streaming when nodes run in parallel.
  // The errors can be safely ignored - the parallel execution still works correctly.

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define specialized agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'researcher',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a research assistant. Break down complex topics into key areas for analysis.',
      maxContextTokens: 28000,
    },
    {
      agentId: 'analyst1',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a financial analyst. Focus on economic and financial aspects.',
      maxContextTokens: 28000,
    },
    {
      agentId: 'analyst2',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a technical analyst. Focus on technological and implementation aspects.',
      maxContextTokens: 28000,
    },
    {
      agentId: 'analyst3',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a market analyst. Focus on market trends and competitive landscape.',
      maxContextTokens: 28000,
    },
    {
      agentId: 'summarizer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a summary expert. Synthesize insights from multiple analysts into a coherent report.',
      maxContextTokens: 28000,
    },
  ];

  // Define parallel edges (fan-out and fan-in)
  const edges: t.GraphEdge[] = [
    {
      from: 'researcher',
      to: ['analyst1', 'analyst2', 'analyst3'], // Fan-out to multiple analysts
      description: 'Distribute research to specialist analysts',
      edgeType: 'parallel', // Explicitly set as parallel for simultaneous execution
    },
    {
      from: ['analyst1', 'analyst2', 'analyst3'], // Fan-in from multiple sources
      to: 'summarizer',
      description: 'Aggregate analysis results',
      edgeType: 'parallel', // Fan-in is also parallel
    },
  ];

  // Track which agents are active
  const activeAgents = new Set<string>();

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        const runStepData = data as any;
        if (runStepData?.name) {
          activeAgents.delete(runStepData.name);
          console.log(`[${runStepData.name}] Completed analysis`);
        }
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP ======');
        const runStepData = data as any;
        if (runStepData?.name) {
          activeAgents.add(runStepData.name);
          console.log(`[${runStepData.name}] Starting analysis...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        // console.log('====== ON_MESSAGE_DELTA ======');
        // console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `parallel-multi-agent-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
      // Add compile options to help with parallel execution
      compileOptions: {
        // checkpointer: new MemorySaver(), // Uncomment if needed
      },
    },
    customHandlers,
    returnContent: true,
  };

  try {
    // Create and execute the run
    const run = await Run.create(runConfig);

    // Debug: Log the graph structure
    console.log('=== DEBUG: Graph Structure ===');
    const graph = (run as any).Graph;
    console.log('Graph exists:', !!graph);
    if (graph) {
      console.log('Graph type:', graph.constructor.name);
      console.log('AgentContexts exists:', !!graph.agentContexts);
      if (graph.agentContexts) {
        console.log('AgentContexts size:', graph.agentContexts.size);
        for (const [agentId, context] of graph.agentContexts) {
          console.log(`\nAgent: ${agentId}`);
          console.log(
            `Tools: ${context.tools?.map((t: any) => t.name || 'unnamed').join(', ') || 'none'}`
          );
        }
      }
    }
    console.log('=== END DEBUG ===\n');

    const userMessage = `I need a comprehensive analysis of AI's impact on the global economy over the next decade. 

CRITICAL: This analysis must be conducted by ALL THREE specialist teams working in parallel:
1. Financial analyst team - focus on monetary policy, investment flows, and economic metrics
2. Technical analyst team - focus on AI implementation, infrastructure, and technological barriers  
3. Market analyst team - focus on industry disruption, competitive dynamics, and consumer adoption

Each team should provide independent analysis that will be synthesized into a final report. Do NOT provide a single unified response - delegate this work to the specialist analyst teams for parallel processing.`;
    conversationHistory.push(new HumanMessage(userMessage));

    console.log('Invoking parallel multi-agent graph...\n');

    const config = {
      configurable: {
        thread_id: 'parallel-conversation-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    // Process with streaming
    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    console.log('\n\nActive agents during execution:', activeAgents.size);
    console.log('Final content parts:', contentParts.length, 'parts');
    console.dir(contentParts, { depth: null });
  } catch (error) {
    console.error('Error in parallel multi-agent test:', error);
  }
}

// Run the test
testParallelMultiAgent();
