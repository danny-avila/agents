import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Example of parallel multi-agent system that starts with parallel execution immediately
 *
 * Graph structure:
 * START -> [analyst1, analyst2] -> END (parallel from start, both run simultaneously)
 *
 * This demonstrates getting a parallel stream from the very beginning,
 * with two agents running simultaneously. Useful for testing how different
 * models respond to the same input.
 */
async function testParallelFromStart() {
  console.log('Testing Parallel From Start Multi-Agent System...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define two agents - both have NO incoming edges, so they run in parallel from the start
  const agents: t.AgentInputs[] = [
    {
      agentId: 'analyst1',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a CREATIVE ANALYST. Analyze the user's query from a creative and innovative perspective. Focus on novel ideas, unconventional approaches, and imaginative possibilities. Keep your response concise (100-150 words). Start with "🎨 CREATIVE:"`,
    },
    {
      agentId: 'analyst2',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a PRACTICAL ANALYST. Analyze the user's query from a logical and practical perspective. Focus on feasibility, metrics, and actionable steps. Keep your response concise (100-150 words). Start with "📊 PRACTICAL:"`,
    },
  ];

  // No edges needed - both agents have no incoming edges, so both are start nodes
  // They will run in parallel and end when both complete
  const edges: t.GraphEdge[] = [];

  // Track which agents are active and their timing
  const activeAgents = new Set<string>();
  const agentTimings: Record<string, { start?: number; end?: number }> = {};
  const startTime = Date.now();

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          agentTimings[nodeName] = agentTimings[nodeName] || {};
          agentTimings[nodeName].end = elapsed;
          console.log(`⏱️  [${nodeName}] COMPLETED at ${elapsed}ms`);
        }
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          agentTimings[nodeName] = agentTimings[nodeName] || {};
          agentTimings[nodeName].start = elapsed;
          activeAgents.add(nodeName);
          console.log(`⏱️  [${nodeName}] STARTED at ${elapsed}ms`);
        }
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
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
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `parallel-start-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
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

    const userMessage = `What are the best approaches to learning a new programming language?`;
    conversationHistory.push(new HumanMessage(userMessage));

    console.log('Invoking parallel-from-start multi-agent graph...\n');
    console.log('Both analyst1 and analyst2 should start simultaneously!\n');

    const config = {
      configurable: {
        thread_id: 'parallel-start-conversation-1',
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

    console.log('\n\n========== TIMING SUMMARY ==========');
    for (const [agent, timing] of Object.entries(agentTimings)) {
      const duration =
        timing.end && timing.start ? timing.end - timing.start : 'N/A';
      console.log(
        `${agent}: started=${timing.start}ms, ended=${timing.end}ms, duration=${duration}ms`
      );
    }

    // Check if parallel
    const agents = Object.keys(agentTimings);
    if (agents.length >= 2) {
      const [a1, a2] = agents;
      const t1 = agentTimings[a1];
      const t2 = agentTimings[a2];
      if (t1.start && t2.start && t1.end && t2.end) {
        const overlap = Math.min(t1.end, t2.end) - Math.max(t1.start, t2.start);
        if (overlap > 0) {
          console.log(
            `\n✅ PARALLEL EXECUTION CONFIRMED: ${overlap}ms overlap`
          );
        } else {
          console.log(`\n❌ SEQUENTIAL EXECUTION: no overlap`);
        }
      }
    }
    console.log('====================================\n');

    console.log('Final content parts:', contentParts.length, 'parts');
    console.dir(contentParts, { depth: null });
    await sleep(3000);
  } catch (error) {
    console.error('Error in parallel-from-start multi-agent test:', error);
  }
}

// Run the test
testParallelFromStart();
