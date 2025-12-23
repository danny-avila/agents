import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';
import { Calculator } from '@/tools/Calculator';

const conversationHistory: BaseMessage[] = [];

/**
 * Test ASYMMETRIC parallel execution:
 * - agent1: NO tools (will finish quickly in step 1)
 * - agent2: HAS tools (will go step 1 → step 2 → step 3)
 *
 * This tests whether langgraph_step can still detect parallel execution
 * when agents have different tool-calling patterns.
 */
async function testAsymmetricParallelTools() {
  console.log(
    'Testing ASYMMETRIC Parallel Agents (one with tools, one without)...\n'
  );

  const { contentParts, aggregateContent } = createContentAggregator();

  // Track metadata for analysis
  const metadataLog: Array<{
    event: string;
    langgraph_step: number;
    langgraph_node: string;
    timestamp: number;
  }> = [];
  const startTime = Date.now();

  // Define two agents - one WITH tools, one WITHOUT
  const agents: t.AgentInputs[] = [
    {
      agentId: 'simple_agent',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      // NO TOOLS - will complete in single step
      instructions: `You are a simple assistant. Just answer the question directly in 1-2 sentences. Start with "🗣️ SIMPLE:". Do NOT try to use any tools.`,
    },
    {
      agentId: 'math_agent',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      tools: [new Calculator()],
      instructions: `You are a MATH SPECIALIST. ALWAYS use the calculator tool to perform calculations, even simple ones. Start your response with "🧮 MATH:". Keep your response concise.`,
    },
  ];

  // No edges - both run in parallel from start
  const edges: t.GraphEdge[] = [];

  const agentTimings: Record<string, { start?: number; end?: number }> = {};

  // Helper to log metadata
  const logMetadata = (
    eventName: string,
    metadata?: Record<string, unknown>
  ) => {
    if (metadata) {
      const entry = {
        event: eventName,
        langgraph_step: metadata.langgraph_step as number,
        langgraph_node: metadata.langgraph_node as string,
        timestamp: Date.now() - startTime,
      };
      metadataLog.push(entry);
      console.log(
        `📊 [${entry.timestamp}ms] ${eventName}: step=${entry.langgraph_step}, node=${entry.langgraph_node}`
      );
    }
  };

  const customHandlers = {
    [GraphEvents.TOOL_END]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== TOOL_END ======');
        logMetadata('TOOL_END', metadata);
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== TOOL_START ======');
        logMetadata('TOOL_START', metadata);
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_END ======');
        logMetadata('CHAT_MODEL_END', metadata);
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          agentTimings[nodeName] = agentTimings[nodeName] || {};
          agentTimings[nodeName].end = elapsed;
        }
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_START ======');
        logMetadata('CHAT_MODEL_START', metadata);
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          agentTimings[nodeName] = agentTimings[nodeName] || {};
          if (!agentTimings[nodeName].start) {
            agentTimings[nodeName].start = elapsed;
          }
        }
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== ON_RUN_STEP ======');
        logMetadata('ON_RUN_STEP', metadata);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        // Don't log these to reduce noise
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        // Don't log these to reduce noise
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  const runConfig: t.RunConfig = {
    runId: `asymmetric-parallel-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  try {
    const run = await Run.create(runConfig);

    // Ask a question that will trigger only the math agent to use tools
    const userMessage = `What is 42 * 17?`;

    conversationHistory.push(new HumanMessage(userMessage));

    console.log('User message:', userMessage);
    console.log('\nExpected behavior:');
    console.log('  - simple_agent: Step 1 only (no tools)');
    console.log('  - math_agent: Step 1 → Step 2 (tool) → Step 3 (response)');
    console.log('\n');

    const config = {
      configurable: {
        thread_id: 'asymmetric-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);

    // Analysis
    console.log('\n\n========== METADATA ANALYSIS ==========');
    console.log('\nAll events by step and node:');
    console.table(metadataLog);

    // Group by step
    const stepGroups = new Map<number, Map<string, string[]>>();
    for (const entry of metadataLog) {
      if (!stepGroups.has(entry.langgraph_step)) {
        stepGroups.set(entry.langgraph_step, new Map());
      }
      const nodeMap = stepGroups.get(entry.langgraph_step)!;
      if (!nodeMap.has(entry.langgraph_node)) {
        nodeMap.set(entry.langgraph_node, []);
      }
      nodeMap.get(entry.langgraph_node)!.push(entry.event);
    }

    console.log('\n\n========== STEP BREAKDOWN ==========');
    for (const [step, nodeMap] of stepGroups) {
      console.log(`\nStep ${step}:`);
      for (const [node, events] of nodeMap) {
        console.log(`  ${node}: ${events.join(', ')}`);
      }
      console.log(`  → ${nodeMap.size} unique node(s) at this step`);
    }

    console.log('\n\n========== PARALLEL DETECTION CHALLENGE ==========');
    console.log('\nAt which steps can we detect parallel execution?');
    for (const [step, nodeMap] of stepGroups) {
      if (nodeMap.size > 1) {
        console.log(
          `  ✅ Step ${step}: ${nodeMap.size} agents detected - PARALLEL`
        );
      } else {
        const [nodeName] = nodeMap.keys();
        console.log(
          `  ⚠️  Step ${step}: Only 1 agent (${nodeName}) - looks sequential!`
        );
      }
    }

    console.log('\n\n========== KEY INSIGHT ==========');
    console.log(
      'If we only look at step 2 or 3, we miss the parallel context!'
    );
    console.log(
      'We need to detect parallelism EARLY (at step 1) and carry that forward.'
    );

    console.log('\n\nFinal content parts:');
    console.dir(contentParts, { depth: null });

    await sleep(2000);
  } catch (error) {
    console.error('Error:', error);
  }
}

testAsymmetricParallelTools();
