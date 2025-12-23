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
import { Tool } from '@langchain/core/tools';

const conversationHistory: BaseMessage[] = [];

// Create a simple "WordCount" tool for the second agent
class WordCounter extends Tool {
  static lc_name(): string {
    return 'WordCounter';
  }

  name = 'word_counter';

  description =
    'Useful for counting the number of words, characters, and sentences in a given text. Input should be the text to analyze.';

  async _call(input: string): Promise<string> {
    const words = input.trim().split(/\s+/).filter(Boolean).length;
    const characters = input.length;
    const sentences = input.split(/[.!?]+/).filter(Boolean).length;
    return JSON.stringify({ words, characters, sentences });
  }
}

/**
 * Example of parallel multi-agent system with tools
 *
 * Graph structure:
 * START -> [math_agent, text_agent] -> END (parallel from start, both run simultaneously)
 *
 * Both agents have tools they can use. This tests how langgraph_step behaves
 * when parallel agents call tools.
 */
async function testParallelWithTools() {
  console.log('Testing Parallel From Start Multi-Agent System WITH TOOLS...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Track metadata for analysis
  const metadataLog: Array<{
    event: string;
    langgraph_step: number;
    langgraph_node: string;
    timestamp: number;
  }> = [];
  const startTime = Date.now();

  // Define two agents with different tools
  const agents: t.AgentInputs[] = [
    {
      agentId: 'math_agent',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      tools: [new Calculator()],
      instructions: `You are a MATH SPECIALIST. When asked about numbers or calculations, ALWAYS use the calculator tool to perform the calculation. Start your response with "🧮 MATH:". Keep your response concise.`,
    },
    {
      agentId: 'text_agent',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      tools: [new WordCounter()],
      instructions: `You are a TEXT ANALYST. When asked about text or content, ALWAYS use the word_counter tool to analyze the text. Start your response with "📝 TEXT:". Keep your response concise.`,
    },
  ];

  // No edges - both agents run in parallel from start
  const edges: t.GraphEdge[] = [];

  // Track active agents and timing
  const activeAgents = new Set<string>();
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

  // Create custom handlers with metadata logging
  const customHandlers = {
    [GraphEvents.TOOL_END]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== TOOL_END ======');
        logMetadata('TOOL_END', metadata);
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
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
          activeAgents.add(nodeName);
        }
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== ON_RUN_STEP_COMPLETED ======');
        logMetadata('ON_RUN_STEP_COMPLETED', metadata);
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
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
        logMetadata('ON_RUN_STEP_DELTA', metadata);
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        logMetadata('ON_MESSAGE_DELTA', metadata);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `parallel-tools-${Date.now()}`,
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

    // User message that should trigger both agents to use their tools
    const userMessage = `I have two tasks:
1. Calculate 1234 + 5678 * 2
2. Analyze this text: "The quick brown fox jumps over the lazy dog"

Please help with both!`;

    conversationHistory.push(new HumanMessage(userMessage));

    console.log(
      'Invoking parallel-from-start multi-agent graph WITH TOOLS...\n'
    );
    console.log(
      'Both math_agent and text_agent should start simultaneously and use tools!\n'
    );

    const config = {
      configurable: {
        thread_id: 'parallel-tools-test-1',
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

    // Analysis output
    console.log('\n\n========== METADATA ANALYSIS ==========');
    console.log('\nAll metadata entries by timestamp:');
    console.table(metadataLog);

    // Group by langgraph_step
    const stepGroups = new Map<number, typeof metadataLog>();
    for (const entry of metadataLog) {
      if (!stepGroups.has(entry.langgraph_step)) {
        stepGroups.set(entry.langgraph_step, []);
      }
      stepGroups.get(entry.langgraph_step)!.push(entry);
    }

    console.log('\n\nGrouped by langgraph_step:');
    for (const [step, entries] of stepGroups) {
      const nodes = [...new Set(entries.map((e) => e.langgraph_node))];
      console.log(
        `\n  Step ${step}: ${nodes.length} unique nodes: ${nodes.join(', ')}`
      );
      console.log(`    Events: ${entries.map((e) => e.event).join(', ')}`);
    }

    // Identify parallel groups (same step, different nodes)
    console.log('\n\n========== PARALLEL DETECTION ANALYSIS ==========');
    for (const [step, entries] of stepGroups) {
      const uniqueNodes = [...new Set(entries.map((e) => e.langgraph_node))];
      if (uniqueNodes.length > 1) {
        console.log(
          `✅ Step ${step} has MULTIPLE agents: ${uniqueNodes.join(', ')} - PARALLEL!`
        );
      } else {
        console.log(`   Step ${step} has single agent: ${uniqueNodes[0]}`);
      }
    }

    console.log('\n\n========== TIMING SUMMARY ==========');
    for (const [agent, timing] of Object.entries(agentTimings)) {
      const duration =
        timing.end && timing.start ? timing.end - timing.start : 'N/A';
      console.log(
        `${agent}: started=${timing.start}ms, ended=${timing.end}ms, duration=${duration}ms`
      );
    }

    // Check overlap for parallel confirmation
    const agentList = Object.keys(agentTimings);
    if (agentList.length >= 2) {
      const [a1, a2] = agentList;
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
    await sleep(2000);
  } catch (error) {
    console.error('Error in parallel-with-tools test:', error);
  }
}

// Run the test
testParallelWithTools();
