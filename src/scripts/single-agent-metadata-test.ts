import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { v4 as uuidv4 } from 'uuid';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Single agent test with extensive metadata logging
 * Compare with multi-agent-parallel-start.ts to see metadata differences
 */
async function testSingleAgent() {
  console.log('Testing Single Agent with Metadata Logging...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  const startTime = Date.now();

  // Create custom handlers with extensive metadata logging
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_END METADATA ======');
        console.dir(metadata, { depth: null });
        const elapsed = Date.now() - startTime;
        console.log(`⏱️  COMPLETED at ${elapsed}ms`);
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_START METADATA ======');
        console.dir(metadata, { depth: null });
        const elapsed = Date.now() - startTime;
        console.log(`⏱️  STARTED at ${elapsed}ms`);
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
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
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
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== ON_RUN_STEP_DELTA ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== ON_MESSAGE_DELTA ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create single-agent run configuration (standard graph, not multi-agent)
  const runConfig: t.RunConfig = {
    runId: `single-agent-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a helpful AI assistant. Keep your response concise (50-100 words).`,
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

    console.log('Invoking single-agent graph...\n');

    const config = {
      configurable: {
        thread_id: 'single-agent-conversation-1',
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

    console.log('\n\n========== SUMMARY ==========');
    console.log('Final content parts:', contentParts.length, 'parts');
    console.dir(contentParts, { depth: null });
    console.log('====================================\n');

    await sleep(3000);
  } catch (error) {
    console.error('Error in single-agent test:', error);
  }
}

// Run the test
testSingleAgent();
