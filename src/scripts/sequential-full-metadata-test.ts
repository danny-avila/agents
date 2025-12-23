import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Dump ALL metadata for SEQUENTIAL execution to compare with parallel
 */
async function testSequentialMetadata() {
  console.log('Dumping FULL metadata for SEQUENTIAL execution...\n');

  const { contentParts, aggregateContent } = createContentAggregator();

  const allMetadata: Array<{
    event: string;
    timestamp: number;
    metadata: Record<string, unknown>;
  }> = [];
  const startTime = Date.now();

  // Sequential chain: agent_a -> agent_b
  const agents: t.AgentInputs[] = [
    {
      agentId: 'agent_a',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are Agent A. Just say "Hello from A" in one sentence.`,
    },
    {
      agentId: 'agent_b',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are Agent B. Just say "Hello from B" in one sentence.`,
    },
  ];

  // Sequential edge: A -> B (using edgeType not type)
  const edges: t.GraphEdge[] = [
    { from: 'agent_a', to: 'agent_b', edgeType: 'direct' },
  ];

  const captureMetadata = (
    eventName: string,
    metadata?: Record<string, unknown>
  ) => {
    if (metadata) {
      allMetadata.push({
        event: eventName,
        timestamp: Date.now() - startTime,
        metadata: { ...metadata },
      });
    }
  };

  const customHandlers = {
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('CHAT_MODEL_END', metadata);
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('CHAT_MODEL_START', metadata);
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('ON_RUN_STEP', metadata);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  const runConfig: t.RunConfig = {
    runId: `sequential-metadata-${Date.now()}`,
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

    const userMessage = `Hi`;
    conversationHistory.push(new HumanMessage(userMessage));

    const config = {
      configurable: {
        thread_id: 'sequential-metadata-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    await run.processStream({ messages: conversationHistory }, config);

    // Print ALL CHAT_MODEL_START metadata (don't dedupe)
    console.log(
      '\n\n========== ALL CHAT_MODEL_START EVENTS (SEQUENTIAL) ==========\n'
    );
    for (const entry of allMetadata) {
      if (entry.event === 'CHAT_MODEL_START') {
        const node = entry.metadata.langgraph_node as string;
        console.log(`\n--- ${node} (at ${entry.timestamp}ms) ---`);
        console.dir(entry.metadata, { depth: null });
      }
    }

    console.log('\n\n========== ALL EVENTS ==========\n');
    for (const entry of allMetadata) {
      console.log(
        `[${entry.timestamp}ms] ${entry.event}: ${entry.metadata.langgraph_node}`
      );
    }

    // Key comparison
    console.log(
      '\n\n========== KEY FIELDS COMPARISON (SEQUENTIAL) ==========\n'
    );

    const agentMetadataMap = new Map<string, Record<string, unknown>>();
    for (const entry of allMetadata) {
      if (entry.event === 'CHAT_MODEL_START') {
        const node = entry.metadata.langgraph_node as string;
        if (!agentMetadataMap.has(node)) {
          agentMetadataMap.set(node, entry.metadata);
        }
      }
    }

    for (const [node, meta] of agentMetadataMap) {
      console.log(`${node}:`);
      console.log(`  langgraph_step: ${meta.langgraph_step}`);
      console.log(
        `  langgraph_triggers: ${JSON.stringify(meta.langgraph_triggers)}`
      );
      console.log(`  checkpoint_ns: ${meta.checkpoint_ns}`);
      console.log(`  __pregel_task_id: ${meta.__pregel_task_id}`);
      console.log(`  langgraph_path: ${JSON.stringify(meta.langgraph_path)}`);
      console.log();
    }

    await sleep(1000);
  } catch (error) {
    console.error('Error:', error);
  }
}

testSequentialMetadata();
