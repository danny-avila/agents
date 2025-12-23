import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';
import { Calculator } from '@/tools/Calculator';

const conversationHistory: BaseMessage[] = [];

/**
 * Dump ALL metadata fields to understand what LangSmith uses
 * to detect parallel execution
 */
async function testFullMetadata() {
  console.log('Dumping FULL metadata to find parallel detection fields...\n');

  const { contentParts, aggregateContent } = createContentAggregator();

  // Collect ALL metadata from all events
  const allMetadata: Array<{
    event: string;
    timestamp: number;
    metadata: Record<string, unknown>;
  }> = [];
  const startTime = Date.now();

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
      tools: [new Calculator()],
      instructions: `You are Agent B. Calculate 2+2 using the calculator tool.`,
    },
  ];

  const edges: t.GraphEdge[] = [];

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
    [GraphEvents.TOOL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('TOOL_END', metadata);
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('TOOL_START', metadata);
      },
    },
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
        const runStep = data as t.RunStep;
        console.log(
          `\n🔍 ON_RUN_STEP: agentId=${runStep.agentId}, groupId=${runStep.groupId}`
        );
        aggregateContent({ event, data: runStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('ON_RUN_STEP_DELTA', metadata);
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        captureMetadata('ON_MESSAGE_DELTA', metadata);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  const runConfig: t.RunConfig = {
    runId: `full-metadata-${Date.now()}`,
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

    const userMessage = `Hi, and calculate 2+2`;
    conversationHistory.push(new HumanMessage(userMessage));

    const config = {
      configurable: {
        thread_id: 'full-metadata-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    await run.processStream({ messages: conversationHistory }, config);

    // Analysis - find ALL unique metadata keys
    console.log('\n\n========== ALL UNIQUE METADATA KEYS ==========\n');
    const allKeys = new Set<string>();
    for (const entry of allMetadata) {
      for (const key of Object.keys(entry.metadata)) {
        allKeys.add(key);
      }
    }
    console.log('Keys found:', [...allKeys].sort());

    // Print first CHAT_MODEL_START for each agent with FULL metadata
    console.log(
      '\n\n========== FULL METADATA FROM CHAT_MODEL_START ==========\n'
    );
    const seenAgents = new Set<string>();
    for (const entry of allMetadata) {
      if (entry.event === 'CHAT_MODEL_START') {
        const node = entry.metadata.langgraph_node as string;
        if (!seenAgents.has(node)) {
          seenAgents.add(node);
          console.log(`\n--- ${node} ---`);
          console.dir(entry.metadata, { depth: null });
        }
      }
    }

    // Look specifically at checkpoint_ns and __pregel_task_id
    console.log('\n\n========== POTENTIAL PARALLEL INDICATORS ==========\n');
    console.log(
      'Comparing checkpoint_ns and __pregel_task_id across agents:\n'
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
      console.log(`  langgraph_checkpoint_ns: ${meta.langgraph_checkpoint_ns}`);
      console.log();
    }

    // Check langgraph_triggers specifically
    console.log('\n========== LANGGRAPH_TRIGGERS ANALYSIS ==========\n');
    for (const [node, meta] of agentMetadataMap) {
      const triggers = meta.langgraph_triggers as string[];
      console.log(`${node}: triggers = ${JSON.stringify(triggers)}`);
    }

    console.log('\n\nFinal content parts:');
    console.dir(contentParts, { depth: null });

    await sleep(1000);
  } catch (error) {
    console.error('Error:', error);
  }
}

testFullMetadata();
