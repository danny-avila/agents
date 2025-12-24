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
 * Example of simple sequential multi-agent system
 *
 * Graph structure:
 * START -> agent_a -> agent_b -> agent_c -> END
 *
 * No conditions, no tools, just automatic sequential flow
 */
async function testSequentialMultiAgent() {
  console.log('Testing Sequential Multi-Agent System (A → B → C)...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define three simple agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'agent_a',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are Agent A, the first agent in a sequential workflow.
      Your job is to:
      1. Receive the initial user request
      2. Process it and add your perspective (keep it brief, 2-3 sentences)
      3. Pass it along to Agent B
      
      Start your response with "AGENT A:" and end with "Passing to Agent B..."`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'agent_b',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are Agent B, the second agent in a sequential workflow.
      Your job is to:
      1. Receive the context from Agent A
      2. Add your own analysis or perspective (keep it brief, 2-3 sentences)
      3. Pass it along to Agent C
      
      Start your response with "AGENT B:" and end with "Passing to Agent C..."`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'agent_c',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are Agent C, the final agent in a sequential workflow.
      Your job is to:
      1. Receive the context from Agents A and B
      2. Provide a final summary or conclusion based on all previous inputs
      3. Complete the workflow
      
      Start your response with "AGENT C:" and end with "Workflow complete."`,
      maxContextTokens: 8000,
    },
  ];

  // Define sequential edges using 'direct' type to avoid tool creation
  // This creates automatic transitions without requiring agents to call tools
  const edges: t.GraphEdge[] = [
    {
      from: 'agent_a',
      to: 'agent_b',
      edgeType: 'direct', // This creates direct edges without tools
      description: 'Automatic transition from A to B',
    },
    {
      from: 'agent_b',
      to: 'agent_c',
      edgeType: 'direct', // This creates direct edges without tools
      description: 'Automatic transition from B to C',
    },
  ];

  // Track agent progression
  let currentAgent = '';

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        const runStepData = data as any;
        if (runStepData?.name) {
          currentAgent = runStepData.name;
          console.log(`\n→ ${currentAgent} is processing...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        const runStepData = data as any;
        if (runStepData?.name) {
          console.log(`✓ ${runStepData.name} completed`);
        }
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
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
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `sequential-multi-agent-${Date.now()}`,
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

    // Test with a simple question
    const userMessage =
      'What are the key considerations for building a recommendation system?';
    conversationHistory.push(new HumanMessage(userMessage));

    console.log(`User: "${userMessage}"\n`);
    console.log('Starting sequential workflow...\n');

    const config = {
      configurable: {
        thread_id: 'sequential-conversation-1',
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

    console.log('\n\n=== Final Output ===');
    console.log('Sequential flow completed successfully!');
    console.log(`Total content parts: ${contentParts.length}`);
    console.log('\n=== Content Parts with Metadata ===');
    console.dir(contentParts, { depth: null });

    // Display the sequential responses
    const aiMessages = conversationHistory.filter(
      (msg) => msg._getType() === 'ai'
    );
    aiMessages.forEach((msg, index) => {
      console.log(`\n--- Response ${index + 1} ---`);
      console.log(msg.content);
    });
  } catch (error) {
    console.error('Error in sequential multi-agent test:', error);
  }
}

// Run the test
testSequentialMultiAgent();
