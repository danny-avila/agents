import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

async function testMultiAgentHandoff() {
  console.log('Testing Multi-Agent Handoff System...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define agent configurations
  const agents: t.AgentInputs[] = [
    {
      agentId: 'flight_assistant',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a flight booking assistant. Help users book flights between airports.',
      maxContextTokens: 28000,
    },
    {
      agentId: 'hotel_assistant',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a hotel booking assistant. Help users book hotel stays.',
      maxContextTokens: 28000,
    },
  ];

  // Define edges (handoff relationships)
  // These edges create handoff tools that agents can use to transfer control dynamically
  const edges: t.GraphEdge[] = [
    {
      from: 'flight_assistant',
      to: 'hotel_assistant',
      description:
        'Transfer to hotel booking assistant when user needs hotel assistance',
      // edgeType defaults to 'handoff' for single-to-single edges
    },
    {
      from: 'hotel_assistant',
      to: 'flight_assistant',
      description:
        'Transfer to flight booking assistant when user needs flight assistance',
      // edgeType defaults to 'handoff' for single-to-single edges
    },
  ];

  // Create custom handlers similar to examples
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
        console.dir(data, { depth: null });
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
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_REASONING_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `multi-agent-test-${Date.now()}`,
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

    const userMessage =
      'I need to book a flight from Boston to New York, and also need a hotel near Times Square.';
    conversationHistory.push(new HumanMessage(userMessage));

    console.log('Invoking multi-agent graph...\n');

    const config = {
      configurable: {
        thread_id: 'multi-agent-conversation-1',
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
      console.log('\n\nConversation history:');
      console.dir(conversationHistory, { depth: null });
    }

    console.log('\n\nFinal content parts:');
    console.dir(contentParts, { depth: null });
  } catch (error) {
    console.error('Error in multi-agent test:', error);
  }
}

// Run the test
testMultiAgentHandoff();
