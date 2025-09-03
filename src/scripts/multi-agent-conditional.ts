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
 * Example of conditional multi-agent system
 *
 * Graph structure:
 * START -> classifier
 * classifier -> technical_expert (if technical question)
 * classifier -> business_expert (if business question)
 * classifier -> general_assistant (otherwise)
 * [all experts] -> END
 */
async function testConditionalMultiAgent() {
  console.log('Testing Conditional Multi-Agent System...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define specialized agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'classifier',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a query classifier. Analyze user questions and determine if they are technical, business-related, or general.',
      maxContextTokens: 8000,
    },
    {
      agentId: 'technical_expert',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a technical expert. Provide detailed technical answers about programming, systems, and technology.',
      maxContextTokens: 8000,
    },
    {
      agentId: 'business_expert',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a business expert. Provide insights on business strategy, operations, and management.',
      maxContextTokens: 8000,
    },
    {
      agentId: 'general_assistant',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions:
        'You are a helpful general assistant. Answer questions on a wide range of topics.',
      maxContextTokens: 8000,
    },
  ];

  // Define conditional edges
  // These create handoff tools with conditional routing logic
  const edges: t.GraphEdge[] = [
    {
      from: 'classifier',
      to: ['technical_expert', 'business_expert', 'general_assistant'],
      description: 'Route to appropriate expert based on query type',
      condition: (state: t.BaseGraphState) => {
        // Simple keyword-based routing for demo
        // In a real system, this would use the classifier's analysis
        const lastMessage = state.messages[state.messages.length - 1];
        const content = lastMessage.content?.toString().toLowerCase() || '';

        if (
          content.includes('code') ||
          content.includes('programming') ||
          content.includes('technical')
        ) {
          return 'technical_expert';
        } else if (
          content.includes('business') ||
          content.includes('strategy') ||
          content.includes('market')
        ) {
          return 'business_expert';
        } else {
          return 'general_assistant';
        }
      },
    },
  ];

  // Track selected expert
  let selectedExpert = '';

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
        const runStepData = data as any;
        if (runStepData?.name && runStepData.name !== 'classifier') {
          selectedExpert = runStepData.name;
          console.log(`Routing to: ${selectedExpert}`);
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
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `conditional-multi-agent-${Date.now()}`,
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

    // Test with different types of questions
    const testQuestions = [
      'How do I implement a binary search tree in Python?',
      'What are the key strategies for market expansion?',
      'What is the capital of France?',
    ];

    const config = {
      configurable: {
        thread_id: 'conditional-conversation-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    for (const question of testQuestions) {
      console.log(`\n--- Processing: "${question}" ---\n`);

      // Reset for each question
      selectedExpert = '';
      conversationHistory.length = 0;
      conversationHistory.push(new HumanMessage(question));

      // Process with streaming
      const inputs = {
        messages: conversationHistory,
      };

      const finalContentParts = await run.processStream(inputs, config);
      const finalMessages = run.getRunMessages();

      if (finalMessages) {
        conversationHistory.push(...finalMessages);
      }

      console.log(`\n\nExpert used: ${selectedExpert}`);
      console.log('Content parts:', contentParts.length);
      console.log('---');
    }
  } catch (error) {
    console.error('Error in conditional multi-agent test:', error);
  }
}

// Run the test
testConditionalMultiAgent();
