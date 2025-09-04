import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

/**
 * Test the new handoff input feature using the prompt field
 * This demonstrates how supervisors can pass specific instructions to specialists
 */
async function testHandoffInput() {
  console.log('Testing Handoff Input Feature...\n');
  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Track which specialist role was selected
  let selectedRole = '';
  let roleInstructions = '';

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
          console.log(`\n[${runStepData.name}] Processing...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
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
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const toolData = data as any;
        if (toolData?.name?.includes('transfer_to_')) {
          const specialist = toolData.name.replace('transfer_to_', '');
          console.log(`\n🔀 Transferring to ${specialist}...`);
          selectedRole = specialist;
        }
      },
    },
  };

  const runConfig: t.RunConfig = {
    customHandlers,
    runId: `test-handoff-input-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents: [
        {
          agentId: 'supervisor',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-5-sonnet-latest',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Task Supervisor. You have access to two specialist agents:
          
          1. transfer_to_analyst - For data analysis tasks
          2. transfer_to_writer - For content creation tasks
          
          When transferring to a specialist, you MUST provide specific instructions
          in the tool call to guide their work. Be detailed about what you need.`,
          maxContextTokens: 8000,
        },
        {
          agentId: 'analyst',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-5-sonnet-latest',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Data Analyst. Follow the supervisor's instructions carefully.
          When you receive instructions, acknowledge them and perform the requested analysis.`,
          maxContextTokens: 8000,
        },
        {
          agentId: 'writer',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-3-5-sonnet-latest',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Content Writer. Follow the supervisor's instructions carefully.
          When you receive instructions, acknowledge them and create the requested content.`,
          maxContextTokens: 8000,
        },
      ],
      edges: [
        {
          from: 'supervisor',
          to: ['analyst', 'writer'],
          edgeType: 'handoff',
          // This prompt field now serves as the description for the input parameter
          prompt:
            'Specific instructions for the specialist to follow. Be detailed about what analysis to perform, what data to focus on, or what content to create.',
        },
      ],
    },
  };

  const run = await Run.create(runConfig);

  // Test queries that should result in different handoffs with specific instructions
  const testQueries = [
    // 'Analyze our Q4 sales data and identify the top 3 performing products',
    'Write a blog post about the benefits of remote work for software developers',
  ];

  const config = {
    configurable: {
      thread_id: 'handoff-input-test-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  for (const query of testQueries) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(60));

    const inputs = {
      messages: [new HumanMessage(query)],
    };

    const finalContentParts = await run.processStream(inputs, config);

    console.log(`\n${'─'.repeat(60)}`);
    console.log('Notice how the supervisor passes specific instructions');
    console.log('to the specialist through the handoff tool input parameter.');
    console.log('─'.repeat(60));
  }
}

// Run the test
testHandoffInput().catch(console.error);
