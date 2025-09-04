import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents, Constants } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

/**
 * Example of hybrid multi-agent system combining handoff and sequential patterns
 *
 * Graph structure:
 * START -> primary_agent -> agent_b -> agent_c -> END
 *                 |
 *                 └─> standalone_agent -> END
 *
 * Because primary_agent has BOTH handoff and direct edges:
 * - Uses Command-based routing for exclusive execution
 * - The primary agent can either:
 *   1. Handoff to standalone_agent (direct edge to agent_b is cancelled)
 *   2. OR continue to agent_b -> agent_c (if no handoff occurs)
 *
 * This is automatic behavior when an agent has both edge types.
 */
async function testHybridMultiAgent() {
  console.log('Testing Hybrid Multi-Agent System (Sequential + Handoff)...\n');

  // Define agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'primary_agent',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are the Primary Agent in a hybrid workflow.
      
      You have TWO options:
      1. If the request requires specialized expertise (complex analysis, deep technical knowledge, etc.), 
         use the "transfer_to_standalone_agent" tool to hand off to the Standalone Specialist
      2. If the request is straightforward and can be handled through standard processing,
         just provide your initial response and it will automatically continue to Agent B
      
      Be decisive - either handoff immediately or provide your response.
      Start your response with "PRIMARY AGENT:" if you're handling it yourself.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'standalone_agent',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Standalone Specialist Agent.
      You only receive requests that require specialized expertise.
      
      Provide a comprehensive, expert-level response to the request.
      Start your response with "STANDALONE SPECIALIST:"
      End with "Specialized analysis complete."`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'agent_b',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are Agent B in a sequential workflow.
      You receive requests that the Primary Agent decided to handle through standard processing.
      
      Your job is to:
      1. Build upon the Primary Agent's initial response
      2. Add additional processing or analysis (keep it brief, 2-3 sentences)
      3. Prepare the information for final processing by Agent C
      
      Start your response with "AGENT B:" and end with "Passing to final processing..."`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'agent_c',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are Agent C, the final agent in the sequential workflow.
      
      Your job is to:
      1. Review all previous processing from Primary Agent and Agent B
      2. Provide a final summary or conclusion
      3. Complete the standard workflow
      
      Start your response with "AGENT C:" and end with "Standard workflow complete."`,
      maxContextTokens: 8000,
    },
  ];

  // Define edges combining handoff and direct patterns
  const edges: t.GraphEdge[] = [
    // Handoff edge: primary can transfer to standalone
    {
      from: 'primary_agent',
      to: 'standalone_agent',
      edgeType: 'handoff',
      description: 'Transfer to standalone specialist for complex requests',
      prompt: 'Specific instructions for the specialist',
    },
    // Direct edge - exclusive with handoffs (automatic when agent has both types)
    {
      from: 'primary_agent',
      to: 'agent_b',
      edgeType: 'direct',
      description: 'Continue to Agent B only if no handoff occurs',
    },
    // Direct edge: agent_b automatically continues to agent_c
    {
      from: 'agent_b',
      to: 'agent_c',
      edgeType: 'direct',
      description: 'Automatic progression from B to C',
    },
  ];

  try {
    // Test with different queries
    const testQueries = [
      {
        query: 'What is the capital of France?',
        expectedPath: 'sequential',
        description: 'Simple query - should go through sequential flow',
      },
      // {
      //   query: 'Design a distributed microservices architecture for a real-time trading platform with sub-millisecond latency requirements',
      //   expectedPath: 'handoff',
      //   description: 'Complex query - should handoff to specialist',
      // },
    ];

    const config = {
      configurable: {
        thread_id: 'hybrid-conversation-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    for (const test of testQueries) {
      console.log(`\n${'='.repeat(70)}`);
      console.log(`TEST: ${test.description}`);
      console.log(`QUERY: "${test.query}"`);
      console.log(`EXPECTED PATH: ${test.expectedPath}`);
      console.log('='.repeat(70));

      // Reset state
      conversationHistory.length = 0;
      conversationHistory.push(new HumanMessage(test.query));

      // Create separate content aggregator for each test
      const { contentParts, aggregateContent } = createContentAggregator();

      // Track agent progression for this test
      let currentAgent = '';
      let handoffOccurred = false;

      // Create custom handlers for this test
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
              console.log(`\n[${currentAgent}] Processing...`);
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
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (
            event: GraphEvents.ON_MESSAGE_DELTA,
            data: t.StreamEventData
          ): void => {
            // console.dir(data, { depth: null });
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
            if (toolData?.name?.startsWith(Constants.LC_TRANSFER_TO_)) {
              const specialist = toolData.name.replace(
                Constants.LC_TRANSFER_TO_,
                ''
              );
              console.log(`\n🔀 Transferring to ${specialist}...`);
              handoffOccurred = true;
            }
          },
        },
      };

      // Create a new run configuration for each test
      const runConfig: t.RunConfig = {
        runId: `hybrid-multi-agent-${test.expectedPath}-${Date.now()}`,
        graphConfig: {
          type: 'multi-agent',
          agents,
          edges,
        },
        customHandlers,
        returnContent: true,
      };

      // Create and execute a new run for this test
      const run = await Run.create(runConfig);

      console.log('\nProcessing request...');

      // Process with streaming
      const inputs = {
        messages: conversationHistory,
      };

      const finalContentParts = await run.processStream(inputs, config);
      const finalMessages = run.getRunMessages();

      if (finalMessages) {
        conversationHistory.push(...finalMessages);
      }

      // Show path taken
      console.log(`\n${'─'.repeat(70)}`);
      console.log('PATH ANALYSIS:');
      console.log(`- Query type: ${test.expectedPath}`);
      console.log(`- Handoff occurred: ${handoffOccurred ? 'YES' : 'NO'}`);
      console.log(
        `- Sequential path runs: ${handoffOccurred ? 'NO (exclusive routing)' : 'YES'}`
      );
      console.log(
        `- Result: ${
          (test.expectedPath === 'handoff' &&
            handoffOccurred &&
            !test.query.includes('continue')) ||
          (test.expectedPath === 'sequential' && !handoffOccurred)
            ? '✅ CORRECT'
            : '❌ INCORRECT'
        }`
      );
      console.log('─'.repeat(70));

      // Display the responses
      const aiMessages = conversationHistory.filter(
        (msg) => msg._getType() === 'ai'
      );
      console.log('\n--- Agent Responses ---');
      aiMessages.forEach((msg, index) => {
        console.log(`\nResponse ${index + 1}:`);
        console.log(msg.content);
      });
    }

    // Final summary
    console.log(`\n${'='.repeat(70)}`);
    console.log('HYBRID WORKFLOW TEST COMPLETE');
    console.log('='.repeat(70));
    console.log('\nThis test demonstrates automatic exclusive routing:');
    console.log('- When an agent has BOTH handoff and direct edges');
    console.log('- It uses Command-based routing for exclusive execution');
    console.log('- Either handoff OR direct edges execute, never both');
    console.log(
      '\nThis prevents duplicate processing in delegation scenarios!'
    );
  } catch (error) {
    console.error('Error in hybrid multi-agent test:', error);
  }
}

// Run the test
testHybridMultiAgent();
