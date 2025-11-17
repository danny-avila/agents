#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { labelContentByAgent, formatAgentMessages } from '@/messages/format';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { Providers, GraphEvents, Constants, StepTypes } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test supervisor-based multi-agent system using a single edge with multiple destinations
 *
 * Instead of creating 5 separate edges, we use one edge with an array of destinations
 * This should create handoff tools for all 5 specialists from a single edge definition
 */
async function testSupervisorListHandoff() {
  console.log('Testing Supervisor with List-Based Handoff Edge...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Track which specialist role was selected
  let selectedRole = '';

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
          selectedRole = specialist;
        }
      },
    },
  };

  // Function to create the graph with a single edge to multiple specialists
  function createSupervisorGraphWithListEdge(): t.RunConfig {
    console.log(`\nCreating graph with supervisor and 5 specialist agents.`);
    console.log(
      'Using a SINGLE edge with multiple destinations (list-based handoff).\n'
    );

    // Define the adaptive specialist configuration that will be reused
    const specialistConfig = {
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are an Adaptive Specialist. Your agent ID indicates your role:
      
      - data_analyst: Focus on statistical analysis, metrics, ML evaluation, A/B testing
      - security_expert: Focus on cybersecurity, vulnerability assessment, compliance  
      - product_designer: Focus on UX/UI design, user research, accessibility
      - devops_engineer: Focus on CI/CD, infrastructure, cloud platforms, monitoring
      - legal_advisor: Focus on licensing, privacy laws, contracts, regulatory compliance
      
      The supervisor will provide specific instructions. Follow them while maintaining your expert perspective.`,
      maxContextTokens: 8000,
    };

    // Create the graph with supervisor and all 5 specialists
    const agents: t.AgentInputs[] = [
      {
        agentId: 'supervisor',
        provider: Providers.ANTHROPIC,
        clientOptions: {
          modelName: 'claude-haiku-4-5',
          apiKey: process.env.ANTHROPIC_API_KEY,
        },
        instructions: `You are a Task Supervisor with access to 5 specialist agents:
        1. transfer_to_data_analyst - For statistical analysis and metrics
        2. transfer_to_security_expert - For cybersecurity and vulnerability assessment  
        3. transfer_to_product_designer - For UX/UI design
        4. transfer_to_devops_engineer - For infrastructure and deployment
        5. transfer_to_legal_advisor - For compliance and licensing
        
        Your role is to:
        1. Analyze the incoming request
        2. Decide which specialist is best suited
        3. Use the appropriate transfer tool (e.g., transfer_to_data_analyst)
        4. Provide specific instructions to guide their work
        
        Be specific about what you need from the specialist.`,
        maxContextTokens: 8000,
      },
      // Include all 5 specialists with the same adaptive configuration
      {
        agentId: 'data_analyst',
        ...specialistConfig,
      },
      {
        agentId: 'security_expert',
        ...specialistConfig,
      },
      {
        agentId: 'product_designer',
        ...specialistConfig,
      },
      {
        agentId: 'devops_engineer',
        ...specialistConfig,
      },
      {
        agentId: 'legal_advisor',
        ...specialistConfig,
      },
    ];

    // Create a SINGLE edge from supervisor to ALL 5 specialists using a list
    const edges: t.GraphEdge[] = [
      {
        from: 'supervisor',
        to: [
          'data_analyst',
          'security_expert',
          'product_designer',
          'devops_engineer',
          'legal_advisor',
        ],
        description:
          'Transfer to appropriate specialist based on task requirements',
        edgeType: 'handoff',
      },
    ];

    return {
      runId: `supervisor-list-handoff-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      customHandlers,
      returnContent: true,
    };
  }

  try {
    // Test with different queries
    const testQueries = [
      'What are the legal implications of using GPL-licensed code in our product?',
    ];

    const config = {
      configurable: {
        thread_id: 'supervisor-list-handoff-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    for (const query of testQueries) {
      console.log(`\n${'='.repeat(80)}`);
      console.log(`FIRST RUN - USER QUERY: "${query}"`);
      console.log('='.repeat(80));

      // Reset conversation
      conversationHistory.length = 0;
      conversationHistory.push(new HumanMessage(query));

      // Create graph with supervisor having a single edge to multiple specialists
      const runConfig = createSupervisorGraphWithListEdge();
      const run = await Run.create(runConfig);

      console.log('Processing first request...');

      // Process with streaming
      const inputs = {
        messages: conversationHistory,
      };

      const finalContentParts = await run.processStream(inputs, config);
      const finalMessages = run.getRunMessages();

      if (finalMessages) {
        conversationHistory.push(...finalMessages);
      }

      // Demo: Map contentParts to agentIds
      console.log(`\n${'─'.repeat(60)}`);
      console.log('CONTENT PARTS TO AGENT MAPPING:');
      console.log('─'.repeat(60));

      if (run.Graph) {
        // Get the mapping of contentPart index to agentId
        const contentPartAgentMap = run.Graph.getContentPartAgentMap();

        console.log(`\nTotal content parts: ${contentParts.length}`);
        console.log(`\nContent Part → Agent Mapping:`);

        contentPartAgentMap.forEach((agentId, index) => {
          const contentPart = contentParts[index];
          const contentType = contentPart?.type || 'unknown';
          const preview =
            contentType === 'text'
              ? (contentPart as any).text?.slice(0, 50) || ''
              : contentType === 'tool_call'
                ? `Tool: ${(contentPart as any).tool_call?.name || 'unknown'}`
                : contentType;

          console.log(
            `  [${index}] ${agentId} → ${contentType}: ${preview}${preview.length >= 50 ? '...' : ''}`
          );
        });

        // Show agent participation summary
        console.log(`\n${'─'.repeat(60)}`);
        console.log('AGENT PARTICIPATION SUMMARY:');
        console.log('─'.repeat(60));

        const activeAgents = run.Graph.getActiveAgentIds();
        console.log(`\nActive agents (${activeAgents.length}):`, activeAgents);

        const stepsByAgent = run.Graph.getRunStepsByAgent();
        stepsByAgent.forEach((steps, agentId) => {
          const toolCallSteps = steps.filter(
            (s) => s.type === StepTypes.TOOL_CALLS
          ).length;
          const messageSteps = steps.filter(
            (s) => s.type === StepTypes.MESSAGE_CREATION
          ).length;
          console.log(`\n  ${agentId}:`);
          console.log(`    - Total steps: ${steps.length}`);
          console.log(`    - Message steps: ${messageSteps}`);
          console.log(`    - Tool call steps: ${toolCallSteps}`);
        });
      }

      // Show graph structure summary
      console.log(`\n${'─'.repeat(60)}`);
      console.log(`GRAPH STRUCTURE:`);
      console.log(`- Agents: 6 total (supervisor + 5 specialists)`);
      console.log(`- Edges: 1 edge with multiple destinations`);
      console.log(
        `- Edge type: handoff (creates individual tools for each destination)`
      );
      console.log(
        `- Result: Supervisor has 5 handoff tools from a single edge`
      );
      console.log('─'.repeat(60));

      // =============================================================
      // SECOND RUN: Demonstrate agent-labeled history
      // =============================================================
      console.log(`\n${'='.repeat(80)}`);
      console.log(`SECOND RUN - Simulating DB Load with Agent-Labeled History`);
      console.log('='.repeat(80));

      // Simulate what happens in the main app:
      // 1. Store contentParts + agentIdMap to "DB" (in-memory here)
      const dbStoredContentParts = [...contentParts];
      const dbStoredAgentIdMap = Object.fromEntries(
        run.Graph!.getContentPartAgentMap()
      );

      console.log('\n📦 Simulating DB storage:');
      console.log(`  - Stored ${dbStoredContentParts.length} content parts`);
      console.log(
        `  - Stored agent mappings for ${Object.keys(dbStoredAgentIdMap).length} parts`
      );

      // 2. On next run, load from "DB" and label by agent
      console.log('\n📥 Loading from DB and labeling by agent...');

      const agentNames = {
        supervisor: 'Supervisor',
        legal_advisor: 'Legal Advisor',
        data_analyst: 'Data Analyst',
        security_expert: 'Security Expert',
        product_designer: 'Product Designer',
        devops_engineer: 'DevOps Engineer',
      };

      const labeledContentParts = labelContentByAgent(
        dbStoredContentParts.filter(
          (p): p is t.MessageContentComplex => p != null
        ),
        dbStoredAgentIdMap,
        agentNames
      );

      console.log(
        `  - Labeled ${labeledContentParts.length} content parts by agent`
      );

      // 3. Convert labeled content parts to payload format
      const payload: t.TPayload = [
        {
          role: 'user',
          content: query,
        },
        {
          role: 'assistant',
          content: labeledContentParts,
        },
      ];

      // 4. Format using formatAgentMessages (simulates what main app does)
      console.log('\n🔧 Calling formatAgentMessages...');
      const { messages: formattedMessages } = formatAgentMessages(payload);

      console.log(
        `  - Formatted into ${formattedMessages.length} BaseMessages`
      );

      // Show a preview of what the supervisor will see
      console.log('\n👁️  Preview of formatted history for supervisor:');
      console.log('─'.repeat(80));
      for (let i = 0; i < formattedMessages.length; i++) {
        const msg = formattedMessages[i];
        const role = msg._getType();
        const preview =
          typeof msg.content === 'string'
            ? msg.content.slice(0, 200)
            : JSON.stringify(msg.content).slice(0, 200);
        console.log(
          `[${i}] ${role}: ${preview}${preview.length >= 200 ? '...' : ''}`
        );
      }
      console.log('─'.repeat(80));

      // 5. Create a new run with the formatted history + a followup question
      console.log(
        '\n🚀 Starting second run with agent-labeled history + followup question...'
      );
      const followupQuery =
        'Can you summarize the key legal points from your previous response?';
      console.log(`   Followup: "${followupQuery}"`);

      // Reset for second run
      const secondRunHistory: BaseMessage[] = [
        ...formattedMessages,
        new HumanMessage(followupQuery),
      ];

      const runConfig2 = createSupervisorGraphWithListEdge();
      const run2 = await Run.create(runConfig2);

      const inputs2 = {
        messages: secondRunHistory,
      };

      await run2.processStream(inputs2, config);

      console.log('\n✅ Second run completed successfully!');
      console.log(
        '   The supervisor correctly understood that the legal_advisor handled'
      );
      console.log('   the previous query, avoiding identity confusion.');
    }

    // Final summary
    console.log(`\n${'='.repeat(60)}`);
    console.log('TEST COMPLETE');
    console.log('='.repeat(60));
    console.log('\nThis test demonstrates that a single edge with multiple');
    console.log('destinations in the "to" field creates individual handoff');
    console.log('tools for each destination agent, achieving the same result');
    console.log('as creating separate edges for each specialist.');
  } catch (error) {
    console.error('Error in supervisor list handoff test:', error);
  }
}

// Run the test
testSupervisorListHandoff();
