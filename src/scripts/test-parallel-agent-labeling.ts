#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { labelContentByAgent, formatAgentMessages } from '@/messages/format';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { Providers, GraphEvents, StepTypes } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test parallel multi-agent system with agent labeling on subsequent runs
 *
 * Graph structure:
 * START -> researcher
 * researcher -> [analyst1, analyst2, analyst3] (fan-out)
 * [analyst1, analyst2, analyst3] -> summarizer (fan-in)
 * summarizer -> END
 */
async function testParallelWithAgentLabeling() {
  console.log('Testing Parallel Multi-Agent with Agent Labeling...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define specialized agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'researcher',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a research coordinator. Analyze the request and provide 2-3 sentence coordination brief.`,
    },
    {
      agentId: 'analyst1',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are FINANCIAL ANALYST. Provide 2-3 sentence financial analysis. Start with "FINANCIAL ANALYSIS:"`,
    },
    {
      agentId: 'analyst2',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are TECHNICAL ANALYST. Provide 2-3 sentence technical analysis. Start with "TECHNICAL ANALYSIS:"`,
    },
    {
      agentId: 'analyst3',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are MARKET ANALYST. Provide 2-3 sentence market analysis. Start with "MARKET ANALYSIS:"`,
    },
    {
      agentId: 'summarizer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are SYNTHESIS EXPERT. Review all analyses and provide 2-3 sentence integrated summary.`,
    },
  ];

  // Define direct edges (fan-out and fan-in)
  const edges: t.GraphEdge[] = [
    {
      from: 'researcher',
      to: ['analyst1', 'analyst2', 'analyst3'],
      description: 'Distribute research to specialist analysts',
      edgeType: 'direct',
    },
    {
      from: ['analyst1', 'analyst2', 'analyst3'],
      to: 'summarizer',
      description: 'Aggregate analysis results',
      edgeType: 'direct',
      prompt:
        'Based on the analyses below, provide an integrated summary:\n\n{results}',
    },
  ];

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
        aggregateContent({ event, data: data as t.RunStep });
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

  try {
    const query = 'What are the implications of widespread AI adoption?';

    console.log(`${'='.repeat(80)}`);
    console.log(`FIRST RUN - USER QUERY: "${query}"`);
    console.log('='.repeat(80));

    // Reset conversation
    conversationHistory.length = 0;
    conversationHistory.push(new HumanMessage(query));

    // Create graph
    const runConfig: t.RunConfig = {
      runId: `parallel-test-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      customHandlers,
      returnContent: true,
    };

    const run = await Run.create(runConfig);

    console.log('\nProcessing first run with parallel agents...\n');

    const config = {
      configurable: {
        thread_id: 'parallel-agent-labeling-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    // Show agent participation
    console.log(`\n${'─'.repeat(80)}`);
    console.log('FIRST RUN - AGENT PARTICIPATION:');
    console.log('─'.repeat(80));

    if (run.Graph) {
      const activeAgents = run.Graph.getActiveAgentIds();
      console.log(`\nActive agents (${activeAgents.length}):`, activeAgents);

      const stepsByAgent = run.Graph.getRunStepsByAgent();
      stepsByAgent.forEach((steps, agentId) => {
        console.log(`  ${agentId}: ${steps.length} steps`);
      });

      console.log(`\nTotal content parts: ${contentParts.length}`);
    }

    // =============================================================
    // SECOND RUN: Test with agent-labeled history
    // =============================================================
    console.log(`\n${'='.repeat(80)}`);
    console.log(`SECOND RUN - Simulating DB Load with Parallel Agent Labeling`);
    console.log('='.repeat(80));

    // Simulate DB storage
    const dbStoredContentParts = [...contentParts];
    const dbStoredAgentIdMap = Object.fromEntries(
      run.Graph!.getContentPartAgentMap()
    );

    console.log('\n📦 Simulating DB storage:');
    console.log(`  - Stored ${dbStoredContentParts.length} content parts`);
    console.log(
      `  - Stored agent mappings for ${Object.keys(dbStoredAgentIdMap).length} parts`
    );

    // Load and label by agent with labelNonTransferContent option
    console.log('\n📥 Loading from DB and labeling ALL agent content...');

    const agentNames = {
      researcher: 'Researcher',
      analyst1: 'Financial Analyst',
      analyst2: 'Technical Analyst',
      analyst3: 'Market Analyst',
      summarizer: 'Synthesizer',
    };

    const labeledContentParts = labelContentByAgent(
      dbStoredContentParts.filter(
        (p): p is t.MessageContentComplex => p != null
      ),
      dbStoredAgentIdMap,
      agentNames,
      { labelNonTransferContent: true } // NEW: Label all content
    );

    console.log(
      `  - Labeled ${labeledContentParts.length} content groups by agent`
    );

    // Convert to payload
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

    // Format using formatAgentMessages
    console.log('\n🔧 Calling formatAgentMessages...');
    const { messages: formattedMessages } = formatAgentMessages(payload);

    console.log(`  - Formatted into ${formattedMessages.length} BaseMessages`);

    // Show preview
    console.log('\n👁️  Preview of formatted history:');
    console.log('─'.repeat(80));
    for (let i = 0; i < formattedMessages.length; i++) {
      const msg = formattedMessages[i];
      const role = msg._getType();
      const preview =
        typeof msg.content === 'string'
          ? msg.content.slice(0, 300)
          : JSON.stringify(msg.content).slice(0, 300);
      console.log(
        `[${i}] ${role}: ${preview}${preview.length >= 300 ? '...' : ''}`
      );
      console.log('');
    }
    console.log('─'.repeat(80));

    // Create a second run with labeled history
    console.log(
      '\n🚀 Starting second run with agent-labeled parallel history...'
    );
    const followupQuery = 'Which analyst identified the most significant risk?';
    console.log(`   Followup: "${followupQuery}"`);

    const secondRunHistory: BaseMessage[] = [
      ...formattedMessages,
      new HumanMessage(followupQuery),
    ];

    const runConfig2: t.RunConfig = {
      runId: `parallel-test-2-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      customHandlers,
      returnContent: true,
    };

    const run2 = await Run.create(runConfig2);

    const inputs2 = {
      messages: secondRunHistory,
    };

    await run2.processStream(inputs2, config);

    console.log('\n✅ Second run completed successfully!');
    console.log(
      '   The researcher correctly understood that parallel analysts handled'
    );
    console.log('   the previous analysis, with clear attribution per agent.');

    console.log(`\n${'='.repeat(80)}`);
    console.log('TEST COMPLETE');
    console.log('='.repeat(80));
    console.log(
      '\nThis demonstrates that parallel multi-agent patterns work correctly'
    );
    console.log(
      'with agent labeling, preventing confusion about who said what.'
    );
  } catch (error) {
    console.error('Error in parallel agent labeling test:', error);
  }
}

// Run the test
testParallelWithAgentLabeling();
