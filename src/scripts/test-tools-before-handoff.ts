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
 * Test edge case: Agent performs 2 web searches before handing off
 *
 * This tests how the system behaves when an agent with handoff capabilities
 * uses tools before transferring control to another agent.
 */
async function testToolsBeforeHandoff() {
  console.log('Testing Tools Before Handoff Edge Case...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Track tool calls and handoffs
  let toolCallCount = 0;
  let handoffOccurred = false;

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(undefined, (name?: string) => {
      console.log(`\n✅ Tool completed: ${name}`);
      return true;
    }),
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
        // console.log('====== ON_MESSAGE_DELTA ======');
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
        console.log(`\n🔧 Tool started:`);
        console.dir({ toolData, metadata }, { depth: null });

        if (toolData?.output?.name?.includes('transfer_to_')) {
          handoffOccurred = true;
          const specialist = toolData.name.replace('transfer_to_', '');
          console.log(`\n🔀 Handoff initiated to: ${specialist}`);
        }
      },
    },
  };

  // Create the graph with research agent and report writer
  function createGraphWithToolsAndHandoff(): t.RunConfig {
    const agents: t.AgentInputs[] = [
      {
        agentId: 'research_coordinator',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        tools: [],
        instructions: `You are a Research Coordinator with access to a report writer specialist.
        
        Your workflow MUST follow these steps IN ORDER:
        1. FIRST: Write an initial response acknowledging the request
           - Explain what you understand about the topic
           - Provide any general knowledge you have
        2. FINALLY: Transfer to the report writer
           - Provide the report writer with a summary of the information
        
        CRITICAL: You MUST write your initial response before transferring to the report writer.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'report_writer',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-5-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Report Writer specialist. Your role is to:
        1. Receive research findings from the Research Coordinator
        2. Create a well-structured, comprehensive report
        3. Include all key findings from the research
        4. Format the report with clear sections and bullet points
        5. Add a brief executive summary at the beginning
        
        Focus on clarity, completeness, and professional presentation.`,
        maxContextTokens: 8000,
      },
    ];

    // Create edge from research coordinator to report writer
    const edges: t.GraphEdge[] = [
      {
        from: 'research_coordinator',
        to: 'report_writer',
        description: 'Transfer to report writer after completing research',
        edgeType: 'handoff',
      },
    ];

    return {
      runId: `tools-before-handoff-${Date.now()}`,
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
    // Single test query that requires handoff to report writer
    const query = `Tell me about quantum computing developments, 
    including major breakthroughs and commercial applications. 
    I need a comprehensive report.`;

    console.log('='.repeat(60));
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(60));

    // Create the graph
    const runConfig = createGraphWithToolsAndHandoff();
    const run = await Run.create(runConfig);

    console.log('\nExpected behavior:');
    console.log('1. Research Coordinator writes initial response');
    console.log('2. Research Coordinator hands off to Report Writer');
    console.log('3. Report Writer creates final report\n');

    // Process with streaming
    conversationHistory.push(new HumanMessage(query));
    const inputs = {
      messages: conversationHistory,
    };

    const config = {
      configurable: {
        thread_id: 'tools-handoff-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const finalContentParts = await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    // Show results summary
    console.log(`\n${'─'.repeat(60)}`);
    console.log('EDGE CASE TEST RESULTS:');
    console.log('─'.repeat(60));
    console.log(`Tool calls before handoff: ${toolCallCount}`);
    console.log(`Expected tool calls: 0 (no web search available)`);
    console.log(`Handoff occurred: ${handoffOccurred ? 'Yes ✅' : 'No ❌'}`);
    console.log(`Test status: ${handoffOccurred ? 'PASSED ✅' : 'FAILED ❌'}`);
    console.log('─'.repeat(60));

    // Display conversation history
    console.log('\nConversation History:');
    console.log('─'.repeat(60));
    conversationHistory.forEach((msg, idx) => {
      const role = msg.constructor.name.replace('Message', '');
      console.log(`\n[${idx}] ${role}:`);
      if (typeof msg.content === 'string') {
        console.log(
          msg.content.substring(0, 200) +
            (msg.content.length > 200 ? '...' : '')
        );
      }
    });
  } catch (error) {
    console.error('Error in tools-before-handoff test:', error);
  }
}

// Run the test
testToolsBeforeHandoff();
