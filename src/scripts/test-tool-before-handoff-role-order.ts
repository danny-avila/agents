import { config } from 'dotenv';
config();

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test: Tool call followed by handoff (role order validation)
 *
 * Reproduces the bug from issue #54:
 * When a router agent runs a non-handoff tool (e.g. list_upload_sessions)
 * and then hands off to another agent in the same turn, the receiving agent
 * gets a message sequence of `... tool → user` which many chat APIs reject
 * with: "400 Unexpected role 'user' after role 'tool'"
 *
 * The fix ensures handoff instructions are injected into the last ToolMessage
 * (instead of appending a new HumanMessage) when the filtered messages end
 * with a ToolMessage.
 */
async function testToolBeforeHandoffRoleOrder(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Test: Tool Call Before Handoff (Role Order Validation)');
  console.log('='.repeat(60));
  console.log('\nThis test verifies that:');
  console.log('1. Router calls a regular tool AND then hands off');
  console.log('2. The receiving agent does NOT get tool → user role sequence');
  console.log('3. No 400 API error occurs after the handoff\n');

  const { contentParts, aggregateContent } = createContentAggregator();

  let currentAgent = '';
  let toolCallCount = 0;
  let handoffOccurred = false;

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(undefined, (name?: string) => {
      toolCallCount++;
      console.log(`\n  Tool completed: ${name} (total: ${toolCallCount})`);
      return true;
    }),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        const runStep = data as t.RunStep;
        if (runStep.agentId) {
          currentAgent = runStep.agentId;
          console.log(`\n[Agent: ${currentAgent}] Processing...`);
        }
        aggregateContent({ event, data: runStep });
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
        _metadata?: Record<string, unknown>
      ): void => {
        const toolData = data as { name?: string };
        if (toolData?.name?.includes('transfer_to_')) {
          handoffOccurred = true;
          const specialist = toolData.name.replace('lc_transfer_to_', '');
          console.log(`\n  Handoff initiated to: ${specialist}`);
        }
      },
    },
  };

  /**
   * Create a simple tool for the router agent.
   * This simulates the list_upload_sessions scenario from issue #54:
   * the router calls a regular tool and THEN hands off in the same turn.
   */
  const listSessions = tool(
    async () => {
      return JSON.stringify({
        sessions: [
          { id: 'sess_1', name: 'Q4 Report', status: 'ready' },
          { id: 'sess_2', name: 'Budget Analysis', status: 'pending' },
        ],
      });
    },
    {
      name: 'list_upload_sessions',
      description: 'List available upload sessions for data analysis',
      schema: z.object({}),
    }
  );

  const agents: t.AgentInputs[] = [
    {
      agentId: 'router',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      tools: [listSessions],
      instructions: `You are a Router agent with access to upload sessions and a data analysis specialist.

Your workflow for data-related requests:
1. FIRST: Call list_upload_sessions to check available data
2. THEN: Transfer to the data_analyst with your findings

CRITICAL: You MUST call list_upload_sessions first, then immediately transfer to data_analyst.
Do NOT write a long response. Just call the tool and hand off.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'data_analyst',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Data Analyst specialist. When you receive a request:
1. Review any data or context provided
2. Provide a concise analysis or recommendation
3. Keep your response brief and focused`,
      maxContextTokens: 8000,
    },
  ];

  const edges: t.GraphEdge[] = [
    {
      from: 'router',
      to: 'data_analyst',
      description: 'Transfer to data analyst after checking sessions',
      edgeType: 'handoff',
      prompt:
        'Provide specific instructions for the data analyst about what to analyze',
      promptKey: 'instructions',
    },
  ];

  const runConfig: t.RunConfig = {
    runId: `tool-before-handoff-role-order-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  const run = await Run.create(runConfig);

  const streamConfig: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    streamMode: string;
  } = {
    configurable: {
      thread_id: 'tool-before-handoff-role-order-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  try {
    const query =
      'I want to visualize my CSV data. Can you check what upload sessions are available and have the analyst help me?';

    console.log('\n' + '-'.repeat(60));
    console.log(`USER QUERY: "${query}"`);
    console.log('-'.repeat(60));
    console.log('\nExpected behavior:');
    console.log('1. Router calls list_upload_sessions tool');
    console.log('2. Router hands off to data_analyst');
    console.log('3. data_analyst responds WITHOUT 400 error\n');

    conversationHistory.push(new HumanMessage(query));
    const inputs = { messages: conversationHistory };

    await run.processStream(inputs, streamConfig);
    const finalMessages = run.getRunMessages();
    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    /** Results */
    console.log(`\n${'='.repeat(60)}`);
    console.log('TEST RESULTS:');
    console.log('='.repeat(60));
    console.log(`Tool calls made: ${toolCallCount}`);
    console.log(`Handoff occurred: ${handoffOccurred ? 'Yes' : 'No'}`);
    console.log(
      `Test status: ${toolCallCount > 0 && handoffOccurred ? 'PASSED' : 'FAILED'}`
    );

    if (toolCallCount === 0) {
      console.log('\nNote: Router did not call any tools before handoff.');
      console.log(
        'The bug only occurs when a non-handoff tool is called in the same turn as the handoff.'
      );
      console.log('Try running again - the model may need stronger prompting.');
    }

    console.log('='.repeat(60));

    /** Show conversation history */
    console.log('\nConversation History:');
    console.log('-'.repeat(60));
    conversationHistory.forEach((msg, idx) => {
      const role = msg.getType();
      const content =
        typeof msg.content === 'string'
          ? msg.content.substring(0, 150) +
            (msg.content.length > 150 ? '...' : '')
          : '[complex content]';
      console.log(`  [${idx}] ${role}: ${content}`);
    });
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('\nTest FAILED with error:', errorMsg);

    if (errorMsg.includes('Unexpected role') || errorMsg.includes('400')) {
      console.error('\n>>> This is the exact bug from issue #54! <<<<');
      console.error(
        '>>> The tool→user role sequence caused a 400 API error. <<<'
      );
    }

    console.log('\nConversation history at failure:');
    console.dir(conversationHistory, { depth: null });
  }
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('\nConversation history at failure:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

testToolBeforeHandoffRoleOrder().catch((err) => {
  console.error('Test failed:', err);
  console.log('\nConversation history at failure:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
