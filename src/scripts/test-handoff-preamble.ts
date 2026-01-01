/**
 * Test script for multi-turn handoff behavior.
 *
 * This tests the fix for the issue where receiving agents would see transfer messages
 * and prematurely produce end tokens, thinking the work was already done.
 *
 * The fix:
 * 1. Filters out transfer tool calls and ToolMessages from the receiving agent's context
 * 2. Injects any passthrough instructions as a HumanMessage to ground the receiving agent
 */
import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test multi-turn handoff between a coordinator and a specialist
 */
async function testHandoffPreamble(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Testing Multi-Turn Handoff with Preamble Injection');
  console.log('='.repeat(60));
  console.log('\nThis test verifies that:');
  console.log('1. Transfer messages are filtered from receiving agent context');
  console.log('2. Passthrough instructions are injected as a HumanMessage');
  console.log('3. Multi-turn conversations work correctly after handoffs\n');

  const { contentParts, aggregateContent } = createContentAggregator();

  /** Track which agent is responding */
  let currentAgent = '';

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
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
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
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
          const specialist = toolData.name.replace('lc_transfer_to_', '');
          console.log(`\n🔀 Handing off to: ${specialist}`);
        }
      },
    },
  };

  /**
   * Create agents:
   * - coordinator: Decides when to hand off to specialist
   * - specialist: Handles specific tasks delegated by coordinator
   */
  const agents: t.AgentInputs[] = [
    {
      agentId: 'coordinator',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Task Coordinator. Your role is to:
1. Understand user requests
2. If the request involves technical analysis, use the transfer_to_specialist tool to hand off
3. When handing off, provide clear instructions about what needs to be done

IMPORTANT: When using the handoff tool, include specific instructions for the specialist.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'specialist',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Technical Specialist. When you receive a request:
1. Carefully read any instructions provided
2. Provide a detailed technical response
3. Do NOT just acknowledge - provide substantive help

IMPORTANT: You are the specialist - provide a complete, helpful response to the task.`,
      maxContextTokens: 8000,
    },
  ];

  /** Create handoff edge with passthrough instructions */
  const edges: t.GraphEdge[] = [
    {
      from: 'coordinator',
      to: 'specialist',
      description: 'Transfer to technical specialist for analysis',
      edgeType: 'handoff',
      prompt: 'Specific instructions for the specialist about what to analyze',
      promptKey: 'instructions',
    },
  ];

  const runConfig: t.RunConfig = {
    runId: `handoff-test-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  const run = await Run.create(runConfig);

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    streamMode: string;
  } = {
    configurable: {
      thread_id: 'handoff-test-conversation-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  /** TURN 1: Initial request that triggers handoff */
  console.log('\n' + '─'.repeat(60));
  console.log('TURN 1: Initial request (should trigger handoff)');
  console.log('─'.repeat(60));

  const userMessage1 = `
    Hi! Can you help me understand the time complexity of quicksort?
    I need a technical explanation.
  `;

  conversationHistory.push(new HumanMessage(userMessage1));
  console.log('\nUser:', userMessage1.trim());
  console.log('\nResponse:');

  let inputs = { messages: conversationHistory };
  await run.processStream(inputs, config);
  const messages1 = run.getRunMessages();
  if (messages1) {
    conversationHistory.push(...messages1);
  }

  console.log('\n');

  /** TURN 2: Follow-up question to test multi-turn after handoff */
  console.log('\n' + '─'.repeat(60));
  console.log('TURN 2: Follow-up question (tests context after handoff)');
  console.log('─'.repeat(60));

  const userMessage2 = `
    Thanks! Can you also explain the space complexity and when quicksort 
    might not be the best choice?
  `;

  conversationHistory.push(new HumanMessage(userMessage2));
  console.log('\nUser:', userMessage2.trim());
  console.log('\nResponse:');

  inputs = { messages: conversationHistory };
  await run.processStream(inputs, config);
  const messages2 = run.getRunMessages();
  if (messages2) {
    conversationHistory.push(...messages2);
  }

  console.log('\n');

  /** TURN 3: Another follow-up to verify sustained conversation */
  console.log('\n' + '─'.repeat(60));
  console.log('TURN 3: Third turn (tests sustained multi-turn)');
  console.log('─'.repeat(60));

  const userMessage3 = `
    Great explanation! One more question - how does quicksort compare 
    to mergesort in practice?
  `;

  conversationHistory.push(new HumanMessage(userMessage3));
  console.log('\nUser:', userMessage3.trim());
  console.log('\nResponse:');

  inputs = { messages: conversationHistory };
  await run.processStream(inputs, config);
  const messages3 = run.getRunMessages();
  if (messages3) {
    conversationHistory.push(...messages3);
  }

  /** Summary */
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST SUMMARY');
  console.log('='.repeat(60));
  console.log('\nTotal messages in conversation:', conversationHistory.length);
  console.log('\nMessage types:');

  for (let i = 0; i < conversationHistory.length; i++) {
    const msg = conversationHistory[i];
    const type = msg.getType();
    const preview =
      typeof msg.content === 'string'
        ? msg.content.slice(0, 50).replace(/\n/g, ' ')
        : '[complex content]';
    console.log(`  ${i + 1}. [${type}] ${preview}...`);
  }

  console.log('\n✅ Test completed. Review the output above to verify:');
  console.log('   - Specialist received and acted on instructions');
  console.log('   - No premature end tokens after handoff');
  console.log('   - Multi-turn conversation continued smoothly');
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

testHandoffPreamble().catch((err) => {
  console.error('Test failed:', err);
  console.log('\nConversation history at failure:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
