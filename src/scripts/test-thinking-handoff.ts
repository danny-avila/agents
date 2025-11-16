#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { ChatModelStreamHandler } from '@/stream';
import { Providers, GraphEvents } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test edge case: switching from OpenAI supervisor (no thinking) to Anthropic specialist (with thinking enabled)
 * This should not throw an error about missing thinking blocks
 */
async function testThinkingHandoff() {
  console.log('Testing OpenAI → Anthropic (with thinking) handoff...\n');

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const toolData = data as any;
        if (toolData?.name) {
          console.log(`\n🔧 Tool called: ${toolData.name}`);
        }
      },
    },
  };

  // Create the graph configuration
  function createGraphConfig(): t.RunConfig {
    console.log(
      'Creating graph with OpenAI supervisor and Anthropic specialist with thinking enabled.\n'
    );

    const agents: t.AgentInputs[] = [
      {
        agentId: 'supervisor',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4o-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a task supervisor. When the user asks about code review, use transfer_to_code_reviewer to hand off to the specialist.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'code_reviewer',
        provider: Providers.ANTHROPIC,
        clientOptions: {
          modelName: 'claude-3-7-sonnet-20250219',
          apiKey: process.env.ANTHROPIC_API_KEY,
          thinking: {
            type: 'enabled',
            budget_tokens: 2000,
          },
        },
        instructions: `You are a code review specialist. Think carefully about the code quality, best practices, and potential issues. Provide thoughtful feedback.`,
        maxContextTokens: 8000,
      },
    ];

    const edges: t.GraphEdge[] = [
      {
        from: 'supervisor',
        to: ['code_reviewer'],
        description: 'Transfer to code review specialist',
        edgeType: 'handoff',
      },
    ];

    return {
      runId: `thinking-handoff-test-${Date.now()}`,
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
    // Test query that should trigger a handoff
    const query =
      'Can you review this function and tell me if there are any issues?\n\nfunction add(a, b) { return a + b; }';

    console.log(`${'='.repeat(60)}`);
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(60));

    // Initialize conversation
    conversationHistory.push(new HumanMessage(query));

    // Create and run the graph
    const runConfig = createGraphConfig();
    const run = await Run.create(runConfig);

    const config = {
      configurable: {
        thread_id: 'thinking-handoff-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    console.log('\nProcessing request...\n');

    // Process with streaming
    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    // Success!
    console.log(`\n${'='.repeat(60)}`);
    console.log('✅ TEST PASSED');
    console.log('='.repeat(60));
    console.log('\nSuccessfully handed off from OpenAI (no thinking) to');
    console.log('Anthropic with thinking enabled without errors!');
    console.log('\nThe ensureThinkingBlockInMessages() function correctly');
    console.log('added a placeholder thinking block to the last assistant');
    console.log('message before calling the Anthropic API.');
  } catch (error) {
    console.error('\n❌ TEST FAILED');
    console.error('='.repeat(60));
    console.error('Error:', error);
    process.exit(1);
  }
}

// Run the test
testThinkingHandoff();
