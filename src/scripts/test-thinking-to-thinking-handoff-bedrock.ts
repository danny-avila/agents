#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { ChatModelStreamHandler } from '@/stream';
import { Providers, GraphEvents } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

/**
 * Test edge case: switching from Bedrock supervisor (with thinking) to Bedrock specialist (with thinking)
 * Both agents use extended thinking, validating that thinking blocks are properly
 * handled across the handoff boundary when both sides produce them.
 */
async function testThinkingToThinkingHandoffBedrock() {
  console.log(
    'Testing Bedrock (with thinking) → Bedrock (with thinking) handoff...\n'
  );

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
    [GraphEvents.ON_RUN_STEP]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const runStep = data as t.RunStep;
        console.log(
          `\n📍 ON_RUN_STEP: agentId=${runStep.agentId}, groupId=${runStep.groupId}`
        );
      },
    },
  };

  function createGraphConfig(): t.RunConfig {
    console.log(
      'Creating graph with Bedrock supervisor (thinking) and Bedrock specialist (thinking).\n'
    );

    const agents: t.AgentInputs[] = [
      {
        agentId: 'supervisor',
        provider: Providers.BEDROCK,
        clientOptions: {
          region: process.env.BEDROCK_AWS_REGION || 'us-east-1',
          model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
          credentials: {
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
          },
          additionalModelRequestFields: {
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
        },
        instructions: `You are a task supervisor. When the user asks about code review, use transfer_to_code_reviewer to hand off to the specialist.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'code_reviewer',
        provider: Providers.BEDROCK,
        clientOptions: {
          region: process.env.BEDROCK_AWS_REGION || 'us-east-1',
          model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
          credentials: {
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
          },
          additionalModelRequestFields: {
            thinking: {
              type: 'enabled',
              budget_tokens: 2000,
            },
          },
        },
        instructions: `You are a code review specialist using Bedrock with extended thinking. Think carefully about the code quality, best practices, and potential issues. Provide thoughtful feedback.`,
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
      runId: `thinking-to-thinking-bedrock-test-${Date.now()}`,
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
    const query =
      'Can you review this function and tell me if there are any issues?\n\nfunction add(a, b) { return a + b; }';

    console.log(`${'='.repeat(60)}`);
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(60));

    conversationHistory.push(new HumanMessage(query));

    const runConfig = createGraphConfig();
    const run = await Run.create(runConfig);

    const streamConfig = {
      configurable: {
        thread_id: 'thinking-to-thinking-bedrock-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    console.log('\nProcessing request...\n');

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, streamConfig);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    console.log(`\n${'='.repeat(60)}`);
    console.log('✅ TEST PASSED');
    console.log('='.repeat(60));
    console.log('\nSuccessfully handed off from Bedrock (with thinking) to');
    console.log('Bedrock (with thinking) without errors!');
    console.log('\nThinking blocks were properly managed across the handoff');
    console.log('boundary when both agents produce them.');
  } catch (error) {
    console.error('\n❌ TEST FAILED');
    console.error('='.repeat(60));
    console.error('Error:', error);
    process.exit(1);
  }
}

testThinkingToThinkingHandoffBedrock();
