#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { ChatModelStreamHandler } from '@/stream';
import { Providers, GraphEvents } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';
import { z } from 'zod';

const conversationHistory: BaseMessage[] = [];

async function reproduceBug() {
  class DelayedModelEndHandler extends ModelEndHandler {
    async handle(
      event: string,
      data: t.ModelEndData,
      metadata?: Record<string, unknown>,
      graph?: any
    ): Promise<void> {
      console.log('[MODEL_END] simulating delay');
      await new Promise((resolve) => setTimeout(resolve, 100));
      console.log('[MODEL_END] delay complete');
      return super.handle(event, data, metadata, graph);
    }
  }

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new DelayedModelEndHandler(),
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

  const calculatorTool = {
    name: 'calculator',
    description:
      'Perform basic arithmetic operations. Use this when you need to calculate numbers.',
    schema: z.object({
      expression: z
        .string()
        .describe(
          'The mathematical expression to evaluate, e.g., "2 + 2" or "10 * 5"'
        ),
    }),
  };

  function createGraphConfig(): t.RunConfig {
    console.log('Creating graph with Bedrock provider and calculator tool.\n');

    const agents: t.AgentInputs[] = [
      {
        agentId: 'calculator_agent',
        provider: Providers.BEDROCK,
        clientOptions: {
          region: process.env.BEDROCK_AWS_REGION || 'us-east-1',
          model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
          credentials: {
            accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
            secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
          },
        },
        instructions:
          'You are a helpful calculator assistant. Use the calculator tool when you need to perform calculations.',
        tools: [calculatorTool],
        maxContextTokens: 8000,
      },
    ];

    return {
      runId: `bedrock-tool-bug-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents,
      },
      customHandlers,
      returnContent: true,
    };
  }

  try {
    const query = 'What is 157 multiplied by 83?';

    console.log(`${'='.repeat(70)}`);
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(70));
    console.log(
      '   ToolEndHandler will fire BEFORE ModelEndHandler can register tool_call_id'
    );
    console.log(
      '   This will cause: "No stepId found for tool_call_id" error\n'
    );

    conversationHistory.push(new HumanMessage(query));

    const runConfig = createGraphConfig();
    const run = await Run.create(runConfig);

    const config = {
      configurable: {
        thread_id: 'bedrock-tool-bug-test',
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

    console.log(`\n${'='.repeat(70)}`);
    console.log('NO ERROR OCCURRED');
    console.log('='.repeat(70));
  } catch (error: any) {
    process.exit(1);
  }
}

// Run the reproduction
reproduceBug();
