// src/scripts/thinking-bedrock.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];
const collectedUsage: UsageMetadata[] = [];

async function testBedrockThinking(): Promise<void> {
  const { userName } = await getArgs();
  const instructions = `You are a helpful AI assistant for ${userName}. When answering questions, be thorough in your reasoning.`;
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];

  // Set up event handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.RunStep) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.RunStepDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.MessageDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.ReasoningDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
  };

  const baseLlmConfig = getLLMConfig(Providers.BEDROCK);

  // Enable thinking with token budget for Bedrock
  const llmConfig = {
    ...baseLlmConfig,
    model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    maxTokens: 5000,
    additionalModelRequestFields: {
      thinking: { type: 'enabled', budget_tokens: 2000 },
    },
  };

  const run = await Run.create<t.IState>({
    runId: 'test-bedrock-thinking-id',
    graphConfig: {
      instructions,
      type: 'standard',
      tools: [createCodeExecutionTool()],
      llmConfig,
    },
    returnContent: true,
    customHandlers: customHandlers as t.RunConfig['customHandlers'],
  });

  const config = {
    configurable: {
      thread_id: 'bedrock-thinking-test-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // Test 1: Regular thinking mode
  console.log('\n\nTest 1: Bedrock Regular thinking mode');
  const userMessage1 = `Please print 'hello world' in python`;
  conversationHistory.push(new HumanMessage(userMessage1));

  console.log('Running first query with Bedrock thinking enabled...');
  const firstInputs = { messages: [...conversationHistory] };
  await run.processStream(firstInputs, config);

  // Extract and display thinking blocks
  const finalMessages = run.getRunMessages();
  console.log('\n\nFinal messages after Test 1:');
  console.dir(finalMessages, { depth: null });

  // Test 2: Try multi-turn conversation
  console.log(
    '\n\nTest 2: Multi-turn conversation with Bedrock thinking enabled'
  );
  const userMessage2 = `Given your previous analysis, what would be the most significant technical challenges in making this transition?`;
  conversationHistory.push(new HumanMessage(userMessage2));

  console.log('Running second query with Bedrock thinking enabled...');
  const secondInputs = { messages: [...conversationHistory] };
  await run.processStream(secondInputs, config);

  // Display thinking blocks for second response
  const finalMessages2 = run.getRunMessages();
  console.log('\n\nBedrock thinking feature test completed!');
  console.dir(finalMessages2, { depth: null });

  console.log('\n\nContent parts:');
  console.dir(_contentParts, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

testBedrockThinking().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
