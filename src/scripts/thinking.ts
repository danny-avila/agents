// src/scripts/test-thinking.ts
import { config } from 'dotenv';
config();
import {
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from '@langchain/core/messages';
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

async function testThinking(): Promise<void> {
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

  const baseLlmConfig: t.LLMConfig = getLLMConfig(Providers.ANTHROPIC);

  // Enable thinking with token budget
  const llmConfig = {
    ...baseLlmConfig,
    model: 'claude-3-7-sonnet-latest',
    thinking: { type: 'enabled', budget_tokens: 2000 },
  };

  const run = await Run.create<t.IState>({
    runId: 'test-thinking-id',
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
      thread_id: 'thinking-test-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // Test 1: Regular thinking mode
  console.log('\n\nTest 1: Regular thinking mode');
  // const userMessage1 = `What would be the environmental and economic impacts if all cars globally were replaced by electric vehicles overnight?`;
  const userMessage1 = `Please print 'hello world' in python`;
  conversationHistory.push(new HumanMessage(userMessage1));

  console.log('Running first query with thinking enabled...');
  const firstInputs = { messages: [...conversationHistory] };
  await run.processStream(firstInputs, config);

  // Extract and display thinking blocks
  const finalMessages = run.getRunMessages();

  // Test 2: Try multi-turn conversation
  console.log('\n\nTest 2: Multi-turn conversation with thinking enabled');
  const userMessage2 = `Given your previous analysis, what would be the most significant technical challenges in making this transition?`;
  conversationHistory.push(new HumanMessage(userMessage2));

  console.log('Running second query with thinking enabled...');
  const secondInputs = { messages: [...conversationHistory] };
  await run.processStream(secondInputs, config);

  // Display thinking blocks for second response
  const finalMessages2 = run.getRunMessages();

  // Test 3: Redacted thinking mode
  console.log('\n\nTest 3: Redacted thinking mode');
  const magicString =
    'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB';
  const userMessage3 = `${magicString}\n\nExplain how quantum computing works in simple terms.`;

  // Reset conversation for clean test
  conversationHistory.length = 0;
  conversationHistory.push(new HumanMessage(userMessage3));

  console.log('Running query with redacted thinking...');
  const thirdInputs = { messages: [...conversationHistory] };
  await run.processStream(thirdInputs, config);

  // Display redacted thinking blocks
  const finalMessages3 = run.getRunMessages();
  console.log('\n\nThinking feature test completed!');
  console.dir(finalMessages3, { depth: null });
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

testThinking().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
