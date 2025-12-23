/* eslint-disable no-console */
// src/scripts/cli.ts
import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { Calculator } from '@/tools/Calculator';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
async function testStandardStreaming(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();
  const { contentParts, aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(undefined, (name?: string) => {
      return true;
    }),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_END METADATA ======');
        console.dir(metadata, { depth: null });
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_START METADATA ======');
        console.dir(metadata, { depth: null });
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== ON_RUN_STEP ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
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
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ) => {
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  if (llmConfig.provider === Providers.BEDROCK) {
    (llmConfig as t.BedrockAnthropicInput).promptCache = true;
  }

  const run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [new Calculator()],
      instructions:
        'You are a friendly AI assistant. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      maxContextTokens: 89000,
    },
    indexTokenCountMap: { 0: 35 },
    returnContent: true,
    customHandlers,
  });

  const config = {
    configurable: {
      provider,
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Test 1: Calculation query');

  const userMessage = `What is 1123123 + 123123 / 20348? After that, run some interesting calculations based off the result`;

  conversationHistory.push(new HumanMessage(userMessage));

  const inputs = {
    messages: conversationHistory,
  };
  const finalContentParts = await run.processStream(inputs, config);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
    console.dir(conversationHistory, { depth: null });
  }
  console.dir(finalContentParts, { depth: null });
  console.log('\n\n====================\n\n');
  // console.dir(contentParts, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  process.exit(1);
});

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
