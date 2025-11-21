/* eslint-disable no-console */
// src/scripts/ant_web_search_error_edge_case.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';

import { getArgs } from '@/scripts/args';
import { Run } from '@/run';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';

const conversationHistory: BaseMessage[] = [];
let _contentParts: (t.MessageContentComplex | undefined)[] = [];
async function testStandardStreaming(): Promise<void> {
  const { userName, location, currentDate } = await getArgs();
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts;
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        // console.dir(data, { depth: null });
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
        console.log('====== ON_RUN_STEP ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        // console.log('====== ON_MESSAGE_DELTA ======');
        // console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        // console.dir(data, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(
    Providers.ANTHROPIC
  ) as t.AnthropicClientOptions & t.SharedLLMConfig;
  llmConfig.model = 'claude-haiku-4-5';

  const run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [
        {
          type: 'web_search_20250305',
          name: 'web_search',
          max_uses: 5,
        },
      ],
      instructions: 'You are a helpful AI research assistant.',
    },
    returnContent: true,
    customHandlers,
  });

  const config = {
    configurable: {
      provider: Providers.ANTHROPIC,
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Test: Web search with multiple searches (error edge case test)');

  // This prompt should trigger multiple web searches which may result in errors
  const userMessage =
    'Do a deep deep research on CoreWeave. I need you to perform multiple searches before you generate the answer. The basis of our research should be to investigate if this is a solid long term investment.';

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
  // console.dir(finalContentParts, { depth: null });
  console.log('\n\n====================\n\n');
  // console.dir(contentParts, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Content Parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content Parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
