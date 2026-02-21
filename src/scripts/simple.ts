// src/scripts/cli.ts
import { config } from 'dotenv';
import { v4 as uuidv4 } from 'uuid';
config();
import {
  HumanMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import {
  ToolEndHandler,
  ModelEndHandler,
  createMetadataAggregator,
} from '@/events';
import { GraphEvents, Providers, TitleMethod } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { sleep } from '@/utils/run';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];
let collectedUsage: UsageMetadata[] = [];

async function testStandardStreaming(): Promise<void> {
  const {
    userName,
    location,
    provider: _provider,
    currentDate,
  } = await getArgs();
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];
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
        console.dir(data, { depth: null });
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
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_REASONING_DELTA ======');
        console.dir(data, { depth: null });
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
        // console.dir(data, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(_provider);
  if (
    'configuration' in llmConfig &&
    (llmConfig as t.OpenAIClientOptions).configuration != null
  ) {
    const openAIConfig = llmConfig as t.OpenAIClientOptions;
    if (openAIConfig.configuration) {
      openAIConfig.configuration.fetch = (
        url: string | URL | Request,
        init?: RequestInit
      ) => {
        console.log('Fetching:', url);
        return fetch(url, init);
      };
    }
  }
  const provider = llmConfig.provider;

  if (provider === Providers.ANTHROPIC) {
    (llmConfig as t.AnthropicClientOptions).clientOptions = {
      defaultHeaders: {
        'anthropic-beta':
          'token-efficient-tools-2025-02-19,output-128k-2025-02-19,prompt-caching-2024-07-31',
      },
    };
  }

  const run = await Run.create<t.IState>({
    runId: uuidv4(),
    graphConfig: {
      type: 'standard',
      llmConfig,
      // tools: [],
      // reasoningKey: 'reasoning',
      instructions:
        'You are a friendly AI assistant. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
    },
    returnContent: true,
    customHandlers,
  });

  const config = {
    runId: uuidv4(),
    configurable: {
      user_id: 'user-123',
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Test 1: Simple message test');

  const userMessage = `hi`;

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
  console.dir(contentParts, { depth: null });
  const { handleLLMEnd, collected } = createMetadataAggregator();
  const titleOptions: t.RunTitleOptions = {
    provider,
    inputText: userMessage,
    contentParts,
    // titleMethod: TitleMethod.STRUCTURED,
    chainOptions: {
      configurable: {
        user_id: 'user-123',
        thread_id: 'conversation-num-1',
      },
      callbacks: [
        {
          handleLLMEnd,
        },
      ],
    },
  };
  if (provider === Providers.ANTHROPIC) {
    titleOptions.clientOptions = {
      model: 'claude-haiku-4-5',
    };
  }
  const titleResult = await run.generateTitle(titleOptions);
  console.log('Collected usage metadata:', collectedUsage);
  console.log('Generated Title:', titleResult);
  console.log('Collected title usage metadata:', collected);
  await sleep(5000);
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

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
