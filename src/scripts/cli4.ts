/* eslint-disable no-console */
// src/scripts/cli.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ModelEndHandler, ToolEndHandler } from '@/events';
import { ChatModelStreamHandler } from '@/stream';

import { getArgs } from '@/scripts/args';
import { Run } from '@/run';
import { GraphEvents, Callback, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';

const conversationHistory: BaseMessage[] = [];
async function testStandardStreaming(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      },
    },
    // [GraphEvents.LLM_STREAM]: new LLMStreamHandler(),
    // [GraphEvents.LLM_START]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== LLM_START ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
    // [GraphEvents.LLM_END]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== LLM_END ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
    /*
    [GraphEvents.CHAIN_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== CHAIN_START ======');
        // console.dir(data, { depth: null });
      }
    },
    [GraphEvents.CHAIN_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== CHAIN_END ======');
        // console.dir(data, { depth: null });
      }
    },
    */
    // [GraphEvents.CHAT_MODEL_START]: {
    //   handle: (_event: string, _data: t.StreamEventData): void => {
    //     console.log('====== CHAT_MODEL_START ======');
    //     console.dir(_data, { depth: null });
    //     // Intentionally left empty
    //   }
    // },
  };

  // const llmConfig = getLLMConfig(provider);
  let llmConfig = getLLMConfig(Providers.OPENAI);

  const graphConfig: t.LegacyGraphConfig = {
    type: 'standard',
    llmConfig,
    tools: [],
    instructions:
      'You are a friendly AI assistant. Always address the user by their name.',
    additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
  };

  let run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig,
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

  console.log(' Test 1: OpenAI Tool Usage');

  // conversationHistory.push(new HumanMessage(`Hi I'm ${userName}.`));
  conversationHistory.push(
    new HumanMessage(`search for good sunrise hikes near ${location}
then search weather in ${location} for today which is ${currentDate}`)
  );
  let inputs = {
    messages: conversationHistory,
  };
  const contentParts = await run.processStream(
    inputs,
    config
    //   {
    //   [Callback.TOOL_START]: (graph, ...args) => {
    //       console.log('TOOL_START callback');
    //   },
    //   [Callback.TOOL_END]: (graph, ...args) => {
    //       console.log('TOOL_END callback');
    //   },
    // }
  );
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log(' Test 2: Anthropic Follow-up Response');

  // const userMessage = `
  // Make a search for the weather in ${location} today, which is ${currentDate}.
  // Make sure to always refer to me by name.
  // After giving me a thorough summary, tell me a joke about the weather forecast we went over.
  // `;
  const userMessage = `Thanks!`;

  conversationHistory.push(new HumanMessage(userMessage));

  inputs = {
    messages: conversationHistory,
  };

  llmConfig = getLLMConfig(Providers.ANTHROPIC);
  graphConfig.llmConfig = llmConfig;
  config.configurable.provider = Providers.ANTHROPIC;

  run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig,
    customHandlers,
  });

  const contentParts2 = await run.processStream(inputs, config);
  const finalMessages2 = run.getRunMessages();
  if (finalMessages2) {
    conversationHistory.push(...finalMessages2);
    // console.dir(conversationHistory, { depth: null });
  }
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
