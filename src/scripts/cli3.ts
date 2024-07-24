/* eslint-disable no-console */
// src/scripts/cli.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage, AIMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import type * as t from '@/types';
import {
  ChatModelStreamHandler,
  LLMStreamHandler,
} from '@/stream';

import { getArgs } from '@/scripts/args';
import { Processor } from '@/processor';
import { GraphEvents } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';

const conversationHistory: BaseMessage[] = [];
async function testStandardStreaming(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();
  const customHandlers = {
    // [GraphEvents.LLM_STREAM]: new LLMStreamHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP ======');
        console.dir(data, { depth: null });
      }
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        console.dir(data, { depth: null });
      }
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
      }
    },
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
    // [GraphEvents.CHAT_MODEL_END]: {
    //   handle: (_event: string, _data: t.StreamEventData): void => {
    //     console.log('====== CHAT_MODEL_END ======');
    //     console.dir(_data, { depth: null });
    //     // Intentionally left empty
    //   }
    // },
    // [GraphEvents.TOOL_START]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== TOOL_START ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
    // [GraphEvents.TOOL_END]: {
    //   handle: (_event: string, data: t.StreamEventData): void => {
    //     console.log('====== TOOL_END ======');
    //     console.dir(data, { depth: null });
    //   }
    // },
  };

  const llmConfig = getLLMConfig(provider);

  const processor = await Processor.create<t.IState>({
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [new TavilySearchResults()],
    },
    customHandlers,
  });

  const config = {
    configurable: {
      provider,
      thread_id: 'conversation-num-1',
      instructions: 'You are a friendly AI assistant. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}.`
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log(' Test 1: Initial greeting');

  conversationHistory.push(new HumanMessage(`Hi I'm ${userName}.`));
  let inputs = {
    messages: conversationHistory,
  };
  const finalMessage = await processor.processStream(inputs, config);
  if (finalMessage) {
    conversationHistory.push(finalMessage);
  }

  console.log(' Test 2: Weather query');

  const userMessage = `
  Make a search for the weather in ${location} today, which is ${currentDate}.
  Make sure to always refer to me by name.
  After giving me a thorough summary, tell me a joke about the weather forecast we went over.
  `;

  conversationHistory.push(new HumanMessage(userMessage));

  inputs = {
    messages: conversationHistory,
  };
  const finalMessage2 = await processor.processStream(inputs, config);
  if (finalMessage2) {
    conversationHistory.push(finalMessage2);
    console.dir(conversationHistory, { depth: null });
  }
}

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
