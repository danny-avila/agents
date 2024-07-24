import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
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
import type { RunnableConfig } from '@langchain/core/runnables';
import * as readline from 'readline';

const conversationHistory: BaseMessage[] = [];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function promptUser(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer);
    });
  });
}

async function interactiveConversation(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();
  
  const customHandlers = {
    [GraphEvents.LLM_STREAM]: new LLMStreamHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.LLM_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== LLM_START ======');
        console.dir(data, { depth: null });
      }
    },
    [GraphEvents.LLM_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== LLM_END ======');
        console.dir(data, { depth: null });
      }
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (_event: string, _data: t.StreamEventData): void => {
        console.log('====== CHAT_MODEL_START ======');
        console.dir(_data, { depth: null });
      }
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event: string, _data: t.StreamEventData): void => {
        console.log('====== CHAT_MODEL_END ======');
        console.dir(_data, { depth: null });
      }
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      }
    },
    [GraphEvents.TOOL_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== TOOL_END ======');
        console.dir(data, { depth: null });
      }
    },
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

  const sessionConfig: Partial<RunnableConfig> & { version: 'v2' | 'v1' } = {
    configurable: {
      provider,
      thread_id: `${userName}-session-${Date.now()}`,
      instructions: `You are a knowledgeable and friendly AI assistant. Tailor your responses to ${userName}'s interests in ${location}.`,
      additional_instructions: `Today is ${currentDate}. Use the tools available when necessary to provide accurate and up-to-date information. Maintain a warm, personalized tone throughout.`
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log(`Starting interactive conversation for ${userName} in ${location}`);

  // Initial system message
  const systemMessage = new HumanMessage(`Hello! I'm ${userName}, currently in ${location}. Today's date is ${currentDate}. I'm looking forward to our conversation!`);
  conversationHistory.push(systemMessage);

  while (true) {
    const userInput = await promptUser("You: ");
    if (userInput.toLowerCase() === 'exit') {
      console.log("Ending conversation. Goodbye!");
      rl.close();
      break;
    }

    const userMessage = new HumanMessage(userInput);
    conversationHistory.push(userMessage);

    const processorInput: t.IState = {
      messages: conversationHistory,
    };

    try {
      const aiResponse = await processor.processStream(processorInput, sessionConfig);
      console.log("Debug - AI Response:", aiResponse);
      console.log("Debug - AI Response Type:", typeof aiResponse);
      if (aiResponse) {
        console.log("Debug - AI Content Type:", typeof aiResponse.content);
        conversationHistory.push(aiResponse);
        if (typeof aiResponse.content === 'string') {
          console.log("AI: " + aiResponse.content);
        } else if (aiResponse.content && typeof aiResponse.content === 'object') {
          console.log("AI: " + JSON.stringify(aiResponse.content, null, 2));
        } else {
          console.log("AI: [Unable to display response]");
        }
      } else {
        console.log("AI: [No response]");
      }
    } catch (error) {
      console.error("An error occurred during processing:", error);
    }
  }
}

interactiveConversation().catch((error) => {
  console.error("An error occurred during the conversation:", error);
  console.log("Final conversation state:");
  console.dir(conversationHistory, { depth: null });
  rl.close();
  process.exit(1);
});