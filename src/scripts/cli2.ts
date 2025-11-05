/* eslint-disable no-console */
// src/scripts/cli2.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler } from '@/stream';
import { TestLLMStreamHandler } from '@/events';

import { getArgs } from '@/scripts/args';
import { Run } from '@/run';
import { GraphEvents } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';

const conversationHistory: BaseMessage[] = [];

async function executePersonalizedQuerySuite(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();

  const customHandlers = {
    [GraphEvents.LLM_STREAM]: new TestLLMStreamHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.LLM_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== LLM_START ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.LLM_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== LLM_END ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (_event: string, _data: t.StreamEventData): void => {
        console.log('====== CHAT_MODEL_START ======');
        console.dir(_data, { depth: null });
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event: string, _data: t.StreamEventData): void => {
        console.log('====== CHAT_MODEL_END ======');
        console.dir(_data, { depth: null });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.TOOL_END]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        console.log('====== TOOL_END ======');
        console.dir(data, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'test-run-id',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [],
    },
    customHandlers,
  });

  const sessionConfig = {
    configurable: {
      provider,
      thread_id: `${userName}-session-${Date.now()}`,
      instructions: `You are a knowledgeable and friendly AI assistant. Tailor your responses to ${userName}'s interests in ${location}.`,
      additional_instructions: `Ensure each topic is thoroughly researched. Today is ${currentDate}. Maintain a warm, personalized tone throughout.`,
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log(`Initiating personalized query suite for ${userName}`);

  const queryTopics = [
    {
      task: 'current weather',
      description: 'Provide a detailed weather forecast',
    },
    {
      task: 'popular tourist attraction',
      description: 'Describe a notable sight',
    },
    {
      task: 'upcoming events',
      description: 'List major events or festivals this week',
    },
    // { task: "famous local dish", description: "Share a recipe for a regional specialty" },
    // { task: "local humor", description: "Tell a joke related to the area or findings" }
  ];

  const userPrompt = `
  Greetings! I'm ${userName}, currently in ${location}. Today's date is ${currentDate}.
  I'm seeking information on various aspects of ${location}. Please address the following:

  ${queryTopics.map((topic, index) => `${index + 1}. ${topic.description} in ${location}.`).join('\n  ')}

  For each topic, conduct a separate search to ensure accuracy and depth.
  In your response, please address me as ${userName} and maintain a friendly, informative tone.
  `;

  conversationHistory.push(new HumanMessage(userPrompt));

  const runInput = {
    messages: conversationHistory,
  };

  const contentParts = await run.processStream(runInput, sessionConfig);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
    console.log("AI Assistant's Response:");
    console.dir(conversationHistory, { depth: null });
  }
}

executePersonalizedQuerySuite().catch((error) => {
  console.error('An error occurred during the query suite execution:', error);
  console.log('Final conversation state:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
