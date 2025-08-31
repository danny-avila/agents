/* eslint-disable no-console */
// src/scripts/supervised.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';

const conversationHistory: BaseMessage[] = [];

async function testSupervised(): Promise<void> {
  const { userName, location, provider, currentDate } = await getArgs();
  const { contentParts, aggregateContent } = createContentAggregator();

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
        aggregateContent({ event, data: data as unknown as { result: t.ToolEndEvent } });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData): void => {
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
  } as t.RunConfig['customHandlers'];

  // Base provider determines initial model; router can override per stage
  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'supervised-run-id',
    graphConfig: {
      type: 'supervised',
      llmConfig,
      tools: [new TavilySearchResults()],
      routerEnabled: true,
      routingPolicies: [
        { stage: 'reasoning', model: Providers.ANTHROPIC },
        { stage: 'summarize', model: Providers.OPENAI },
      ],
      // Per-stage model configs ensure correct provider/model for each stage
      models: {
        reasoning: {
          provider: Providers.ANTHROPIC,
          model: 'claude-3-5-sonnet-20240620',
          streaming: true,
          streamUsage: true,
        },
        summarize: {
          provider: Providers.OPENAI,
          model: 'gpt-4o',
          streaming: true,
          streamUsage: true,
        },
      },
      instructions:
        'You are a helpful assistant. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
    },
    returnContent: true,
    customHandlers,
  });

  const configV2 = {
    configurable: {
      provider, // initial provider
      thread_id: 'supervised-conversation-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // Turn 1
  conversationHistory.push(new HumanMessage(`Hi, I'm ${userName}.`));
  await run.processStream({ messages: conversationHistory }, configV2);
  const fm1 = run.getRunMessages();
  if (fm1) conversationHistory.push(...fm1);

  // Turn 2 triggers search + summary across router stages
  const userMessage = `
  Make a search for the weather in ${location} today, which is ${currentDate}.
  Provide a concise summary and end with a short joke.
  `;
  conversationHistory.push(new HumanMessage(userMessage));
  await run.processStream({ messages: conversationHistory }, configV2);
  const fm2 = run.getRunMessages();
  if (fm2) {
    conversationHistory.push(...fm2);
    console.dir(conversationHistory, { depth: null });
  }
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});

testSupervised().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});


