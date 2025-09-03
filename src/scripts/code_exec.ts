// src/scripts/cli.ts
import { config } from 'dotenv';
config();
import { HumanMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import {
  ToolEndHandler,
  ModelEndHandler,
  createMetadataAggregator,
} from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents } from '@/common';
import { Run } from '@/run';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';

const conversationHistory: BaseMessage[] = [];

async function testCodeExecution(): Promise<void> {
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
        // console.log('====== ON_RUN_STEP_DELTA ======');
        // console.dir(data, { depth: null });
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
        console.dir(data, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'message-num-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [new TavilySearchResults(), createCodeExecutionTool()],
      instructions:
        'You are a friendly AI assistant with coding capabilities. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
    },
    returnContent: true,
    customHandlers,
  });

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    run_id?: string;
    streamMode: string;
  } = {
    configurable: {
      provider,
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
    // recursionLimit: 3,
  };

  console.log('Test 1: Sorting Algorithm Comparison');

  const userMessage1 = `
  Hi ${userName} here. I need a Python script that compares different sorting algorithms. Can you write a script that:
  1. Implements three sorting algorithms: Bubble Sort, Insertion Sort, and Merge Sort
  2. Generates three random lists of integers:
     - A small list (20 elements)
     - A medium list (100 elements)
     - A large list (1000 elements)
  3. Applies each sorting algorithm to each list and measures the execution time
  4. Prints a comparison table showing the time taken by each algorithm for each list size
  5. Determines and announces the fastest algorithm for each list size
  Please write the script and then execute it to show the results.
  `;

  conversationHistory.push(new HumanMessage(userMessage1));

  let inputs = {
    messages: conversationHistory,
  };
  const finalContentParts1 = await run.processStream(inputs, config);
  const finalMessages1 = run.getRunMessages();
  if (finalMessages1) {
    conversationHistory.push(...finalMessages1);
  }
  console.log('\n\n====================\n\n');
  console.dir(contentParts, { depth: null });

  console.log('Test 2: Text File Analysis and Processing');

  const userMessage2 = `
  Great job on the sorting algorithms! Now, let's solve a popular LeetCode problem using the Merge Sort algorithm we implemented. The problem is "Merge Intervals". Here's what I need:

  1. Implement a solution to the Merge Intervals problem:
     - Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals.
     - Return an array of the non-overlapping intervals that cover all the intervals in the input.

  2. Use the Merge Sort algorithm as part of the solution to sort the intervals based on their start times.

  3. Implement a function to generate a random list of intervals for testing.

  4. Create test cases:
     - A small case with 5 intervals
     - A medium case with 20 intervals
     - A large case with 100 intervals

  5. Apply your solution to each test case and print:
     - The original intervals
     - The merged intervals
     - The time taken to solve each case

  Please write the script and execute it to demonstrate the results for all three test cases.
  `;

  conversationHistory.push(new HumanMessage(userMessage2));

  inputs = {
    messages: conversationHistory,
  };
  const finalContentParts2 = await run.processStream(inputs, config);
  const finalMessages2 = run.getRunMessages();
  if (finalMessages2) {
    conversationHistory.push(...finalMessages2);
  }
  console.log('\n\n====================\n\n');
  console.dir(finalContentParts2, { depth: null });

  const { handleLLMEnd, collected } = createMetadataAggregator();
  const titleResult = await run.generateTitle({
    provider,
    inputText: userMessage2,
    contentParts,
    chainOptions: {
      callbacks: [
        {
          handleLLMEnd,
        },
      ],
    },
  });
  console.log('Generated Title:', titleResult);
  console.log('Collected metadata:', collected);
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

testCodeExecution().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
