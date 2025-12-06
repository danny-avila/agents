// src/scripts/code_exec_ptc.ts
/**
 * Live LLM test for Programmatic Tool Calling (PTC).
 * Run with: npm run code_exec_ptc
 *
 * Tests PTC with a real LLM in the loop, demonstrating:
 * 1. LLM decides when to use PTC
 * 2. LLM writes Python code that calls tools programmatically
 * 3. ToolNode automatically injects programmatic tools
 * 4. Tools filtered by allowed_callers
 */
import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
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
import { createProgrammaticToolCallingTool } from '@/tools/ProgrammaticToolCalling';
import {
  createGetTeamMembersTool,
  createGetExpensesTool,
  createGetWeatherTool,
  createProgrammaticToolRegistry,
} from '@/test/mockTools';

const conversationHistory: BaseMessage[] = [];

async function testProgrammaticToolCalling(): Promise<void> {
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

  // Create mock tool instances
  const teamTool = createGetTeamMembersTool();
  const expensesTool = createGetExpensesTool();
  const weatherTool = createGetWeatherTool();

  // Create special tools
  const codeExecTool = createCodeExecutionTool();
  const ptcTool = createProgrammaticToolCallingTool();

  // Build complete tool list and map
  const allTools = [teamTool, expensesTool, weatherTool, codeExecTool, ptcTool];
  const toolMap = new Map(allTools.map((t) => [t.name, t]));

  // Create tool registry with allowed_callers configuration
  // Only includes business logic tools (not special tools)
  // Special tools (execute_code, PTC) are always bound directly to LLM
  const toolRegistry = createProgrammaticToolRegistry();

  console.log('\n' + '='.repeat(70));
  console.log('Tool Configuration Summary:');
  console.log('='.repeat(70));
  console.log('Total tools:', allTools.length);
  console.log(
    'Programmatic-allowed:',
    Array.from(toolRegistry.values())
      .filter((t) => t.allowed_callers?.includes('code_execution'))
      .map((t) => t.name)
      .join(', ')
  );
  console.log(
    'Direct-callable:',
    Array.from(toolRegistry.values())
      .filter((t) => !t.allowed_callers || t.allowed_callers.includes('direct'))
      .map((t) => t.name)
      .join(', ')
  );
  console.log('='.repeat(70) + '\n');

  const run = await Run.create<t.IState>({
    runId: 'ptc-test-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      agents: [
        {
          agentId: 'default',
          provider: llmConfig.provider,
          clientOptions: llmConfig,
          tools: allTools,
          toolMap,
          toolRegistry,
          instructions:
            'You are a friendly AI assistant with advanced coding capabilities. ' +
            'You have access to team and expense management tools, but ONLY through programmatic code execution. ' +
            'When you need to analyze expenses or process team data, use the programmatic_code_execution tool ' +
            'to write Python code that calls get_team_members(), get_expenses(), and get_weather() functions. ' +
            'These functions are async - use await. Use asyncio.gather() for parallel execution.',
          additional_instructions: `The user's name is ${userName} and they are located in ${location}. Today is ${currentDate}.`,
        },
      ],
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
      thread_id: 'ptc-conversation-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Test 1: Team Expense Analysis with PTC');
  console.log('='.repeat(70) + '\n');

  const userMessage1 = `Hi ${userName}! I need you to analyze our team's expenses. Please:

1. Get the list of all team members
2. For each member, fetch their expense records
3. Calculate the total expenses per member
4. Identify anyone who spent more than $500
5. Show me a summary report

IMPORTANT: Use the programmatic_code_execution tool to do this efficiently. 
Don't call each tool separately - write Python code that orchestrates all the calls!`;

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
  console.log('Content Parts:');
  console.dir(contentParts, { depth: null });

  console.log('\n\n' + '='.repeat(70));
  console.log('Test 2: Conditional Logic and Parallel Execution');
  console.log('='.repeat(70) + '\n');

  const userMessage2 = `Great job! Now let's test some advanced patterns. Please:

1. Check the weather in both San Francisco and New York (in parallel!)
2. Based on which city has better weather (warmer), fetch the team members
3. For the Engineering team members only, calculate their travel expenses
4. Show me the results

Again, use programmatic_code_execution for maximum efficiency. Use asyncio.gather() 
to check both cities' weather at the same time!`;

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
  console.log('Final Content Parts:');
  console.dir(finalContentParts2, { depth: null });

  console.log('\n\n' + '='.repeat(70));
  console.log('Generating conversation title...');
  console.log('='.repeat(70) + '\n');

  const { handleLLMEnd, collected } = createMetadataAggregator();
  const titleResult = await run.generateTitle({
    provider,
    inputText: userMessage1,
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

testProgrammaticToolCalling().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
