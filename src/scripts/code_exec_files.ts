// src/scripts/code_exec_files.ts
/**
 * Tests automatic session tracking for code execution file persistence.
 * Files created in one execution are automatically available in subsequent executions
 * without the LLM needing to track or pass session_id.
 *
 * Run with: npm run code_exec_files
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
import { Constants, GraphEvents } from '@/common';
import { Run } from '@/run';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';

const conversationHistory: BaseMessage[] = [];

/**
 * Prints session context from the graph for debugging
 */
function printSessionContext(run: Run<t.IState>): void {
  const graph = run.Graph;
  if (!graph) {
    console.log('[Session] No graph available');
    return;
  }

  const session = graph.sessions.get(Constants.EXECUTE_CODE) as
    | t.CodeSessionContext
    | undefined;

  if (!session) {
    console.log('[Session] No session context stored yet');
    return;
  }

  console.log('[Session] Current session context:');
  console.log(`  - session_id: ${session.session_id}`);
  console.log(`  - files: ${JSON.stringify(session.files, null, 2)}`);
  console.log(
    `  - lastUpdated: ${new Date(session.lastUpdated).toISOString()}`
  );
}

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
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        _metadata?: Record<string, unknown>
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
      tools: [createCodeExecutionTool()],
      instructions:
        'You are a friendly AI assistant with coding capabilities. Always address the user by their name.',
      additional_instructions: `The user's name is ${userName} and they are located in ${location}. The current date is ${currentDate}.`,
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  });

  const streamConfig: Partial<RunnableConfig> & {
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
  };

  console.log('\n========== Test 1: Create Project Plan ==========\n');
  console.log(
    'Creating initial file - this establishes the session context.\n'
  );

  const userMessage1 = `
  Hi ${userName} here. We are testing your file capabilities.
  
  1. Create a text file named "project_plan.txt" that contains: "This is a project plan for a new software development project."
  
  Please generate this file so I can review it.
  `;

  conversationHistory.push(new HumanMessage(userMessage1));

  let inputs = {
    messages: conversationHistory,
  };
  await run.processStream(inputs, streamConfig);
  const finalMessages1 = run.getRunMessages();
  if (finalMessages1) {
    conversationHistory.push(...finalMessages1);
  }

  console.log('\n\n========== Session Context After Test 1 ==========\n');
  printSessionContext(run);
  console.dir(contentParts, { depth: null });

  console.log('\n========== Test 2: Edit Project Plan ==========\n');
  console.log(
    'Editing the file from Test 1 - session_id is automatically injected.\n'
  );

  const userMessage2 = `
  Thanks for creating the project plan. Now I'd like you to edit the same plan to:
  
  1. Read the existing project_plan.txt file
  2. Add a new section called "Technology Stack" that contains: "The technology stack for this project includes the following technologies" and nothing more.
  3. Save this as a new file called "project_plan_v2.txt" (remember files are read-only)
  4. Print the contents of both files to verify
`;

  conversationHistory.push(new HumanMessage(userMessage2));

  inputs = {
    messages: conversationHistory,
  };
  await run.processStream(inputs, streamConfig);
  const finalMessages2 = run.getRunMessages();
  if (finalMessages2) {
    conversationHistory.push(...finalMessages2);
  }

  console.log('\n\n========== Session Context After Test 2 ==========\n');
  printSessionContext(run);
  console.dir(contentParts, { depth: null });

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
