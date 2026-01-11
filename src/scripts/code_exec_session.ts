// src/scripts/code_exec_session.ts
/**
 * Test script for automatic session tracking in code execution tools.
 *
 * This tests the automatic session_id injection feature where:
 * 1. First code execution generates files and returns a session_id
 * 2. Session context is stored in Graph.sessions
 * 3. Subsequent code executions automatically have access to previous files
 *    without the LLM needing to explicitly pass session_id
 *
 * Run with: npm run code_exec_session
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
 * Prints a formatted section header for test output
 */
function printSection(title: string): void {
  console.log('\n' + '='.repeat(60));
  console.log(` ${title}`);
  console.log('='.repeat(60) + '\n');
}

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

async function testAutomaticSessionTracking(): Promise<void> {
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
        _metadata?: Record<string, unknown>
      ): void => {
        console.log('====== TOOL_START ======');
        console.dir(data, { depth: null });
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'session-tracking-test-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [createCodeExecutionTool()],
      instructions: `You are an AI assistant testing automatic file persistence.
When writing Python code:
- Use print() for all outputs
- Files from previous executions are automatically available in /mnt/data/
- Files are READ-ONLY; write modifications to NEW filenames
- IMPORTANT: Do NOT include session_id in your tool calls - it's handled automatically.`,
      additional_instructions: `User: ${userName}, Location: ${location}, Date: ${currentDate}.`,
    },
    returnContent: true,
    customHandlers,
  });

  const streamConfig: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    run_id?: string;
    streamMode: string;
  } = {
    configurable: {
      provider,
      thread_id: 'session-tracking-test',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // =========================================================================
  // Test 1: Create initial file (establishes session)
  // =========================================================================
  printSection('Test 1: Create Initial File');
  console.log(
    'This test creates a file, which should establish a session context.\n'
  );

  const userMessage1 = `
Create a Python file that writes a simple JSON config file named "app_config.json" with the following content:
{
  "app_name": "TestApp",
  "version": "1.0.0",
  "debug": true
}

After writing, print the contents to confirm it was created correctly.
`;

  conversationHistory.push(new HumanMessage(userMessage1));
  await run.processStream({ messages: conversationHistory }, streamConfig);

  const finalMessages1 = run.getRunMessages();
  if (finalMessages1) {
    conversationHistory.push(...finalMessages1);
  }

  printSection('Session Context After Test 1');
  printSessionContext(run);

  // =========================================================================
  // Test 2: Access previously created file (uses automatic session injection)
  // =========================================================================
  printSection('Test 2: Access Previous File (Automatic Session)');
  console.log('This test reads the file created in Test 1.');
  console.log(
    'The LLM does NOT need to provide session_id - it should be injected automatically.\n'
  );

  const userMessage2 = `
Now read the app_config.json file that was just created and:
1. Print its contents
2. Confirm the version is "1.0.0"

Note: You should be able to access this file from the previous execution automatically.
`;

  conversationHistory.push(new HumanMessage(userMessage2));
  await run.processStream({ messages: conversationHistory }, streamConfig);

  const finalMessages2 = run.getRunMessages();
  if (finalMessages2) {
    conversationHistory.push(...finalMessages2);
  }

  printSection('Session Context After Test 2');
  printSessionContext(run);

  // =========================================================================
  // Test 3: Modify file (write to new filename)
  // =========================================================================
  printSection('Test 3: Modify File (Write to New Filename)');
  console.log(
    'This test modifies the config by reading the old file and writing a new one.\n'
  );

  const userMessage3 = `
Read app_config.json, update the version to "2.0.0" and debug to false, 
then save it as "app_config_v2.json". Print both the old and new contents.
`;

  conversationHistory.push(new HumanMessage(userMessage3));
  await run.processStream({ messages: conversationHistory }, streamConfig);

  const finalMessages3 = run.getRunMessages();
  if (finalMessages3) {
    conversationHistory.push(...finalMessages3);
  }

  printSection('Session Context After Test 3');
  printSessionContext(run);

  // =========================================================================
  // Summary
  // =========================================================================
  printSection('Test Summary');
  console.log('The automatic session tracking feature should have:');
  console.log('1. Stored the session_id after the first code execution');
  console.log('2. Automatically injected it into subsequent executions');
  console.log('3. Accumulated file references across all executions');
  console.log('\nCheck the session context output above to verify.\n');

  // Generate title
  const { handleLLMEnd, collected } = createMetadataAggregator();
  const titleResult = await run.generateTitle({
    provider,
    inputText: 'Testing automatic session tracking for code execution',
    contentParts,
    chainOptions: {
      callbacks: [{ handleLLMEnd }],
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

testAutomaticSessionTracking().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
