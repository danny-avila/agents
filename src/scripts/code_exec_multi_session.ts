// src/scripts/code_exec_multi_session.ts
/**
 * Tests multi-session file tracking for code execution.
 * Verifies that:
 * 1. Files from multiple executions are accumulated
 * 2. Each file tracks its source session_id
 * 3. Edited/recreated files replace older versions (latest preferred)
 *
 * Run with: npm run code_exec_multi_session
 */
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Constants, GraphEvents } from '@/common';
import { Run } from '@/run';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';

const conversationHistory: BaseMessage[] = [];

/**
 * Prints session context from the graph
 */
function printSessionContext(run: Run<t.IState>, label: string): void {
  const graph = run.Graph;
  if (!graph) {
    console.log(`\n[${label}] No graph available`);
    return;
  }

  const session = graph.sessions.get(Constants.EXECUTE_CODE) as
    | t.CodeSessionContext
    | undefined;

  console.log(`\n========== ${label} ==========`);
  if (!session) {
    console.log('  No session context stored yet');
    return;
  }

  console.log(`  Latest session_id: ${session.session_id}`);
  console.log(`  Files tracked: ${session.files.length}`);
  for (const file of session.files) {
    console.log(`    - ${file.name} (session: ${file.session_id})`);
  }
}

async function testMultiSessionFiles(): Promise<void> {
  const { provider } = await getArgs();

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
  };

  const llmConfig = getLLMConfig(provider);

  const run = await Run.create<t.IState>({
    runId: 'multi-session-test',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [createCodeExecutionTool()],
      instructions: `You are a coding assistant. Execute code exactly as requested.
When asked to create files, use Python and save to /mnt/data/.
When reading files, print their contents.
Be concise in responses.`,
    },
    returnContent: true,
    customHandlers,
  });

  const streamConfig: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    streamMode: string;
  } = {
    configurable: {
      provider,
      thread_id: 'multi-session-test',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // ========== TEST 1: Create first file ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST 1: Create first file (file_a.txt)');
  console.log('='.repeat(60));

  conversationHistory.push(
    new HumanMessage(`
Create a file called "file_a.txt" with the content:
"This is file A, version 1"
Print confirmation when done.
`)
  );

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const messages1 = run.getRunMessages();
  if (messages1) conversationHistory.push(...messages1);

  printSessionContext(run, 'After Test 1');

  // ========== TEST 2: Create second file (different session) ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST 2: Create second file (file_b.txt)');
  console.log('Expecting: Both file_a.txt and file_b.txt tracked');
  console.log('='.repeat(60));

  conversationHistory.push(
    new HumanMessage(`
Create a NEW file called "file_b.txt" with the content:
"This is file B"
Print confirmation when done.
`)
  );

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const messages2 = run.getRunMessages();
  if (messages2) conversationHistory.push(...messages2);

  printSessionContext(run, 'After Test 2');

  // ========== TEST 3: Read BOTH files (verifies accumulation) ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST 3: Read BOTH files from previous executions');
  console.log('This verifies multi-session file accumulation works');
  console.log('='.repeat(60));

  conversationHistory.push(
    new HumanMessage(`
Read and print the contents of BOTH files:
1. file_a.txt
2. file_b.txt

Show me what's in each file.
`)
  );

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const messages3 = run.getRunMessages();
  if (messages3) conversationHistory.push(...messages3);

  printSessionContext(run, 'After Test 3');

  // ========== TEST 4: Edit file_a.txt (verifies latest-wins) ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST 4: Edit file_a.txt (create new version)');
  console.log('Expecting: Old file_a.txt replaced with new version');
  console.log('='.repeat(60));

  conversationHistory.push(
    new HumanMessage(`
Create an UPDATED version of "file_a.txt" with the content:
"This is file A, version 2 - UPDATED"
Print confirmation when done.
`)
  );

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const messages4 = run.getRunMessages();
  if (messages4) conversationHistory.push(...messages4);

  printSessionContext(run, 'After Test 4');

  // ========== TEST 5: Read file_a.txt (verifies latest version) ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST 5: Read file_a.txt to verify it has the UPDATED content');
  console.log('Expected: "version 2 - UPDATED" NOT "version 1"');
  console.log('='.repeat(60));

  conversationHistory.push(
    new HumanMessage(`
Read and print the contents of file_a.txt.
Tell me what version it shows.
`)
  );

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const messages5 = run.getRunMessages();
  if (messages5) conversationHistory.push(...messages5);

  printSessionContext(run, 'Final Session State');

  // ========== SUMMARY ==========
  console.log('\n\n' + '='.repeat(60));
  console.log('TEST SUMMARY');
  console.log('='.repeat(60));

  const finalSession = run.Graph?.sessions.get(Constants.EXECUTE_CODE) as
    | t.CodeSessionContext
    | undefined;

  if (finalSession) {
    const uniqueSessionIds = new Set(
      finalSession.files.map((f) => f.session_id)
    );
    console.log(`\nTotal files tracked: ${finalSession.files.length}`);
    console.log(`Unique session_ids: ${uniqueSessionIds.size}`);
    console.log('\nFiles:');
    for (const file of finalSession.files) {
      console.log(
        `  - ${file.name} (session: ${file.session_id?.slice(0, 20)}...)`
      );
    }

    // Verify expectations
    const fileACount = finalSession.files.filter(
      (f) => f.name === 'file_a.txt'
    ).length;
    const fileBCount = finalSession.files.filter(
      (f) => f.name === 'file_b.txt'
    ).length;

    console.log('\n✓ Checks:');
    console.log(`  file_a.txt count: ${fileACount} (expected: 1, latest wins)`);
    console.log(`  file_b.txt count: ${fileBCount} (expected: 1)`);

    if (fileACount === 1 && fileBCount === 1) {
      console.log('\n✅ All tests passed! Multi-session tracking works.');
    } else {
      console.log('\n❌ Test failed - unexpected file counts');
    }
  }
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

testMultiSessionFiles().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
