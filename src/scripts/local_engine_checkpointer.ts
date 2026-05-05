/**
 * src/scripts/local_engine_checkpointer.ts
 *
 * Live demonstration of the file checkpointer. We:
 *
 *   1. Build the local-coding-tool bundle ourselves with
 *      `createLocalCodingToolBundle({ fileCheckpointing: true })` so we
 *      can keep a handle on the checkpointer.
 *   2. Pass those tools to the run as `graphConfig.tools`. We do NOT
 *      use `toolExecution.engine: 'local'` here — that path auto-binds
 *      its own internal tools without exposing a checkpointer back to
 *      us. The point is to show how a host wires checkpointing on top
 *      of the local tools when it cares about rewind.
 *   3. Seed two files with known contents.
 *   4. Ask the model to MUTATE both via `edit_file` / `write_file`.
 *   5. Verify the contents changed on disk.
 *   6. Call `checkpointer.rewind()` and verify the original contents
 *      came back.
 *
 * Run with: `npm run local:checkpointer`
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createLocalCodingToolBundle } from '@/tools/local';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

const FILE_A = 'config.json';
const FILE_B = 'README.md';
const ORIGINAL_A = '{\n  "version": "1.0.0",\n  "name": "demo"\n}\n';
const ORIGINAL_B = '# Demo\n\nThis README is intentionally tiny.\n';

async function main(): Promise<void> {
  const { userName, provider } = await getArgs();
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-cp-'));
  console.log(`[local_engine_checkpointer] workspace: ${cwd}`);

  await writeFile(join(cwd, FILE_A), ORIGINAL_A, 'utf8');
  await writeFile(join(cwd, FILE_B), ORIGINAL_B, 'utf8');

  // Build the local-engine tool bundle with file checkpointing on so
  // we can call `bundle.checkpointer.rewind()` after the run.
  const bundle = createLocalCodingToolBundle({
    cwd,
    fileCheckpointing: true,
    timeoutMs: 30_000,
  });
  if (bundle.checkpointer == null) {
    throw new Error(
      'expected bundle.checkpointer to be defined when fileCheckpointing is true'
    );
  }

  const { contentParts, aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    // Forward ON_RUN_STEP so the aggregator's stepMap is seeded
    // before ON_RUN_STEP_COMPLETED arrives — without this the
    // aggregator logs "No run step or runId found for completed step
    // event" once per tool call (issue #142).
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        const cast = data as unknown as { result: t.ToolEndEvent };
        const tc = (cast.result as { tool_call?: { name?: string } } | undefined)
          ?.tool_call;
        console.log(`====== ON_RUN_STEP_COMPLETED tool=${tc?.name ?? '?'} ======`);
        aggregateContent({ event, data: cast });
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
      handle: (_event: string, data: t.StreamEventData): void => {
        const obj = data as unknown as { name?: string; input?: unknown };
        console.log(
          `====== TOOL_START tool=${obj.name ?? '?'} input=${JSON.stringify(
            obj.input
          )} ======`
        );
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const runConfig: t.RunConfig = {
    runId: 'local-engine-cp-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: bundle.tools,
      instructions:
        'You are a coding assistant with local file tools. Make exactly the changes ' +
        'the user asks for, no more. Use `edit_file` for single-line literal ' +
        'replacements and `write_file` for full overwrites.',
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-cp-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi ${userName}. Please do exactly the following:\n\n` +
      `1. Use \`edit_file\` on \`${FILE_A}\` to replace \`"version": "1.0.0"\` with \`"version": "2.0.0"\`.\n` +
      `2. Use \`write_file\` on \`${FILE_B}\` to overwrite it with the contents:\n` +
      '   "# Demo\\n\\nThis README has been overwritten by the agent.\\n"\n\n' +
      'Confirm both changes succeeded. Do not touch any other files.'
  );
  conversationHistory.push(userMessage);
  console.log('====== USER ======\n' + userMessage.content + '\n');

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log('\n====== FINAL CONTENT PARTS ======');
  console.dir(contentParts, { depth: null });

  // After the run: verify the agent actually changed the files.
  const afterA = await readFile(join(cwd, FILE_A), 'utf8');
  const afterB = await readFile(join(cwd, FILE_B), 'utf8');
  console.log('\n====== AFTER AGENT EDITS ======');
  console.log(`--- ${FILE_A} ---\n${afterA}`);
  console.log(`--- ${FILE_B} ---\n${afterB}`);

  const aChanged = afterA !== ORIGINAL_A;
  const bChanged = afterB !== ORIGINAL_B;
  console.log(`changed ${FILE_A}: ${aChanged}`);
  console.log(`changed ${FILE_B}: ${bChanged}`);

  console.log('\n====== CAPTURED CHECKPOINTS ======');
  console.log(bundle.checkpointer.capturedPaths());

  // Now rewind and verify originals come back.
  const restoredCount = await bundle.checkpointer.rewind();
  console.log(`\n====== AFTER REWIND (restored ${restoredCount} files) ======`);
  const rewoundA = await readFile(join(cwd, FILE_A), 'utf8');
  const rewoundB = await readFile(join(cwd, FILE_B), 'utf8');
  console.log(`--- ${FILE_A} ---\n${rewoundA}`);
  console.log(`--- ${FILE_B} ---\n${rewoundB}`);

  const aBack = rewoundA === ORIGINAL_A;
  const bBack = rewoundB === ORIGINAL_B;
  console.log(`${FILE_A} restored to original: ${aBack}`);
  console.log(`${FILE_B} restored to original: ${bBack}`);

  if (!(aBack && bBack)) {
    console.error('\n[checkpointer] rewind did not restore both files (BUG)');
    process.exitCode = 1;
  } else {
    console.log('\n[checkpointer] all files restored to pre-run state ✔');
  }

  await rm(cwd, { recursive: true, force: true });
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
