/**
 * src/scripts/local_engine.ts
 *
 * Live end-to-end exercise of the local execution engine. Spins up a
 * temporary workspace, points an agent at it via
 * `toolExecution: { engine: 'local' }`, and asks the model to do a
 * small coding task that exercises `write_file`, `read_file`,
 * `edit_file`, `bash`, and `list_directory`. After the run, dumps the
 * workspace contents so you can verify the tools actually mutated
 * disk.
 *
 * Run with: `npm run local`
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, readdir, readFile, rm } from 'fs/promises';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

async function main(): Promise<void> {
  const { userName, provider } = await getArgs();
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-engine-'));
  console.log(`[local_engine] workspace: ${cwd}`);

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
        const cast = data as unknown as { result: t.ToolEndEvent };
        const tc = (cast.result as { tool_call?: { name?: string } } | undefined)
          ?.tool_call;
        console.log(`tool=${tc?.name ?? '<unknown>'}`);
        aggregateContent({ event, data: cast });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
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
    runId: 'local-engine-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions:
        'You are a friendly AI coding assistant with access to local file and shell tools. ' +
        'Always operate inside the configured working directory and prefer the smallest tool ' +
        'that gets the job done (read_file before grep, write_file before bash heredocs, etc.). ' +
        `Address the user by their name (${userName}).`,
    },
    toolExecution: {
      engine: 'local',
      local: {
        cwd,
        // Sandbox is opt-in; leave off for this baseline script. The engine
        // logs a one-time warning when it runs without sandboxing.
        bashAst: 'auto',
        timeoutMs: 30_000,
      },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi, I'm ${userName}. In the current working directory, please do the following step by step:\n\n` +
      '1. Create a file called `greet.py` with a tiny Python function `greet(name)` that returns "Hello, <name>!".\n' +
      '2. Use `bash` to run `python3 -c "from greet import greet; print(greet(\\"world\\"))"` and confirm it prints "Hello, world!".\n' +
      '3. Use `edit_file` to change the greeting to "Hi, <name>!" (only that one literal change).\n' +
      '4. Re-run the same bash command and confirm the new output.\n' +
      '5. Use `list_directory` to show what files exist now.\n\n' +
      'Before exiting, briefly summarise the steps you took.'
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

  console.log('\n====== WORKSPACE ON DISK ======');
  const entries = await readdir(cwd, { withFileTypes: true });
  for (const e of entries) {
    if (!e.isFile()) continue;
    const path = join(cwd, e.name);
    const body = await readFile(path, 'utf8').catch(() => '<binary>');
    console.log(`--- ${e.name} ---\n${body}\n`);
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
