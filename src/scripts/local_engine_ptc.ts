/**
 * src/scripts/local_engine_ptc.ts
 *
 * Live exercise of the local engine's programmatic tool calling. Asks
 * the model to use `run_tools_with_code` (Python) to do a multi-step
 * coding workflow that calls `write_file`, `read_file`, and
 * `edit_file` from inside a single Python program. The local engine
 * stands up an in-process loopback HTTP bridge guarded by a per-Run
 * bearer token (verified via `crypto.timingSafeEqual`), and the
 * generated Python sends `x-librechat-bridge-token` on every call.
 *
 * Run with: `npm run local:ptc`
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
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-ptc-'));
  console.log(`[local_engine_ptc] workspace: ${cwd}`);

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
    runId: 'local-engine-ptc-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions:
        'You are a coding assistant. Whenever you need to run multiple file/edit ' +
        'operations together, prefer the `run_tools_with_code` tool with `lang: "py"` ' +
        'over making many separate tool calls — it lets you sequence read_file/write_file/' +
        'edit_file in a single Python program. Always use absolute or workspace-relative paths.',
    },
    toolExecution: {
      engine: 'local',
      local: { cwd, timeoutMs: 60_000 },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-ptc-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi ${userName}. Use the \`run_tools_with_code\` tool with lang: "py" to do all of the following inside a SINGLE Python program (one tool call):\n\n` +
      '1. Call `write_file` to create `notes.md` with the contents `# Notes\\n- first item\\n`.\n' +
      '2. Call `read_file` to read it back.\n' +
      '3. Call `edit_file` to change `first item` to `FIRST item`.\n' +
      '4. Call `read_file` again and print the final contents.\n\n' +
      'Print every intermediate value with `print(...)` so I can see them. Do not make multiple tool calls — everything must happen inside one `run_tools_with_code` invocation. After it returns, summarise what you did in one sentence.'
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
