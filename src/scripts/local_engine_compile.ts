/**
 * src/scripts/local_engine_compile.ts
 *
 * Live demonstration of the cheap typecheck integration shipped as
 * an alternative to full LSP:
 *
 *   1. Post-edit syntax check (`local.postEditSyntaxCheck: 'auto'`)
 *      runs `node --check` after every edit_file / write_file on
 *      `.js`/`.mjs` files — the model sees the parse error in the
 *      tool result and self-corrects on the next turn.
 *
 *   2. `compile_check` tool runs the project's standard typecheck
 *      command (auto-detected from project markers, or supplied via
 *      the `command` arg) and surfaces the result.
 *
 * The script seeds a tiny package.json + tsconfig + index.ts in a
 * temp dir, then asks the model to:
 *   (a) `write_file` an intentionally broken `index.js`,
 *   (b) call `compile_check` to see the error,
 *   (c) fix the file via `edit_file`,
 *   (d) call `compile_check` again to confirm green.
 *
 * Run with: `npm run local:compile`
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm, symlink, writeFile } from 'fs/promises';
import { resolve } from 'path';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
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
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-compile-'));
  console.log(`[local_engine_compile] workspace: ${cwd}`);

  // Seed a tiny TypeScript project so `compile_check` can actually
  // resolve a real toolchain (tsc). We include a healthy index.ts
  // baseline; the LLM will be asked to add a broken file alongside.
  await writeFile(
    join(cwd, 'tsconfig.json'),
    JSON.stringify(
      {
        compilerOptions: {
          target: 'ES2020',
          module: 'commonjs',
          strict: true,
          noEmit: true,
          skipLibCheck: true,
        },
        include: ['*.ts'],
      },
      null,
      2
    ),
    'utf8'
  );
  await writeFile(
    join(cwd, 'index.ts'),
    'export function add(a: number, b: number): number {\n  return a + b;\n}\n',
    'utf8'
  );

  // Make tsc available in the temp workspace by symlinking the
  // surrounding repo's node_modules. This avoids a slow first-run
  // npm install while the model waits, and lets `npx --no-install
  // tsc` resolve typescript instantly. `process.cwd()` is the
  // package root when running via `npm run`. We fall back
  // gracefully on platforms where the symlink isn't possible.
  const repoNodeModules = resolve(process.cwd(), 'node_modules');
  try {
    await symlink(repoNodeModules, join(cwd, 'node_modules'), 'dir');
  } catch (err) {
    console.warn(
      '[local_engine_compile] could not symlink node_modules:',
      (err as Error).message
    );
  }
  await writeFile(
    join(cwd, 'package.json'),
    JSON.stringify({ name: 'lc-local-compile-demo', private: true }, null, 2),
    'utf8'
  );

  const { aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const obj = data as unknown as { name?: string; input?: unknown };
        const inputStr = JSON.stringify(obj.input);
        console.log(
          `====== TOOL_START tool=${obj.name ?? '?'} input=${
            inputStr.length > 200 ? inputStr.slice(0, 200) + '…' : inputStr
          } ======`
        );
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const runConfig: t.RunConfig = {
    runId: 'local-engine-compile-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions:
        `You are ${userName}'s coding assistant. The host has wired a ` +
        'post-edit syntax check (you will see `[syntax-check warning ...]` blocks ' +
        'in tool results) and the `compile_check` tool. Use them to verify your ' +
        'edits. When a syntax check warns you of an error, FIX IT before doing ' +
        'anything else. Keep responses tight.',
    },
    toolExecution: {
      engine: 'local',
      local: {
        cwd,
        postEditSyntaxCheck: 'auto',
        timeoutMs: 30_000,
      },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-compile-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi ${userName}. Here's the workflow. The workspace is a tiny TypeScript project (tsconfig.json + index.ts).\n\n` +
      '1. First, exercise the post-edit syntax check. Use `write_file` to create `broken.js` with EXACTLY this content (it is intentionally broken):\n' +
      '```\n' +
      'function broken(a,) {\n' +
      '  return a +;\n' +
      '}\n' +
      '```\n' +
      '2. After the write, the post-edit syntax check should warn you with `[syntax-check warning ...]`. Read the warning carefully and fix the file via `edit_file` or `write_file`. The smallest valid fix is fine.\n\n' +
      '3. Now exercise `compile_check`. Use `write_file` to create `broken.ts` with EXACTLY this content (it has a TYPE error, not a syntax error — `tsc` will catch it):\n' +
      '```\n' +
      'export const x: number = "not a number";\n' +
      '```\n' +
      '4. Call `compile_check` (no args). Anthropic\'s tsc should fail with a type error.\n' +
      '5. Fix the type error in `broken.ts` (e.g. change the literal to a number) via `edit_file`.\n' +
      '6. Call `compile_check` again to confirm green.\n\n' +
      'Tell me what each `compile_check` call reported. Be concise.'
  );
  conversationHistory.push(userMessage);
  console.log('====== USER ======\n' + userMessage.content + '\n');

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  let sawSyntaxWarning = false;
  let compileCheckCalls = 0;
  let compileCheckFailedAtLeastOnce = false;
  let compileCheckPassedAtLeastOnce = false;
  console.log('\n====== TOOL MESSAGES IN HISTORY ======');
  for (const msg of conversationHistory) {
    if (msg instanceof ToolMessage) {
      const content =
        typeof msg.content === 'string'
          ? msg.content
          : JSON.stringify(msg.content);
      const head = content.slice(0, 240);
      console.log(`- ${msg.name}: ${head.replace(/\n/g, ' ⏎ ')}`);
      if (content.includes('[syntax-check')) sawSyntaxWarning = true;
      if (msg.name === 'compile_check') {
        compileCheckCalls += 1;
        if (content.includes('PASSED')) compileCheckPassedAtLeastOnce = true;
        if (content.includes('FAILED')) compileCheckFailedAtLeastOnce = true;
      }
    }
  }

  console.log('\n====== ASSISTANT FINAL TEXT ======');
  const lastAssistant = [...conversationHistory]
    .reverse()
    .find((m) => m._getType() === 'ai');
  if (lastAssistant) {
    const c = lastAssistant.content;
    console.log(
      typeof c === 'string'
        ? c
        : Array.isArray(c)
          ? c
            .map((b) => ('text' in b ? b.text : `<${b.type}>`))
            .join(' ')
          : JSON.stringify(c)
    );
  }

  console.log('\n====== ASSERTIONS ======');
  console.log(
    `post-edit syntax-check warning observed: ${sawSyntaxWarning} ${
      sawSyntaxWarning ? '✔' : '✖'
    }`
  );
  console.log(
    `compile_check called ${compileCheckCalls} time(s); FAILED-then-PASSED cycle observed: ${
      compileCheckFailedAtLeastOnce && compileCheckPassedAtLeastOnce
    } ${compileCheckFailedAtLeastOnce && compileCheckPassedAtLeastOnce ? '✔' : '✖'}`
  );

  if (
    !sawSyntaxWarning ||
    !(compileCheckFailedAtLeastOnce && compileCheckPassedAtLeastOnce)
  ) {
    process.exitCode = 1;
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
