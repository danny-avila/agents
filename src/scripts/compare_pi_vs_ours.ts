/**
 * src/scripts/compare_pi_vs_ours.ts
 *
 * Side-by-side runs: pi-mono's `pi` CLI vs our local engine, same
 * task, same model, two parallel temp workspaces. We track:
 *
 *   - tool calls (name + args length, ordered)
 *   - wall-clock time
 *   - total Anthropic input/output tokens (when reported)
 *   - whether the final on-disk state matches the expected outcome
 *
 * The tasks intentionally probe areas where we expect the local
 * engine to behave differently:
 *
 *   T1 simple-edit       — both should one-shot
 *   T2 fuzzy-edit        — model emits an `oldText` with off-by-
 *                          whitespace; our `editStrategies` chain
 *                          should recover without re-reading;
 *                          pi should also handle it (its edit tool
 *                          has a similar fallback chain)
 *   T3 syntax-error-fix  — pre-seed broken JS; ours surfaces the
 *                          parse error in the write_file tool result
 *                          via post-edit syntax check; pi has to read
 *                          the file (or run bash node --check) to
 *                          notice
 *
 * Run: PI_BIN=path/to/cli.js npm run compare:pi
 * Defaults to ~/Projects/pi-mono/packages/coding-agent/dist/cli.js.
 */
import { config } from 'dotenv';
config();
import { spawn } from 'child_process';
import { homedir, tmpdir } from 'os';
import { join, resolve } from 'path';
import {
  cp,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from 'fs/promises';
import { performance } from 'perf_hooks';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const PROVIDER = Providers.ANTHROPIC;
const MODEL = 'claude-sonnet-4-5';
const PI_BIN =
  process.env.PI_BIN ??
  resolve(
    homedir(),
    'Projects/pi-mono/packages/coding-agent/dist/cli.js'
  );

interface Task {
  name: string;
  description: string;
  /** Files seeded into the workspace before the run. */
  seed: Record<string, string>;
  /** Prompt sent to both agents. */
  prompt: string;
  /** Function that returns true if the workspace ended in the right state. */
  verify: (cwd: string) => Promise<{ ok: boolean; detail: string }>;
  /** Optional pre-run hook (e.g. symlink node_modules so `tsc` is available). */
  setup?: (cwd: string) => Promise<void>;
}

interface ToolCallObservation {
  name: string;
  argsBytes: number;
  isError: boolean;
}

interface RunOutcome {
  toolCalls: ToolCallObservation[];
  wallMs: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  cost: number;
  finalAssistant: string;
  errored: boolean;
  errorMessage?: string;
}

const TASKS: Task[] = [
  {
    name: 'T1 simple-edit',
    description: 'Single literal substitution in an existing file.',
    seed: {
      'greet.py':
        'def greet(name):\n    return f"Hello, {name}!"\n',
    },
    prompt:
      'Edit greet.py: change the greeting from "Hello" to "Hi". ' +
      'Keep the rest of the file identical. Reply with "done" when finished.',
    verify: async (cwd) => {
      const text = await readFile(join(cwd, 'greet.py'), 'utf8').catch(
        () => ''
      );
      const ok = text.includes('"Hi, {name}!"') && !text.includes('Hello,');
      return { ok, detail: ok ? '' : `actual: ${JSON.stringify(text)}` };
    },
  },
  {
    name: 'T2 fuzzy-edit',
    description:
      'Original file has trailing whitespace + tabs; the model is asked to make a literal change without seeing the trailing whitespace.',
    seed: {
      // trailing spaces are intentional here
      'config.ts':
        'export const config = {  \n' +
        '\tport: 3000,\n' +
        '\thost: "localhost",  \n' +
        '};\n',
    },
    prompt:
      'In config.ts, change the port from 3000 to 4242. The file may have ' +
      'unusual whitespace; do the smallest correct change. Reply with "done".',
    verify: async (cwd) => {
      const text = await readFile(join(cwd, 'config.ts'), 'utf8').catch(
        () => ''
      );
      const ok = /port:\s*4242/.test(text) && !/3000/.test(text);
      return { ok, detail: ok ? '' : `actual:\n${text}` };
    },
  },
  {
    name: 'T4 type-error-fix-loop',
    description:
      'Pre-seeded TS file with a type error in a tiny tsconfig project. Ours can call `compile_check`; pi can run `npx tsc --noEmit` via bash.',
    seed: {
      'tsconfig.json': JSON.stringify(
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
      'package.json': JSON.stringify(
        { name: 'lc-compare-t4', private: true },
        null,
        2
      ),
      'broken.ts':
        'export const port: number = "not a number";\n',
    },
    prompt:
      'broken.ts has a type error. Fix it so the project typechecks cleanly. ' +
      'After fixing, verify by running the project\'s typecheck (or `compile_check` if available). ' +
      'Reply with "done".',
    verify: async (cwd) => {
      const text = await readFile(join(cwd, 'broken.ts'), 'utf8').catch(
        () => ''
      );
      const ok =
        /port:\s*number\s*=\s*\d/.test(text) && !/"not a number"/.test(text);
      return { ok, detail: ok ? '' : `actual: ${text}` };
    },
    setup: symlinkRepoNodeModules,
  },
  {
    name: 'T3 syntax-error-fix',
    description:
      'Pre-seeded broken JS file. Ours surfaces the parse error in the write_file/edit_file tool result; pi has to discover it via bash/read.',
    seed: {
      'broken.js':
        'function add(a, b) {\n  return a + ;\n}\nconsole.log(add(1, 2));\n',
    },
    prompt:
      'broken.js is syntactically invalid. Fix it so `node --check broken.js` passes. ' +
      'The intended behaviour is that add(1, 2) prints 3. Reply with "done".',
    verify: async (cwd) => {
      const text = await readFile(join(cwd, 'broken.js'), 'utf8').catch(
        () => ''
      );
      // Does not include the broken token
      const cleaned = !/return\s+a\s*\+\s*;/.test(text);
      // Should still console.log the result
      const hasLog = /console\.log/.test(text);
      const ok = cleaned && hasLog;
      return { ok, detail: ok ? '' : `actual:\n${text}` };
    },
  },
];

/* ------------------------------------------------------------------ */
/* pi runner                                                           */
/* ------------------------------------------------------------------ */

async function runPi(task: Task, cwd: string): Promise<RunOutcome> {
  const start = performance.now();
  const args = [
    PI_BIN,
    '--print',
    '--mode',
    'json',
    '--no-session',
    '--provider',
    PROVIDER,
    '--model',
    MODEL,
    task.prompt,
  ];
  return new Promise<RunOutcome>((resolveOutcome) => {
    const child = spawn('node', args, {
      cwd,
      env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });

    child.on('close', (code) => {
      const wallMs = performance.now() - start;
      if (code !== 0) {
        resolveOutcome({
          toolCalls: [],
          wallMs,
          inputTokens: 0,
          outputTokens: 0,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          cost: 0,
          finalAssistant: '',
          errored: true,
          errorMessage:
            stderr.trim().slice(-500) || `exit ${code ?? 'unknown'}`,
        });
        return;
      }

      const toolCalls: ToolCallObservation[] = [];
      let inputTokens = 0;
      let outputTokens = 0;
      let cacheReadTokens = 0;
      let cacheWriteTokens = 0;
      let cost = 0;
      let finalAssistant = '';

      for (const line of stdout.split('\n')) {
        if (line === '') continue;
        let event: { type?: string; message?: unknown };
        try {
          event = JSON.parse(line);
        } catch {
          continue;
        }
        if (event.type === 'message_end') {
          const m = event.message as {
            role?: string;
            content?: Array<{
              type?: string;
              name?: string;
              arguments?: unknown;
              text?: string;
            }>;
            usage?: {
              input?: number;
              output?: number;
              cost?: { total?: number };
            };
          };
          if (m.role === 'assistant') {
            const usage = m.usage as
              | {
                  input?: number;
                  output?: number;
                  cacheRead?: number;
                  cacheWrite?: number;
                  cost?: { total?: number };
                }
              | undefined;
            inputTokens += usage?.input ?? 0;
            outputTokens += usage?.output ?? 0;
            cacheReadTokens += usage?.cacheRead ?? 0;
            cacheWriteTokens += usage?.cacheWrite ?? 0;
            cost += usage?.cost?.total ?? 0;
            for (const block of m.content ?? []) {
              if (block.type === 'toolCall' && block.name != null) {
                toolCalls.push({
                  name: block.name,
                  argsBytes: JSON.stringify(block.arguments ?? {}).length,
                  isError: false,
                });
              }
              if (block.type === 'text' && block.text != null) {
                finalAssistant = block.text;
              }
            }
          }
          if (m.role === 'toolResult') {
            const tr = m as unknown as { isError?: boolean };
            if (tr.isError && toolCalls.length > 0) {
              toolCalls[toolCalls.length - 1].isError = true;
            }
          }
        }
      }

      resolveOutcome({
        toolCalls,
        wallMs,
        inputTokens,
        outputTokens,
        cacheReadTokens,
        cacheWriteTokens,
        cost,
        finalAssistant,
        errored: false,
      });
    });
  });
}

/* ------------------------------------------------------------------ */
/* Our local-engine runner                                             */
/* ------------------------------------------------------------------ */

async function runOurs(task: Task, cwd: string): Promise<RunOutcome> {
  const start = performance.now();
  const conversation: BaseMessage[] = [];
  const observedToolCalls: ToolCallObservation[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let cacheWriteTokens = 0;

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
  };

  const llmConfig = getLLMConfig(PROVIDER);
  const runConfig: t.RunConfig = {
    runId: `compare-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      llmConfig: { ...llmConfig, model: MODEL },
      instructions:
        'You are a coding assistant with local file tools. Use read_file, ' +
        'edit_file, write_file, bash. Be concise.',
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

  let errored = false;
  let errorMessage: string | undefined;
  try {
    const run = await Run.create<t.IState>(runConfig);
    conversation.push(new HumanMessage(task.prompt));
    const streamConfig = {
      configurable: { provider: PROVIDER, thread_id: `compare-${Date.now()}` },
      streamMode: 'values',
      version: 'v2' as const,
    };
    await run.processStream(
      { messages: conversation },
      streamConfig as Parameters<typeof run.processStream>[1]
    );
    const finalMessages = run.getRunMessages();
    if (finalMessages) {
      conversation.push(...finalMessages);
    }
  } catch (err) {
    errored = true;
    errorMessage = (err as Error).message.slice(0, 500);
  }

  // Walk the conversation: tool calls live on AIMessage as `tool_calls`,
  // tool results are ToolMessage entries (already chronologically next to them).
  for (const msg of conversation) {
    if (msg._getType() === 'ai') {
      const ai = msg as unknown as {
        tool_calls?: Array<{ name?: string; args?: unknown }>;
        usage_metadata?: { input_tokens?: number; output_tokens?: number };
      };
      if (ai.tool_calls != null) {
        for (const tc of ai.tool_calls) {
          observedToolCalls.push({
            name: tc.name ?? '?',
            argsBytes: JSON.stringify(tc.args ?? {}).length,
            isError: false,
          });
        }
      }
      if (ai.usage_metadata != null) {
        inputTokens += ai.usage_metadata.input_tokens ?? 0;
        outputTokens += ai.usage_metadata.output_tokens ?? 0;
        const idu =
          (ai.usage_metadata as unknown as {
            input_token_details?: {
              cache_read?: number;
              cache_creation?: number;
            };
          }).input_token_details;
        cacheReadTokens += idu?.cache_read ?? 0;
        cacheWriteTokens += idu?.cache_creation ?? 0;
      }
    }
    if (msg instanceof ToolMessage) {
      if (msg.status === 'error' && observedToolCalls.length > 0) {
        observedToolCalls[observedToolCalls.length - 1].isError = true;
      }
    }
  }

  const lastAssistant = [...conversation]
    .reverse()
    .find((m) => m._getType() === 'ai');
  let finalAssistant = '';
  if (lastAssistant) {
    const c = lastAssistant.content;
    finalAssistant =
      typeof c === 'string'
        ? c
        : Array.isArray(c)
          ? c
            .map((b) => ('text' in b ? b.text : ''))
            .filter(Boolean)
            .join(' ')
          : '';
  }

  return {
    toolCalls: observedToolCalls,
    wallMs: performance.now() - start,
    inputTokens,
    outputTokens,
    cacheReadTokens,
    cacheWriteTokens,
    cost: 0,
    finalAssistant: finalAssistant.slice(0, 500),
    errored,
    errorMessage,
  };
}

/* ------------------------------------------------------------------ */
/* Harness                                                             */
/* ------------------------------------------------------------------ */

async function setupWorkspace(task: Task): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'lc-compare-'));
  for (const [name, content] of Object.entries(task.seed)) {
    await writeFile(join(dir, name), content, 'utf8');
  }
  if (task.setup != null) {
    await task.setup(dir);
  }
  return dir;
}

async function symlinkRepoNodeModules(cwd: string): Promise<void> {
  const { symlink } = await import('fs/promises');
  const repo = resolve(process.cwd(), 'node_modules');
  await symlink(repo, join(cwd, 'node_modules'), 'dir').catch(() => {
    /* fall through; tsc just won't be available */
  });
}

function summariseToolCalls(calls: ToolCallObservation[]): string {
  if (calls.length === 0) return '<none>';
  const grouped = new Map<string, number>();
  for (const c of calls) {
    grouped.set(c.name, (grouped.get(c.name) ?? 0) + 1);
  }
  const inline = [...grouped.entries()]
    .map(([n, c]) => `${n}×${c}`)
    .join(', ');
  const errors = calls.filter((c) => c.isError).length;
  return `${calls.length} call(s) [${inline}]${errors > 0 ? ` (${errors} errored)` : ''}`;
}

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

async function main(): Promise<void> {
  console.log(`pi binary: ${PI_BIN}`);
  console.log(`model:     ${MODEL}`);
  console.log(`provider:  ${PROVIDER}`);

  const results: Array<{
    task: Task;
    pi: RunOutcome;
    piVerify: { ok: boolean; detail: string };
    ours: RunOutcome;
    oursVerify: { ok: boolean; detail: string };
  }> = [];

  for (const task of TASKS) {
    console.log(`\n========== ${task.name} ==========`);
    console.log(task.description);

    // Run pi
    const piCwd = await setupWorkspace(task);
    console.log(`[pi] workspace=${piCwd}`);
    const pi = await runPi(task, piCwd);
    console.log(
      `[pi] ${pi.errored ? 'ERROR' : 'ok'} ${fmtMs(pi.wallMs)} ${summariseToolCalls(pi.toolCalls)} ` +
        `in=${pi.inputTokens} out=${pi.outputTokens} cacheR=${pi.cacheReadTokens} cacheW=${pi.cacheWriteTokens} $${pi.cost.toFixed(4)}`
    );
    if (pi.errored) console.log(`[pi] err: ${pi.errorMessage}`);
    const piVerify = await task.verify(piCwd);
    console.log(`[pi] verify: ${piVerify.ok ? '✔' : '✖'} ${piVerify.detail}`);
    await rm(piCwd, { recursive: true, force: true });

    // Run ours
    const oursCwd = await setupWorkspace(task);
    console.log(`[ours] workspace=${oursCwd}`);
    const ours = await runOurs(task, oursCwd);
    console.log(
      `[ours] ${ours.errored ? 'ERROR' : 'ok'} ${fmtMs(ours.wallMs)} ${summariseToolCalls(ours.toolCalls)} ` +
        `in=${ours.inputTokens} out=${ours.outputTokens} cacheR=${ours.cacheReadTokens} cacheW=${ours.cacheWriteTokens}`
    );
    if (ours.errored) console.log(`[ours] err: ${ours.errorMessage}`);
    const oursVerify = await task.verify(oursCwd);
    console.log(
      `[ours] verify: ${oursVerify.ok ? '✔' : '✖'} ${oursVerify.detail}`
    );
    await rm(oursCwd, { recursive: true, force: true });

    results.push({ task, pi, piVerify, ours, oursVerify });
  }

  /* Summary table ---------------------------------------------------- */
  console.log('\n\n================ SUMMARY ================\n');
  const cols: Array<[string, string, string, string]> = [
    ['task', 'metric', 'pi', 'ours'],
  ];
  for (const r of results) {
    cols.push([r.task.name, 'verify', r.piVerify.ok ? '✔' : '✖', r.oursVerify.ok ? '✔' : '✖']);
    cols.push([
      '',
      'wall',
      fmtMs(r.pi.wallMs),
      fmtMs(r.ours.wallMs),
    ]);
    cols.push([
      '',
      'tool calls',
      String(r.pi.toolCalls.length),
      String(r.ours.toolCalls.length),
    ]);
    // Anthropic's accounting: total billed input = input + cache_creation;
    // cache_read is rebated. We display new+miss separately for fairness.
    const piNew = r.pi.inputTokens + r.pi.cacheWriteTokens;
    const oursNew = r.ours.inputTokens + r.ours.cacheWriteTokens;
    cols.push([
      '',
      'input new',
      String(piNew),
      String(oursNew),
    ]);
    cols.push([
      '',
      'cache read',
      String(r.pi.cacheReadTokens),
      String(r.ours.cacheReadTokens),
    ]);
    cols.push([
      '',
      'output tok',
      String(r.pi.outputTokens),
      String(r.ours.outputTokens),
    ]);
  }

  const widths = [0, 0, 0, 0].map((_, i) =>
    Math.max(...cols.map((row) => row[i].length))
  );
  for (const row of cols) {
    console.log(
      row
        .map((cell, i) => cell.padEnd(widths[i]))
        .join('  ')
    );
  }

  // Confirm everywhere succeeded
  const allOk =
    results.every((r) => r.piVerify.ok) &&
    results.every((r) => r.oursVerify.ok);
  console.log(
    `\nOverall: pi ${results.every((r) => r.piVerify.ok) ? 'all ✔' : 'some ✖'}, ` +
      `ours ${results.every((r) => r.oursVerify.ok) ? 'all ✔' : 'some ✖'}.`
  );
  if (!allOk) process.exitCode = 1;
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

// Silence unused readdir import in some bundlers.
void readdir;
void cp;
