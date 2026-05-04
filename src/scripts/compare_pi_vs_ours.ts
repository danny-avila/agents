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
  /** Optional binary files seeded into the workspace (key = path, value = bytes). */
  seedBinary?: Record<string, Buffer>;
  /** Prompt sent to both agents. */
  prompt: string;
  /** Function that returns true if the workspace ended in the right state. */
  verify: (cwd: string) => Promise<{ ok: boolean; detail: string }>;
  /** Optional pre-run hook (e.g. symlink node_modules so `tsc` is available). */
  setup?: (cwd: string) => Promise<void>;
  /**
   * Optional setup specific to our local engine (extra `local.*` config knobs)
   * — lets us toggle e.g. `attachReadAttachments` per-task without
   *   making the default surface noisier than necessary.
   */
  oursLocalConfigOverrides?: Partial<t.LocalExecutionConfig>;
  /**
   * Some tasks aren't realistically supportable on one side. When set,
   * skip the named runner and report N/A in the table.
   */
  skip?: 'pi' | 'ours';
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
  {
    name: 'T5 multi-file-rename',
    description:
      'Rename a function across three files. Tests how the agent finds + applies the rename.',
    seed: {
      'src/lib.ts':
        'export function calc_total(a: number, b: number): number {\n' +
        '  return a + b;\n' +
        '}\n',
      'src/index.ts':
        'import { calc_total } from "./lib";\n' +
        'console.log(calc_total(2, 3));\n',
      'src/index.test.ts':
        'import { calc_total } from "./lib";\n' +
        'if (calc_total(1, 1) !== 2) throw new Error("fail");\n' +
        'console.log("ok");\n',
    },
    prompt:
      'Rename the exported function `calc_total` to `calculateTotal` across src/lib.ts, ' +
      'src/index.ts, and src/index.test.ts. Update every reference. Reply "done" when finished.',
    verify: async (cwd) => {
      const lib = await readFile(join(cwd, 'src/lib.ts'), 'utf8').catch(() => '');
      const idx = await readFile(join(cwd, 'src/index.ts'), 'utf8').catch(() => '');
      const tst = await readFile(join(cwd, 'src/index.test.ts'), 'utf8').catch(
        () => ''
      );
      const allRenamed =
        /function\s+calculateTotal/.test(lib) &&
        /calculateTotal\(/.test(idx) &&
        /calculateTotal\(/.test(tst);
      const noOldName =
        !/calc_total/.test(lib) &&
        !/calc_total/.test(idx) &&
        !/calc_total/.test(tst);
      const ok = allRenamed && noOldName;
      return {
        ok,
        detail: ok
          ? ''
          : `lib:\n${lib}\nindex:\n${idx}\ntest:\n${tst}`,
      };
    },
  },
  {
    name: 'T6 image-read-and-describe',
    description:
      'Reads a PNG and describes it. Ours embeds via attachReadAttachments + image_url block; pi has no equivalent and is skipped.',
    seed: {},
    setup: async (cwd) => {
      const { copyFile } = await import('fs/promises');
      // Use a real PNG (Anthropic refuses tiny 1x1 PNGs with "Could not
      // process image"). Try a few well-known macOS app icons; fall back to
      // any *.png we can find under /System.
      const candidates = [
        '/System/Library/CoreServices/Certificate Assistant.app/Contents/Resources/droppedImage.png',
        '/System/Library/CoreServices/Certificate Assistant.app/Contents/Resources/shapeimage_1.png',
        '/System/Library/CoreServices/BluetoothUIServer.app/Contents/Resources/handoff.png',
      ];
      for (const path of candidates) {
        try {
          await copyFile(path, join(cwd, 'sample.png'));
          return;
        } catch {
          // try next
        }
      }
      throw new Error('No system PNG available for T6 image task');
    },
    prompt:
      'Read sample.png and briefly describe what the image shows. Reply with "done" at the end.',
    verify: async (cwd) => {
      // The verify step is soft — we just check the file is still on disk
      // (the agent shouldn't have deleted it) and the script-level error
      // tracking will fail this task if Anthropic refused the request.
      const { stat } = await import('fs/promises');
      try {
        await stat(join(cwd, 'sample.png'));
        return { ok: true, detail: '' };
      } catch {
        return { ok: false, detail: 'sample.png missing' };
      }
    },
    oursLocalConfigOverrides: { attachReadAttachments: 'images-only' },
    skip: 'pi',
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

async function runOurs(
  task: Task,
  cwd: string,
  overrides: Partial<t.LocalExecutionConfig> = {}
): Promise<RunOutcome> {
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
    // ON_RUN_STEP must be forwarded too — without it the aggregator's
    // `stepMap` is empty when ON_RUN_STEP_COMPLETED arrives and you
    // get a "No run step or runId found for completed step event"
    // warn for every tool call. The harness doesn't actually use the
    // aggregated content, but feeding both events keeps logs clean.
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
      // NB: in the legacy path Run.createLegacyGraph rebuilds
      // `clientOptions` from llmConfig (it ignores graphConfig.clientOptions),
      // so promptCache lives here and not on a separate clientOptions field.
      llmConfig: { ...llmConfig, model: MODEL, promptCache: true },
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
        ...overrides,
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
        const reportedInput = ai.usage_metadata.input_tokens ?? 0;
        outputTokens += ai.usage_metadata.output_tokens ?? 0;
        const idu =
          (ai.usage_metadata as unknown as {
            input_token_details?: {
              cache_read?: number;
              cache_creation?: number;
            };
          }).input_token_details;
        const cacheRead = idu?.cache_read ?? 0;
        const cacheCreate = idu?.cache_creation ?? 0;
        cacheReadTokens += cacheRead;
        cacheWriteTokens += cacheCreate;
        // The Anthropic adapter at src/llm/anthropic/utils/message_outputs.ts:31
        // reports usage_metadata.input_tokens as the TOTAL prompt
        // (input + cache_creation + cache_read), not just the uncached
        // portion. Subtract cached fields so `inputTokens` here is
        // apples-to-apples with pi's `input` field (uncached only).
        const trulyUncached = Math.max(
          0,
          reportedInput - cacheRead - cacheCreate
        );
        inputTokens += trulyUncached;
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

  // Sonnet 4.5 pricing (USD per 1M tokens). Pi computes its own cost; we
  // compute ours from the same per-turn breakdown so the cost columns are
  // comparable. Source: anthropic.com/pricing as of model ship.
  const PRICE_INPUT = 3.0 / 1_000_000;
  const PRICE_OUTPUT = 15.0 / 1_000_000;
  const PRICE_CACHE_WRITE = 3.75 / 1_000_000;
  const PRICE_CACHE_READ = 0.3 / 1_000_000;
  const cost =
    inputTokens * PRICE_INPUT +
    outputTokens * PRICE_OUTPUT +
    cacheWriteTokens * PRICE_CACHE_WRITE +
    cacheReadTokens * PRICE_CACHE_READ;

  return {
    toolCalls: observedToolCalls,
    wallMs: performance.now() - start,
    inputTokens,
    outputTokens,
    cacheReadTokens,
    cacheWriteTokens,
    cost,
    finalAssistant: finalAssistant.slice(0, 500),
    errored,
    errorMessage,
  };
}

/* ------------------------------------------------------------------ */
/* Harness                                                             */
/* ------------------------------------------------------------------ */

async function setupWorkspace(task: Task): Promise<string> {
  const { mkdir } = await import('fs/promises');
  const dir = await mkdtemp(join(tmpdir(), 'lc-compare-'));
  for (const [relPath, content] of Object.entries(task.seed)) {
    const abs = join(dir, relPath);
    await mkdir(join(abs, '..'), { recursive: true });
    await writeFile(abs, content, 'utf8');
  }
  for (const [relPath, bytes] of Object.entries(task.seedBinary ?? {})) {
    const abs = join(dir, relPath);
    await mkdir(join(abs, '..'), { recursive: true });
    await writeFile(abs, bytes);
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

interface AggregatedSide {
  outcomes: RunOutcome[];
  verifies: boolean[];
}
function emptySide(): AggregatedSide {
  return { outcomes: [], verifies: [] };
}
function avg(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}
function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

async function runOnce(
  task: Task,
  side: 'pi' | 'ours'
): Promise<{ outcome: RunOutcome; verify: { ok: boolean; detail: string } } | null> {
  if (task.skip === side) return null;
  const cwd = await setupWorkspace(task);
  const outcome =
    side === 'pi'
      ? await runPi(task, cwd)
      : await runOurs(task, cwd, task.oursLocalConfigOverrides ?? {});
  let verify = await task.verify(cwd);
  if (outcome.errored) {
    // Force-fail verify when the runner errored — otherwise a soft
    // verify can mask a real provider rejection or a crash.
    verify = {
      ok: false,
      detail: `runner errored: ${outcome.errorMessage ?? 'unknown'}`,
    };
  }
  await rm(cwd, { recursive: true, force: true });
  return { outcome, verify };
}

async function main(): Promise<void> {
  const ITERS = Math.max(1, Number(process.env.COMPARE_ITERS ?? '1'));
  console.log(`pi binary: ${PI_BIN}`);
  console.log(`model:     ${MODEL}`);
  console.log(`provider:  ${PROVIDER}`);
  console.log(`iters:     ${ITERS}`);

  const results: Array<{
    task: Task;
    pi: AggregatedSide;
    ours: AggregatedSide;
  }> = [];

  for (const task of TASKS) {
    console.log(`\n========== ${task.name} ==========`);
    console.log(task.description);

    const pi = emptySide();
    const ours = emptySide();

    for (let i = 0; i < ITERS; i++) {
      const tag = ITERS > 1 ? ` (iter ${i + 1}/${ITERS})` : '';
      const piRes = await runOnce(task, 'pi');
      if (piRes != null) {
        pi.outcomes.push(piRes.outcome);
        pi.verifies.push(piRes.verify.ok);
        console.log(
          `[pi]${tag} ${piRes.outcome.errored ? 'ERROR' : piRes.verify.ok ? 'ok' : 'fail'} ` +
            `${fmtMs(piRes.outcome.wallMs)} ${summariseToolCalls(piRes.outcome.toolCalls)} ` +
            `in=${piRes.outcome.inputTokens} out=${piRes.outcome.outputTokens} ` +
            `cacheR=${piRes.outcome.cacheReadTokens} cacheW=${piRes.outcome.cacheWriteTokens} ` +
            `$${piRes.outcome.cost.toFixed(4)}`
        );
        if (piRes.outcome.errored) console.log(`  err: ${piRes.outcome.errorMessage}`);
      } else {
        console.log(`[pi]${tag} (skipped)`);
      }

      const oursRes = await runOnce(task, 'ours');
      if (oursRes != null) {
        ours.outcomes.push(oursRes.outcome);
        ours.verifies.push(oursRes.verify.ok);
        console.log(
          `[ours]${tag} ${oursRes.outcome.errored ? 'ERROR' : oursRes.verify.ok ? 'ok' : 'fail'} ` +
            `${fmtMs(oursRes.outcome.wallMs)} ${summariseToolCalls(oursRes.outcome.toolCalls)} ` +
            `in=${oursRes.outcome.inputTokens} out=${oursRes.outcome.outputTokens} ` +
            `cacheR=${oursRes.outcome.cacheReadTokens} cacheW=${oursRes.outcome.cacheWriteTokens}`
        );
        if (oursRes.outcome.errored) console.log(`  err: ${oursRes.outcome.errorMessage}`);
      } else {
        console.log(`[ours]${tag} (skipped)`);
      }
    }

    results.push({ task, pi, ours });
  }

  /* Summary table ---------------------------------------------------- */
  console.log('\n\n================ SUMMARY ================');
  if (ITERS > 1) {
    console.log(`(metrics are mean over ${ITERS} iterations)\n`);
  } else {
    console.log();
  }

  function fmtSide(side: AggregatedSide, key: keyof RunOutcome): string {
    if (side.outcomes.length === 0) return 'N/A';
    const vals = side.outcomes.map((o) => Number(o[key] ?? 0));
    return Math.round(avg(vals)).toString();
  }
  function fmtSideMs(side: AggregatedSide): string {
    if (side.outcomes.length === 0) return 'N/A';
    return fmtMs(avg(side.outcomes.map((o) => o.wallMs)));
  }
  function fmtSideCalls(side: AggregatedSide): string {
    if (side.outcomes.length === 0) return 'N/A';
    return avg(side.outcomes.map((o) => o.toolCalls.length)).toFixed(1);
  }
  function fmtVerify(side: AggregatedSide): string {
    if (side.verifies.length === 0) return 'N/A';
    const passed = side.verifies.filter(Boolean).length;
    return passed === side.verifies.length
      ? '✔'
      : `${passed}/${side.verifies.length}`;
  }
  function fmtCost(side: AggregatedSide): string {
    if (side.outcomes.length === 0) return 'N/A';
    const c = avg(side.outcomes.map((o) => o.cost));
    return c === 0 ? '-' : `$${c.toFixed(4)}`;
  }

  const cols: Array<[string, string, string, string]> = [
    ['task', 'metric', 'pi', 'ours'],
  ];
  for (const r of results) {
    cols.push([r.task.name, 'verify', fmtVerify(r.pi), fmtVerify(r.ours)]);
    cols.push(['', 'wall', fmtSideMs(r.pi), fmtSideMs(r.ours)]);
    cols.push(['', 'tool calls', fmtSideCalls(r.pi), fmtSideCalls(r.ours)]);
    cols.push(['', 'input new', fmtSide(r.pi, 'inputTokens'), fmtSide(r.ours, 'inputTokens')]);
    cols.push(['', 'cache read', fmtSide(r.pi, 'cacheReadTokens'), fmtSide(r.ours, 'cacheReadTokens')]);
    cols.push(['', 'cache write', fmtSide(r.pi, 'cacheWriteTokens'), fmtSide(r.ours, 'cacheWriteTokens')]);
    cols.push(['', 'output tok', fmtSide(r.pi, 'outputTokens'), fmtSide(r.ours, 'outputTokens')]);
    cols.push(['', 'cost', fmtCost(r.pi), fmtCost(r.ours)]);
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

  // Aggregate verify counts across all iters of all non-skipped tasks.
  const piVerifies = results.flatMap((r) => r.pi.verifies);
  const oursVerifies = results.flatMap((r) => r.ours.verifies);
  const piPassed = piVerifies.filter(Boolean).length;
  const oursPassed = oursVerifies.filter(Boolean).length;
  console.log(
    `\nOverall: pi ${piPassed}/${piVerifies.length}, ours ${oursPassed}/${oursVerifies.length}.`
  );
  if (piPassed < piVerifies.length || oursPassed < oursVerifies.length) {
    process.exitCode = 1;
  }
  void sum;
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
