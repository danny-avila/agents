/* eslint-disable no-console */
/**
 * SDK-side activity-label eval runner, ported from LibreChat #14527
 * (scripts/activity-labels/). corpus and report are byte-identical to the
 * LibreChat originals so results compare across repos; checks carries one
 * tool-echo normalization fix pending backport (see checks.cjs header).
 * The user prompt is rendered by the REAL `buildActivityLabelPrompt` from
 * `src/prompts/activityLabel.ts` instead of a hand-port. That is the point
 * of the SDK version: a change to the builder (sections, truncation,
 * framing) is measured as it will ship.
 *
 * Replays each case against the production wire shape (system = variant
 * instruction, user = built prompt, max_tokens 256). Multi-step cases run
 * serially and chain each generated label into the next step's
 * `previousLabels`, exercising the builder's own sanitation and 3-label cap.
 *
 * Usage (from the repo root):
 *   npm run label:eval -- [--variants sdk-default,host-shipped]
 *     [--cases sandbox-probe-run,fib-rapid] [--samples 3] [--model id]
 *     [--concurrency 6] [--dry]
 *
 * Needs ANTHROPIC_API_KEY (env or repo .env). Results land in
 * src/scripts/activity-labels/results/ (gitignored): a timestamped JSON of
 * every record plus latest.md. The aggregate table is a regression guard;
 * the per-case tables read by eye are the real instrument.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import type { ActivityLabelToolEntry } from '@/types/activityLabel';
import type { Variant } from './variants';
import { buildActivityLabelPrompt } from '@/prompts/activityLabel';
import { variants } from './variants';

/** The subset of an entry the echo checker reads; verbatim captured steps
 *  only recover tool names, not full entries. */
type EchoEntry = Pick<ActivityLabelToolEntry, 'toolName'>;

type LabelCheck = {
  flags: string[];
  wordCount: number;
  firstWord?: string;
  maxOverlap?: number;
};

/** Aggregate rows are produced and consumed by the ported CJS modules;
 *  run.ts only pipes them from aggregate() into markdownReport(). */
type AggregateRow = Readonly<Record<string, string | number>>;

const require = createRequire(import.meta.url);
const HARNESS_DIR = path.dirname(fileURLToPath(import.meta.url));
const { cases, stepEntries } = require('./corpus.cjs') as {
  cases: CorpusCase[];
  stepEntries: (step: CorpusStep) => EchoEntry[];
};
const { checkLabel } = require('./checks.cjs') as {
  checkLabel: (
    label: string,
    context: { entries: EchoEntry[]; previousLabels: string[] }
  ) => LabelCheck;
};
const { aggregate, markdownReport } = require('./report.cjs') as {
  aggregate: (records: RunRecord[], model: string) => AggregateRow[];
  markdownReport: (input: {
    records: RunRecord[];
    aggregates: AggregateRow[];
    runCases: CorpusCase[];
    variantNames: string[];
    model: string;
    samples: number;
  }) => string;
};

const ROOT = path.resolve(HARNESS_DIR, '..', '..', '..');
const RESULTS_DIR = path.join(HARNESS_DIR, 'results');
const CHAR_LIMIT = 600;
const MAX_TOKENS = 256;

type CorpusStep = {
  id?: string;
  verbatim?: string;
  productionLabel?: string;
  payload?: {
    entries: ActivityLabelToolEntry[];
    thinkingExcerpts?: string[];
    lastAssistantText?: string;
  };
};

type CorpusCase = { id: string; notes?: string; steps: CorpusStep[] };

type RunArgs = {
  samples: number;
  concurrency: number;
  model: string;
  dry: boolean;
  variants?: string[];
  cases?: string[];
};

function parseArgs(argv: string[]): RunArgs {
  const args: RunArgs = {
    samples: 1,
    concurrency: 6,
    model: 'claude-haiku-4-5',
    dry: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--dry') {
      args.dry = true;
    } else if (key === '--variants') {
      args.variants = argv[++i].split(',');
    } else if (key === '--cases') {
      args.cases = argv[++i].split(',');
    } else if (key === '--samples') {
      args.samples = Number(argv[++i]);
    } else if (key === '--model') {
      args.model = argv[++i];
    } else if (key === '--concurrency') {
      args.concurrency = Number(argv[++i]);
    } else {
      /** A mistyped `--dryy` must not fall through to a fully billed
       *  default sweep. */
      throw new Error(`unknown option: ${key}`);
    }
  }
  if (!Number.isInteger(args.samples) || args.samples < 1) {
    throw new Error(
      `--samples must be a positive integer, got ${args.samples}`
    );
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1) {
    throw new Error(
      `--concurrency must be a positive integer, got ${args.concurrency}`
    );
  }
  return args;
}

function loadKey(): string {
  if (process.env.ANTHROPIC_API_KEY) {
    return process.env.ANTHROPIC_API_KEY;
  }
  const envPath = path.join(ROOT, '.env');
  const line = fs.existsSync(envPath)
    ? fs
        .readFileSync(envPath, 'utf8')
        .split('\n')
        .find((entry) => entry.startsWith('ANTHROPIC_API_KEY='))
    : undefined;
  if (!line) {
    throw new Error(
      `ANTHROPIC_API_KEY not set and not found in ${envPath}.\n` +
        'Pass it inline:  ANTHROPIC_API_KEY=sk-… npm run label:eval'
    );
  }
  return line
    .slice('ANTHROPIC_API_KEY='.length)
    .trim()
    .replace(/^["']|["']$/g, '');
}

const BUILDER_TERMINAL = '\n\nLabel:';

/**
 * Renders the continuity section through the REAL builder (empty batch +
 * previousLabels yields `<section>\n\n<terminal>`) so verbatim captured
 * steps get exactly the sanitation and 3-label cap production applies —
 * not a reimplementation of it. The terminal is derived from an empty
 * render rather than assumed, so this keeps working if the builder ever
 * renames `Label:` — the exact change the framing variants evaluate.
 */
function continuitySection(previousLabels: string[]): string | null {
  if (previousLabels.length === 0) {
    return null;
  }
  const terminal = buildActivityLabelPrompt({
    entries: [],
    charLimit: CHAR_LIMIT,
  });
  const rendered = buildActivityLabelPrompt({
    entries: [],
    charLimit: CHAR_LIMIT,
    previousLabels,
  });
  return rendered === terminal
    ? null
    : rendered.slice(0, -(terminal.length + '\n\n'.length));
}

/**
 * Applies a framing hypothesis as marker-exact substitutions on a built (or
 * captured) prompt. Strict: the heading is rewritten when the prompt STARTS
 * with the section (a bare batch with no previous labels, intent, or
 * excerpts puts `Tool calls:` first) or when the section-boundary marker
 * occurs exactly once; the terminal only when it is the final line. A
 * prompt failing every check is returned unchanged, so a framing variant
 * degrades to the control rather than corrupting the sample.
 */
function applyFraming(
  prompt: string,
  { entriesHeading, terminal }: Pick<Variant, 'entriesHeading' | 'terminal'>
): string {
  let text = prompt;
  if (entriesHeading != null && entriesHeading !== 'Tool calls:') {
    const lead = 'Tool calls:\n';
    const marker = '\n\nTool calls:\n';
    const first = text.indexOf(marker);
    if (text.startsWith(lead)) {
      text = `${entriesHeading}\n` + text.slice(lead.length);
    } else if (first !== -1 && text.indexOf(marker, first + 1) === -1) {
      text =
        text.slice(0, first) +
        `\n\n${entriesHeading}\n` +
        text.slice(first + marker.length);
    }
  }
  if (
    terminal != null &&
    terminal !== 'Label:' &&
    text.endsWith(BUILDER_TERMINAL)
  ) {
    text = text.slice(0, -'Label:'.length) + terminal;
  }
  return text;
}

function renderStepPrompt(
  step: CorpusStep,
  variant: Variant,
  previousLabels: string[] | null
): string {
  let prompt: string;
  if (step.verbatim != null) {
    const section =
      previousLabels != null ? continuitySection(previousLabels) : null;
    prompt = section != null ? `${section}\n\n${step.verbatim}` : step.verbatim;
  } else {
    prompt = buildActivityLabelPrompt({
      entries: step.payload?.entries ?? [],
      thinkingExcerpts: step.payload?.thinkingExcerpts,
      lastAssistantText: step.payload?.lastAssistantText,
      charLimit: CHAR_LIMIT,
      previousLabels: previousLabels ?? undefined,
    });
  }
  return applyFraming(prompt, variant);
}

type LabelSuccess = {
  label: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
};

type LabelFailure = { error: string; latencyMs: number };

type LabelResult = LabelSuccess | LabelFailure;

/** An error no retry or later task can recover from — thrown through the
 *  pool so the run stops instead of recording it per task. */
class FatalRunError extends Error {}

/**
 * One label request with three attempts. The whole attempt — fetch AND the
 * body reads, which reject on a connection reset after headers arrive —
 * sits inside one try, so any transport rejection is retried like a
 * 429/500/529 instead of escaping to the top level, where a single flake
 * would discard every completed record of an otherwise finished run.
 * `latencyMs` spans the whole sequence including backoff — the cost a
 * variant actually paid.
 */
async function requestLabel({
  apiKey,
  model,
  instruction,
  prompt,
}: {
  apiKey: string;
  model: string;
  instruction: string;
  prompt: string;
}): Promise<LabelResult> {
  const started = Date.now();
  let lastError = 'exhausted retries';
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model,
          max_tokens: MAX_TOKENS,
          system: instruction,
          messages: [{ role: 'user', content: prompt }],
        }),
      });
      if (response.ok) {
        const json = (await response.json()) as {
          content?: Array<{ text?: string }>;
          usage?: { input_tokens?: number; output_tokens?: number };
        };
        /** Same normalization as the production extractor (src/run.ts): a
         *  label renders as a single row and re-enters later prompts as
         *  continuity context, so newlines must not survive here either —
         *  and a raw newline would break the per-case Markdown tables. */
        const label = (json.content ?? [])
          .map((block) => block.text ?? '')
          .join('')
          .replace(/\s+/g, ' ')
          .trim()
          .replace(/^["']|["']$/g, '');
        return {
          label,
          latencyMs: Date.now() - started,
          inputTokens: json.usage?.input_tokens ?? 0,
          outputTokens: json.usage?.output_tokens ?? 0,
        };
      }
      const body = await response.text();
      if (response.status === 401 || response.status === 403) {
        /** Every remaining task would send the same doomed request —
         *  with the default corpus, hundreds of them — and produce an
         *  all-error report. Kill the run instead. */
        throw new FatalRunError(
          `authentication failed (HTTP ${response.status}): ${body.slice(0, 160)}`
        );
      }
      lastError = `HTTP ${response.status}: ${body.slice(0, 160)}`;
      if (attempt < 3 && [429, 500, 529].includes(response.status)) {
        const retryAfter = Number(response.headers.get('retry-after'));
        const waitMs =
          Number.isFinite(retryAfter) && retryAfter > 0
            ? retryAfter * 1000
            : attempt * 2000;
        await new Promise((resolve) =>
          setTimeout(resolve, Math.min(waitMs, 15000))
        );
        continue;
      }
      break;
    } catch (error) {
      if (error instanceof FatalRunError) {
        throw error;
      }
      lastError = `request failed: ${error instanceof Error ? error.message : String(error)}`;
      if (attempt < 3) {
        await new Promise((resolve) => setTimeout(resolve, attempt * 2000));
        continue;
      }
      break;
    }
  }
  return { error: lastError, latencyMs: Date.now() - started };
}

type RecordBase = {
  variant: string;
  sample: number;
  caseId: string;
  stepId: string;
};

type DryRunRecord = RecordBase & { prompt: string };

type ErrorRecord = RecordBase & { error: string };

type LabelRecord = RecordBase & {
  label: string;
  production?: string;
  flags: string[];
  wordCount: number;
  firstWord?: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
};

type RunRecord = DryRunRecord | ErrorRecord | LabelRecord;

/** One case chain: steps serial, labels feeding forward. */
async function runCase({
  apiKey,
  model,
  variant,
  sample,
  testCase,
  dry,
  records,
}: {
  apiKey: string;
  model: string;
  variant: Variant;
  sample: number;
  testCase: CorpusCase;
  dry: boolean;
  records: RunRecord[];
}): Promise<void> {
  const chain: string[] = [];
  for (const step of testCase.steps) {
    const prompt = renderStepPrompt(
      step,
      variant,
      variant.usePreviousLabels ? chain : null
    );
    const stepId = step.id ?? testCase.id;
    if (dry) {
      records.push({
        variant: variant.name,
        sample,
        caseId: testCase.id,
        stepId,
        prompt,
      });
      continue;
    }
    const result = await requestLabel({
      apiKey,
      model,
      instruction: variant.instruction,
      prompt,
    });
    if ('error' in result) {
      records.push({
        variant: variant.name,
        sample,
        caseId: testCase.id,
        stepId,
        error: result.error,
      });
      continue;
    }
    const { flags, wordCount, firstWord } = checkLabel(result.label, {
      entries: stepEntries(step),
      previousLabels: chain,
    });
    chain.push(result.label);
    records.push({
      variant: variant.name,
      sample,
      caseId: testCase.id,
      stepId,
      label: result.label,
      production: step.productionLabel,
      flags,
      wordCount,
      firstWord,
      latencyMs: result.latencyMs,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
    });
  }
}

async function pool(
  tasks: Array<() => Promise<void>>,
  size: number
): Promise<void> {
  /** Shared index instead of queue.shift(): shifting reindexes the whole
   *  remaining array per task, quadratic over a large sweep. */
  let next = 0;
  const workers = Array.from(
    { length: Math.min(size, tasks.length) },
    async () => {
      while (next < tasks.length) {
        const task = tasks[next];
        next += 1;
        await task();
      }
    }
  );
  await Promise.all(workers);
}

/** A typo in a selection must fail up front, not silently drop the
 *  requested column and complete a billed run without its control; a
 *  duplicate would schedule the same paid arm twice and merge both runs
 *  under one aggregate row. */
function resolveSelection<T>(
  requested: string[] | undefined,
  available: T[],
  nameOf: (item: T) => string,
  flag: string
): T[] {
  if (requested == null) {
    return available;
  }
  const byName = new Map(available.map((item) => [nameOf(item), item]));
  const unknown = requested.filter((name) => !byName.has(name));
  if (unknown.length > 0) {
    throw new Error(
      `unknown ${flag}: ${unknown.join(', ')} (available: ${[...byName.keys()].join(', ')})`
    );
  }
  const duplicates = requested.filter(
    (name, index) => requested.indexOf(name) !== index
  );
  if (duplicates.length > 0) {
    throw new Error(
      `duplicate ${flag} selection: ${[...new Set(duplicates)].join(', ')}`
    );
  }
  return requested.map((name) => byName.get(name)!);
}

(async () => {
  const args = parseArgs(process.argv.slice(2));
  const runVariants = resolveSelection(
    args.variants,
    variants,
    (variant) => variant.name,
    '--variants'
  );
  const runCases = resolveSelection(args.cases, cases, (c) => c.id, '--cases');
  if (runVariants.length === 0 || runCases.length === 0) {
    throw new Error('nothing selected — check --variants / --cases names');
  }
  const apiKey = args.dry ? '' : loadKey();
  const records: RunRecord[] = [];
  const tasks: Array<() => Promise<void>> = [];
  /** Variants innermost, so adjacent queue slots cycle through the arms:
   *  a variant-major queue would correlate variant identity with elapsed
   *  run time, letting later arms inherit rate-limit or provider-load
   *  conditions the first arm never saw. */
  for (let sample = 1; sample <= args.samples; sample++) {
    for (const testCase of runCases) {
      for (const variant of runVariants) {
        tasks.push(() =>
          runCase({
            apiKey,
            model: args.model,
            variant,
            sample,
            testCase,
            dry: args.dry,
            records,
          })
        );
      }
    }
  }
  const totalSteps = runCases.reduce((sum, c) => sum + c.steps.length, 0);
  console.log(
    `${args.dry ? 'DRY RUN — rendering only' : `model ${args.model}`} · ${runVariants.length} variants × ${args.samples} samples × ${runCases.length} cases (${totalSteps} steps each pass)`
  );
  const started = Date.now();
  await pool(tasks, args.concurrency);
  console.log(`done in ${((Date.now() - started) / 1000).toFixed(1)}s\n`);

  if (args.dry) {
    for (const record of records.slice(0, 3) as DryRunRecord[]) {
      console.log(
        `--- ${record.variant} / ${record.caseId} / ${record.stepId} ---`
      );
      console.log(record.prompt);
      console.log('');
    }
    console.log(`rendered ${records.length} prompts (showing 3)`);
    return;
  }

  const aggregates = aggregate(records, args.model);
  const variantNames = runVariants.map((variant) => variant.name);
  const report = markdownReport({
    records,
    aggregates,
    runCases,
    variantNames,
    model: args.model,
    samples: args.samples,
  });
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  /** Lets the rescorer refuse a stored run whose corpus has since
   *  drifted (renamed steps, changed tool names). */
  const corpusFingerprint = crypto
    .createHash('sha256')
    .update(JSON.stringify(cases))
    .digest('hex');
  fs.writeFileSync(
    path.join(RESULTS_DIR, `${stamp}.json`),
    JSON.stringify({ args, corpusFingerprint, records }, null, 2)
  );
  fs.writeFileSync(path.join(RESULTS_DIR, 'latest.md'), report);

  console.log(report.split('## Per-case')[0]);
  console.log(
    'full per-case tables: src/scripts/activity-labels/results/latest.md'
  );
})().catch((error: Error) => {
  console.error('ERR', error.message);
  process.exit(1);
});
