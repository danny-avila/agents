/**
 * Live, reproducible benchmark: single tail prompt-cache breakpoint (new
 * default) vs. the legacy "last two user messages" strategy.
 *
 * It replays realistic harness conversations against a real provider and, for
 * each model call, records the cache token breakdown the API reports. The two
 * strategies are run over the SAME conversations (only the cache MARKING
 * differs) under distinct cache namespaces, then compared.
 *
 * What it demonstrates
 * --------------------
 *  - Agent tool loop (one user turn, many tool rounds): the legacy strategy
 *    pins its only message breakpoint on the lone user message, so every
 *    appended assistant/tool turn is re-sent UNCACHED on the next call — cache
 *    write/fresh ≫ read. The tail strategy rides the true tail, so the growing
 *    transcript is written once and read back. This is the dominant agent shape
 *    and where the legacy approach breaks down hardest.
 *  - Multi-turn chat (frequent user messages): legacy's two rolling markers do
 *    fine here; the tail strategy ties (never worse).
 *  - Realistic agent (user turns interleaved with tool rounds): tail wins.
 *
 * Metrics (per strategy, summed over all calls in a scenario)
 *  - cache_read   : tokens served from cache (HIGHER is better).
 *  - cache_write  : tokens written to cache (cache_creation).
 *  - fresh        : uncached input processed at full price
 *                   (= input_tokens - cache_read - cache_write); this is what
 *                   balloons when caching fails to cover the transcript.
 *  - effective    : a cost proxy in input-token-equivalents using Anthropic's
 *                   published multipliers — read x0.1, write x1.25, fresh x1.0.
 *                   LOWER is better.
 *
 * Usage
 *   # Anthropic (default). Needs ANTHROPIC_API_KEY in .env (or BENCH_ENV_FILE).
 *   npm run bench:cache
 *   # Bedrock. Needs BEDROCK_AWS_* creds.
 *   npm run bench:cache -- --provider bedrock
 *   # Options: --provider anthropic|bedrock  --rounds <N>  --model <id>
 *
 * Not a unit test (no `.test.` suffix) so CI never runs it; it makes real,
 * paid API calls.
 */
import { config } from 'dotenv';
config({ path: process.env.BENCH_ENV_FILE || '.env' });

import {
  HumanMessage,
  AIMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { CustomAnthropic } from '@/llm/anthropic';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import {
  addCacheControl,
  addTailCacheControl,
  addBedrockCacheControl,
  addBedrockTailCacheControl,
} from '@/messages/cache';

type ProviderName = 'anthropic' | 'bedrock';

interface Args {
  provider: ProviderName;
  rounds: number;
  model?: string;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const out: Args = { provider: 'anthropic', rounds: 6 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--provider') out.provider = argv[++i] as ProviderName;
    else if (a === '--rounds') out.rounds = Number(argv[++i]);
    else if (a === '--model') out.model = argv[++i];
  }
  return out;
}

/** Deterministic filler of roughly `tokens` tokens (~0.75 words/token). */
function filler(tokens: number, tag: string): string {
  const words = Math.max(1, Math.round(tokens * 0.75));
  const out: string[] = [];
  for (let i = 0; i < words; i++) {
    out.push(`${tag}${i % 97}`);
  }
  return out.join(' ');
}

// ---------------------------------------------------------------------------
// Scenarios. Each returns the message list for every model call (call `i`
// sends step `i`; the transcript grows append-only between calls), built under
// a per-run nonce so the two strategy runs never share a cache namespace.
// ---------------------------------------------------------------------------

const STABLE_TOKENS = 2000; // big stable context (instructions / first request)
const TOOL_RESULT_TOKENS = 600; // realistic agent tool output (file/search)

function processToolCall(id: string, batch: number) {
  return { id, name: 'process_records', args: { batch } };
}

/** Agent tool loop: ONE user turn, then `rounds` assistant→tool rounds. */
function toolLoopScenario(nonce: string, rounds: number): BaseMessage[][] {
  const steps: BaseMessage[][] = [];
  const conv: BaseMessage[] = [
    new HumanMessage(
      `Session ${nonce}. Reference data follows.\n${filler(STABLE_TOKENS, `ref${nonce}`)}\n\n` +
        'Process every batch using the process_records tool until done.'
    ),
  ];
  for (let i = 1; i <= rounds; i++) {
    steps.push([...conv]);
    conv.push(
      new AIMessage({
        content: `Processing batch ${i}.`,
        tool_calls: [processToolCall(`tl_${nonce}_${i}`, i)],
      })
    );
    conv.push(
      new ToolMessage({
        tool_call_id: `tl_${nonce}_${i}`,
        content: `Batch ${i} of session ${nonce} complete. ${filler(TOOL_RESULT_TOKENS, `out${i}`)}`,
      })
    );
  }
  return steps;
}

/** Multi-turn chat: frequent user messages, no tools (legacy's good case). */
function chatScenario(nonce: string, rounds: number): BaseMessage[][] {
  const steps: BaseMessage[][] = [];
  const conv: BaseMessage[] = [
    new HumanMessage(
      `Session ${nonce}.\n${filler(STABLE_TOKENS, `doc${nonce}`)}\n\nQuestion 1: summarize.`
    ),
  ];
  for (let i = 1; i <= rounds; i++) {
    steps.push([...conv]);
    conv.push(new AIMessage(`Answer ${i}. ${filler(120, `ans${i}`)}`));
    conv.push(
      new HumanMessage(`Question ${i + 1}: ${filler(60, `q${i + 1}`)}`)
    );
  }
  return steps;
}

/** Realistic agent: each user turn triggers two tool rounds, then a new user. */
function agentMixedScenario(nonce: string, rounds: number): BaseMessage[][] {
  const steps: BaseMessage[][] = [];
  const conv: BaseMessage[] = [
    new HumanMessage(
      `Session ${nonce}. Project context:\n${filler(STABLE_TOKENS, `ctx${nonce}`)}\n\nTask 1: investigate.`
    ),
  ];
  let tc = 0;
  for (let turn = 1; turn <= rounds; turn++) {
    // two tool rounds within this user turn
    for (let r = 0; r < 2; r++) {
      steps.push([...conv]);
      tc++;
      const id = `am_${nonce}_${tc}`;
      conv.push(
        new AIMessage({
          content: `Turn ${turn} step ${r + 1}.`,
          tool_calls: [{ id, name: 'process_records', args: { step: tc } }],
        })
      );
      conv.push(
        new ToolMessage({
          tool_call_id: id,
          content: `Result ${tc} (${nonce}). ${filler(TOOL_RESULT_TOKENS, `r${tc}`)}`,
        })
      );
    }
    // model summarizes, user asks the next task
    steps.push([...conv]);
    conv.push(new AIMessage(`Turn ${turn} summary. ${filler(80, `s${turn}`)}`));
    conv.push(
      new HumanMessage(`Task ${turn + 1}: ${filler(60, `t${turn + 1}`)}`)
    );
  }
  return steps;
}

const SCENARIOS: Array<{
  name: string;
  build: (nonce: string, rounds: number) => BaseMessage[][];
}> = [
  {
    name: 'Agent tool loop (1 user turn, N tool rounds)',
    build: toolLoopScenario,
  },
  { name: 'Multi-turn chat (frequent user messages)', build: chatScenario },
  {
    name: 'Realistic agent (user turns + tool rounds)',
    build: agentMixedScenario,
  },
];

// ---------------------------------------------------------------------------
// Provider plumbing.
// ---------------------------------------------------------------------------

const PROCESS_TOOL = {
  type: 'function' as const,
  function: {
    name: 'process_records',
    description: 'Process a batch of records.',
    parameters: {
      type: 'object',
      properties: { batch: { type: 'number' }, step: { type: 'number' } },
    },
  },
};

interface StrategyPair {
  legacy: (m: BaseMessage[]) => BaseMessage[];
  tail: (m: BaseMessage[]) => BaseMessage[];
}

function makeProvider(args: Args): {
  invoke: (messages: BaseMessage[]) => Promise<Usage | undefined>;
  strategies: StrategyPair;
  label: string;
} {
  if (args.provider === 'bedrock') {
    const model = args.model ?? 'us.anthropic.claude-sonnet-4-5-20250929-v1:0';
    const llm = new CustomChatBedrockConverse({
      model,
      region:
        process.env.BEDROCK_AWS_REGION ??
        process.env.AWS_DEFAULT_REGION ??
        'us-east-1',
      credentials: {
        accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      },
      streaming: true,
      streamUsage: true,
      maxTokens: 32,
      promptCache: true,
    }).bindTools([PROCESS_TOOL]);
    return {
      label: `bedrock:${model}`,
      invoke: async (messages) =>
        (await llm.invoke(messages)).usage_metadata as Usage,
      strategies: {
        legacy: (m) => addBedrockCacheControl<BaseMessage>(m),
        tail: (m) => addBedrockTailCacheControl<BaseMessage>(m),
      },
    };
  }

  const model = args.model ?? 'claude-sonnet-4-5';
  const llm = new CustomAnthropic({
    model,
    apiKey: process.env.ANTHROPIC_API_KEY,
    maxTokens: 32,
    promptCache: true,
    streaming: true,
    streamUsage: true,
  } as never).bindTools([PROCESS_TOOL]);
  return {
    label: `anthropic:${model}`,
    invoke: async (messages) =>
      (await llm.invoke(messages)).usage_metadata as Usage,
    strategies: {
      legacy: (m) => addCacheControl<BaseMessage>(m),
      tail: (m) => addTailCacheControl<BaseMessage>(m),
    },
  };
}

type Usage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_token_details?: { cache_creation?: number; cache_read?: number };
};

interface Totals {
  read: number;
  write: number;
  fresh: number;
  effective: number;
}

function emptyTotals(): Totals {
  return { read: 0, write: 0, fresh: 0, effective: 0 };
}

function addUsage(t: Totals, u: Usage | undefined): void {
  const d = u?.input_token_details ?? {};
  const read = d.cache_read ?? 0;
  const write = d.cache_creation ?? 0;
  // Provider-agnostic fresh: total prompt tokens minus cached buckets. Avoids
  // the `input_tokens` ambiguity — Anthropic folds cache tokens INTO
  // input_tokens, while Bedrock reports input_tokens as fresh-only with cache
  // tokens separate. `total_tokens - output_tokens` is the full prompt size on
  // both, so subtracting read+write leaves the truly fresh (full-price) input.
  const promptTotal = (u?.total_tokens ?? 0) - (u?.output_tokens ?? 0);
  const fresh = Math.max(0, promptTotal - read - write);
  t.read += read;
  t.write += write;
  t.fresh += fresh;
  // Anthropic/Bedrock price multipliers: read 0.1x, write 1.25x, fresh 1x.
  t.effective += fresh + write * 1.25 + read * 0.1;
}

async function runStrategy(
  steps: BaseMessage[][],
  apply: (m: BaseMessage[]) => BaseMessage[],
  invoke: (m: BaseMessage[]) => Promise<Usage | undefined>
): Promise<Totals> {
  const totals = emptyTotals();
  for (const step of steps) {
    const usage = await invoke(apply(step));
    addUsage(totals, usage);
  }
  return totals;
}

function pct(legacy: number, tail: number): string {
  if (legacy === 0) return tail === 0 ? '0%' : 'n/a';
  const delta = ((tail - legacy) / legacy) * 100;
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(0)}%`;
}

function uniqueNonce(tag: string): string {
  return `${tag}-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

async function main(): Promise<void> {
  const args = parseArgs();
  if (args.provider === 'anthropic' && !process.env.ANTHROPIC_API_KEY) {
    console.error('Set ANTHROPIC_API_KEY (in .env or via BENCH_ENV_FILE).');
    process.exit(1);
  }
  if (args.provider === 'bedrock' && !process.env.BEDROCK_AWS_ACCESS_KEY_ID) {
    console.error(
      'Set BEDROCK_AWS_ACCESS_KEY_ID / BEDROCK_AWS_SECRET_ACCESS_KEY.'
    );
    process.exit(1);
  }

  const { invoke, strategies, label } = makeProvider(args);
  console.log(`\nProvider: ${label}   rounds=${args.rounds}`);
  console.log(
    'Metrics summed over all calls in a scenario. read↑ better; fresh↓ and effective↓ better.\n'
  );

  let tailWins = 0;
  let scenarioCount = 0;

  for (const scenario of SCENARIOS) {
    // Distinct nonce per strategy run so legacy and tail never share a cache.
    const legacySteps = scenario.build(uniqueNonce('legacy'), args.rounds);
    const legacy = await runStrategy(legacySteps, strategies.legacy, invoke);
    const tailSteps = scenario.build(uniqueNonce('tail'), args.rounds);
    const tail = await runStrategy(tailSteps, strategies.tail, invoke);

    console.log(`SCENARIO: ${scenario.name}  (${legacySteps.length} calls)`);
    const row = (name: string, t: Totals): string =>
      `  ${name.padEnd(8)} read=${String(t.read).padStart(7)}  write=${String(
        t.write
      ).padStart(7)}  fresh=${String(t.fresh).padStart(7)}  effective=${String(
        Math.round(t.effective)
      ).padStart(7)}`;
    console.log(row('legacy', legacy));
    console.log(row('tail', tail));
    console.log(
      `  Δ tail vs legacy:  read ${pct(legacy.read, tail.read)}   ` +
        `fresh ${pct(legacy.fresh, tail.fresh)}   ` +
        `effective ${pct(legacy.effective, tail.effective)} (lower=cheaper)`
    );

    const better = tail.effective <= legacy.effective;
    const tie =
      Math.abs(tail.effective - legacy.effective) / (legacy.effective || 1) <
      0.03;
    console.log(
      `  → ${better ? (tie ? '≈ TIE' : '✅ TAIL WINS') : '❌ legacy better'}\n`
    );
    scenarioCount++;
    if (better) tailWins++;
  }

  console.log(
    `RESULT: tail strategy is better-or-equal in ${tailWins}/${scenarioCount} scenarios.`
  );
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
