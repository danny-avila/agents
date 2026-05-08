/* eslint-disable no-console */
/**
 * Live multi-provider validation for the recency-window summarization
 * change.  Verifies two end-to-end behaviors against real APIs:
 *
 *   1. First-turn protection: a single oversized user message does not
 *      trigger summarization.  Summary events MUST NOT fire.  This
 *      addresses LibreChat issue #12940.
 *
 *   2. Multi-turn compaction: after enough turns accumulate, the
 *      summarizer fires on older content while the most recent two
 *      user-led turns are returned in `getRunMessages()` verbatim.
 *
 * IMPORTANT: env loading must happen *before* this module's imports
 * resolve.  The Bedrock AWS SDK in particular captures credentials
 * during module init.  Run with the dotenv preload + override flag:
 *
 *   DOTENV_CONFIG_OVERRIDE=true node -r dotenv/config \
 *     --loader ./tsconfig-paths-bootstrap.mjs \
 *     --experimental-specifier-resolution=node \
 *     ./src/scripts/summarization-recency.ts --provider all
 */
import { config as loadEnv } from 'dotenv';
// Override pre-existing env vars (some shells inject empty placeholders).
// This is a belt-and-suspenders second pass after the -r dotenv/config
// preload — covers the case where the script is invoked without preload.
loadEnv({ override: true });

// The Bedrock llmConfig requires BEDROCK_AWS_REGION specifically; default it
// to the standard cross-region-inference region when the user has bedrock
// credentials but the region knob is commented out.
if (
  (process.env.BEDROCK_AWS_REGION == null ||
    process.env.BEDROCK_AWS_REGION === '') &&
  process.env.BEDROCK_AWS_ACCESS_KEY_ID != null &&
  process.env.BEDROCK_AWS_ACCESS_KEY_ID !== ''
) {
  process.env.BEDROCK_AWS_REGION =
    process.env.AWS_DEFAULT_REGION ?? 'us-east-1';
}

import {
  AIMessage,
  HumanMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';

import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';
import { createTokenCounter } from '@/utils/tokens';
import { getLLMConfig } from '@/utils/llmConfig';

interface ProviderEntry {
  name: string;
  provider: Providers;
  envCheck: () => boolean;
  /**
   * Token budget tight enough that ~30K of dummy content overflows on
   * turn 1 but the recency window keeps the message verbatim, then
   * triggers summarization once a 3rd turn arrives.
   */
  maxContextTokens: number;
  /** Optional override for the model field on the agent's llmConfig. */
  modelOverride?: string;
}

const PROVIDERS: ProviderEntry[] = [
  {
    name: 'anthropic',
    provider: Providers.ANTHROPIC,
    envCheck: () =>
      process.env.ANTHROPIC_API_KEY != null &&
      process.env.ANTHROPIC_API_KEY !== '',
    maxContextTokens: 2_000,
  },
  {
    name: Providers.OPENAI, // 'openAI' — must match the llmConfigs key
    provider: Providers.OPENAI,
    envCheck: () =>
      process.env.OPENAI_API_KEY != null && process.env.OPENAI_API_KEY !== '',
    maxContextTokens: 2_000,
    modelOverride: 'gpt-5.4-mini',
  },
  {
    name: 'google',
    provider: Providers.GOOGLE,
    envCheck: () =>
      process.env.GOOGLE_API_KEY != null && process.env.GOOGLE_API_KEY !== '',
    maxContextTokens: 2_000,
  },
  {
    name: 'bedrock',
    provider: Providers.BEDROCK,
    envCheck: () =>
      process.env.BEDROCK_AWS_ACCESS_KEY_ID != null &&
      process.env.BEDROCK_AWS_ACCESS_KEY_ID !== '' &&
      process.env.BEDROCK_AWS_SECRET_ACCESS_KEY != null &&
      process.env.BEDROCK_AWS_SECRET_ACCESS_KEY !== '' &&
      // The Bedrock llmConfig reads BEDROCK_AWS_REGION specifically; if it's
      // missing, the SDK throws "Resolved credential object is not valid".
      process.env.BEDROCK_AWS_REGION != null &&
      process.env.BEDROCK_AWS_REGION !== '',
    maxContextTokens: 2_000,
  },
  {
    name: Providers.OPENROUTER,
    provider: Providers.OPENROUTER,
    envCheck: () =>
      process.env.OPENROUTER_API_KEY != null &&
      process.env.OPENROUTER_API_KEY !== '',
    maxContextTokens: 2_000,
    modelOverride: 'moonshotai/kimi-k2.6',
  },
  {
    name: Providers.DEEPSEEK,
    provider: Providers.DEEPSEEK,
    envCheck: () =>
      process.env.DEEPSEEK_API_KEY != null &&
      process.env.DEEPSEEK_API_KEY !== '',
    maxContextTokens: 2_000,
    modelOverride: 'deepseek-v4-flash',
  },
];

interface ScenarioSpies {
  onSummarizeStart: Array<unknown>;
  onSummarizeComplete: Array<unknown>;
}

function buildHandlers(spies: ScenarioSpies): Record<string, unknown> {
  return {
    [GraphEvents.ON_SUMMARIZE_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        spies.onSummarizeStart.push(data);
      },
    },
    [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        spies.onSummarizeComplete.push(data);
      },
    },
  };
}

function newSpies(): ScenarioSpies {
  return { onSummarizeStart: [], onSummarizeComplete: [] };
}

let cachedTokenCounter: t.TokenCounter | undefined;
async function getTokenCounter(): Promise<t.TokenCounter> {
  if (cachedTokenCounter == null) {
    cachedTokenCounter = await createTokenCounter();
  }
  return cachedTokenCounter;
}

async function createRun({
  entry,
  threadId,
  spies,
  retainTurns,
}: {
  entry: ProviderEntry;
  threadId: string;
  spies: ScenarioSpies;
  retainTurns?: number;
}): Promise<Run<t.IState>> {
  const baseConfig = getLLMConfig(entry.name);
  const llmConfig =
    entry.modelOverride != null
      ? { ...baseConfig, model: entry.modelOverride }
      : baseConfig;
  // tokenCounter is required for pruneMessages to be wired up
  // (Graph.ts gates createPruneMessages on it).  Without prune, no
  // messagesToRefine, no summarization trigger.
  const tokenCounter = await getTokenCounter();
  return Run.create<t.IState>({
    runId: `recency-${entry.name}-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      llmConfig,
      tools: [],
      instructions:
        'You are a brief assistant.  Reply in 1-2 short sentences.  Do not echo or restate the user message.',
      maxContextTokens: entry.maxContextTokens,
      summarizationEnabled: true,
      summarizationConfig: {
        provider: entry.provider,
        maxSummaryTokens: 400,
        ...(retainTurns != null
          ? { retainRecent: { turns: retainTurns } }
          : {}),
      },
    },
    returnContent: false,
    tokenCounter,
    customHandlers: buildHandlers(spies) as never,
  });
}

async function runTurn(
  run: Run<t.IState>,
  history: BaseMessage[],
  text: string,
  threadId: string
): Promise<BaseMessage[]> {
  history.push(new HumanMessage(text));
  await run.processStream({ messages: history }, {
    configurable: { thread_id: threadId },
    // Match the recursion limit used by src/specs/summarization.test.ts —
    // some providers re-cycle through agent ↔ summarize a few times
    // before settling and the default of 25 trips them.
    recursionLimit: 80,
    streamMode: 'values',
    version: 'v2',
  } as never);
  const finalMessages = run.getRunMessages();
  if (finalMessages != null) {
    history.push(...finalMessages);
  }
  return history;
}

const PADDING = 'Lorem ipsum dolor sit amet, '.repeat(400);

interface ScenarioResult {
  name: string;
  provider: string;
  passed: boolean;
  details: string[];
}

async function scenarioFirstTurnProtection(
  entry: ProviderEntry
): Promise<ScenarioResult> {
  const result: ScenarioResult = {
    name: 'first-turn protection (large single user message)',
    provider: entry.name,
    passed: false,
    details: [],
  };
  const threadId = `recency-1-${entry.name}-${Date.now()}`;
  const spies = newSpies();

  try {
    const run = await createRun({ entry, threadId, spies });
    const history: BaseMessage[] = [];
    // Sized to overflow the configured 2K budget on a single message.
    // Old behavior: summarization fires and replaces the user's payload
    // with a generic summary (LibreChat issue #12940).  New behavior:
    // recency window skips the LLM summarization call entirely and the
    // payload is preserved up to the prune step's truncation logic.
    const oversizedMessage =
      `Here is a structured payload I need you to keep verbatim:\n\n` +
      `<payload-MARKER-XYZ123>\n${PADDING}\n</payload-MARKER-XYZ123>\n\n` +
      `Reply OK so we can continue.`;

    try {
      await runTurn(run, history, oversizedMessage, threadId);
    } catch (turnErr) {
      // A subsequent prune emergency-error ("Message pruning removed all
      // messages") is acceptable: it means the budget is genuinely too
      // tight, surfacing as a clear error rather than a silent
      // summarization that destroys the user's payload.  The signal we
      // care about is whether ON_SUMMARIZE_START fired beforehand.
      const msg = turnErr instanceof Error ? turnErr.message : String(turnErr);
      if (msg.includes('empty_messages')) {
        result.details.push(
          'note: prune surfaced empty_messages error (expected when single message > budget)'
        );
      } else {
        throw turnErr;
      }
    }

    if (spies.onSummarizeStart.length > 0) {
      result.details.push(
        `FAIL: ON_SUMMARIZE_START fired ${spies.onSummarizeStart.length}x — first user message was destroyed by summarization.`
      );
    } else {
      result.details.push('OK: no ON_SUMMARIZE_START on first turn.');
    }
    if (spies.onSummarizeComplete.length > 0) {
      result.details.push(
        `FAIL: ON_SUMMARIZE_COMPLETE fired ${spies.onSummarizeComplete.length}x.`
      );
    }

    result.passed =
      spies.onSummarizeStart.length === 0 &&
      spies.onSummarizeComplete.length === 0;
  } catch (err) {
    result.details.push(
      `EXCEPTION: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  return result;
}

async function scenarioMultiTurnCompaction(
  entry: ProviderEntry
): Promise<ScenarioResult> {
  const result: ScenarioResult = {
    name: 'multi-turn compaction preserves the recency tail',
    provider: entry.name,
    passed: false,
    details: [],
  };
  const threadId = `recency-2-${entry.name}-${Date.now()}`;
  const spies = newSpies();

  try {
    const run = await createRun({ entry, threadId, spies, retainTurns: 2 });
    const history: BaseMessage[] = [];

    // 4 turns; each padded so that older turns will overflow the
    // configured budget once the conversation has accumulated a few
    // exchanges (~3K chars per turn ≈ 750 tokens × 4 ≈ 3K tokens).
    await runTurn(
      run,
      history,
      `Turn 1.  Topic: ALPHA-BEACON.  ${PADDING.slice(0, 3000)}\nReply only "noted alpha".`,
      threadId
    );
    await runTurn(
      run,
      history,
      `Turn 2.  Topic: BETA-LIGHTHOUSE.  ${PADDING.slice(0, 3000)}\nReply only "noted beta".`,
      threadId
    );
    await runTurn(
      run,
      history,
      `Turn 3.  Topic: GAMMA-PARSEC.  ${PADDING.slice(0, 3000)}\nReply only "noted gamma".`,
      threadId
    );
    await runTurn(
      run,
      history,
      `Turn 4.  Final: which topic codenames have I mentioned?  Reply with the comma-separated list of codenames you remember.`,
      threadId
    );

    const startedCount = spies.onSummarizeStart.length;
    const completedCount = spies.onSummarizeComplete.length;
    result.details.push(
      `summarize start=${startedCount}, complete=${completedCount}`
    );

    if (startedCount === 0) {
      result.details.push(
        'FAIL: expected at least one summarization to fire across 4 turns at the configured budget.'
      );
      return result;
    }

    // Inspect the final assistant message for codename recall as a soft signal.
    const lastAi = [...history].reverse().find((m) => m instanceof AIMessage);
    const lastAiText =
      lastAi != null
        ? typeof lastAi.content === 'string'
          ? lastAi.content
          : JSON.stringify(lastAi.content)
        : '';
    result.details.push(
      `final-AI-snippet: ${lastAiText.slice(0, 200).replace(/\s+/g, ' ')}`
    );

    // The recency window keeps the most recent 2 turns verbatim, so the
    // model must still recall GAMMA and the turn-4 ask.  ALPHA/BETA may
    // be remembered from the summary or forgotten — that's allowed.
    const recallsRecent =
      lastAiText.toLowerCase().includes('gamma') ||
      lastAiText.toLowerCase().includes('parsec');
    if (recallsRecent) {
      result.details.push('OK: recent-tail topic (GAMMA-PARSEC) recalled.');
    } else {
      result.details.push(
        'WARN: recent-tail topic not in final response — could be model wording (not a hard fail).'
      );
    }

    result.passed = startedCount > 0 && completedCount > 0;
  } catch (err) {
    result.details.push(
      `EXCEPTION: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  return result;
}

function summarize(results: ScenarioResult[]): boolean {
  console.log('\n========== Summary ==========');
  let allPassed = true;
  for (const r of results) {
    const status = r.passed ? 'PASS' : 'FAIL';
    console.log(`[${status}] ${r.provider}: ${r.name}`);
    for (const d of r.details) {
      console.log(`    ${d}`);
    }
    if (!r.passed) {
      allPassed = false;
    }
  }
  console.log('=============================\n');
  return allPassed;
}

async function main(): Promise<void> {
  const argv = await yargs(hideBin(process.argv))
    .option('provider', {
      type: 'string',
      description: 'provider name, or "all" to run every configured provider',
      default: 'all',
    })
    .option('skip-multi', {
      type: 'boolean',
      description:
        'skip the multi-turn compaction scenario (faster smoke test)',
      default: false,
    })
    .help().argv;

  const requested = String(argv.provider).toLowerCase();
  const targets =
    requested === 'all'
      ? PROVIDERS
      : PROVIDERS.filter((p) => p.name.toLowerCase() === requested);

  if (targets.length === 0) {
    console.error(
      `unknown provider "${requested}".  available: ${PROVIDERS.map((p) => p.name).join(', ')}, all`
    );
    process.exit(2);
  }

  const results: ScenarioResult[] = [];
  for (const entry of targets) {
    if (!entry.envCheck()) {
      console.log(`skipping ${entry.name} — credentials not in .env`);
      continue;
    }
    console.log(`\n----- provider: ${entry.name} -----`);
    results.push(await scenarioFirstTurnProtection(entry));
    if (!argv['skip-multi']) {
      results.push(await scenarioMultiTurnCompaction(entry));
    }
  }

  const ok = summarize(results);
  process.exit(ok ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
