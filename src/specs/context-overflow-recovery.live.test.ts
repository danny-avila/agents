/**
 * Live end-to-end verification that a real provider rejection for an
 * over-limit prompt is absorbed by forced compaction instead of surfacing.
 *
 * Each case deliberately configures `maxContextTokens` far above the model's
 * real window, so the proactive pruner does not prevent the overflow and the
 * reactive recovery path is what gets exercised.
 *
 * Run with:
 * RUN_CONTEXT_OVERFLOW_LIVE_TESTS=1 OPENROUTER_API_KEY=... npm test -- context-overflow-recovery.live.test.ts --runInBand
 *
 * Cases self-skip when their provider credentials are absent. Models are the
 * smallest-window, cheapest option per provider — the prompt has to exceed
 * the window for the test to mean anything, so a large window is a large bill.
 *
 * The Bedrock case needs `NODE_OPTIONS='--experimental-vm-modules'`; the AWS
 * SDK uses a dynamic import that jest otherwise refuses.
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createTokenCounter } from '@/utils/tokens';
import { hasAnyEnv, hasEnv } from '@/specs/spec.utils';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

jest.setTimeout(300_000);

const liveEnabled = process.env.RUN_CONTEXT_OVERFLOW_LIVE_TESTS === '1';

interface LiveCase {
  label: string;
  provider: Providers;
  model: string;
  /** The model's real input window. */
  contextWindow: number;
  envKeys: readonly string[];
  clientOptions: Record<string, unknown>;
}

const LIVE_CASES: readonly LiveCase[] = [
  {
    label: 'openrouter / qwen-2.5-7b (32k window)',
    provider: Providers.OPENROUTER,
    model: 'qwen/qwen-2.5-7b-instruct',
    contextWindow: 32_768,
    envKeys: ['OPENROUTER_API_KEY'],
    clientOptions: {
      apiKey: process.env.OPENROUTER_API_KEY,
      configuration: {
        baseURL:
          process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1',
      },
    },
  },
  {
    /**
     * Google's text models all carry a ~1M window, which would make this the
     * most expensive case in the file by an order of magnitude. This model
     * accepts ordinary text and caps its input at 64k, so it exercises the
     * same Gemini API rejection for a fraction of the tokens.
     */
    label: 'google / gemini-3.1-flash-image (64k window)',
    provider: Providers.GOOGLE,
    model: 'gemini-3.1-flash-image',
    contextWindow: 65_536,
    envKeys: ['GOOGLE_API_KEY', 'GEMINI_API_KEY'],
    clientOptions: {
      apiKey: process.env.GOOGLE_API_KEY ?? process.env.GEMINI_API_KEY,
    },
  },
  {
    label: 'anthropic / haiku (200k window)',
    provider: Providers.ANTHROPIC,
    model: 'claude-haiku-4-5-20251001',
    contextWindow: 200_000,
    envKeys: ['ANTHROPIC_API_KEY'],
    clientOptions: { apiKey: process.env.ANTHROPIC_API_KEY },
  },
  {
    label: 'bedrock / haiku (200k window)',
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    contextWindow: 200_000,
    envKeys: ['BEDROCK_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID'],
    clientOptions: {
      region:
        process.env.BEDROCK_AWS_REGION ?? process.env.AWS_REGION ?? 'us-east-1',
    },
  },
];

/**
 * Single-token words, so an N-word message is at least N tokens under every
 * provider's tokenizer — the prompt must clear the window with certainty,
 * since a request that squeaks under it is billed in full and proves nothing.
 */
const FILLER =
  'the quick brown fox jumps over lazy dog and then runs past'.split(' ');

function buildOversizedHistory(totalWords: number): BaseMessage[] {
  const perMessage = Math.ceil(totalWords / 4);
  const messages: BaseMessage[] = [];
  for (let turn = 0; turn < 4; turn++) {
    const words: string[] = new Array(perMessage);
    for (let i = 0; i < perMessage; i++) {
      words[i] = FILLER[(i + turn) % FILLER.length];
    }
    messages.push(new HumanMessage(`Notes part ${turn}: ${words.join(' ')}`));
  }
  messages.push(
    new HumanMessage('In one short sentence, what were those notes about?')
  );
  return messages;
}

const describeIfLive = liveEnabled ? describe : describe.skip;

describeIfLive('context overflow recovery (live)', () => {
  for (const testCase of LIVE_CASES) {
    const runnable = hasAnyEnv(testCase.envKeys);
    const maybeIt = runnable ? it : it.skip;

    maybeIt(
      `recovers without surfacing an error — ${testCase.label}`,
      async () => {
        const summarizeEvents: t.SummarizeCompleteEvent[] = [];
        const tokenCounter = await createTokenCounter();

        const run = await Run.create<t.IState>({
          runId: `overflow-live-${testCase.provider}-${Date.now()}`,
          graphConfig: {
            type: 'standard',
            llmConfig: {
              provider: testCase.provider,
              ...testCase.clientOptions,
              model: testCase.model,
            } as t.LLMConfig,
            /**
             * Deliberately wrong, by a wide margin: the pruner will happily
             * build a prompt the provider cannot accept, which is exactly the
             * situation the recovery path exists for.
             */
            maxContextTokens: testCase.contextWindow * 2,
            summarizationEnabled: true,
            summarizationConfig: {
              provider: testCase.provider,
              model: testCase.model,
            },
          },
          returnContent: true,
          /**
           * The assertions below read post-run `AgentContext` state, and the
           * default cleanup path calls `clearHeavyState()` → `reset()`, which
           * deliberately undoes the overflow correction for the next turn.
           */
          skipCleanup: true,
          tokenCounter,
          customHandlers: {
            [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
              handle: (_event: string, data: t.StreamEventData): void => {
                summarizeEvents.push(
                  data as unknown as t.SummarizeCompleteEvent
                );
              },
            },
          },
        });

        const messages = buildOversizedHistory(
          Math.ceil(testCase.contextWindow * 1.3)
        );

        const content = await run.processStream({ messages }, {
          configurable: { thread_id: `overflow-live-${testCase.provider}` },
          streamMode: 'values',
          version: 'v2',
        } as never);

        expect(content).toBeDefined();
        expect(Array.isArray(content)).toBe(true);
        expect((content ?? []).length).toBeGreaterThan(0);

        const agentContext = run.Graph?.agentContexts.get('default');
        expect(agentContext?.overflowRecoveryAttempts).toBeGreaterThan(0);
        /** The corrected budget must be below the fiction we started with. */
        expect(agentContext?.maxContextTokens).toBeLessThan(
          testCase.contextWindow * 2
        );
        expect(summarizeEvents.length).toBeGreaterThan(0);
      }
    );
  }

  it('has at least one runnable provider configured', () => {
    const anyRunnable = LIVE_CASES.some((testCase) =>
      testCase.envKeys.some(hasEnv)
    );
    expect(anyRunnable).toBe(true);
  });
});
