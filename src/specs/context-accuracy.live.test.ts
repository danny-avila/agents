// src/specs/context-accuracy.live.test.ts
/**
 * Live ACCURACY verification for ON_CONTEXT_USAGE against real provider
 * counts in the hard scenarios: tool loops (where calibration engages),
 * prompt caching (cache fields for cost math), and pruning under context
 * pressure (calibrated remaining-units math). Logs measured ratios.
 *
 * Run with:
 * RUN_CONTEXT_USAGE_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- context-accuracy.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { createTokenCounter, TokenEncoderManager } from '@/utils/tokens';
import { GraphEvents, Providers } from '@/common';
import { ModelEndHandler } from '@/events';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_CONTEXT_USAGE_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;
const modelName =
  process.env.ANTHROPIC_CONTEXT_LIVE_MODEL ?? 'claude-haiku-4-5';

function createStreamConfig(threadId: string): Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} {
  return {
    configurable: { thread_id: threadId },
    streamMode: 'values',
    version: 'v2',
  };
}

interface Captured {
  contextEvents: t.ContextUsageEvent[];
  collectedUsage: Array<{
    input_tokens?: number;
    output_tokens?: number;
    input_token_details?: { cache_creation?: number; cache_read?: number };
  }>;
  handlers: Record<string, t.EventHandler>;
}

function createCapture(): Captured {
  const contextEvents: t.ContextUsageEvent[] = [];
  const collectedUsage: Captured['collectedUsage'] = [];
  const handlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage as never),
    [GraphEvents.ON_CONTEXT_USAGE]: {
      handle: (_event, data): void => {
        contextEvents.push(data as unknown as t.ContextUsageEvent);
      },
    },
  };
  return { contextEvents, collectedUsage, handlers };
}

function estimatedUsed(event: t.ContextUsageEvent): number {
  return (event.contextBudget ?? 0) - (event.remainingContextTokens ?? 0);
}

const addTool = tool(
  async ({ a, b }: { a: number; b: number }) => String(a + b),
  {
    name: 'add',
    description: 'Add two numbers and return the sum',
    schema: z.object({ a: z.number(), b: z.number() }),
  }
);

/** ~5K tokens so the system prompt clears the haiku prompt-cache minimum;
 *  salted per run so the first call always writes a cold cache entry */
function buildLongInstructions(salt: string): string {
  return [
    `Session ${salt}: you are a precise assistant. Use the add tool for any arithmetic, then reply with only the number.`,
    ...Array.from(
      { length: 200 },
      (_, i) =>
        `Rule ${i}: always answer precisely, verify arithmetic results twice, and keep every response to a single line without commentary.`
    ),
  ].join(' ');
}

describeIfLive('Context accuracy live integration', () => {
  jest.setTimeout(240_000);

  let tokenCounter: t.TokenCounter;

  beforeAll(async () => {
    tokenCounter = await createTokenCounter();
  });

  afterAll(() => {
    TokenEncoderManager.reset();
  });

  it('tracks provider counts through a cached tool loop and tightens with calibration', async () => {
    const capture = createCapture();
    const run = await Run.create<t.IState>({
      runId: `ctx-acc-loop-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          modelName,
          apiKey: process.env.ANTHROPIC_API_KEY,
          temperature: 0,
          maxTokens: 128,
          streaming: true,
          streamUsage: true,
          promptCache: true,
        },
        instructions: buildLongInstructions(`${Date.now()}`),
        maxContextTokens: 16000,
        tools: [addTool],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: capture.handlers,
      tokenCounter,
      indexTokenCountMap: {},
    });

    await run.processStream(
      {
        messages: [
          new HumanMessage(
            'Use the add tool to compute 1742 + 2581, then reply with only the number.'
          ),
        ],
      },
      createStreamConfig(`ctx-acc-loop-${Date.now()}`)
    );

    expect(capture.contextEvents.length).toBeGreaterThanOrEqual(2);
    expect(capture.collectedUsage.length).toBe(capture.contextEvents.length);

    const ratios = capture.contextEvents.map((event, index) => {
      const usage = capture.collectedUsage[index];
      /** LangChain-normalized Anthropic usage reports cache fields as
       *  subsets of input_tokens; only add them when genuinely additive
       *  (same heuristic as calculateTotalTokens) */
      const baseInput = usage.input_tokens ?? 0;
      const cacheSum =
        (usage.input_token_details?.cache_creation ?? 0) +
        (usage.input_token_details?.cache_read ?? 0);
      const providerInput = baseInput + (cacheSum > baseInput ? cacheSum : 0);
      console.log(`[ctx-accuracy] call ${index + 1}:`, {
        estimated: estimatedUsed(event),
        providerInput,
        providerParts: {
          input: usage.input_tokens,
          cacheWrite: usage.input_token_details?.cache_creation,
          cacheRead: usage.input_token_details?.cache_read,
        },
        effectiveInstructionTokens: event.effectiveInstructionTokens,
        systemMessageTokens: event.breakdown.systemMessageTokens,
        toolSchemaTokens: event.breakdown.toolSchemaTokens,
        messageTokens: event.breakdown.messageTokens,
        remaining: event.remainingContextTokens,
        budget: event.contextBudget,
        calibrationRatio: event.calibrationRatio,
      });
      return estimatedUsed(event) / providerInput;
    });
    console.log('[ctx-accuracy] tool-loop estimate/provider ratios:', ratios);

    /** Call 1: uncalibrated local estimate must be in the right ballpark */
    expect(ratios[0]).toBeGreaterThan(0.5);
    expect(ratios[0]).toBeLessThan(2);

    /** Call 2+: calibration has real provider counts — tighter band, and
     *  no further from truth than the uncalibrated call (noise epsilon) */
    const last = ratios[ratios.length - 1];
    expect(last).toBeGreaterThan(0.6);
    expect(last).toBeLessThan(1.7);
    expect(Math.abs(last - 1)).toBeLessThanOrEqual(
      Math.abs(ratios[0] - 1) + 0.15
    );

    /** Prompt caching engaged: write on the first call, read on the next */
    const cacheWrites = capture.collectedUsage.map(
      (u) => u.input_token_details?.cache_creation ?? 0
    );
    const cacheReads = capture.collectedUsage.map(
      (u) => u.input_token_details?.cache_read ?? 0
    );
    expect(Math.max(...cacheWrites)).toBeGreaterThan(0);
    expect(Math.max(...cacheReads)).toBeGreaterThan(0);

    const text = (run.getRunMessages() ?? [])
      .filter((message) => message.getType() === 'ai')
      .map((message) =>
        typeof message.content === 'string'
          ? message.content
          : JSON.stringify(message.content)
      )
      .join(' ');
    expect(text).toContain('4323');
  });

  it('stays accurate when pruning drops history under a small budget', async () => {
    const capture = createCapture();
    const history: BaseMessage[] = [];
    for (let i = 0; i < 8; i++) {
      history.push(
        new HumanMessage(
          `Question ${i}: please summarize the following filler passage. ` +
            `${'The quick brown fox jumps over the lazy dog while counting tokens carefully. '.repeat(12)}`
        ),
        new AIMessage(
          `Answer ${i}: the passage repeats a pangram about a fox and a dog while counting tokens. `.repeat(
            4
          )
        )
      );
    }
    history.push(new HumanMessage('Reply with exactly one word: pruned'));

    const indexTokenCountMap: Record<string, number> = {};
    for (let i = 0; i < history.length; i++) {
      indexTokenCountMap[i] = tokenCounter(history[i]);
    }

    const run = await Run.create<t.IState>({
      runId: `ctx-acc-prune-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          modelName,
          apiKey: process.env.ANTHROPIC_API_KEY,
          temperature: 0,
          maxTokens: 64,
          streaming: true,
          streamUsage: true,
        },
        instructions: 'You follow instructions exactly.',
        maxContextTokens: 1500,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: capture.handlers,
      tokenCounter,
      indexTokenCountMap,
    });

    await run.processStream(
      { messages: history },
      createStreamConfig(`ctx-acc-prune-${Date.now()}`)
    );

    expect(capture.contextEvents).toHaveLength(1);
    const event = capture.contextEvents[0];
    const usage = capture.collectedUsage[0];

    /** Pruning engaged: the full history exceeds what was sent */
    expect(event.prePruneContextTokens).toBeGreaterThan(estimatedUsed(event));
    expect(event.remainingContextTokens).toBeGreaterThanOrEqual(0);
    expect(event.remainingContextTokens).toBeLessThan(event.contextBudget ?? 0);

    const providerInput = usage.input_tokens ?? 0;
    const ratio = estimatedUsed(event) / providerInput;
    console.log('[ctx-accuracy] pruned-run estimate/provider ratio:', ratio, {
      estimated: estimatedUsed(event),
      providerInput,
      prePrune: event.prePruneContextTokens,
      budget: event.contextBudget,
    });
    expect(ratio).toBeGreaterThan(0.4);
    expect(ratio).toBeLessThan(2.5);
  });
});
