import { MemorySaver } from '@langchain/langgraph';
import { describe, expect, it } from '@jest/globals';
import {
  AIMessageChunk,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { OVERFLOW_SIGNATURES } from '@/utils/__tests__/fixtures/contextOverflowSignatures';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

/**
 * Every message is claimed to cost the same, so the budget the recovery
 * settles on maps directly onto a message count and the assertions can be
 * about behavior rather than arithmetic.
 */
const TOKENS_PER_MESSAGE = 40_000;

const tokenCounter: t.TokenCounter = () => TOKENS_PER_MESSAGE;

function signatureFor(model: string): Record<string, unknown> {
  const signature = OVERFLOW_SIGNATURES.find((s) => s.model === model);
  if (signature == null) {
    throw new Error(`missing fixture for ${model}`);
  }
  return signature.error;
}

/** Rebuilds a thrown provider error from a captured signature. */
function throwable(fields: Record<string, unknown>): Error {
  const error = new Error(String(fields.message));
  return Object.assign(error, fields);
}

/**
 * Fails the first N calls with a real captured provider rejection, then
 * answers normally — the shape of a run that overflows and then fits after
 * compaction.
 */
class OverflowThenSucceedModel implements t.ChatModel {
  readonly calls: BaseMessage[][] = [];

  constructor(
    private readonly error: Record<string, unknown>,
    private readonly failures = 1
  ) {}

  private record(messages: BaseMessage[]): void {
    this.calls.push(messages);
    if (this.calls.length <= this.failures) {
      throw throwable(this.error);
    }
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.record(messages);
    return new AIMessageChunk({ content: 'recovered' });
  }

  async stream(
    messages: BaseMessage[],
    _config?: RunnableConfig
  ): Promise<AsyncIterable<AIMessageChunk>> {
    this.record(messages);
    return (async function* chunks(): AsyncGenerator<AIMessageChunk> {
      yield new AIMessageChunk({ content: 'recovered' });
    })();
  }
}

function buildConversation(turns: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(new HumanMessage(`question ${i}`));
    messages.push(new AIMessage(`answer ${i}`));
  }
  messages.push(new HumanMessage('final question'));
  return messages;
}

async function createRun(options: {
  runId: string;
  maxContextTokens: number;
  checkpointer?: boolean;
}): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        disableStreaming: true,
        streamUsage: false,
      },
      maxContextTokens: options.maxContextTokens,
      compileOptions:
        options.checkpointer === true
          ? { checkpointer: new MemorySaver() }
          : undefined,
    },
    returnContent: true,
    skipCleanup: true,
    tokenCounter,
  });
}

const streamConfig = {
  configurable: { thread_id: 'context-overflow-recovery' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

describe('context overflow recovery', () => {
  it('preserves masked tool originals while checkpointed messages survive', async () => {
    const run = await createRun({
      runId: 'overflow-originals-checkpoint',
      maxContextTokens: 1_000_000,
      checkpointer: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    agentContext.preserveOriginalToolContent(new Map([[2, 'full output']]));

    run.Graph.resetValues();

    expect(agentContext.pendingOriginalToolContent).toEqual(
      new Map([[2, 'full output']])
    );
  });

  it('compacts and retries instead of surfacing the provider error', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-numbers',
      /** Deliberately wrong: far above the model's real 200k window. */
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(model.calls).toHaveLength(2);
    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
  });

  it('retargets the budget to the ceiling the provider reported', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-budget',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    run.Graph.overrideModel = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    const agentContext = run.Graph.agentContexts.get('default');
    expect(agentContext?.maxContextTokens).toBeLessThan(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);
  });

  it('sends strictly less on the retry', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-shrinks',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    expect(model.calls[1].length).toBeLessThan(model.calls[0].length);
  });

  it('recovers from a rejection that reported no numbers', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-blind',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(model.calls).toHaveLength(2);
    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
  });

  it('gives up after the bounded number of recoveries rather than looping', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-bounded',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(8) }, streamConfig)
    ).rejects.toThrow(/too long/i);

    /** Initial call plus one retry per allowed recovery. */
    expect(model.calls.length).toBeLessThanOrEqual(4);
  });

  it('restores the configured budget and allowance for the next run', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-reset',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    run.Graph.overrideModel = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );

    await run.processStream({ messages: buildConversation(8) }, streamConfig);

    const agentContext = run.Graph.agentContexts.get('default');
    expect(agentContext?.maxContextTokens).toBeLessThan(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);

    /** What `processStream` runs at the start of every turn. */
    run.Graph.resetValues();

    expect(agentContext?.maxContextTokens).toBe(1_000_000);
    expect(agentContext?.overflowRecoveryAttempts).toBe(0);
  });

  it('recovers again on a later turn of the same run', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-second-turn',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const signature = signatureFor('claude-haiku-4-5-20251001');

    for (const turn of [1, 2, 3]) {
      run.Graph.overrideModel = new OverflowThenSucceedModel(signature);
      await run.processStream(
        { messages: buildConversation(8) },
        {
          ...streamConfig,
          configurable: { thread_id: `context-overflow-recovery-${turn}` },
        }
      );
    }

    const agentContext = run.Graph.agentContexts.get('default');
    /**
     * One recovery per turn, not three accumulated — an allowance that
     * carried across turns would have stopped recovering by turn three.
     */
    expect(agentContext?.overflowRecoveryAttempts).toBe(1);
  });

  it('does not retry when neither pruning nor summarization can shrink the prompt', async () => {
    /**
     * No token counter means no pruner, and summarization is off, so the
     * summarize node deliberately no-ops — a retry would resend a
     * byte-identical prompt.
     */
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-nothing-to-shrink',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000_000,
      },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001'),
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(8) }, streamConfig)
    ).rejects.toThrow(/too long/i);

    expect(model.calls).toHaveLength(1);
    expect(
      run.Graph.agentContexts.get('default')?.overflowRecoveryAttempts
    ).toBe(0);
  });

  it('spends no summarization call on the first recovery', async () => {
    /**
     * The regression this guards: suppressing the summarize node alone is not
     * enough, because it hands control straight back to the agent node, where
     * the *configured* trigger fires by default on the very messages the
     * re-prune produced — spending the model call the staging exists to
     * avoid.
     */
    const summarizeStarts: unknown[] = [];
    const run = await Run.create<t.IState>({
      runId: 'overflow-recovery-compress-first',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.ANTHROPIC,
          disableStreaming: true,
          streamUsage: false,
        },
        maxContextTokens: 1_000_000,
        summarizationEnabled: true,
      },
      returnContent: true,
      skipCleanup: true,
      tokenCounter,
      customHandlers: {
        [GraphEvents.ON_SUMMARIZE_START]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            summarizeStarts.push(data);
          },
        },
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const content = await run.processStream(
      { messages: buildConversation(8) },
      streamConfig
    );

    expect(content).toEqual([{ type: 'text', text: 'recovered' }]);
    expect(model.calls).toHaveLength(2);
    /** Recovered on tool-output compression alone. */
    expect(summarizeStarts).toHaveLength(0);
  });

  it('keeps the corrected budget in provider units and seeds calibration', async () => {
    /**
     * The provider counted 274,468 for a prompt it caps at 200,000 — 1.37×
     * over the provider ceiling. The corrected budget stays in provider units,
     * and the provider/local ratio is installed on the pruner so it is applied
     * exactly once on this retry and later tool-call turns.
     */
    const run = await createRun({
      runId: 'overflow-recovery-proportional',
      maxContextTokens: 1_000_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      signatureFor('claude-haiku-4-5-20251001')
    );
    run.Graph.overrideModel = model;

    const messages = buildConversation(8);
    const agentContext = run.Graph.agentContexts.get('default');
    if (agentContext == null) {
      throw new Error('Expected default agent context');
    }
    agentContext.calibrationRatio = 1.5;
    await run.processStream({ messages }, streamConfig);

    const uncalibratedPrompt = model.calls[0].length * TOKENS_PER_MESSAGE;
    expect(agentContext.maxContextTokens).toBe(Math.floor(200_000 * 0.95));
    expect(agentContext.calibrationRatio).toBeCloseTo(
      274_468 / uncalibratedPrompt
    );
  });

  it('does not intercept errors compaction cannot fix', async () => {
    const run = await createRun({
      runId: 'overflow-recovery-unrelated',
      maxContextTokens: 600_000,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new OverflowThenSucceedModel(
      {
        name: 'AuthenticationError',
        status: 401,
        message: '401 Incorrect API key provided.',
      },
      Number.MAX_SAFE_INTEGER
    );
    run.Graph.overrideModel = model;

    await expect(
      run.processStream({ messages: buildConversation(2) }, streamConfig)
    ).rejects.toThrow(/Incorrect API key/);
    expect(model.calls).toHaveLength(1);
  });
});
