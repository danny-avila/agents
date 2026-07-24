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
import { Providers } from '@/common';
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
    expect(agentContext?.maxContextTokens).toBeLessThanOrEqual(200_000);
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
