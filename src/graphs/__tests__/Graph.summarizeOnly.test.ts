import { Runnable } from '@langchain/core/runnables';
import { describe, expect, it, jest } from '@jest/globals';
import {
  AIMessageChunk,
  HumanMessage,
  AIMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import * as init from '@/llm/init';
import { Run } from '@/run';

const tokenCounter: t.TokenCounter = () => 10;

/** Answers every call with the same text and remembers what it was sent. */
class RecordingModel extends Runnable<BaseMessage[], AIMessageChunk> {
  lc_namespace = ['tests'];
  readonly calls: BaseMessage[][] = [];

  constructor(private readonly reply: string) {
    super();
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessageChunk> {
    this.calls.push(messages);
    return new AIMessageChunk({ content: this.reply });
  }
}

/** A finished conversation: the last message is the assistant's reply. */
function buildHistory(turns: number): BaseMessage[] {
  const messages: BaseMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(new HumanMessage(`question ${i}`));
    messages.push(new AIMessage(`answer ${i}`));
  }
  return messages;
}

const streamConfig = {
  configurable: { thread_id: 'summarize-only' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

interface Harness {
  run: Run<t.IState>;
  agentModel: RecordingModel;
  summarizer: RecordingModel;
  snapshots: t.ContextUsageEvent[];
  completions: t.SummarizeCompleteEvent[];
  restore: () => void;
}

async function createHarness(options: {
  runId: string;
  summarizeOnly?: boolean;
  summarizationEnabled?: boolean;
  summarizationConfig?: t.SummarizationConfig;
  withBudget?: boolean;
}): Promise<Harness> {
  const snapshots: t.ContextUsageEvent[] = [];
  const completions: t.SummarizeCompleteEvent[] = [];
  const withBudget = options.withBudget !== false;
  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        disableStreaming: true,
        streamUsage: false,
      },
      ...(withBudget ? { maxContextTokens: 100_000 } : {}),
      summarizationEnabled: options.summarizationEnabled ?? true,
      summarizeOnly: options.summarizeOnly ?? true,
      summarizationConfig: options.summarizationConfig,
    },
    returnContent: true,
    skipCleanup: true,
    ...(withBudget ? { tokenCounter } : {}),
    customHandlers: {
      [GraphEvents.ON_CONTEXT_USAGE]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          snapshots.push(data as t.ContextUsageEvent);
        },
      },
      [GraphEvents.ON_SUMMARIZE_COMPLETE]: {
        handle: (_event: string, data: t.StreamEventData): void => {
          completions.push(data as t.SummarizeCompleteEvent);
        },
      },
    },
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const agentModel = new RecordingModel('the agent must not answer');
  run.Graph.overrideModel = agentModel;
  const summarizer = new RecordingModel('CHECKPOINT: everything so far');
  const spy = jest
    .spyOn(init, 'initializeModel')
    .mockReturnValue(
      summarizer as unknown as ReturnType<typeof init.initializeModel>
    );
  return {
    run,
    agentModel,
    summarizer,
    snapshots,
    completions,
    restore: () => spy.mockRestore(),
  };
}

describe('summarize-only runs', () => {
  it('summarizes the whole history and ends without a model call', async () => {
    const harness = await createHarness({ runId: 'summarize-only-basic' });
    try {
      const history = buildHistory(3);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      /** Every history message plus the summarization instruction. */
      expect(harness.summarizer.calls[0]).toHaveLength(history.length + 1);

      expect(harness.completions).toHaveLength(1);
      expect(harness.completions[0].summary?.content?.[0]).toMatchObject({
        text: 'CHECKPOINT: everything so far',
      });

      /** Exactly one snapshot: the post-summary state the host persists. */
      expect(harness.snapshots).toHaveLength(1);
      expect(harness.snapshots[0].breakdown.summaryTokens).toBeGreaterThan(0);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(0);
    } finally {
      harness.restore();
    }
  });

  it('keeps only an explicitly configured recency window', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-retain',
      summarizationConfig: { retainRecent: { turns: 1 } },
    });
    try {
      const history = buildHistory(3);
      await harness.run.processStream({ messages: history }, streamConfig);

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      /** The last turn (two messages) stays out of the summary. */
      expect(harness.summarizer.calls[0]).toHaveLength(history.length - 2 + 1);
      expect(harness.snapshots).toHaveLength(1);
      expect(harness.snapshots[0].breakdown.messageCount).toBe(2);
    } finally {
      harness.restore();
    }
  });

  it('ends immediately when summarization is not enabled', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-disabled',
      summarizationEnabled: false,
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(2) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(0);
      expect(harness.completions).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('requests the summary even without a pruning budget', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-no-budget',
      withBudget: false,
    });
    try {
      await harness.run.processStream(
        { messages: buildHistory(2) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(1);
      expect(harness.completions).toHaveLength(1);
      expect(harness.snapshots).toHaveLength(0);
    } finally {
      harness.restore();
    }
  });

  it('re-arms for the next run on the same graph', async () => {
    const harness = await createHarness({ runId: 'summarize-only-rearm' });
    try {
      await harness.run.processStream(
        { messages: buildHistory(2) },
        streamConfig
      );
      await harness.run.processStream(
        { messages: buildHistory(4) },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(0);
      expect(harness.summarizer.calls).toHaveLength(2);
      expect(harness.completions).toHaveLength(2);
    } finally {
      harness.restore();
    }
  });

  it('leaves ordinary runs on the trigger path', async () => {
    const harness = await createHarness({
      runId: 'summarize-only-off',
      summarizeOnly: false,
    });
    try {
      const content = await harness.run.processStream(
        { messages: [...buildHistory(2), new HumanMessage('one more')] },
        streamConfig
      );

      expect(harness.agentModel.calls).toHaveLength(1);
      expect(harness.summarizer.calls).toHaveLength(0);
      expect(content).toEqual([
        { type: 'text', text: 'the agent must not answer' },
      ]);
    } finally {
      harness.restore();
    }
  });
});
