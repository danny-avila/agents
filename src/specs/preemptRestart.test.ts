// src/specs/preemptRestart.test.ts
/**
 * End-to-end discard-and-restart flow through a real `Run`.
 *
 * The companion to `preemptSeal.test.ts`: that file covers a preempt request
 * that arrives once the turn has an answer worth keeping. This one covers the
 * window BEFORE that — the model is silent, or streaming reasoning it has not
 * turned into text yet — where a seal is impossible and the request used to
 * wait out the whole turn. Here the in-flight call is torn down, its partial
 * output dropped, and the model called again with the boundary's injection
 * appended to the SAME prompt.
 *
 * Everything runs the dispatch-synchronous loop in `attemptInvoke` (no
 * registered CHAT_MODEL_STREAM handler), which is the only loop allowed to
 * seal or restart.
 */
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage } from '@langchain/core/messages';
import type { HookCallback } from '@/hooks/types';
import type * as t from '@/types';
import { HookRegistry } from '@/hooks/HookRegistry';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { Run } from '@/run';

const RESTARTED_RESPONSE = 'Answering the steer instead.';

/**
 * A reasoning-only chunk. The seal gate rejects these by design — several
 * providers strip or sign reasoning, so it cannot stand in for the visible
 * assistant turn a sealed sequence needs — which is exactly why a turn made of
 * nothing else is discardable rather than sealable.
 */
function thinkingChunk(text: string): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text: '',
    message: new AIMessageChunk({
      content: [{ type: 'thinking', thinking: text }],
    }),
  });
}

const streamConfig = {
  configurable: { thread_id: 'preempt-restart-e2e' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

/**
 * A host arming an interrupt, modelled the way a real one behaves: the flag is
 * level-triggered and stays true until a boundary drains it, and `wake` is a
 * hint the SDK may act on or ignore.
 */
function createHost(restartGraceMs: number): {
  arm: () => void;
  disarm: () => void;
  preemption: t.StreamPreemption;
} {
  let armed = false;
  const wakes = new Set<() => void>();
  return {
    arm: (): void => {
      armed = true;
      for (const wake of wakes) {
        wake();
      }
    },
    disarm: (): void => {
      armed = false;
    },
    preemption: {
      shouldPreempt: () => armed,
      subscribe: (wake) => {
        wakes.add(wake);
        return () => {
          wakes.delete(wake);
        };
      },
      restartGraceMs,
      maxSeals: 2,
    },
  };
}

/** Records every prompt the provider was actually asked to complete. */
class PromptRecordingModel extends FakeChatModel {
  prompts: BaseMessage[][] = [];
  /**
   * Fired from inside the first stream, so a test can arm the interrupt at a
   * realistic moment: while the provider is already working. Arming before
   * `processStream` instead exercises a different path entirely — the request
   * is outstanding before the call is issued, and the call never goes out.
   */
  onFirstStream: () => void = () => {};

  protected recordPrompt(messages: BaseMessage[]): number {
    this.prompts.push(messages);
    return this.prompts.length;
  }
}

/**
 * First call hangs without yielding anything — the silent window between a
 * request and its first chunk, where no per-chunk poll can ever run. Later
 * calls answer normally.
 */
class SilentThenAnsweringModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) === 1) {
      this.onFirstStream();
      await new Promise<never>((_resolve, reject) => {
        const { signal } = options;
        if (signal == null) {
          reject(new Error('Expected the attempt to carry an abort signal'));
          return;
        }
        signal.addEventListener(
          'abort',
          () => reject(new Error('provider stream aborted')),
          { once: true }
        );
      });
      return;
    }
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

/**
 * First call streams reasoning and nothing else, then hangs — the long
 * thinking stretch. `thinkingChunkCount` is deliberately more than one so the
 * per-chunk trigger is exercised repeatedly against the grace window.
 */
class ThinkingThenAnsweringModel extends PromptRecordingModel {
  thinkingChunkCount = 3;

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) !== 1) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }
    for (let i = 0; i < this.thinkingChunkCount; i++) {
      yield thinkingChunk(`step ${i}`);
      this.onFirstStream();
    }
    await new Promise<never>((_resolve, reject) => {
      options.signal?.addEventListener(
        'abort',
        () => reject(new Error('provider stream aborted')),
        { once: true }
      );
    });
  }
}

/** Streams reasoning and then ends, never producing text. */
class ThinkingOnlyModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.recordPrompt(messages);
    yield thinkingChunk('first');
    this.onFirstStream();
    yield thinkingChunk('second');
  }
}

/** Streams reasoning first, then real text — the turn a seal must win. */
class ThinkingThenTextModel extends PromptRecordingModel {
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    if (this.recordPrompt(messages) !== 1) {
      yield* super._streamResponseChunks(messages, options, runManager);
      return;
    }
    yield thinkingChunk('almost there');
    this.onFirstStream();
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

/** A host on the seal-only contract: armed, but with no wake channel. */
function createSealOnlyHost(): {
  arm: () => void;
  disarm: () => void;
  preemption: t.StreamPreemption;
  } {
  let armed = false;
  return {
    arm: (): void => {
      armed = true;
    },
    disarm: (): void => {
      armed = false;
    },
    preemption: { shouldPreempt: () => armed, maxSeals: 2 },
  };
}

async function createRestartRun(options: {
  runId: string;
  hook: HookCallback<'PreemptBoundary'>;
  preemption: t.StreamPreemption;
  model: FakeChatModel;
}): Promise<Run<t.IState>> {
  const registry = new HookRegistry();
  registry.register('PreemptBoundary', { hooks: [options.hook] });
  const run = await Run.create<t.IState>({
    runId: options.runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o-mini',
        apiKey: 'test-key',
      },
      instructions: 'Answer plainly.',
    },
    hooks: registry,
    preemption: options.preemption,
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  run.Graph.overrideModel = options.model;
  return run;
}

function userTexts(messages: BaseMessage[]): string[] {
  const texts: string[] = [];
  for (const message of messages) {
    if (message.getType() !== 'human') {
      continue;
    }
    texts.push(
      typeof message.content === 'string'
        ? message.content
        : JSON.stringify(message.content)
    );
  }
  return texts;
}

describe('cooperative restart (end-to-end via Run)', () => {
  it('discards a silent turn and re-issues it with the steer appended', async () => {
    const host = createHost(0);
    /**
     * The screenshot case: the provider has accepted the request and gone
     * quiet. No chunk will ever arrive to poll on, so the wake is the only
     * thing that can reach this turn.
     */
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-silent',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [
            { role: 'user' as const, content: 'actually, be brief' },
          ],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    /**
     * The whole safety argument in one assertion: the discarded turn leaves no
     * assistant message behind, so the injected turn sits directly after the
     * original question. Adjacent user turns are native on Anthropic, OpenAI
     * and Gemini, and normalized for the strict-alternation providers.
     */
    expect(userTexts(model.prompts[1])).toEqual([
      'tell me a long story',
      'actually, be brief',
    ]);
    expect(
      model.prompts[1].some((message) => message.getType() === 'ai')
    ).toBe(false);
  });

  it('discards a thinking-only turn once the request outlives the grace', async () => {
    const host = createHost(0);
    const model = new ThinkingThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-thinking',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop there' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    expect(userTexts(model.prompts[1])).toEqual([
      'think it through',
      'stop there',
    ]);
  });

  it('prefers a seal when text arrives inside the grace window', async () => {
    const host = createHost(60_000);
    const model = new ThinkingThenTextModel({
      responses: ['Partial answer.', RESTARTED_RESPONSE],
    });
    const boundaryPrompts: number[] = [];
    const run = await createRestartRun({
      runId: 'restart-defers-to-seal',
      preemption: host.preemption,
      model,
      hook: () => {
        boundaryPrompts.push(model.prompts.length);
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'go on' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think then answer')] },
      streamConfig
    );

    expect(boundaryPrompts).toHaveLength(1);
    expect(model.prompts).toHaveLength(2);
    /**
     * A seal KEEPS the turn, so the resumed prompt carries the partial
     * assistant answer. That is the whole difference from a restart, and the
     * grace window is what guarantees it: reasoning alone must not convert a
     * turn whose text was moments away.
     */
    expect(
      model.prompts[1].some((message) => message.getType() === 'ai')
    ).toBe(true);
  });

  it('re-issues the call when a restart boundary injects nothing', async () => {
    const host = createHost(0);
    const model = new SilentThenAnsweringModel({
      responses: [RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'restart-empty-boundary',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {};
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('tell me a long story')] },
      streamConfig
    );

    /**
     * A cancelled interrupt must not cost the user their answer. A seal cannot
     * self-loop here (it would leave a trailing model turn with no new input),
     * but a discard left graph state untouched, so re-entering the node simply
     * re-issues the original call.
     */
    expect(model.prompts).toHaveLength(2);
    expect(userTexts(model.prompts[1])).toEqual(['tell me a long story']);
  });
});

/**
 * Everything a restart depends on is opt-in through `subscribe`, and the lane
 * it names is where the discard's boundary gets dispatched. Without it a
 * discard would return no message AND record no lane, so the node would inject
 * nothing and end the turn empty with the steer still queued.
 */
describe('seal-only hosts', () => {
  it('never discards a thinking turn when no wake channel was supplied', async () => {
    const host = createSealOnlyHost();
    const model = new ThinkingThenTextModel({
      responses: ['Partial answer.', RESTARTED_RESPONSE],
    });
    const run = await createRestartRun({
      runId: 'seal-only-no-restart',
      preemption: host.preemption,
      model,
      hook: () => {
        host.disarm();
        return {
          injectedMessages: [{ role: 'user' as const, content: 'go on' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think then answer')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(2);
    /**
     * Sealed, not discarded: the resumed prompt carries the partial assistant
     * answer. An empty `messages[1]` with no assistant turn would mean the
     * turn was thrown away with no boundary to catch it.
     */
    expect(model.prompts[1].some((message) => message.getType() === 'ai')).toBe(
      true
    );
  });

  /**
   * The turn a seal can never accept, on a host that cannot restart. It must
   * end as an ordinary reasoning-only turn — one model call, no boundary — and
   * NOT be discarded into an empty turn with the steer still queued.
   */
  it('lets a thinking-only turn finish rather than discarding it', async () => {
    const host = createSealOnlyHost();
    const model = new ThinkingOnlyModel({ responses: [RESTARTED_RESPONSE] });
    const boundaries: number[] = [];
    const run = await createRestartRun({
      runId: 'seal-only-thinking-only',
      preemption: host.preemption,
      model,
      hook: () => {
        boundaries.push(model.prompts.length);
        return {
          injectedMessages: [{ role: 'user' as const, content: 'stop there' }],
        };
      },
    });
    model.onFirstStream = host.arm;

    await run.processStream(
      { messages: [new HumanMessage('think it through')] },
      streamConfig
    );

    expect(model.prompts).toHaveLength(1);
    expect(boundaries).toHaveLength(0);
  });
});
