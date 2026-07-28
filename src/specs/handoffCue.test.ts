// src/specs/handoffCue.test.ts
/**
 * End-to-end #345: a bare direct-edge successor on a prefill-semantics
 * provider must receive a user-turn handoff cue after the predecessor's
 * trailing assistant output — and tolerant providers must not.
 */
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type * as t from '@/types';
import { Providers } from '@/common';
import { PREDECESSOR_HANDOFF_CUE } from '@/messages/handoffCue';
import { FakeChatModel } from '@/llm/fake';
import { Run } from '@/run';

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

const streamConfig = {
  configurable: { thread_id: 'handoff-cue-e2e' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

async function runTwoAgents(
  provider: Providers,
  modelId = 'test-model'
): Promise<{ criticPayload: BaseMessage[]; messages: BaseMessage[] }> {
  const run = await Run.create<t.IState>({
    runId: `handoff-cue-${provider}`,
    graphConfig: {
      type: 'multi-agent',
      agents: [
        {
          agentId: 'writer',
          provider,
          clientOptions: { model: modelId, apiKey: 'test-key' },
          instructions: 'You are the writer.',
        },
        {
          agentId: 'critic',
          provider,
          clientOptions: { model: modelId, apiKey: 'test-key' },
          instructions: 'You are the critic.',
        },
      ],
      edges: [{ from: 'writer', to: 'critic', edgeType: 'direct' }],
    },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const model = new CapturingChatModel({
    responses: ['The essay text.', 'The critique text.'],
  });
  run.Graph.overrideModel = model;

  await run.processStream(
    { messages: [new HumanMessage('write the essay')] },
    streamConfig
  );

  expect(model.invocations).toHaveLength(2);
  return {
    criticPayload: model.invocations[1],
    messages: run.getRunMessages() ?? [],
  };
}

describe('bare direct-edge handoff cue (#345)', () => {
  jest.setTimeout(15000);

  it('anthropic successor payload ends with the cue, not the predecessor prefill', async () => {
    const { criticPayload, messages } = await runTwoAgents(
      Providers.ANTHROPIC
    );
    const last = criticPayload[criticPayload.length - 1];
    expect(last.getType()).toBe('human');
    expect(last.content).toBe(PREDECESSOR_HANDOFF_CUE);
    expect(criticPayload[criticPayload.length - 2].getType()).toBe('ai');
    /**
     * Wire-only: the cue exists in the provider payload, never in run
     * state or host-visible messages.
     */
    expect(
      messages.some((m) => m.content === PREDECESSOR_HANDOFF_CUE)
    ).toBe(false);
  });

  it('bedrock successor payload also gets the cue', async () => {
    /** Bedrock anthropic-likeness is model-dependent — needs a Claude id. */
    const { criticPayload } = await runTwoAgents(
      Providers.BEDROCK,
      'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    );
    const last = criticPayload[criticPayload.length - 1];
    expect(last.content).toBe(PREDECESSOR_HANDOFF_CUE);
  });

  it('openAI successor payload is untouched', async () => {
    const { criticPayload } = await runTwoAgents(Providers.OPENAI);
    const last = criticPayload[criticPayload.length - 1];
    expect(last.getType()).toBe('ai');
    expect(
      criticPayload.some((m) => m.content === PREDECESSOR_HANDOFF_CUE)
    ).toBe(false);
  });
});
