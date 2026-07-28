// src/llm/invoke.handoffCue.test.ts
/**
 * The handoff cue is keyed on the provider ACTUALLY serving the call, at the
 * `attemptInvoke` funnel — the seam every fallback and summarization send
 * passes through. An OpenAI primary falling back to a Claude surface must
 * gain the cue; an Anthropic primary falling back to OpenAI must not ship it.
 */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import { Providers } from '@/common';
import { PREDECESSOR_HANDOFF_CUE } from '@/messages/handoffCue';
import { FakeChatModel } from '@/llm/fake';
import { attemptInvoke, type InvokeContext } from './invoke';

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];
  /** Serving model id, as a real provider client would expose it. */
  model?: string;

  constructor(modelId?: string) {
    super({ responses: ['ok'] });
    this.model = modelId;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    yield* super._streamResponseChunks(messages, options, runManager);
  }
}

const runTail = new AIMessage({ content: 'predecessor output', id: 'run-1' });
const payload = (): BaseMessage[] => [new HumanMessage('go'), runTail];
const context = {
  getLastRunMessage: (): BaseMessage => runTail,
  getOrCreateToolOutputRegistry: (): undefined => undefined,
} as unknown as InvokeContext;

async function sentBy(
  provider: Providers,
  modelId?: string
): Promise<BaseMessage[]> {
  const model = new CapturingChatModel(modelId);
  await attemptInvoke({
    model: model as never,
    messages: payload(),
    provider,
    context,
    onChunk: async () => undefined,
  });
  return model.invocations[0];
}

describe('attemptInvoke handoff-cue funnel', () => {
  it('applies the cue when the serving provider is Anthropic', async () => {
    const sent = await sentBy(Providers.ANTHROPIC);
    expect(sent.at(-1)?.content).toBe(PREDECESSOR_HANDOFF_CUE);
  });

  it('applies the cue for Bedrock serving a Claude model', async () => {
    const sent = await sentBy(
      Providers.BEDROCK,
      'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
    );
    expect(sent.at(-1)?.content).toBe(PREDECESSOR_HANDOFF_CUE);
  });

  it('does not apply the cue for Bedrock serving a non-Claude model', async () => {
    const sent = await sentBy(Providers.BEDROCK, 'us.amazon.nova-pro-v1:0');
    expect(sent.at(-1)?.getType()).toBe('ai');
  });

  it('does not ship the Claude-only turn to a tolerant serving provider', async () => {
    const sent = await sentBy(Providers.OPENAI);
    expect(sent.at(-1)?.getType()).toBe('ai');
    expect(sent.some((m) => m.content === PREDECESSOR_HANDOFF_CUE)).toBe(
      false
    );
  });
});
