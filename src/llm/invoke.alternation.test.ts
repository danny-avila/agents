// src/llm/invoke.alternation.test.ts
/**
 * `attemptInvoke` is the single funnel for primary, fallback and
 * summarization model calls, and it keys the alternation pass on the provider
 * ACTUALLY serving the call. This is the seam that protects a fallback: an
 * OpenAI primary that failed after a boundary injected two human turns hands
 * the same array to a Bedrock fallback, which rejects consecutive user turns.
 */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import { Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { attemptInvoke } from './invoke';
import type * as t from '@/types';

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  constructor() {
    super({ responses: ['ok'] });
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

const adjacentUserPayload = (): BaseMessage[] => [
  new HumanMessage({ content: 'question' }),
  new AIMessage({ content: 'partial answer' }),
  new HumanMessage({ content: 'hook context' }),
  new HumanMessage({ content: 'steer' }),
];

async function invokeAs(provider: Providers): Promise<BaseMessage[]> {
  const model = new CapturingChatModel();
  await attemptInvoke({
    model: model as unknown as t.ChatModel,
    messages: adjacentUserPayload(),
    provider,
    onChunk: async () => undefined,
  });
  expect(model.invocations).toHaveLength(1);
  return model.invocations[0];
}

describe('attemptInvoke alternation funnel', () => {
  it('coalesces adjacent user turns when the serving provider is strict', async () => {
    const sent = await invokeAs(Providers.BEDROCK);
    expect(sent.map((m) => m.getType())).toEqual(['human', 'ai', 'human']);
    expect(sent[2].content).toBe('hook context\n\nsteer');
  });

  it('leaves the payload alone for a tolerant provider', async () => {
    const sent = await invokeAs(Providers.OPENAI);
    expect(sent.map((m) => m.getType())).toEqual([
      'human',
      'ai',
      'human',
      'human',
    ]);
  });

  it('is idempotent when the graph already coalesced for the primary', async () => {
    const model = new CapturingChatModel();
    const once = await invokeAs(Providers.BEDROCK);
    await attemptInvoke({
      model: model as unknown as t.ChatModel,
      messages: once,
      provider: Providers.BEDROCK,
      onChunk: async () => undefined,
    });
    expect(model.invocations[0].map((m) => m.getType())).toEqual([
      'human',
      'ai',
      'human',
    ]);
    expect(model.invocations[0][2].content).toBe('hook context\n\nsteer');
  });
});
