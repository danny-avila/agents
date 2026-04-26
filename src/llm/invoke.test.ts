import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { describe, it, expect, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { Providers } from '@/common';

/**
 * Minimal stub model that records what `messages` get passed into
 * `invoke`/`stream`. Extends `BaseChatModel` would pull in too much
 * surface for a focused invoke-path test, so we shape just the fields
 * `attemptInvoke` reads.
 */
function buildCapturingModel(): {
  invokeMessages: BaseMessage[][];
  streamMessages: BaseMessage[][];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: any;
  } {
  const invokeMessages: BaseMessage[][] = [];
  const streamMessages: BaseMessage[][] = [];

  const responseMsg = new AIMessage({ content: 'ok' });

  const model = {
    invoke: jest.fn(async (messages: BaseMessage[]): Promise<AIMessage> => {
      invokeMessages.push(messages);
      return responseMsg;
    }),
  } as Record<string, unknown>;

  return { invokeMessages, streamMessages, model };
}

function buildStreamingCapturingModel(): {
  streamMessages: BaseMessage[][];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: any;
  } {
  const streamMessages: BaseMessage[][] = [];
  const model = {
    stream: jest.fn(async function* (
      messages: BaseMessage[]
    ): AsyncGenerator<AIMessageChunk> {
      streamMessages.push(messages);
      yield new AIMessageChunk({ content: 'ok' });
    }),
  } as Record<string, unknown>;
  return { streamMessages, model };
}

describe('attemptInvoke applies lazy ref annotation', () => {
  it('annotates ToolMessages with live _refKey before sending to provider (non-streaming)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-1', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-1' } }
    );

    expect(invokeMessages).toHaveLength(1);
    const sent = invokeMessages[0];
    expect(sent[1].content).toBe('[ref: tool0turn0]\noutput');

    const original = messages[1] as ToolMessage;
    expect(original.content).toBe('output');
    expect(original.additional_kwargs._refKey).toBe('tool0turn0');
    expect(messages[1]).not.toBe(sent[1]);
  });

  it('annotates messages passed to model.stream (streaming path)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-2', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { streamMessages, model } = buildStreamingCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
        onChunk: () => {
          /* swallow */
        },
      },
      { configurable: { run_id: 'run-2' } }
    );

    expect(streamMessages).toHaveLength(1);
    expect(streamMessages[0][0].content).toBe('[ref: tool0turn0]\noutput');
    expect(messages[0].content).toBe('output');
  });

  it('passes messages unchanged when no registry is exposed on context (e.g. summarization)', async () => {
    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
    });

    expect(invokeMessages).toHaveLength(1);
    expect(invokeMessages[0][0].content).toBe('output');
  });

  it('skips annotation for stale _refKey not present in current run registry (cross-run scenario)', async () => {
    const registry = new ToolOutputReferenceRegistry();
    // run-3 registry holds tool0turn0 - the current run's live ref
    registry.set('run-3', 'tool0turn0', 'live-stored');

    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      // Stale ToolMessage from a hydrated prior run - its _refKey points
      // at a key that exists in registry, but conceptually different
      // semantics. For this test, use a key that doesn't exist in the
      // current registry to demonstrate the no-op behavior.
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'old',
        status: 'success',
        content: 'old-output',
        additional_kwargs: { _refKey: 'tool5turn5' },
      }),
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'new',
        status: 'success',
        content: 'new-output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-3' } }
    );

    const sent = invokeMessages[0];
    expect(sent[0].content).toBe('old-output');
    expect(sent[1].content).toBe('[ref: tool0turn0]\nnew-output');
  });

  it('applies unresolved-refs annotation regardless of registry presence', async () => {
    const registry = new ToolOutputReferenceRegistry();
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'error',
        content: 'Error: bad ref',
        additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-err' } }
    );

    expect(invokeMessages[0][0].content).toBe(
      'Error: bad ref\n[unresolved refs: tool9turn9]'
    );
  });
});

describe('tryFallbackProviders applies the same lazy annotation transform', () => {
  it('threads context through to attemptInvoke so fallback messages are annotated', async () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('run-fb', 'tool0turn0', 'stored');
    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();
    /**
     * Mock `initializeModel` indirectly by stubbing the LLM init via
     * Jest's manual `mock` so the fallback path returns our capturing
     * model. Skipping this here would require pulling in the real
     * provider init chain (Anthropic, etc.) which the rest of this
     * test layer does not bring in.
     */
    jest.doMock('@/llm/init', () => ({
      initializeModel: (): unknown => model,
    }));

    // Reset the module so the doMock takes effect.
    jest.resetModules();
    const { tryFallbackProviders: freshTry } = (await import(
      '@/llm/invoke'
    )) as { tryFallbackProviders: typeof tryFallbackProviders };

    await freshTry({
      fallbacks: [{ provider: Providers.ANTHROPIC }],
      messages,
      primaryError: new Error('primary failed'),
      context,
      config: { configurable: { run_id: 'run-fb' } },
    });

    expect(invokeMessages.length).toBeGreaterThanOrEqual(1);
    expect(invokeMessages[invokeMessages.length - 1][0].content).toBe(
      '[ref: tool0turn0]\noutput'
    );

    jest.dontMock('@/llm/init');
    jest.resetModules();
  });
});
