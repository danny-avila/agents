import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { describe, it, expect, jest } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import { ToolNode } from '@/tools/ToolNode';
import { Providers } from '@/common';

/**
 * Minimal stub model shape `attemptInvoke` reads. Either `invoke` or
 * `stream` is populated depending on which path the test exercises;
 * extending the real `BaseChatModel` would pull in too much surface.
 */
type StubModel = {
  invoke?: (messages: BaseMessage[], config?: unknown) => Promise<AIMessage>;
  stream?: (
    messages: BaseMessage[],
    config?: unknown
  ) => AsyncGenerator<AIMessageChunk>;
};

function buildCapturingModel(): {
  invokeMessages: BaseMessage[][];
  model: StubModel;
  } {
  const invokeMessages: BaseMessage[][] = [];
  const responseMsg = new AIMessage({ content: 'ok' });
  const model: StubModel = {
    invoke: jest.fn(async (messages: BaseMessage[]): Promise<AIMessage> => {
      invokeMessages.push(messages);
      return responseMsg;
    }),
  };
  return { invokeMessages, model };
}

function buildStreamingCapturingModel(): {
  streamMessages: BaseMessage[][];
  model: StubModel;
  } {
  const streamMessages: BaseMessage[][] = [];
  const model: StubModel = {
    stream: jest.fn(async function* (
      messages: BaseMessage[]
    ): AsyncGenerator<AIMessageChunk> {
      streamMessages.push(messages);
      yield new AIMessageChunk({ content: 'ok' });
    }),
  };
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

  it('annotates refs registered under an anonymous-batch scope (no run_id)', async () => {
    /**
     * Regression: anonymous ToolNode invocations register refs under
     * a synthetic per-batch scope (`\0anon-<n>`) that
     * `config.configurable.run_id` cannot recover. The transform must
     * read the message-stamped `_refScope` rather than relying on the
     * config-derived runId, otherwise the registry lookup misses and
     * the LLM never sees the `[ref: …]` marker.
     */
    const registry = new ToolOutputReferenceRegistry();
    const anonScope = '\0anon-0';
    registry.set(anonScope, 'tool0turn0', 'stored');

    const context = {
      getOrCreateToolOutputRegistry: () => registry,
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const messages: BaseMessage[] = [
      new ToolMessage({
        name: 'echo',
        tool_call_id: 'tc1',
        status: 'success',
        content: 'output',
        additional_kwargs: {
          _refKey: 'tool0turn0',
          _refScope: anonScope,
        },
      }),
    ];

    const { invokeMessages, model } = buildCapturingModel();

    await attemptInvoke({
      model: model as t.ChatModel,
      messages,
      provider: Providers.ANTHROPIC,
      context,
    });

    expect(invokeMessages[0][0].content).toBe('[ref: tool0turn0]\noutput');
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

describe('cross-run hydration through ToolNode + attemptInvoke', () => {
  it('annotates run 2 refs but leaves hydrated run 1 ToolMessages untouched', async () => {
    /**
     * Smoke test for the headline scenario: ToolMessages produced in
     * run 1 are persisted with clean content + `_refKey`/`_refScope`
     * metadata. When those messages are hydrated into run 2's state
     * and run 2 produces its own tool output, the annotation transform
     * must (a) annotate run 2's fresh tool message because its
     * `_refScope` is live in run 2's registry, and (b) leave run 1's
     * tool message clean because run 1's scope is not in run 2's
     * registry. Same `tool0turn0` key collides across runs without any
     * confusion.
     */
    const echo = tool(async (input) => (input as { command: string }).command, {
      name: 'echo',
      description: 'echoes its command back',
      schema: z.object({ command: z.string() }),
    }) as unknown as StructuredToolInterface;

    /* Run 1 */
    const run1Node = new ToolNode({
      tools: [echo],
      toolOutputReferences: { enabled: true },
    });
    const run1Result = (await run1Node.invoke(
      {
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'r1c1', name: 'echo', args: { command: 'run-1-output' } },
            ],
          }),
        ],
      },
      { configurable: { run_id: 'run-1' } }
    )) as { messages: ToolMessage[] };

    const run1ToolMsg = run1Result.messages[0];
    expect(run1ToolMsg.content).toBe('run-1-output');
    expect(run1ToolMsg.additional_kwargs._refKey).toBe('tool0turn0');
    expect(run1ToolMsg.additional_kwargs._refScope).toBe('run-1');

    /* Run 2 - fresh ToolNode and registry, simulating a new session */
    const run2Node = new ToolNode({
      tools: [echo],
      toolOutputReferences: { enabled: true },
    });
    const run2Result = (await run2Node.invoke(
      {
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              { id: 'r2c1', name: 'echo', args: { command: 'run-2-output' } },
            ],
          }),
        ],
      },
      { configurable: { run_id: 'run-2' } }
    )) as { messages: ToolMessage[] };

    const run2ToolMsg = run2Result.messages[0];
    expect(run2ToolMsg.content).toBe('run-2-output');
    expect(run2ToolMsg.additional_kwargs._refKey).toBe('tool0turn0');
    expect(run2ToolMsg.additional_kwargs._refScope).toBe('run-2');

    /* Hydrate run 1's message + run 2's message into a single state */
    const hydrated: BaseMessage[] = [
      new HumanMessage('first request'),
      run1ToolMsg,
      new HumanMessage('second request'),
      run2ToolMsg,
    ];

    /* attemptInvoke with run 2's registry */
    const context = {
      getOrCreateToolOutputRegistry: () =>
        run2Node._unsafeGetToolOutputRegistry(),
    } as unknown as Parameters<typeof attemptInvoke>[0]['context'];

    const { invokeMessages, model } = buildCapturingModel();
    await attemptInvoke(
      {
        model: model as t.ChatModel,
        messages: hydrated,
        provider: Providers.ANTHROPIC,
        context,
      },
      { configurable: { run_id: 'run-2' } }
    );

    const sent = invokeMessages[0];
    /* Run 1's hydrated tool message stays clean — its scope is stale */
    expect(sent[1].content).toBe('run-1-output');
    /* Run 2's tool message gets annotated — its scope is live */
    expect(sent[3].content).toBe('[ref: tool0turn0]\nrun-2-output');

    /* Persisted state is unchanged */
    expect(hydrated[1].content).toBe('run-1-output');
    expect(hydrated[3].content).toBe('run-2-output');
  });
});
