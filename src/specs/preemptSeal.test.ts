// src/specs/preemptSeal.test.ts
/**
 * End-to-end cooperative seal flow through a real `Run`: fake model streams,
 * the host requests a preempt, the stream seals at a safe chunk, the
 * `PreemptBoundary` drain decides what happens next. Everything below runs
 * the dispatch-synchronous loop in `attemptInvoke` (no registered
 * CHAT_MODEL_STREAM handler), which is the only loop allowed to seal.
 */
import { HumanMessage } from '@langchain/core/messages';
import { RunnableBinding } from '@langchain/core/runnables';
import {
  type OpenAIClient,
  convertMessagesToResponsesInput,
  convertResponsesDeltaToChatGenerationChunk,
} from '@langchain/openai';
import type { BaseMessage } from '@langchain/core/messages';
import type { HookCallback } from '@/hooks/types';
import type * as t from '@/types';
import { HookRegistry } from '@/hooks/HookRegistry';
import { FakeChatModel } from '@/llm/fake';
import { ChatOpenAI } from '@/llm/openai';
import { Providers } from '@/common';
import { Run } from '@/run';

const FULL_RESPONSE = 'Alpha beta gamma delta epsilon zeta';
const RESUMED_RESPONSE = 'Continuing after the steer.';

const streamConfig = {
  configurable: { thread_id: 'preempt-seal-e2e' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

async function createSealRun(options: {
  runId: string;
  hook: HookCallback<'PreemptBoundary'>;
  responses: string[];
  stopHook?: HookCallback<'Stop'>;
  modelCallbacks?: FakeChatModel['callbacks'];
}): Promise<Run<t.IState>> {
  const registry = new HookRegistry();
  registry.register('PreemptBoundary', { hooks: [options.hook] });
  if (options.stopHook) {
    registry.register('Stop', { hooks: [options.stopHook] });
  }
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
    preemption: { shouldPreempt: () => true, maxSeals: 1 },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) {
    throw new Error('Expected graph to be initialized');
  }
  const model = new FakeChatModel({
    responses: options.responses,
  });
  if (options.modelCallbacks != null) {
    model.callbacks = options.modelCallbacks;
  }
  /**
   * Wrapped, not bare, when the test watches model-level callbacks: with
   * tools bound, production hands `attemptInvoke` a `RunnableBinding` (and a
   * system runnable pipes a sequence on top), while `clientOptions.callbacks`
   * lives on the chat model at the bottom. A bare override would let a
   * naive `model.callbacks` property read pass the detector while missing
   * every real tool-enabled run.
   */
  run.Graph.overrideModel = (
    options.modelCallbacks != null
      ? new RunnableBinding({ bound: model, kwargs: {}, config: {} })
      : model
  ) as typeof model;
  return run;
}

class CountingChatModel extends FakeChatModel {
  invocations = 0;

  override async *_streamResponseChunks(
    ...args: Parameters<FakeChatModel['_streamResponseChunks']>
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    this.invocations += 1;
    yield* super._streamResponseChunks(...args);
  }
}

class ResponsesReasoningChatModel extends FakeChatModel {
  invocations: BaseMessage[][] = [];
  includeServerToolResult = false;

  _useResponsesApi(): boolean {
    return true;
  }

  override async *_streamResponseChunks(
    ...args: Parameters<FakeChatModel['_streamResponseChunks']>
  ): ReturnType<FakeChatModel['_streamResponseChunks']> {
    const [messages] = args;
    this.invocations.push(messages);
    if (this.invocations.length !== 1) {
      yield* super._streamResponseChunks(...args);
      return;
    }

    const outputOffset = this.includeServerToolResult ? 1 : 0;
    const events: Parameters<
      typeof convertResponsesDeltaToChatGenerationChunk
    >[0][] = [
      {
        type: 'response.created',
        sequence_number: 0,
        response: {
          id: 'resp_interrupted',
          created_at: 0,
          output_text: '',
          error: null,
          incomplete_details: null,
          instructions: null,
          metadata: null,
          model: 'gpt-5.6',
          object: 'response',
          output: [],
          parallel_tool_calls: true,
          temperature: null,
          tool_choice: 'auto',
          tools: [],
          top_p: null,
          status: 'in_progress',
        },
      },
      ...(this.includeServerToolResult
        ? [
          {
            type: 'response.output_item.done' as const,
            sequence_number: 1,
            output_index: 0,
            item: {
              id: 'ci_interrupted',
              type: 'code_interpreter_call' as const,
              status: 'completed' as const,
              code: 'print("server result")',
              container_id: 'container_interrupted',
              outputs: [
                { type: 'logs' as const, logs: 'server result' },
                {
                  type: 'image' as const,
                  url: 'https://example.com/ephemeral-chart.png',
                },
              ],
            },
          },
        ]
        : []),
      {
        type: 'response.output_item.added',
        sequence_number: 1 + outputOffset,
        output_index: outputOffset,
        item: {
          id: 'rs_interrupted',
          type: 'reasoning',
          status: 'in_progress',
          summary: [],
        },
      },
      {
        type: 'response.output_item.done',
        sequence_number: 2 + outputOffset,
        output_index: outputOffset,
        item: {
          id: 'rs_interrupted',
          type: 'reasoning',
          status: 'completed',
          summary: [],
          encrypted_content: 'encrypted-reasoning',
        },
      },
      {
        type: 'response.output_item.added',
        sequence_number: 3 + outputOffset,
        output_index: 1 + outputOffset,
        item: {
          id: 'msg_interrupted',
          type: 'message',
          role: 'assistant',
          status: 'in_progress',
          content: [],
        },
      },
      {
        type: 'response.output_text.delta',
        sequence_number: 4 + outputOffset,
        output_index: 1 + outputOffset,
        content_index: 0,
        item_id: 'msg_interrupted',
        delta: 'Partial answer.',
        logprobs: [],
      },
    ];
    for (const event of events) {
      const chunk = convertResponsesDeltaToChatGenerationChunk(event);
      if (chunk != null) {
        yield chunk;
      }
    }
  }
}

type StreamingResponsesDelegate = {
  completionWithRetry: (
    request: OpenAIClient.Responses.ResponseCreateParamsStreaming
  ) => Promise<AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent>>;
};

const aiContents = (messages: BaseMessage[]): string[] =>
  messages
    .filter((message) => message.getType() === 'ai')
    .map((message) =>
      typeof message.content === 'string'
        ? message.content
        : JSON.stringify(message.content)
    );

describe('cooperative seal (end-to-end via Run)', () => {
  jest.setTimeout(15000);

  it('surfaces an empty boundary as preempt_incomplete instead of a natural finish', async () => {
    const run = await createSealRun({
      runId: 'seal-empty-boundary',
      hook: async () => ({}),
      responses: [FULL_RESPONSE],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    /**
     * The answer really was cut short: the host asked to preempt, the seal
     * took the budget, and the drain had nothing to resume with. A terminal
     * consumer reading only completion events would persist a truncated
     * answer as finished — `getHaltReason()` is the channel that prevents
     * that (AgentSession emits `run.halted` off it).
     */
    expect(run.getHaltReason()).toBe('preempt_incomplete');
    expect(run.Graph?.preemptIncomplete).toBe(true);
    expect(run.Graph?.preemptEmptyBoundaries).toBe(1);

    const contents = aiContents(run.getRunMessages() ?? []);
    expect(contents).toHaveLength(1);
    expect(contents[0].length).toBeGreaterThan(0);
    expect(contents[0].length).toBeLessThan(FULL_RESPONSE.length);
    expect(FULL_RESPONSE.startsWith(contents[0])).toBe(true);
  });

  it('forwards a halting hook\'s own stopReason to Stop hooks and getHaltReason', async () => {
    let stopReasonSeen: string | undefined;
    const run = await createSealRun({
      runId: 'seal-halt-reason',
      hook: async () => ({
        preventContinuation: true,
        stopReason: 'host_policy_stop',
      }),
      responses: [FULL_RESPONSE],
      stopHook: async (input) => {
        stopReasonSeen = input.stopReason;
        return {};
      },
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    /**
     * The hook-supplied reason must win end to end: a persistence/audit Stop
     * hook records the actual cause, not the generic preempt_incomplete
     * label, and getHaltReason() reports the same string afterward. A
     * halting boundary that injected nothing also counts as an empty
     * boundary in the truncated-seal telemetry.
     */
    expect(stopReasonSeen).toBe('host_policy_stop');
    expect(run.getHaltReason()).toBe('host_policy_stop');
    expect(run.Graph?.preemptIncomplete).toBe(true);
    expect(run.Graph?.preemptEmptyBoundaries).toBe(1);
  });

  it('closes model-level callbacks for the sealed run, not just config-level ones', async () => {
    let starts = 0;
    let ends = 0;
    const run = await createSealRun({
      runId: 'seal-model-callbacks',
      hook: async () => ({
        injectedMessages: [
          { role: 'user' as const, content: 'Shorter.', source: 'steer' },
        ],
      }),
      responses: [FULL_RESPONSE, RESUMED_RESPONSE],
      /**
       * A handler supplied on the MODEL (clientOptions.callbacks) gets
       * handleChatModelStart from the real run, so the sealed turn's
       * synthetic close must reach it too — otherwise its span for the
       * sealed run never closes. Two runs (sealed + resumed): both must
       * balance.
       */
      modelCallbacks: [
        {
          handleChatModelStart: (): void => {
            starts += 1;
          },
          handleLLMEnd: (): void => {
            ends += 1;
          },
        },
      ],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    expect(run.Graph?.preemptSealCount).toBe(1);
    expect(starts).toBe(2);
    expect(ends).toBe(2);
  });

  it('a halting boundary stops multi-agent successors, not just the sealed subgraph', async () => {
    const registry = new HookRegistry();
    registry.register('PreemptBoundary', {
      hooks: [
        async () => ({
          preventContinuation: true,
          stopReason: 'stop_everything',
        }),
      ],
    });
    const run = await Run.create<t.IState>({
      runId: 'seal-halt-multiagent',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent_a',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'You are agent A.',
          },
          {
            agentId: 'agent_b',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'You are agent B.',
          },
        ],
        edges: [{ from: 'agent_a', to: 'agent_b', edgeType: 'direct' }],
      },
      hooks: registry,
      preemption: { shouldPreempt: () => true, maxSeals: 1 },
      returnContent: true,
      skipCleanup: true,
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }
    const model = new CountingChatModel({
      responses: [FULL_RESPONSE, 'agent B should never say this'],
    });
    run.Graph.overrideModel = model;

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    /**
     * The registry halt is cleared to protect the sealed commit, so nothing
     * in processStream's poll stops the outer workflow — the createCallModel
     * entry guard is what keeps the direct-edge successor from taking a
     * model turn after the halting boundary.
     */
    expect(model.invocations).toBe(1);
    expect(run.getHaltReason()).toBe('stop_everything');
    expect(run.Graph.preemptIncomplete).toBe(true);
    const contents = aiContents(run.getRunMessages() ?? []);
    expect(contents.some((c) => c.includes('agent B should never'))).toBe(
      false
    );
  });

  it('resumes after an injecting boundary and completes without a halt reason', async () => {
    const run = await createSealRun({
      runId: 'seal-inject-resume',
      hook: async () => ({
        injectedMessages: [
          {
            role: 'user' as const,
            content: 'Make it shorter.',
            source: 'steer',
          },
        ],
      }),
      responses: [FULL_RESPONSE, RESUMED_RESPONSE],
    });

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    expect(run.getHaltReason()).toBeUndefined();
    expect(run.Graph?.preemptIncomplete).toBe(false);
    expect(run.Graph?.preemptSealCount).toBe(1);

    const messages = run.getRunMessages() ?? [];
    const steer = messages.find(
      (message) => message.additional_kwargs.source === 'steer'
    );
    expect(steer).toBeDefined();
    expect(steer?.content).toBe('Make it shorter.');

    /**
     * Two assistant turns: the sealed partial and the post-steer
     * continuation, which must have run to completion — the seal budget was
     * spent, so the second stream cannot seal again even though
     * `shouldPreempt` still answers true.
     */
    const contents = aiContents(messages);
    expect(contents).toHaveLength(2);
    expect(FULL_RESPONSE.startsWith(contents[0])).toBe(true);
    expect(contents[0].length).toBeLessThan(FULL_RESPONSE.length);
    expect(contents[1]).toBe(RESUMED_RESPONSE);
  });

  it.each(['v0', 'v1'] as const)(
    'does not replay interrupted OpenAI Responses item ids on %s resume',
    async (outputVersion) => {
      const run = await createSealRun({
        runId: `seal-openai-responses-reasoning-${outputVersion}`,
        hook: async () => ({
          injectedMessages: [
            { role: 'user' as const, content: 'Go on.', source: 'steer' },
          ],
        }),
        responses: [],
      });
      const model = new ResponsesReasoningChatModel({
        responses: [RESUMED_RESPONSE],
      });
      model.outputVersion = outputVersion;
      run.Graph!.overrideModel = model;

      await run.processStream(
        { messages: [new HumanMessage('hello there')] },
        streamConfig
      );

      expect(model.invocations).toHaveLength(2);
      const sealedMessage = model.invocations[1].find(
        (message) => message.getType() === 'ai'
      );
      expect(sealedMessage).toBeDefined();
      expect(sealedMessage?.text).toBe('Partial answer.');
      expect(sealedMessage?.response_metadata).not.toHaveProperty('id');
      const actualOutputVersion = (
        sealedMessage?.response_metadata as { output_version?: unknown }
      ).output_version;
      expect(actualOutputVersion).toBe(
        outputVersion === 'v1' ? 'v1' : undefined
      );

      const providerInput = convertMessagesToResponsesInput({
        messages: model.invocations[1],
        model: 'gpt-5.6',
        zdrEnabled: false,
      });
      const unsafeReasoning = providerInput.find(
        (item) =>
          item.type === 'reasoning' &&
          item.id === 'rs_interrupted' &&
          (typeof item.encrypted_content !== 'string' ||
            item.encrypted_content.length === 0)
      );
      expect(unsafeReasoning).toBeUndefined();
      expect(JSON.stringify(providerInput)).toContain('Partial answer.');
    }
  );

  it.each(['v0', 'v1'] as const)(
    'preserves completed Responses server results on %s resume without ids',
    async (outputVersion) => {
      const run = await createSealRun({
        runId: `seal-openai-responses-server-result-${outputVersion}`,
        hook: async () => ({
          injectedMessages: [
            { role: 'user' as const, content: 'Go on.', source: 'steer' },
          ],
        }),
        responses: [],
      });
      const model = new ResponsesReasoningChatModel({
        responses: [RESUMED_RESPONSE],
      });
      model.outputVersion = outputVersion;
      model.includeServerToolResult = true;
      run.Graph!.overrideModel = model;

      await run.processStream(
        { messages: [new HumanMessage('hello there')] },
        streamConfig
      );

      expect(model.invocations).toHaveLength(2);
      const sealedMessage = model.invocations[1].find(
        (message) => message.getType() === 'ai'
      );
      expect(sealedMessage).toBeDefined();
      expect(sealedMessage?.text).toContain('Partial answer.');
      expect(sealedMessage?.text).toContain('server result');
      expect(sealedMessage?.text).toContain('ephemeral-chart.png');
      expect(
        (sealedMessage?.response_metadata as { output_version?: unknown })
          .output_version
      ).toBe('v1');

      const providerInput = convertMessagesToResponsesInput({
        messages: model.invocations[1],
        model: 'gpt-5.6',
        zdrEnabled: false,
      });
      const serializedProviderInput = JSON.stringify(providerInput);
      expect(serializedProviderInput).toContain('server result');
      expect(serializedProviderInput).toContain('ephemeral-chart.png');
      expect(serializedProviderInput).not.toContain('ci_interrupted');
      expect(serializedProviderInput).not.toContain('code_interpreter_call');
      expect(serializedProviderInput).not.toContain('function_call_output');
    }
  );

  it('preserves dropped raw Responses results through a real model resume', async () => {
    const run = await createSealRun({
      runId: 'seal-openai-responses-raw-result',
      hook: async () => ({
        injectedMessages: [
          { role: 'user' as const, content: 'Go on.', source: 'steer' },
        ],
      }),
      responses: [],
    });
    const model = new ChatOpenAI({
      model: 'gpt-5.6',
      apiKey: 'test-key',
      useResponsesApi: true,
    });
    const responses = (
      model as unknown as { responses: StreamingResponsesDelegate }
    ).responses;
    const requests: OpenAIClient.Responses.ResponseCreateParamsStreaming[] = [];
    responses.completionWithRetry = async (request) => {
      requests.push(request);
      const invocation = requests.length;
      return (async function* () {
        if (invocation === 1) {
          yield {
            type: 'response.output_item.done',
            sequence_number: 0,
            output_index: 0,
            item: {
              id: 'local_output_item',
              type: 'local_shell_call_output',
              status: 'completed',
              output: 'local shell result',
            },
          } as OpenAIClient.Responses.ResponseStreamEvent;
          yield {
            type: 'response.output_item.added',
            sequence_number: 1,
            output_index: 1,
            item: {
              id: 'rs_interrupted',
              type: 'reasoning',
              status: 'in_progress',
              summary: [],
            },
          } as OpenAIClient.Responses.ResponseStreamEvent;
          yield {
            type: 'response.output_text.delta',
            sequence_number: 2,
            output_index: 2,
            content_index: 0,
            item_id: 'msg_interrupted',
            delta: 'Partial answer.',
            logprobs: [],
          } as OpenAIClient.Responses.ResponseStreamEvent;
          return;
        }
        yield {
          type: 'response.output_text.delta',
          sequence_number: 0,
          output_index: 0,
          content_index: 0,
          item_id: 'msg_resumed',
          delta: RESUMED_RESPONSE,
          logprobs: [],
        } as OpenAIClient.Responses.ResponseStreamEvent;
      })();
    };
    run.Graph!.overrideModel = model;

    await run.processStream(
      { messages: [new HumanMessage('hello there')] },
      streamConfig
    );

    expect(requests).toHaveLength(2);
    const resumedInput = JSON.stringify(requests[1].input);
    expect(resumedInput).toContain('Partial answer.');
    expect(resumedInput).toContain('local shell result');
    expect(resumedInput).toContain('serverToolResult');
    expect(resumedInput.indexOf('local shell result')).toBeLessThan(
      resumedInput.indexOf('Partial answer.')
    );
    expect(resumedInput).not.toContain('local_output_item');
    expect(resumedInput).not.toContain('local_shell_call_output');
    expect(resumedInput).not.toContain('rs_interrupted');
    expect(run.getHaltReason()).toBeUndefined();
  });
});
