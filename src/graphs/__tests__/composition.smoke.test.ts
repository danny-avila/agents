import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { HumanMessage, getBufferString } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { getProviderMessageProvenance } from '@/messages/provenance';
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Constants, Providers } from '@/common';
import { FakeChatModel } from '@/llm/fake';
import { StandardGraph } from '../Graph';

const CHAIN_PROMPT_PREFIX = 'Previous context:\n';

const makeAgent = (agentId: string): t.AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  instructions: `You are ${agentId}.`,
});

const makeConfig = (threadId: string): RunnableConfig => ({
  configurable: {
    thread_id: threadId,
  },
});

const makeStreamConfig = (threadId: string): t.WorkflowValuesStreamConfig => ({
  ...makeConfig(threadId),
  streamMode: 'values' as const,
});

const countMessageChars: t.TokenCounter = (message) =>
  Math.ceil(
    (typeof message.content === 'string'
      ? message.content.length
      : JSON.stringify(message.content).length) / 4
  );

const getAiContents = (messages: t.BaseGraphState['messages']): string[] =>
  messages
    .filter((message) => message.getType() === 'ai')
    .map((message) => message.content)
    .filter((content): content is string => typeof content === 'string');

const getChainPromptContent = (messages: BaseMessage[]): string => {
  const promptMessage = messages.find(
    (message) =>
      message.getType() === 'human' &&
      typeof message.content === 'string' &&
      message.content.startsWith(CHAIN_PROMPT_PREFIX)
  );
  if (promptMessage == null || typeof promptMessage.content !== 'string') {
    throw new Error('Expected chain prompt message');
  }
  return promptMessage.content;
};

class CapturingChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];

  constructor(responses: string[]) {
    super({ responses });
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

class GatedMessageCountChatModel extends FakeChatModel {
  private responseIndex = 0;

  constructor(
    private readonly gatedAiMessageCount: number,
    private readonly onGatedStart: () => void,
    private readonly releaseGate: Promise<void>
  ) {
    super({ responses: ['unused'] });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const aiMessageCount = messages.filter(
      (message) => message.getType() === 'ai'
    ).length;
    if (aiMessageCount === this.gatedAiMessageCount) {
      this.onGatedStart();
      await this.releaseGate;
    }
    const output = `response-${this.responseIndex++}`;
    yield this._createResponseChunk(output);
    void runManager?.handleLLMNewToken(output);
  }
}

class HandoffBridgeChatModel extends FakeChatModel {
  readonly invocations: BaseMessage[][] = [];
  private invocationIndex = 0;

  constructor() {
    super({ responses: ['unused'] });
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.invocations.push(messages);
    const invocationIndex = this.invocationIndex++;
    if (invocationIndex < 2) {
      const call =
        invocationIndex === 0
          ? { id: 'lookup_1', name: 'lookup', args: '{}' }
          : {
            id: 'transfer_1',
            name: `${Constants.LC_TRANSFER_TO_}B`,
            args: JSON.stringify({ instructions: 'Take over.' }),
          };
      yield this._createResponseChunk('', [
        {
          ...call,
          index: 0,
          type: 'tool_call_chunk',
        },
      ]);
      void runManager?.handleLLMNewToken('');
      return;
    }
    yield this._createResponseChunk('handoff complete');
    void runManager?.handleLLMNewToken('handoff complete');
  }
}

type AgentInvocation = {
  agentId: string;
  messages: BaseMessage[];
};

const expectCompiledWorkflow = (
  workflow: t.CompiledWorkflow | t.CompiledMultiAgentWorkflow
): void => {
  expect(typeof workflow.invoke).toBe('function');
  expect(typeof workflow.stream).toBe('function');
};

describe('LangGraph composition smoke tests', () => {
  it('getToolCount includes direct graph tools (host-supplied graphTools) alongside instances and definitions', () => {
    const askLikeTool = { name: 'ask_user_question' } as unknown as NonNullable<
      t.AgentInputs['graphTools']
    >[number];
    const graph = new StandardGraph({
      runId: 'toolcount-smoke',
      agents: [
        {
          ...makeAgent('agent'),
          toolDefinitions: [
            { name: 'evt1', description: 'event tool one' },
            { name: 'evt2', description: 'event tool two' },
          ],
          graphTools: [askLikeTool],
        },
      ],
    });
    // 2 schema-only event tools + 1 in-process direct tool — all bound to the
    // model and token-accounted, so all must be counted (Codex #289 P3).
    expect(graph.getToolCount()).toBe(3);
  });

  it('keeps ordinary tools executable when graphTools are added without a host toolMap (traditional mode)', () => {
    type HostTool = NonNullable<t.AgentInputs['graphTools']>[number];
    const echoTool = { name: 'echo_tool' } as unknown as HostTool;
    const askLikeTool = { name: 'ask_user_question' } as unknown as HostTool;
    const graph = new StandardGraph({
      runId: 'toolmap-merge-smoke',
      agents: [
        {
          ...makeAgent('agent'),
          tools: [echoTool],
          graphTools: [askLikeTool],
        },
      ],
    });
    const agentContext = graph.agentContexts.get('agent');
    const node = graph.initializeTools({
      currentTools: agentContext?.tools,
      currentToolMap: undefined,
      agentContext,
    });
    /**
     * ToolNode treats a provided toolMap as authoritative — if the merged map
     * built for graphTools drops the base tools, they stay bound to the model
     * but every call fails as an unknown tool (Codex #289 round 2).
     */
    const toolMap = (node as unknown as { toolMap: Map<string, unknown> })
      .toolMap;
    expect(toolMap.has('echo_tool')).toBe(true);
    expect(toolMap.has('ask_user_question')).toBe(true);
  });

  it('clears run-scoped eager tool state on reset', () => {
    const graph = new StandardGraph({
      runId: 'standard-eager-reset',
      agents: [makeAgent('agent')],
    });
    const executions = graph.eagerEventToolExecutions;
    const usageCount = graph.eagerEventToolUsageCount;
    const scopedUsageCount = graph.getEagerEventToolUsageCount('agent');
    const chunks = graph.eagerEventToolCallChunks;
    const suppressions = graph.eagerEventToolSuppressions;

    graph.eagerEventToolExecutions.set(
      'call_weather',
      {} as t.EagerEventToolExecution
    );
    graph.eagerEventToolUsageCount.set('weather', 1);
    scopedUsageCount.set('weather', 1);
    graph.eagerEventToolCallChunks.set('0', { argsText: '{"city":"NYC"}' });
    graph.eagerEventToolSuppressions.add('weather');

    graph.resetValues();

    expect(graph.eagerEventToolExecutions).toBe(executions);
    expect(graph.eagerEventToolUsageCount).toBe(usageCount);
    expect(graph.getEagerEventToolUsageCount('agent')).toBe(scopedUsageCount);
    expect(graph.eagerEventToolCallChunks).toBe(chunks);
    expect(graph.eagerEventToolSuppressions).toBe(suppressions);
    expect(graph.eagerEventToolExecutions.size).toBe(0);
    expect(graph.eagerEventToolUsageCount.size).toBe(0);
    expect(scopedUsageCount.size).toBe(0);
    expect(graph.eagerEventToolCallChunks.size).toBe(0);
    expect(graph.eagerEventToolSuppressions.size).toBe(0);
  });

  it('compiles and invokes the standard single-agent graph', async () => {
    const graph = new StandardGraph({
      runId: 'standard-smoke',
      agents: [makeAgent('agent')],
    });
    graph.overrideTestModel(['standard ok']);

    const workflow = graph.createWorkflow();
    expectCompiledWorkflow(workflow);

    const result = await workflow.invoke(
      { messages: [new HumanMessage('hello')] },
      makeConfig('standard-smoke')
    );

    expect(getAiContents(result.messages)).toEqual(['standard ok']);
  });

  it('streams values from the standard single-agent graph', async () => {
    const graph = new StandardGraph({
      runId: 'standard-stream-smoke',
      agents: [makeAgent('agent')],
    });
    graph.overrideTestModel(['standard stream ok']);

    const workflow = graph.createWorkflow();
    const stream = (await workflow.stream(
      { messages: [new HumanMessage('hello')] },
      makeStreamConfig('standard-stream-smoke')
    )) as AsyncIterable<t.BaseGraphState>;
    const chunks: t.BaseGraphState[] = [];

    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(0);
    expect(
      chunks.some((chunk) =>
        getAiContents(chunk.messages).includes('standard stream ok')
      )
    ).toBe(true);
  });

  it('compiles and invokes a multi-agent graph with one agent and no edges', async () => {
    const graph = new MultiAgentGraph({
      runId: 'multi-single-smoke',
      agents: [makeAgent('A')],
      edges: [],
    });
    graph.overrideTestModel(['multi ok']);

    const workflow = graph.createWorkflow();
    expectCompiledWorkflow(workflow);

    const result = await workflow.invoke(
      { messages: [new HumanMessage('hello')] },
      makeConfig('multi-single-smoke')
    );

    expect(getAiContents(result.messages)).toEqual(['multi ok']);
  });

  it('compiles and invokes direct sequential edges', async () => {
    const graph = new MultiAgentGraph({
      runId: 'direct-chain-smoke',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: 'A', to: 'B', edgeType: 'direct' }],
    });
    graph.overrideTestModel(['from A', 'from B']);

    const workflow = graph.createWorkflow();
    expectCompiledWorkflow(workflow);

    const result = await workflow.invoke(
      { messages: [new HumanMessage('start')] },
      makeConfig('direct-chain-smoke')
    );

    expect(getAiContents(result.messages)).toEqual(['from A', 'from B']);
  });

  it('does not duplicate excludeResults chain prompt history for downstream agents', async () => {
    const model = new CapturingChatModel(['from A', 'from B', 'from C']);
    const prompt = (messages: BaseMessage[], startIndex: number): string =>
      `${CHAIN_PROMPT_PREFIX}${getBufferString(messages.slice(startIndex))}`;
    const graph = new MultiAgentGraph({
      runId: 'exclude-results-chain-smoke',
      agents: [makeAgent('A'), makeAgent('B'), makeAgent('C')],
      edges: [
        {
          from: 'A',
          to: 'B',
          edgeType: 'direct',
          prompt,
          excludeResults: true,
        },
        {
          from: 'B',
          to: 'C',
          edgeType: 'direct',
          prompt,
          excludeResults: true,
        },
      ],
    });
    graph.overrideModel = model;

    const result = await graph
      .createWorkflow()
      .invoke(
        { messages: [new HumanMessage('start')] },
        makeConfig('exclude-results-chain-smoke')
      );

    expect(getAiContents(result.messages)).toEqual([
      'from A',
      'from B',
      'from C',
    ]);
    expect(model.invocations).toHaveLength(3);

    const downstreamPrompt = getChainPromptContent(model.invocations[2]);
    const previousPromptCount =
      downstreamPrompt.match(/Human: Previous context:/g)?.length ?? 0;
    expect(previousPromptCount).toBe(1);
    const routingPrompt = model.invocations[2].find(
      (message) => message.additional_kwargs.source === 'routing'
    );
    expect(routingPrompt).toBeDefined();
    expect(getProviderMessageProvenance(routingPrompt!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
  });

  it('compiles and invokes a handoff edge using graph-managed transfer tools', async () => {
    const transferToolCall: ToolCall = {
      id: 'call_transfer_to_B',
      name: `${Constants.LC_TRANSFER_TO_}B`,
      args: { instructions: 'Take over from here.' },
      type: 'tool_call',
    };
    const graph = new MultiAgentGraph({
      runId: 'handoff-smoke',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
    });
    graph.overrideTestModel(['routing to B', 'handoff complete'], undefined, [
      transferToolCall,
    ]);

    const workflow = graph.createWorkflow();
    expectCompiledWorkflow(workflow);

    const result = await workflow.invoke(
      { messages: [new HumanMessage('start')] },
      makeConfig('handoff-smoke')
    );

    expect(getAiContents(result.messages)).toContain('handoff complete');
  });

  it('marks handoff routing prompts and tool-tail assistant bridges as synthetic', async () => {
    const lookup = tool(async () => 'lookup result', {
      name: 'lookup',
      description: 'lookup',
      schema: z.object({}),
    });
    const model = new HandoffBridgeChatModel();
    const graph = new MultiAgentGraph({
      runId: 'handoff-provenance-smoke',
      agents: [
        { ...makeAgent('A'), graphTools: [lookup] },
        makeAgent('B'),
      ],
      edges: [
        {
          from: 'A',
          to: 'B',
          edgeType: 'handoff',
          prompt: 'Provide transfer instructions.',
        },
      ],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [new HumanMessage('start')] },
        makeConfig('handoff-provenance-smoke')
      );

    const recipientMessages = model.invocations.find((messages) =>
      messages.some(
        (message) => message.additional_kwargs.source === 'routing'
      )
    );
    const routingPrompt = recipientMessages?.find(
      (message) => message.additional_kwargs.source === 'routing'
    );
    const bridge = recipientMessages?.find(
      (message) =>
        message.getType() === 'ai' &&
        String(message.content).startsWith('[Processed tool result')
    );

    expect(routingPrompt).toBeDefined();
    expect(bridge).toBeDefined();
    expect(getProviderMessageProvenance(routingPrompt!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
    expect(getProviderMessageProvenance(bridge!)?.parts).toEqual([
      { attribution: 'synthetic' },
    ]);
  });

  it('compiles fan-out/fan-in direct composition with prompt wrapping', () => {
    const graph = new MultiAgentGraph({
      runId: 'fan-in-smoke',
      agents: [
        makeAgent('root'),
        makeAgent('left'),
        makeAgent('right'),
        makeAgent('final'),
      ],
      edges: [
        { from: 'root', to: ['left', 'right'], edgeType: 'direct' },
        {
          from: ['left', 'right'],
          to: 'final',
          edgeType: 'direct',
          prompt: 'Summarize these results:\n{results}',
        },
      ],
    });

    expectCompiledWorkflow(graph.createWorkflow());
    expect(graph.getParallelGroupId('root')).toBeUndefined();
    expect(graph.getParallelGroupId('left')).toBe(1);
    expect(graph.getParallelGroupId('right')).toBe(1);
    expect(graph.getParallelGroupId('final')).toBeUndefined();
  });

  it('bounds results before interpolating a routing prompt', async () => {
    const largeResult = `BEGIN\n${'large routed result '.repeat(2000)}\nEND`;
    const model = new CapturingChatModel([largeResult, 'done']);
    const graph = new MultiAgentGraph({
      runId: 'bounded-routing-results',
      tokenCounter: countMessageChars,
      agents: [
        makeAgent('source'),
        {
          ...makeAgent('destination'),
          instructions: 'routing instruction '.repeat(120),
          maxContextTokens: 1000,
        },
      ],
      edges: [
        {
          from: 'source',
          to: 'destination',
          edgeType: 'direct',
          prompt: `${'large static prefix '.repeat(1000)}\n{results}\n{results}`,
          excludeResults: true,
        },
      ],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [new HumanMessage('start')] },
        makeConfig('bounded-routing-results')
      );

    const routingPrompt = model.invocations[1].find(
      (message) => message.additional_kwargs.source === 'routing'
    );
    expect(routingPrompt?.content).toEqual(expect.stringContaining('truncated'));
    expect(String(routingPrompt?.content).length).toBeLessThan(500);
    expect(largeResult).toContain('large routed result');
  });

  it('bounds routing text returned by prompt functions', async () => {
    const largeResult = `BEGIN\n${'large routed result '.repeat(2000)}\nEND`;
    const model = new CapturingChatModel([largeResult, 'done']);
    const graph = new MultiAgentGraph({
      runId: 'bounded-function-routing-results',
      tokenCounter: countMessageChars,
      agents: [
        makeAgent('source'),
        {
          ...makeAgent('destination'),
          instructions: 'routing instruction '.repeat(120),
          maxContextTokens: 1000,
        },
      ],
      edges: [
        {
          from: 'source',
          to: 'destination',
          edgeType: 'direct',
          prompt: (messages) => getBufferString(messages),
          excludeResults: true,
        },
      ],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [new HumanMessage('start')] },
        makeConfig('bounded-function-routing-results')
      );

    const routingPrompt = model.invocations[1].find(
      (message) => message.additional_kwargs.source === 'routing'
    );
    expect(routingPrompt?.content).toEqual(expect.stringContaining('truncated'));
    expect(String(routingPrompt?.content).length).toBeLessThan(500);
    expect(largeResult).toContain('large routed result');
  });

  it('preserves an undefined function prompt as no routing message', async () => {
    const model = new CapturingChatModel(['source result', 'done']);
    const graph = new MultiAgentGraph({
      runId: 'undefined-function-routing-prompt',
      agents: [makeAgent('source'), makeAgent('destination')],
      edges: [
        {
          from: 'source',
          to: 'destination',
          edgeType: 'direct',
          prompt: () => undefined,
          excludeResults: true,
        },
      ],
    });
    graph.overrideModel = model;

    await graph
      .createWorkflow()
      .invoke(
        { messages: [new HumanMessage('start')] },
        makeConfig('undefined-function-routing-prompt')
      );

    expect(
      model.invocations[1].some(
        (message) => message.additional_kwargs.source === 'routing'
      )
    ).toBe(false);
  });

  it.each([
    ['without a prompt wrapper', undefined],
    ['with a prompt wrapper', 'Summarize these results:\n{results}'],
  ])('waits for every explicit fan-in source %s', async (_label, prompt) => {
    const invocations: AgentInvocation[] = [];
    let releaseLeft2!: () => void;
    let markLeft2Started!: () => void;
    const left2Release = new Promise<void>((resolve) => {
      releaseLeft2 = resolve;
    });
    const left2Started = new Promise<void>((resolve) => {
      markLeft2Started = resolve;
    });
    const invocationHandler = BaseCallbackHandler.fromMethods({
      handleChatModelStart: (
        _llm,
        messages,
        _runId,
        _parentRunId,
        _extraParams,
        _tags,
        metadata
      ) => {
        const agentId = metadata?.agentId;
        if (typeof agentId === 'string') {
          invocations.push({ agentId, messages: [...messages[0]] });
        }
      },
    });
    const graph = new MultiAgentGraph({
      runId: 'fan-in-waiting-edge-smoke',
      agents: [
        makeAgent('root'),
        makeAgent('left'),
        makeAgent('left2'),
        makeAgent('right'),
        makeAgent('final'),
      ],
      edges: [
        { from: 'root', to: ['left', 'right'], edgeType: 'direct' },
        { from: 'left', to: 'left2', edgeType: 'direct' },
        {
          from: ['left2', 'right'],
          to: 'final',
          edgeType: 'direct',
          prompt,
        },
      ],
    });
    graph.overrideModel = new GatedMessageCountChatModel(
      3,
      markLeft2Started,
      left2Release
    );

    const invocation = graph.createWorkflow().invoke(
      { messages: [new HumanMessage('start')] },
      {
        ...makeConfig('fan-in-waiting-edge-smoke'),
        callbacks: [invocationHandler],
      }
    );

    let gateTimeout: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        left2Started,
        new Promise<void>((_resolve, reject) => {
          gateTimeout = setTimeout(
            () =>
              reject(
                new Error(
                  `Timed out waiting for the gated branch; started: ${invocations
                    .map(({ agentId }) => agentId)
                    .join(', ')}`
                )
              ),
            5_000
          );
        }),
      ]);
    } finally {
      clearTimeout(gateTimeout);
    }
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(
      invocations.filter(({ agentId }) => agentId === 'final')
    ).toHaveLength(0);
    releaseLeft2();
    await invocation;

    const invokedAgentIds = invocations.map(({ agentId }) => agentId);
    expect(
      invokedAgentIds.filter((agentId) => agentId === 'final')
    ).toHaveLength(1);
    expect(invokedAgentIds.indexOf('final')).toBeGreaterThan(
      invokedAgentIds.indexOf('left2')
    );
    expect(invokedAgentIds.indexOf('final')).toBeGreaterThan(
      invokedAgentIds.indexOf('right')
    );
    const finalInvocation = invocations.find(
      ({ agentId }) => agentId === 'final'
    );
    if (prompt == null) {
      expect(
        finalInvocation?.messages.filter(
          (message) => message.getType() === 'ai'
        )
      ).toHaveLength(4);
    } else {
      expect(getBufferString(finalInvocation?.messages ?? [])).toContain(
        'response-3'
      );
    }
  });

  it('compiles mixed handoff and direct routing from the same agent', () => {
    const graph = new MultiAgentGraph({
      runId: 'mixed-routing-smoke',
      agents: [makeAgent('router'), makeAgent('handoff'), makeAgent('direct')],
      edges: [
        { from: 'router', to: 'handoff', edgeType: 'handoff' },
        { from: 'router', to: 'direct', edgeType: 'direct' },
      ],
    });

    expectCompiledWorkflow(graph.createWorkflow());
  });
});
