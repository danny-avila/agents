import { HumanMessage, getBufferString } from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
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

const expectCompiledWorkflow = (
  workflow: t.CompiledWorkflow | t.CompiledMultiAgentWorkflow
): void => {
  expect(typeof workflow.invoke).toBe('function');
  expect(typeof workflow.stream).toBe('function');
};

describe('LangGraph composition smoke tests', () => {
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
