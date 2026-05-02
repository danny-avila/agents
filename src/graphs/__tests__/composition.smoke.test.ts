import { HumanMessage } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Constants, Providers } from '@/common';
import { StandardGraph } from '../Graph';

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
