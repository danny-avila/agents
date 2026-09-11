import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AgentInputs,
  GraphSubagentConfig,
  LCTool,
  PreparedSubagentContext,
  ResolvedSubagentConfig,
  StandardGraphInput,
  SubagentContextAdapter,
  SubagentContextInput,
  SubagentExecutionContext,
  ToolExecuteBatchRequest,
} from '@/types';
import {
  Constants,
  GraphEvents,
  Providers,
  SUBAGENT_CONTEXT_VERSION,
} from '@/common';
import { SubagentExecutor } from '../subagent/SubagentExecutor';
import { StandardGraph } from '@/graphs/Graph';
import { HandlerRegistry } from '@/events';
import { FakeChatModel } from '@/llm/fake';
import { HookRegistry } from '@/hooks';
import { createGraph } from '@/graphs';

const toolDefinition: LCTool = {
  name: 'inspect_file',
  description: 'Inspect a shared file.',
  parameters: { type: 'object', properties: {} },
};

function agent(agentId = 'worker'): AgentInputs {
  return {
    agentId,
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'unused' },
    maxContextTokens: 8000,
    toolDefinitions: [toolDefinition],
  };
}

function config(inputs = agent()): ResolvedSubagentConfig {
  return {
    type: 'worker',
    name: 'Worker',
    description: 'Analyze files',
    agentInputs: inputs,
  };
}

function executionId(
  context: SubagentExecutionContext | undefined
): string | undefined {
  return context?.ancestry.at(-1)?.subagentRunId;
}

function createHarness({
  adapter,
  childConfig = config(),
  toolCalls,
}: {
  adapter: SubagentContextAdapter;
  childConfig?: ResolvedSubagentConfig | GraphSubagentConfig;
  toolCalls?: (input: StandardGraphInput) => ToolCall[];
}) {
  const batches: ToolExecuteBatchRequest[] = [];
  const hookContexts: SubagentExecutionContext[] = [];
  const graphs: StandardGraph[] = [];
  const graphInputs: StandardGraphInput[] = [];
  const registry = new HandlerRegistry();
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [
      async (input) => {
        if (input.executionContext != null)
          hookContexts.push(input.executionContext);
        return {};
      },
    ],
  });
  registry.register(GraphEvents.ON_TOOL_EXECUTE, {
    handle: (_event, rawData) => {
      const batch = rawData as ToolExecuteBatchRequest;
      batches.push(batch);
      batch.resolve(
        batch.toolCalls.map((call) => ({
          toolCallId: call.id,
          status: 'success',
          content: 'File inspected',
        }))
      );
    },
  });
  const configureGraph = (
    graph: StandardGraph,
    input: StandardGraphInput
  ): StandardGraph => {
    graph.hookRegistry = hooks;
    graph.handlerRegistry = registry;
    graph.eventToolExecutionAvailable = true;
    graph.overrideTestModel(
      ['Inspecting file', 'Analysis complete'],
      0,
      toolCalls?.(input) ?? [
        {
          name: 'inspect_file',
          args: {},
          id: 'inspect-call',
          type: 'tool_call',
        },
      ]
    );
    graph.setSubagentModelOverride(
      new FakeChatModel({
        responses: ['Inspecting nested file', 'Nested analysis complete'],
        toolCalls: [
          {
            name: 'inspect_file',
            args: {},
            id: 'nested-inspect-call',
            type: 'tool_call',
          },
        ],
      })
    );
    graphInputs.push(input);
    graphs.push(graph);
    return graph;
  };
  const executor = new SubagentExecutor({
    configs: new Map([[childConfig.type, childConfig]]),
    parentRunId: 'root-run',
    parentAgentId: 'parent',
    parentHandlerRegistry: registry,
    hookRegistry: hooks,
    maxDepth: 3,
    subagentContext: adapter,
    createChildGraph: (input) =>
      configureGraph(new StandardGraph(input), input),
    createChildGraphByKind: (request) =>
      configureGraph(createGraph(request), request.input),
  });
  const execute = (parentToolCallId = 'spawn-call', signal?: AbortSignal) =>
    executor.execute({
      description: 'Analyze the attached PDF',
      subagentType: childConfig.type,
      parentToolCallId,
      threadId: 'conversation',
      signal,
      parentConfigurable: { run_id: 'root-run', thread_id: 'conversation' },
    });
  return { batches, hookContexts, graphs, graphInputs, executor, execute };
}

describe('host-owned subagent context', () => {
  it('prepares actual file content and carries trusted lineage through tools and hooks', async () => {
    const prepared: SubagentContextInput[] = [];
    let transcript: BaseMessage[] = [];
    const adapter: SubagentContextAdapter = {
      prepare: async (input) => {
        prepared.push(input);
        return {
          messages: [
            new HumanMessage({
              id: 'shared-file',
              content: [
                { type: 'text', text: 'Shared PDF: Quarterly results' },
                {
                  type: 'image_url',
                  image_url: { url: 'data:image/png;base64,aGVsbG8=' },
                },
              ],
            }),
          ],
          configurable: {
            sharedFileGrant: executionId(input.executionContext),
            run_id: 'spoofed',
          },
        };
      },
      complete: async (_input, result) => {
        transcript = result.messages;
        return { content: `${result.content}\nPublished file: durable-csv-id` };
      },
    };
    const harness = createHarness({ adapter });
    const result = await harness.execute();
    expect(SUBAGENT_CONTEXT_VERSION).toBe(1);
    expect(result.error).toBeUndefined();
    expect(result.content).toContain('Published file: durable-csv-id');
    expect(
      transcript.find((message) => message.id === 'shared-file')?.content
    ).toEqual([
      { type: 'text', text: 'Shared PDF: Quarterly results' },
      {
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,aGVsbG8=' },
      },
    ]);
    expect(prepared[0]).toMatchObject({
      parentThreadId: 'conversation',
      memberAgentIds: ['worker'],
      resumed: false,
      executionContext: { rootRunId: 'root-run', depth: 1 },
    });
    expect(harness.batches).toHaveLength(1);
    expect(harness.batches[0].executionContext).toEqual(
      prepared[0].executionContext
    );
    expect(harness.batches[0].metadata?.executionContext).toEqual(
      prepared[0].executionContext
    );
    expect(harness.batches[0].configurable?.executionContext).toEqual(
      prepared[0].executionContext
    );
    expect(harness.batches[0].configurable?.run_id).toBe('root-run');
    expect(harness.batches[0].configurable?.sharedFileGrant).toBe(
      executionId(prepared[0].executionContext)
    );
    expect(harness.hookContexts).toContainEqual(prepared[0].executionContext);
  });

  it('isolates simultaneous copies of the same agent and reauthorizes a completed retry', async () => {
    const prepared: SubagentContextInput[] = [];
    const adapter: SubagentContextAdapter = {
      prepare: async (input) => {
        prepared.push(input);
        return {
          configurable: {
            sharedFileGrant: executionId(input.executionContext),
          },
        };
      },
    };
    const harness = createHarness({ adapter });
    await Promise.all([harness.execute('first'), harness.execute('second')]);
    expect(
      new Set(
        harness.batches.map((batch) => executionId(batch.executionContext))
      ).size
    ).toBe(2);
    expect(harness.batches.map((batch) => batch.agentId)).toEqual([
      'worker',
      'worker',
    ]);
    for (const batch of harness.batches) {
      expect(batch.configurable?.sharedFileGrant).toBe(
        executionId(batch.executionContext)
      );
    }
    await harness.execute('first');
    expect(prepared).toHaveLength(3);
    expect(prepared[2].resumed).toBe(true);
    expect(executionId(prepared[2].executionContext)).toBe(
      executionId(prepared[0].executionContext)
    );
    expect(harness.batches).toHaveLength(2);
  });

  it('replaces inherited sandbox sessions with private execution partitions', async () => {
    const inherited = {
      session_id: 'parent-session',
      lastUpdated: Date.now(),
      files: [{ id: 'private-parent-file', name: 'private.txt' }],
    };
    const inputs: AgentInputs = {
      ...agent(),
      codeSessionKey: 'parent-partition',
      initialSessions: new Map([[Constants.EXECUTE_CODE, inherited]]),
      toolDefinitions: [{ ...toolDefinition, name: Constants.EXECUTE_CODE }],
    };
    const harness = createHarness({
      childConfig: config(inputs),
      adapter: {
        prepare: async (input) => ({
          agentSessions: {
            worker: { codeSessionKey: executionId(input.executionContext) },
          },
        }),
      },
      toolCalls: () => [
        {
          name: Constants.EXECUTE_CODE,
          args: { code: 'print("hello")', lang: 'python' },
          id: 'code-call',
          type: 'tool_call',
        },
      ],
    });
    const results = await Promise.all([
      harness.execute('first'),
      harness.execute('second'),
    ]);
    expect(results.every((result) => result.error == null)).toBe(true);
    expect(harness.batches).toHaveLength(2);
    for (const batch of harness.batches) {
      expect(batch.toolCalls[0].codeSessionContext).toBeUndefined();
    }
    for (const graphInput of harness.graphInputs) {
      expect(graphInput.agents[0].codeSessionKey).toBe(
        executionId(graphInput.subagentExecutionContext)
      );
      expect(graphInput.agents[0].initialSessions).toBeUndefined();
    }
    expect(harness.graphInputs[0].agents[0].codeSessionKey).not.toBe(
      harness.graphInputs[1].agents[0].codeSessionKey
    );
    expect(inputs.codeSessionKey).toBe('parent-partition');
    expect(inputs.initialSessions?.get(Constants.EXECUTE_CODE)).toBe(inherited);
  });

  it('rejects sandbox overrides for agents outside the child graph', async () => {
    const harness = createHarness({
      adapter: { prepare: async () => ({ agentSessions: { outsider: {} } }) },
    });
    expect((await harness.execute()).error).toContain('access was denied');
    expect(harness.graphs).toHaveLength(0);
  });

  it('blocks execution when preparation denies or is aborted', async () => {
    const controller = new AbortController();
    const adapter: SubagentContextAdapter = {
      prepare: async () => {
        throw new Error('Cross-run access denied');
      },
    };
    const harness = createHarness({ adapter });
    const denied = await harness.execute('denied');
    expect(denied.error).toContain('access was denied');
    expect(denied.content).not.toContain('Cross-run');
    expect(harness.graphs).toHaveLength(0);
    adapter.prepare = async (): Promise<PreparedSubagentContext> => {
      controller.abort();
      return { messages: [new HumanMessage('Must never reach a model')] };
    };
    const aborted = await harness.execute('aborted', controller.signal);
    expect(aborted.error).toBeDefined();
    expect(harness.graphs).toHaveLength(0);
  });

  it('retains completed work when result delivery needs retry', async () => {
    let attempts = 0;
    const adapter: SubagentContextAdapter = {
      prepare: async () => ({}),
      complete: async (_input, result) => {
        attempts++;
        if (attempts === 1) throw new Error('Transient publication failure');
        return { content: `${result.content}\nPublished file: durable-id` };
      },
    };
    const harness = createHarness({ adapter });
    expect((await harness.execute()).error).toContain('delivery failed');
    expect((await harness.execute()).content).toContain('durable-id');
    expect(harness.batches).toHaveLength(1);
    expect(harness.graphs).toHaveLength(1);
  });

  it('stamps self-spawn direct tools with the child execution identity', async () => {
    const configurations: RunnableConfig[] = [];
    const direct = tool(
      async (_input, runnableConfig) => {
        configurations.push(runnableConfig);
        return 'Read-only shared input';
      },
      {
        name: 'inspect_file',
        description: 'Inspect a file',
        schema: z.object({}),
      }
    );
    const self = config({
      ...agent('parent'),
      tools: [direct],
      toolDefinitions: undefined,
    });
    self.self = true;
    const harness = createHarness({
      adapter: { prepare: async () => ({}) },
      childConfig: self,
    });
    await harness.execute();
    expect(configurations).toHaveLength(1);
    const identity = configurations[0].metadata
      ?.executionContext as SubagentExecutionContext;
    expect(identity.ancestry[0]).toMatchObject({
      parentAgentId: 'parent',
      parentToolCallId: 'spawn-call',
      subagentAgentId: 'parent',
    });
    expect(configurations[0].configurable?.executionContext).toEqual(identity);
    expect(harness.hookContexts).toContainEqual(identity);
  });

  it('prepares graph members together with one graph execution identity', async () => {
    const prepared: SubagentContextInput[] = [];
    const harness = createHarness({
      adapter: {
        prepare: async (input) => {
          prepared.push(input);
          return {};
        },
      },
      childConfig: {
        kind: 'graph',
        type: 'team',
        name: 'Team',
        description: 'Analyze together',
        agents: [agent('reader'), agent('writer')],
        edges: [{ from: 'reader', to: 'writer', edgeType: 'direct' }],
        entryAgentId: 'reader',
        resultAgentId: 'writer',
      },
    });
    const result = await harness.execute();
    expect(result.error).toBeUndefined();
    expect(prepared[0].memberAgentIds).toEqual(['reader', 'writer']);
    expect(prepared[0].executionContext.ancestry[0].subagentKind).toBe('graph');
    expect(harness.batches.length).toBeGreaterThan(0);
    for (const batch of harness.batches)
      expect(batch.executionContext).toEqual(prepared[0].executionContext);
  });

  it('preserves the complete ancestry through nested delegation', async () => {
    const prepared: SubagentContextInput[] = [];
    const child = config({
      ...agent(),
      subagentConfigs: [config(agent('grandchild'))],
    });
    child.allowNested = true;
    const harness = createHarness({
      adapter: {
        prepare: async (input) => {
          prepared.push(input);
          return {
            configurable: {
              sharedFileGrant: executionId(input.executionContext),
            },
          };
        },
      },
      childConfig: child,
      toolCalls: () => [
        {
          name: Constants.SUBAGENT,
          args: { subagent_type: 'worker', description: 'Inspect the file' },
          id: 'nested-spawn',
          type: 'tool_call',
        },
      ],
    });
    const result = await harness.execute();
    expect(result.error).toBeUndefined();
    expect(prepared).toHaveLength(2);
    expect(prepared[1].executionContext.depth).toBe(2);
    expect(prepared[1].executionContext.ancestry[0]).toEqual(
      prepared[0].executionContext.ancestry[0]
    );
    expect(prepared[1].executionContext.ancestry[1]).toMatchObject({
      parentRunId: executionId(prepared[0].executionContext),
      parentToolCallId: 'nested-spawn',
      subagentAgentId: 'grandchild',
    });
    expect(harness.batches).toHaveLength(1);
    expect(harness.batches[0].executionContext).toEqual(
      prepared[1].executionContext
    );
    expect(harness.batches[0].configurable?.sharedFileGrant).toBe(
      executionId(prepared[1].executionContext)
    );
  });
});
