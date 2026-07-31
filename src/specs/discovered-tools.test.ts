import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { FakeChatModel } from '@/llm/fake';
import { StandardGraph } from '@/graphs';
import { askUserQuestion } from '@/hitl';
import { Providers } from '@/common';
import { Run } from '@/run';

const DISCOVERED_TOOL = 'save_issue_mcp_linear';

const searchTool = tool(
  async ({ query }) => [
    JSON.stringify({
      found: 1,
      tools: [{ name: DISCOVERED_TOOL }],
      query,
    }),
    { tool_references: [{ tool_name: DISCOVERED_TOOL }] },
  ],
  {
    name: 'tool_search',
    description: 'Find a deferred tool.',
    schema: z.object({ query: z.string() }),
    responseFormat: 'content_and_artifact',
  }
);

const askTool = tool(
  async (input) => {
    const { answer } = askUserQuestion(input);
    return answer;
  },
  {
    name: 'ask_user_question',
    description: 'Pause until the user answers a question.',
    schema: z.object({ question: z.string() }),
  }
);

class DiscoveryThenAskModel extends FakeChatModel {
  private turn = 0;

  constructor() {
    super({ responses: ['', ''] });
  }

  async *_streamResponseChunks(
    _messages: BaseMessage[],
    _options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const calls: ToolCall[][] = [
      [
        {
          name: 'tool_search',
          args: { query: 'save_issue' },
          id: 'search-call',
          type: 'tool_call',
        },
      ],
      [
        {
          name: 'ask_user_question',
          args: { question: 'Proceed?' },
          id: 'ask-call',
          type: 'tool_call',
        },
      ],
    ];
    const toolCallChunks = calls[this.turn++].map(
      (call, index): ToolCallChunk => ({
        name: call.name,
        args: JSON.stringify(call.args),
        id: call.id,
        index,
        type: 'tool_call_chunk',
      })
    );
    yield this._createResponseChunk('', toolCallChunks);
  }
}

describe('Run discovered tools', () => {
  it('exposes discoveries made before an ask-user pause', async () => {
    const run = await Run.create<t.IState>({
      runId: 'discovery-pause-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          model: 'gpt-4o-mini',
          streaming: true,
          streamUsage: false,
        },
        instructions: 'Search first, then ask the user.',
        tools: [searchTool, askTool],
        compileOptions: { checkpointer: new MemorySaver() },
      },
      returnContent: true,
      customHandlers: {},
    });
    run.Graph!.overrideModel = new DiscoveryThenAskModel();

    await run.processStream(
      { messages: [new HumanMessage('Create a Linear issue')] },
      {
        configurable: { thread_id: 'discovery-pause-thread' },
        version: 'v2',
      }
    );

    expect(run.getInterrupt()?.payload.type).toBe('ask_user_question');
    // The interrupted inner subgraph has not returned its messages to the outer
    // reducer, which is why reconstructing discoveries from run history fails.
    expect(run.getRunMessages()).toEqual([]);
    expect(run.getDiscoveredTools()).toEqual([DISCOVERED_TOOL]);

    const snapshot = run.getDiscoveredTools();
    snapshot.push('caller-mutation');
    expect(run.getDiscoveredTools()).toEqual([DISCOVERED_TOOL]);
  });

  it('retains the last snapshot when normal cleanup resets agent contexts', async () => {
    const run = await Run.create<t.IState>({
      runId: 'discovery-cleanup-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          model: 'gpt-4o-mini',
        },
        instructions: 'Test discovery cleanup.',
      },
    });
    const graph = run.Graph as StandardGraph;
    graph.agentContexts
      .get(graph.defaultAgentId)
      ?.markToolsAsDiscovered([DISCOVERED_TOOL]);

    graph.clearHeavyState();

    expect(run.getDiscoveredTools()).toEqual([DISCOVERED_TOOL]);
  });

  it('supports per-agent snapshots while returning a deduplicated union by default', async () => {
    const run = await Run.create<t.IState>({
      runId: 'multi-agent-discovery-run',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'researcher',
            provider: Providers.ANTHROPIC,
            clientOptions: {
              modelName: 'claude-haiku-4-5',
              apiKey: 'test-key',
            },
            instructions: 'Research.',
          },
          {
            agentId: 'writer',
            provider: Providers.ANTHROPIC,
            clientOptions: {
              modelName: 'claude-haiku-4-5',
              apiKey: 'test-key',
            },
            instructions: 'Write.',
          },
        ],
        edges: [{ from: 'researcher', to: 'writer', edgeType: 'direct' }],
      },
    });
    const graph = run.Graph as StandardGraph;
    graph.agentContexts
      .get('researcher')
      ?.markToolsAsDiscovered(['shared_tool', 'research_tool']);
    graph.agentContexts
      .get('writer')
      ?.markToolsAsDiscovered(['shared_tool', 'writing_tool']);

    expect(run.getDiscoveredTools('researcher')).toEqual([
      'shared_tool',
      'research_tool',
    ]);
    expect(run.getDiscoveredTools('writer')).toEqual([
      'shared_tool',
      'writing_tool',
    ]);
    expect(run.getDiscoveredTools()).toEqual([
      'shared_tool',
      'research_tool',
      'writing_tool',
    ]);

    graph.clearHeavyState();
    expect(run.getDiscoveredTools('researcher')).toEqual([
      'shared_tool',
      'research_tool',
    ]);
    expect(run.getDiscoveredTools('writer')).toEqual([
      'shared_tool',
      'writing_tool',
    ]);
    expect(run.getDiscoveredTools()).toEqual([
      'shared_tool',
      'research_tool',
      'writing_tool',
    ]);
  });
});
