// src/graphs/__tests__/MultiAgentGraph.test.ts
import { Command } from '@langchain/langgraph';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import {
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
  PROVIDER_MESSAGE_PROVENANCE_LIMITS,
  setProviderMessageProvenance,
} from '@/messages/provenance';
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Constants, Providers } from '@/common';

type HandoffReception = {
  processHandoffReception(
    messages: BaseMessage[],
    agentId: string
  ): { filteredMessages: BaseMessage[] } | null;
};

type InvocableGraphTool = {
  name?: string;
  invoke(input: ToolCall, config?: RunnableConfig): Promise<unknown>;
};

const INVALID_PROVENANCE_CASES = [
  {
    label: 'proxy-backed malformed',
    create: (): unknown => ({
      version: 1,
      parts: new Proxy([{ attribution: 'tool' }], {
        get(target, property, receiver) {
          return property === 'length'
            ? Number.NaN
            : Reflect.get(target, property, receiver);
        },
      }),
    }),
  },
  {
    label: 'plain oversized rehydrated',
    create: (): unknown => ({
      version: 1,
      parts: Array.from(
        { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1 },
        () => ({ attribution: 'tool' })
      ),
    }),
  },
] as const;

function expectCanonicalInvalidProvenance(
  message: BaseMessage,
  sourceProvenance: unknown
): void {
  expect(message.additional_kwargs.provenance).toEqual({
    version: 1,
    parts: null,
  });
  expect(message.additional_kwargs.provenance).not.toBe(sourceProvenance);
  expect(message.lc_kwargs.additional_kwargs).toBe(message.additional_kwargs);
  expect(getProviderMessageProvenance(message)).toBeUndefined();
}

describe('MultiAgentGraph.validateEdgeAgents', () => {
  const makeAgent = (agentId: string): t.AgentInputs => ({
    agentId,
    provider: Providers.OPENAI,
    instructions: 'test',
  });

  it('constructs without error when every edge endpoint has a matching agent', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
    };

    expect(() => new MultiAgentGraph(input)).not.toThrow();
  });

  it('throws a descriptive error when an edge `to` points at an unknown agent', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A')],
      edges: [{ from: 'A', to: 'MISSING', edgeType: 'handoff' }],
    };

    expect(() => new MultiAgentGraph(input)).toThrow(/MISSING/);
    expect(() => new MultiAgentGraph(input)).toThrow(
      /edges reference agent\(s\) not present in agents/
    );
  });

  it('throws when an edge `from` points at an unknown agent', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A')],
      edges: [{ from: 'MISSING', to: 'A', edgeType: 'handoff' }],
    };

    expect(() => new MultiAgentGraph(input)).toThrow(/MISSING/);
  });

  it('reports all unknown agent ids in a single error', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A')],
      edges: [
        { from: 'A', to: 'B', edgeType: 'handoff' },
        { from: 'A', to: 'C', edgeType: 'handoff' },
      ],
    };

    let thrown: Error | undefined;
    try {
      new MultiAgentGraph(input);
    } catch (err) {
      thrown = err as Error;
    }
    expect(thrown).toBeDefined();
    expect(thrown!.message).toMatch(/"B"/);
    expect(thrown!.message).toMatch(/"C"/);
  });

  it('handles array `from` / `to` fields', () => {
    const valid: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A'), makeAgent('B'), makeAgent('C')],
      edges: [{ from: ['A'], to: ['B', 'C'], edgeType: 'direct' }],
    };
    expect(() => new MultiAgentGraph(valid)).not.toThrow();

    const invalid: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: ['A'], to: ['B', 'C'], edgeType: 'direct' }],
    };
    expect(() => new MultiAgentGraph(invalid)).toThrow(/"C"/);
  });

  it('accepts an empty edges array (single-agent case with no handoffs)', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'r1',
      agents: [makeAgent('A')],
      edges: [],
    };
    expect(() => new MultiAgentGraph(input)).not.toThrow();
  });

  it('projects retained content provenance when filtering handoff calls', () => {
    const graph = new MultiAgentGraph({
      runId: 'handoff-provenance',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
    });
    const transferName = `${Constants.LC_TRANSFER_TO_}B`;
    const assistant = new AIMessage({
      id: 'assistant-source',
      content: [
        { type: 'text', text: 'model answer' },
        { type: 'text', text: 'retained server result' },
        {
          type: 'tool_use',
          id: 'transfer-call',
          name: transferName,
          input: {},
        },
      ],
      tool_calls: [
        { id: 'lookup-call', name: 'lookup', args: {} },
        { id: 'transfer-call', name: transferName, args: {} },
      ],
    });
    setProviderMessageProvenance(assistant, [
      {
        attribution: 'model',
        sourceMessageId: 'assistant-source',
        sourceContentPartIndices: [0],
      },
      {
        attribution: 'tool',
        sourceMessageId: 'server-result-source',
        sourceContentPartIndices: [1],
      },
      {
        attribution: 'model',
        sourceMessageId: 'assistant-source',
        sourceContentPartIndices: [2],
      },
    ]);
    const transferResult = new ToolMessage({
      content: 'Successfully transferred to B',
      name: transferName,
      tool_call_id: 'transfer-call',
    });

    const reception = (
      graph as unknown as HandoffReception
    ).processHandoffReception([assistant, transferResult], 'B');
    const filteredAssistant = reception?.filteredMessages[0] as AIMessage;

    expect(filteredAssistant).not.toBe(assistant);
    expect(filteredAssistant.content).toEqual([
      { type: 'text', text: 'model answer' },
      { type: 'text', text: 'retained server result' },
    ]);
    expect(filteredAssistant.tool_calls).toEqual([
      { id: 'lookup-call', name: 'lookup', args: {} },
    ]);
    expect(getProviderMessageProvenance(filteredAssistant)?.parts).toEqual([
      {
        attribution: 'model',
        sourceMessageId: 'assistant-source',
        sourceContentPartIndices: [0],
      },
      {
        attribution: 'tool',
        sourceMessageId: 'server-result-source',
        sourceContentPartIndices: [1],
      },
    ]);
    expect(getProviderSourceMessageIds(filteredAssistant)).toEqual([
      'assistant-source',
      'server-result-source',
    ]);
    expect(getProviderMessageProvenance(assistant)?.parts).toHaveLength(3);
  });

  it.each(INVALID_PROVENANCE_CASES)(
    'preserves $label provenance invalidity while filtering handoff content',
    ({ create }) => {
      const graph = new MultiAgentGraph({
        runId: 'invalid-handoff-provenance',
        agents: [makeAgent('A'), makeAgent('B')],
        edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
      });
      const transferName = `${Constants.LC_TRANSFER_TO_}B`;
      const sourceProvenance = create();
      const assistant = new AIMessage({
        content: [
          { type: 'text', text: 'retained bytes with unknown authorship' },
          {
            type: 'tool_use',
            id: 'transfer-call',
            name: transferName,
            input: {},
          },
        ],
        tool_calls: [
          { id: 'lookup-call', name: 'lookup', args: {} },
          { id: 'transfer-call', name: transferName, args: {} },
        ],
        additional_kwargs: { provenance: sourceProvenance },
      });
      const transferResult = new ToolMessage({
        content: 'Successfully transferred to B',
        name: transferName,
        tool_call_id: 'transfer-call',
      });

      const reception = (
        graph as unknown as HandoffReception
      ).processHandoffReception([assistant, transferResult], 'B');
      const filteredAssistant = reception?.filteredMessages[0] as AIMessage;

      expect(filteredAssistant.content).toEqual([
        { type: 'text', text: 'retained bytes with unknown authorship' },
      ]);
      expectCanonicalInvalidProvenance(
        filteredAssistant,
        sourceProvenance
      );
      expect(assistant.additional_kwargs.provenance).toBe(sourceProvenance);
    }
  );

  it('copies provenance into a parallel handoff tool-call projection', async () => {
    const graph = new MultiAgentGraph({
      runId: 'parallel-handoff-provenance',
      agents: [makeAgent('A'), makeAgent('B')],
      edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
    });
    const transferName = `${Constants.LC_TRANSFER_TO_}B`;
    const transferCall: ToolCall = {
      id: 'transfer-call',
      name: transferName,
      args: {},
      type: 'tool_call',
    };
    const assistant = new AIMessage({
      id: 'parallel-assistant-source',
      content: 'routing with retained model text',
      tool_calls: [
        transferCall,
        { id: 'parallel-call', name: 'lookup', args: {} },
      ],
    });
    setProviderMessageProvenance(assistant, [
      {
        attribution: 'model',
        sourceMessageId: 'parallel-assistant-source',
      },
    ]);
    const graphTools = graph.agentContexts.get('A')?.graphTools as
      | InvocableGraphTool[]
      | undefined;
    const handoffTool = graphTools?.find(
      (candidate) => candidate.name === transferName
    );
    if (handoffTool == null) {
      throw new Error('Expected handoff tool');
    }

    const output = await handoffTool.invoke(transferCall, {
      state: { messages: [assistant] },
    } as unknown as RunnableConfig);
    const command = output as Command<unknown, { messages: BaseMessage[] }>;
    const update = command.update;
    if (update == null || Array.isArray(update)) {
      throw new Error('Expected handoff command update');
    }
    const filteredAssistant = update.messages[0] as AIMessage;

    expect(filteredAssistant).not.toBe(assistant);
    expect(filteredAssistant.tool_calls).toEqual([transferCall]);
    expect(getProviderMessageProvenance(filteredAssistant)?.parts).toEqual([
      {
        attribution: 'model',
        sourceMessageId: 'parallel-assistant-source',
      },
    ]);
    expect(getProviderMessageProvenance(assistant)?.parts).toEqual([
      {
        attribution: 'model',
        sourceMessageId: 'parallel-assistant-source',
      },
    ]);
  });

  it.each(INVALID_PROVENANCE_CASES)(
    'preserves $label provenance invalidity in a parallel handoff projection',
    async ({ create }) => {
      const graph = new MultiAgentGraph({
        runId: 'invalid-parallel-handoff-provenance',
        agents: [makeAgent('A'), makeAgent('B')],
        edges: [{ from: 'A', to: 'B', edgeType: 'handoff' }],
      });
      const transferName = `${Constants.LC_TRANSFER_TO_}B`;
      const transferCall: ToolCall = {
        id: 'transfer-call',
        name: transferName,
        args: {},
        type: 'tool_call',
      };
      const sourceProvenance = create();
      const assistant = new AIMessage({
        content: 'retained bytes with unknown authorship',
        tool_calls: [
          transferCall,
          { id: 'parallel-call', name: 'lookup', args: {} },
        ],
        additional_kwargs: { provenance: sourceProvenance },
      });
      const graphTools = graph.agentContexts.get('A')?.graphTools as
        | InvocableGraphTool[]
        | undefined;
      const handoffTool = graphTools?.find(
        (candidate) => candidate.name === transferName
      );
      if (handoffTool == null) {
        throw new Error('Expected handoff tool');
      }

      const output = await handoffTool.invoke(transferCall, {
        state: { messages: [assistant] },
      } as unknown as RunnableConfig);
      const command = output as Command<unknown, { messages: BaseMessage[] }>;
      const update = command.update;
      if (update == null || Array.isArray(update)) {
        throw new Error('Expected handoff command update');
      }
      const filteredAssistant = update.messages[0] as AIMessage;

      expect(filteredAssistant.tool_calls).toEqual([transferCall]);
      expectCanonicalInvalidProvenance(
        filteredAssistant,
        sourceProvenance
      );
      expect(assistant.additional_kwargs.provenance).toBe(sourceProvenance);
    }
  );

  it('rejects grouped fan-in containing a command-routed source', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'hybrid-fan-in',
      agents: [
        makeAgent('router'),
        makeAgent('worker'),
        makeAgent('result'),
        makeAgent('alternate'),
      ],
      edges: [
        {
          from: ['router', 'worker'],
          to: 'result',
          edgeType: 'direct',
        },
        {
          from: 'router',
          to: 'alternate',
          edgeType: 'handoff',
        },
      ],
    };

    expect(() => new MultiAgentGraph(input)).toThrow(
      /grouped direct edge.*command-routed source.*router/i
    );
  });

  it('rejects a prompted direct edge from a command-routed source', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'hybrid-prompt',
      agents: [
        makeAgent('router'),
        makeAgent('result'),
        makeAgent('alternate'),
      ],
      edges: [
        {
          from: 'router',
          to: 'result',
          edgeType: 'direct',
          prompt: 'Continue through the direct path.',
        },
        {
          from: 'router',
          to: 'alternate',
          edgeType: 'handoff',
        },
      ],
    };

    expect(() => new MultiAgentGraph(input)).toThrow(
      /prompted direct edge.*command-routed source.*router/i
    );
  });

  it('rejects a command-routed source sharing a prompted destination group', () => {
    const input: t.MultiAgentGraphInput = {
      runId: 'hybrid-prompt-group',
      agents: [
        makeAgent('router'),
        makeAgent('worker'),
        makeAgent('result'),
        makeAgent('alternate'),
      ],
      edges: [
        { from: 'router', to: 'result', edgeType: 'direct' },
        {
          from: 'worker',
          to: 'result',
          edgeType: 'direct',
          prompt: 'Combine the inputs.',
        },
        {
          from: 'router',
          to: 'alternate',
          edgeType: 'handoff',
        },
      ],
    };

    expect(() => new MultiAgentGraph(input)).toThrow(
      /prompted direct edge.*command-routed source.*router/i
    );
  });
});
