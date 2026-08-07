// src/graphs/__tests__/MultiAgentGraph.test.ts
import type * as t from '@/types';
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Providers } from '@/common';

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
