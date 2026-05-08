// src/graphs/__tests__/MultiAgentGraph.test.ts
import { MultiAgentGraph } from '../MultiAgentGraph';
import { Providers } from '@/common';
import type * as t from '@/types';

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
});
