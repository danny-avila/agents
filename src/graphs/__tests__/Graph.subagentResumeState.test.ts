import { describe, it, expect } from '@jest/globals';
import type { SubagentToolNodeResumeState } from '@/tools/subagent/SubagentReplay';
import { StandardGraph } from '../Graph';
import { Providers } from '@/common';

type ResumeToolNode = {
  createSubagentResumeState(): SubagentToolNodeResumeState;
  restoreSubagentResumeState(state: SubagentToolNodeResumeState): void;
};

const makeGraph = (runId: string): StandardGraph =>
  new StandardGraph({
    runId,
    agents: [
      {
        agentId: 'child-agent',
        provider: Providers.OPENAI,
        instructions: 'Test child state restoration.',
      },
    ],
  });

const initializeToolNode = (graph: StandardGraph): ResumeToolNode => {
  const agentContext = graph.agentContexts.get('child-agent');
  return graph.initializeTools({
    currentTools: [],
    agentContext,
  }) as unknown as ResumeToolNode;
};

describe('subagent graph resume state', () => {
  it('restores tool identity, turns, references, and sessions into a branch', () => {
    const source = makeGraph('source');
    source.toolOutputReferences = { enabled: true };
    const sourceNode = initializeToolNode(source);
    source.toolCallStepIds.set('call_tool', 'step_tool');
    source.sessions.set('execute_code', {
      session_id: 'sandbox-session',
      lastUpdated: 42,
    });
    source
      .getOrCreateToolOutputRegistry()
      ?.set('source-scope', 'tool0turn0', 'raw output');
    source.getEagerEventToolUsageCount('child-agent').set('calculator', 3);
    source.eagerEventToolSuppressions.add('unstable_search');
    const sourceNodeState = sourceNode.createSubagentResumeState();
    sourceNode.restoreSubagentResumeState({
      ...sourceNodeState,
      toolUsageCounts: [{ toolName: 'calculator', count: 3 }],
      directPathTurns: [{ toolCallId: 'call_tool', turn: 2 }],
    });

    const target = makeGraph('target');
    target.toolOutputReferences = { enabled: true };
    const targetNode = initializeToolNode(target);
    target.restoreSubagentResumeState(
      source.createSubagentResumeState('source-scope'),
      'target-scope'
    );

    expect(target.toolCallStepIds.get('call_tool')).toBe('step_tool');
    expect(target.sessions.get('execute_code')).toMatchObject({
      session_id: 'sandbox-session',
      lastUpdated: 42,
    });
    expect(
      target.getOrCreateToolOutputRegistry()?.get('target-scope', 'tool0turn0')
    ).toBe('raw output');
    expect(targetNode.createSubagentResumeState()).toMatchObject({
      toolUsageCounts: [{ toolName: 'calculator', count: 3 }],
      directPathTurns: [{ toolCallId: 'call_tool', turn: 2 }],
    });
    expect(
      target.getEagerEventToolUsageCount('child-agent').get('calculator')
    ).toBe(3);
    expect(target.eagerEventToolSuppressions).toEqual(
      new Set(['unstable_search'])
    );
  });
});
