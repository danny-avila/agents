import {
  getSubagentResumeManifest,
  attachSubagentResumeManifest,
  requireValidSubagentResumeManifest,
  stripSubagentResumeManifest,
} from '@/tools/subagent/SubagentReplay';

describe('SubagentReplay manifest', () => {
  const execution = {
    parentToolCallId: 'call_parent',
    childRunId: 'child-run',
    approvalExecutionScope: 'child-approval-scope',
    checkpoints: [
      {
        threadId: 'child-thread',
        checkpointId: 'checkpoint-root',
        checkpointNs: '',
      },
      {
        threadId: 'child-thread',
        checkpointId: 'checkpoint-agent',
        checkpointNs: 'agent:task',
      },
    ],
    graphState: {
      toolCallSteps: [{ toolCallId: 'call_tool', stepId: 'step_tool' }],
      toolSessions: [
        {
          toolName: 'execute_code',
          context: { session_id: 'session', lastUpdated: 1 },
        },
      ],
      toolNodes: [
        {
          stateKey: 'child-agent',
          toolUsageCounts: [{ toolName: 'calculator', count: 1 }],
          directPathTurns: [{ toolCallId: 'call_tool', turn: 0 }],
        },
      ],
      eagerToolUsage: [
        {
          agentId: 'child-agent',
          toolUsageCounts: [{ toolName: 'calculator', count: 1 }],
        },
      ],
      eagerToolSuppressions: ['unstable_search'],
      toolOutputReferences: {
        entries: [{ key: 'tool0turn0', value: '42' }],
        turnCounter: 1,
        warnedNonStringTools: [],
      },
    },
    approvalReplays: [
      {
        key: {
          executionScope: 'child-approval-scope',
          agentId: 'child-agent',
          toolUseId: 'call_tool',
        },
        result: {
          decision: 'ask' as const,
          reason: 'review tool',
          additionalContexts: [],
          injectedMessages: [],
          errors: [],
        },
      },
    ],
  };

  it('round-trips a private manifest without exposing it publicly', () => {
    const manifest = { version: 1 as const, executions: [execution] };
    const payload = attachSubagentResumeManifest(
      { type: 'tool_approval', visible: true },
      manifest
    );

    expect(getSubagentResumeManifest(payload)).toEqual(manifest);
    expect(stripSubagentResumeManifest(payload)).toEqual({
      type: 'tool_approval',
      visible: true,
    });
  });

  it.each([
    ['string', 'approve this child'],
    ['number', 42],
    ['null', null],
    ['array', ['approve', { call: 1 }]],
  ])(
    'round-trips a private manifest around a %s interrupt payload',
    (_label, visiblePayload) => {
      const manifest = { version: 1 as const, executions: [execution] };
      const payload = attachSubagentResumeManifest(visiblePayload, manifest);

      expect(getSubagentResumeManifest(payload)).toEqual(manifest);
      expect(stripSubagentResumeManifest(payload)).toEqual(visiblePayload);
    }
  );

  it('rejects ambiguous or cross-execution replay data', () => {
    const mismatchedScope = {
      ...execution,
      approvalReplays: [
        {
          ...execution.approvalReplays[0],
          key: {
            ...execution.approvalReplays[0].key,
            executionScope: 'different-child',
          },
        },
      ],
    };
    const duplicateParent = {
      version: 1 as const,
      executions: [execution, { ...execution }],
    };
    const duplicateToolCall = {
      ...execution,
      graphState: {
        ...execution.graphState,
        toolCallSteps: [
          ...execution.graphState.toolCallSteps,
          { ...execution.graphState.toolCallSteps[0] },
        ],
      },
    };

    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest(
          { type: 'tool_approval' },
          { version: 1, executions: [mismatchedScope] }
        )
      )
    ).toBeUndefined();
    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest({ type: 'tool_approval' }, duplicateParent)
      )
    ).toBeUndefined();
    expect(
      getSubagentResumeManifest(
        attachSubagentResumeManifest(
          { type: 'tool_approval' },
          { version: 1, executions: [duplicateToolCall] }
        )
      )
    ).toBeUndefined();
  });

  it('fails closed for malformed nested references', () => {
    const malformed = {
      __librechat_subagent_resume_manifest: {
        version: 1,
        executions: [{ ...execution, checkpoints: [null] }],
      },
    };
    expect(getSubagentResumeManifest(malformed)).toBeUndefined();
    expect(() => requireValidSubagentResumeManifest(malformed)).toThrow(
      'Invalid subagent resume manifest.'
    );
    expect(
      getSubagentResumeManifest({
        __librechat_subagent_resume_manifest: {
          version: 1,
          executions: [
            {
              ...execution,
              graphState: {
                ...execution.graphState,
                toolCallSteps: [null],
              },
            },
          ],
        },
      })
    ).toBeUndefined();
  });

  it('rejects duplicate checkpoint namespaces and approval replay keys', () => {
    const duplicateCheckpointNamespace = {
      ...execution,
      checkpoints: [execution.checkpoints[0], { ...execution.checkpoints[0] }],
    };
    const duplicateApprovalReplay = {
      ...execution,
      approvalReplays: [
        execution.approvalReplays[0],
        { ...execution.approvalReplays[0] },
      ],
    };

    for (const invalidExecution of [
      duplicateCheckpointNamespace,
      duplicateApprovalReplay,
    ]) {
      expect(
        getSubagentResumeManifest(
          attachSubagentResumeManifest(
            { type: 'tool_approval' },
            { version: 1, executions: [invalidExecution] }
          )
        )
      ).toBeUndefined();
    }
  });

  it('rejects cyclic descendant manifests', () => {
    const cyclic: {
      version: number;
      executions: Array<typeof execution & { descendant?: object }>;
    } = { version: 1, executions: [{ ...execution }] };
    cyclic.executions[0].descendant = cyclic;

    expect(
      getSubagentResumeManifest({
        __librechat_subagent_resume_manifest: cyclic,
      })
    ).toBeUndefined();
  });
});
