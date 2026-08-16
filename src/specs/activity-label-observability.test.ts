import { CallbackHandler } from '@langfuse/langchain';
import { propagateAttributes } from '@langfuse/tracing';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { Providers } from '@/common';
import { Run } from '@/run';

const invoke = jest.fn();

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => ({ invoke })),
}));

jest.mock('@langfuse/langchain', () => ({
  CallbackHandler: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('@langfuse/tracing', () => ({
  ...jest.requireActual('@langfuse/tracing'),
  propagateAttributes: jest.fn((_params, action: () => unknown) => action()),
}));

const MockedCallbackHandler = CallbackHandler as jest.MockedClass<
  typeof CallbackHandler
>;
const MockedPropagateAttributes = propagateAttributes as jest.MockedFunction<
  typeof propagateAttributes
>;

async function createRun(): Promise<Run<never>> {
  const run = await Run.create({
    runId: 'response-1',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          name: 'Changing Model Name',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
  });
  if (run.Graph != null) {
    run.Graph.messages = [new HumanMessage('Why did the run fail?')];
  }
  return run;
}

describe('activity label observability', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    invoke.mockResolvedValue(new AIMessage('Resolved the failing run'));
    process.env = {
      ...originalEnv,
      LANGFUSE_SECRET_KEY: 'sk-test',
      LANGFUSE_PUBLIC_KEY: 'pk-test',
      LANGFUSE_BASE_URL: 'https://langfuse.test',
    };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('uses a stable trace name and source-run metadata for batch labels', async () => {
    const run = await createRun();

    await run.generateActivityLabel({
      provider: Providers.OPENAI,
      entries: [
        {
          toolName: 'inspect_status',
          toolInput: { id: 'one' },
          toolOutput: { fixed: true },
          status: 'success',
        },
      ],
      chainOptions: {
        configurable: {
          thread_id: 'thread-1',
          user_id: 'user-1',
          requestBody: { parentMessageId: 'parent-1' },
        },
      },
    });

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
    expect(MockedCallbackHandler.mock.calls[0][0]).toMatchObject({
      traceMetadata: {
        messageId: 'activity-label-response-1',
        parentMessageId: 'parent-1',
        agentId: 'agent-1',
        agentName: 'Changing Model Name',
        sourceRunId: 'response-1',
        responseId: 'response-1',
        activityIndex: '0',
      },
      tags: ['librechat', 'activity-label'],
    });
    expect(MockedPropagateAttributes.mock.calls[0][0]).toMatchObject({
      traceName: 'LibreChat Activity Label',
      metadata: {
        sourceRunId: 'response-1',
        responseId: 'response-1',
        activityIndex: '0',
      },
    });
    expect(invoke.mock.calls[0][1]).toMatchObject({
      runName: 'LibreChat Activity Label',
      tags: ['librechat', 'activity-label'],
      metadata: {
        sourceRunId: 'response-1',
        responseId: 'response-1',
        activityIndex: 0,
        parentMessageId: 'parent-1',
        agentId: 'agent-1',
        agentName: 'Changing Model Name',
      },
    });
  });

  it('correlates reasoning-label revisions under one visible step', async () => {
    const run = await createRun();

    await run.generateReasoningLabel({
      provider: Providers.OPENAI,
      agentId: 'agent-1',
      visibleReasoning: 'I am tracing the failure through the refresh path.',
      reasoningStepId: 'reasoning-step-1',
      revision: 3,
      status: 'complete',
      sourceRunId: 'response-1',
      sourceTraceId: 'source-trace-1',
      responseId: 'response-1',
      chainOptions: {
        configurable: {
          thread_id: 'thread-1',
          user_id: 'user-1',
          requestBody: { parentMessageId: 'parent-1' },
        },
      },
    });

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
    expect(MockedCallbackHandler.mock.calls[0][0]).toMatchObject({
      traceMetadata: {
        messageId: 'reasoning-label-response-1',
        parentMessageId: 'parent-1',
        agentId: 'agent-1',
        agentName: 'Changing Model Name',
        sourceRunId: 'response-1',
        sourceTraceId: 'source-trace-1',
        responseId: 'response-1',
        reasoningStepId: 'reasoning-step-1',
        revision: '3',
        status: 'complete',
      },
      tags: ['librechat', 'reasoning-label', 'reasoning-step'],
    });
    expect(MockedPropagateAttributes.mock.calls[0][0]).toMatchObject({
      traceName: 'LibreChat Reasoning Label',
      metadata: {
        sourceRunId: 'response-1',
        sourceTraceId: 'source-trace-1',
        responseId: 'response-1',
        reasoningStepId: 'reasoning-step-1',
        revision: '3',
        status: 'complete',
      },
    });
    expect(invoke.mock.calls[0][1]).toMatchObject({
      runName: 'LibreChat Reasoning Label',
      tags: ['librechat', 'reasoning-label', 'reasoning-step'],
      metadata: {
        sourceRunId: 'response-1',
        sourceTraceId: 'source-trace-1',
        responseId: 'response-1',
        reasoningStepId: 'reasoning-step-1',
        revision: 3,
        status: 'complete',
        parentMessageId: 'parent-1',
        agentId: 'agent-1',
        agentName: 'Changing Model Name',
      },
    });
  });

  it('does not trace unattributed reasoning in a multi-agent run', async () => {
    const run = await Run.create({
      runId: 'multi-agent-reasoning-run',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent-1',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
          },
          {
            agentId: 'agent-2',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4.1-mini' },
            tools: [],
          },
        ],
        edges: [],
      },
    });

    await expect(
      run.generateReasoningLabel({
        provider: Providers.OPENAI,
        visibleReasoning: 'Inspecting output owned by an unspecified agent',
        reasoningStepId: 'reasoning-step-1',
        revision: 0,
        chainOptions: {
          configurable: {
            thread_id: 'thread-1',
            user_id: 'user-1',
          },
        },
      })
    ).resolves.toEqual({});

    expect(MockedCallbackHandler).not.toHaveBeenCalled();
    expect(MockedPropagateAttributes).not.toHaveBeenCalled();
    expect(invoke).not.toHaveBeenCalled();
  });

  it('uses a stable trace name for phase summaries', async () => {
    const run = await createRun();

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        { label: 'Inspected session refresh behavior' },
        { label: 'Fixed refresh token validation' },
      ],
      sourceRunId: 'response-1',
      responseId: 'response-1',
      phaseIndex: 0,
      chainOptions: {
        configurable: {
          thread_id: 'thread-1',
          user_id: 'user-1',
          requestBody: { parentMessageId: 'parent-1' },
        },
      },
    });

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
    expect(MockedCallbackHandler.mock.calls[0][0]).toMatchObject({
      tags: ['librechat', 'activity-phase', 'agent-run-summary', 'agent'],
      traceMetadata: {
        sourceRunId: 'response-1',
        responseId: 'response-1',
        phaseIndex: '0',
        activityCount: '2',
      },
    });
    expect(MockedPropagateAttributes.mock.calls[0][0]).toMatchObject({
      traceName: 'LibreChat Activity Phase',
      metadata: {
        sourceRunId: 'response-1',
        responseId: 'response-1',
        phaseIndex: '0',
        activityCount: '2',
      },
    });
  });
});
