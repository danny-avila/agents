import { AIMessage } from '@langchain/core/messages';
import { Providers } from '@/common';
import { LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT } from '@/langfuseToolOutputTracing';
import { Run } from '@/run';

const invoke = jest.fn();

jest.mock('@/llm/init', () => ({
  initializeModel: jest.fn(() => ({ invoke })),
}));

async function createRun(): Promise<Run<never>> {
  return Run.create({
    runId: 'phase-run',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent-1',
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4.1-mini' },
          tools: [],
        },
      ],
    },
  });
}

describe('generateActivityPhaseLabel', () => {
  beforeEach(() => {
    invoke.mockReset();
    invoke.mockResolvedValue(
      new AIMessage('"Fixed session refresh handling and verified auth tests."')
    );
  });

  it('does not spend a model call on one logical activity', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [{ label: 'Inspected session refresh middleware' }],
      })
    ).resolves.toEqual({});
    expect(invoke).not.toHaveBeenCalled();
  });

  it('summarizes two activities and normalizes the persisted row', async () => {
    const run = await createRun();

    await expect(
      run.generateActivityPhaseLabel({
        provider: Providers.OPENAI,
        activities: [
          { label: 'Inspected session refresh middleware' },
          { label: 'Fixed refresh token validation' },
        ],
        assistantContext: ['I am checking the auth path.'],
        closingTextPhase: 'final_answer',
      })
    ).resolves.toEqual({
      label: 'Fixed session refresh handling and verified auth tests',
    });
    expect(invoke).toHaveBeenCalledTimes(1);
    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).toContain(
      'Inspected session refresh middleware'
    );
    expect(String(messages[1].content)).toContain(
      'Fixed refresh token validation'
    );
    const modelConfig = invoke.mock.calls[0][1] as {
      callbacks?: { parentRunId?: string };
      tags?: string[];
    };
    expect(modelConfig.callbacks?.parentRunId).toEqual(expect.any(String));
    expect(modelConfig.tags).toEqual(
      expect.arrayContaining(['activity-phase', 'agent'])
    );
  });

  it('applies every agent redaction policy when any activity is unattributed', async () => {
    const run = await Run.create({
      runId: 'mixed-attribution-phase-run',
      graphConfig: {
        type: 'standard',
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
            langfuse: {
              toolOutputTracing: { redactedToolNames: ['secret_tool'] },
            },
          },
        ],
      },
    });

    await run.generateActivityPhaseLabel({
      provider: Providers.OPENAI,
      activities: [
        { agentId: 'agent-1', label: 'Inspected public session behavior' },
        {
          entries: [
            {
              toolName: 'secret_tool',
              toolInput: { key: 'public-key' },
              toolOutput: 'STRICT_AGENT_SECRET',
              status: 'success',
            },
          ],
        },
      ],
    });

    const messages = invoke.mock.calls[0][0] as AIMessage[];
    expect(String(messages[1].content)).not.toContain('STRICT_AGENT_SECRET');
    expect(String(messages[1].content)).toContain(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });
});
