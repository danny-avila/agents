import { config as dotenvConfig } from 'dotenv';

dotenvConfig(
  process.env.LIVE_ENV_PATH == null
    ? undefined
    : { path: process.env.LIVE_ENV_PATH }
);
dotenvConfig();

import { describe, expect, it, jest } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Constants, GraphEvents, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { Run } from '@/run';

jest.setTimeout(180_000);

const liveEnabled = process.env.RUN_GRAPH_SUBAGENT_LIVE_TESTS === '1';
const requestedProvider = process.env.GRAPH_SUBAGENT_LIVE_PROVIDER;
const describeIfLive = liveEnabled ? describe : describe.skip;

type LiveProviderCase = {
  label: string;
  provider: Providers;
  model: string;
  apiKey?: string;
};

const LIVE_PROVIDER_CASES: readonly LiveProviderCase[] = [
  {
    label: 'Anthropic Haiku',
    provider: Providers.ANTHROPIC,
    model:
      process.env.ANTHROPIC_GRAPH_SUBAGENT_LIVE_MODEL ?? 'claude-haiku-4-5',
    apiKey: process.env.ANTHROPIC_API_KEY,
  },
  {
    label: 'OpenAI GPT-4.1 mini',
    provider: Providers.OPENAI,
    model: process.env.OPENAI_GRAPH_SUBAGENT_LIVE_MODEL ?? 'gpt-4.1-mini',
    apiKey: process.env.OPENAI_API_KEY,
  },
];

const explicitlyRequestedCase =
  requestedProvider == null || requestedProvider === ''
    ? undefined
    : LIVE_PROVIDER_CASES.find(
      ({ provider }) =>
        provider.toLowerCase() === requestedProvider.toLowerCase()
    );

if (
  liveEnabled &&
  requestedProvider != null &&
  requestedProvider !== '' &&
  explicitlyRequestedCase == null
) {
  throw new Error(
    `Unknown GRAPH_SUBAGENT_LIVE_PROVIDER "${requestedProvider}".`
  );
}
if (
  liveEnabled &&
  explicitlyRequestedCase != null &&
  (explicitlyRequestedCase.apiKey == null ||
    explicitlyRequestedCase.apiKey === '')
) {
  throw new Error(
    `Missing credentials for GRAPH_SUBAGENT_LIVE_PROVIDER "${requestedProvider}".`
  );
}
if (
  liveEnabled &&
  explicitlyRequestedCase == null &&
  !LIVE_PROVIDER_CASES.some(({ apiKey }) => apiKey != null && apiKey !== '')
) {
  throw new Error(
    'No graph-subagent live-provider credentials are configured.'
  );
}

function createLiveAgent(
  testCase: LiveProviderCase,
  agentId: string,
  marker: string
): t.AgentInputs {
  return {
    agentId,
    provider: testCase.provider,
    clientOptions: {
      modelName: testCase.model,
      apiKey: testCase.apiKey,
      temperature: 0,
      maxTokens: 64,
      streaming: true,
      streamUsage: true,
    },
    instructions: `You are the ${agentId} validation stage. When asked to proceed, reply with exactly ${marker} and no other text.`,
    maxContextTokens: 8_000,
  };
}

function getGraphSubagentTool(run: Run<t.IState>): t.GenericTool {
  const tools = (run.Graph as StandardGraph).agentContexts.get('parent')
    ?.graphTools as t.GenericTool[] | undefined;
  const subagentTool = tools?.find(
    (tool) => 'name' in tool && tool.name === Constants.SUBAGENT
  );
  if (subagentTool == null) {
    throw new Error('Expected graph subagent tool');
  }
  return subagentTool;
}

describeIfLive('Graph subagent live providers', () => {
  for (const testCase of LIVE_PROVIDER_CASES) {
    const providerSelected =
      requestedProvider == null ||
      requestedProvider === '' ||
      requestedProvider.toLowerCase() === testCase.provider.toLowerCase();
    const itIfAvailable =
      providerSelected && testCase.apiKey != null && testCase.apiKey !== ''
        ? it
        : it.skip;

    itIfAvailable(
      `${testCase.label} runs a bounded team and returns only its result member`,
      async () => {
        const usageEvents: t.SubagentUsageEvent[] = [];
        const updateEvents: t.SubagentUpdateEvent[] = [];
        const entry = createLiveAgent(testCase, 'entry', 'ENTRY_MARKER');
        const worker = createLiveAgent(testCase, 'worker', 'WORKER_MARKER');
        const resultAgent = createLiveAgent(testCase, 'result', 'FINAL_MARKER');
        const parent: t.AgentInputs = {
          ...createLiveAgent(testCase, 'parent', 'PARENT_MARKER'),
          maxSubagentDepth: 1,
          subagentConfigs: [
            {
              kind: 'graph',
              type: 'live-team',
              name: 'Live Team',
              description: 'Runs a three-member live-provider chain.',
              maxTurns: 2,
              agents: [entry, worker, resultAgent],
              edges: [
                {
                  from: 'entry',
                  to: 'worker',
                  edgeType: 'direct',
                  prompt: 'Proceed with the worker validation stage.',
                },
                {
                  from: 'worker',
                  to: 'result',
                  edgeType: 'direct',
                  prompt: 'Proceed with the result validation stage.',
                },
              ],
              entryAgentId: 'entry',
              resultAgentId: 'result',
            },
          ],
        };
        const rootRunId = `graph-subagent-live-${testCase.provider}-${Date.now()}`;
        const run = await Run.create<t.IState>({
          runId: rootRunId,
          graphConfig: { type: 'standard', agents: [parent] },
          returnContent: true,
          skipCleanup: true,
          subagentUsageSink: (event) => {
            usageEvents.push(event);
          },
          customHandlers: {
            [GraphEvents.ON_SUBAGENT_UPDATE]: {
              handle: (_event, data): void => {
                updateEvents.push(data as t.SubagentUpdateEvent);
              },
            },
          },
        });
        const invokeConfig: RunnableConfig = {
          configurable: {
            thread_id: `graph-subagent-live-${testCase.provider}`,
          },
        };

        const output = await getGraphSubagentTool(run).invoke(
          {
            description: 'Begin the entry validation stage.',
            subagent_type: 'live-team',
          },
          invokeConfig
        );
        const outputText = String(output);

        expect(outputText).toContain('FINAL_MARKER');
        expect(outputText).not.toContain('ENTRY_MARKER');
        expect(outputText).not.toContain('WORKER_MARKER');
        expect(usageEvents.map((event) => event.memberAgentId).sort()).toEqual([
          'entry',
          'result',
          'worker',
        ]);
        for (const event of usageEvents) {
          expect(event.runId).toBe(rootRunId);
          expect(event.parentRunId).toBe(rootRunId);
          expect(event.depth).toBe(1);
          expect(event.ancestry?.map((entry) => entry.subagentType)).toEqual([
            'live-team',
          ]);
          expect(event.subagentKind).toBe('graph');
          expect(event.provider).toBe(testCase.provider);
          expect(event.model).toBeTruthy();
          expect(event.usage.total_tokens).toBeGreaterThan(0);
        }
        expect(
          updateEvents.filter((event) => event.phase === 'start')
        ).toHaveLength(1);
        expect(
          updateEvents.filter((event) => event.phase === 'stop')
        ).toHaveLength(1);
        for (const event of updateEvents) {
          expect(event.runId).toBe(rootRunId);
          expect(event.parentRunId).toBe(rootRunId);
          expect(event.depth).toBe(1);
          expect(event.ancestry?.map((entry) => entry.subagentType)).toEqual([
            'live-team',
          ]);
        }
        expect(
          new Set(
            updateEvents
              .map((event) => event.memberAgentId)
              .filter((agentId): agentId is string => agentId != null)
          )
        ).toEqual(new Set(['entry', 'worker', 'result']));
      }
    );
  }
});
