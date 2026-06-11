import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { Run } from '@/run';

/**
 * Live verification for `subagentUsageSink` (host billing of subagent
 * child-run model usage).
 *
 * Runs a supervisor that MUST delegate to a "researcher" subagent, then
 * asserts:
 * 1. The host's CHAT_MODEL_END handler collected the PARENT's calls only.
 * 2. The sink received one event per CHILD model call, tagged with the
 *    subagent type, child run id, and the child's model/provider.
 * 3. Child usage has real token counts (the previously-unbilled tokens).
 *
 * Usage:
 *   OPENAI_API_KEY=... npx ts-node -r tsconfig-paths/register src/scripts/subagent-usage-sink.ts
 *
 * Or with Anthropic:
 *   ANTHROPIC_API_KEY=... npx ts-node -r tsconfig-paths/register src/scripts/subagent-usage-sink.ts --provider anthropic
 */

const useAnthropic =
  process.argv.includes('--provider') &&
  process.argv[process.argv.indexOf('--provider') + 1] === 'anthropic';

const provider = useAnthropic ? Providers.ANTHROPIC : Providers.OPENAI;
const apiKey = useAnthropic
  ? process.env.ANTHROPIC_API_KEY
  : process.env.OPENAI_API_KEY;
const modelName = useAnthropic ? 'claude-sonnet-4-20250514' : 'gpt-4o-mini';

if (!apiKey) {
  console.error(
    `Missing ${useAnthropic ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY'} environment variable`
  );
  process.exit(1);
}

async function main(): Promise<void> {
  console.log('=== Subagent Usage Sink Live Verification ===\n');
  console.log(`Provider: ${provider}`);
  console.log(`Model: ${modelName}\n`);

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider,
    clientOptions: { modelName, apiKey },
    instructions: `You are a supervisor agent. For ANY user question, you MUST delegate to the "researcher" subagent via the subagent tool — never answer directly. After the subagent returns, give the user a one-sentence final answer.`,
    maxContextTokens: 16000,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches questions and returns concise answers.',
        agentInputs: {
          agentId: 'researcher',
          provider,
          clientOptions: { modelName, apiKey },
          instructions:
            'You are a research specialist. Answer the task in one or two sentences.',
          maxContextTokens: 8000,
        },
      },
    ],
  };

  const collectedUsage: UsageMetadata[] = [];
  const sunkEvents: t.SubagentUsageEvent[] = [];

  const runId = `usage-sink-live-${Date.now()}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [parentAgent],
    },
    returnContent: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    },
    subagentUsageSink: (event) => {
      sunkEvents.push(event);
    },
  });

  const callerConfig = {
    configurable: { thread_id: `usage-sink-${Date.now()}` },
    streamMode: 'values' as const,
    version: 'v2' as const,
  };

  await run.processStream(
    {
      messages: [
        new HumanMessage(
          'In what year was the Eiffel Tower completed? Use the researcher subagent.'
        ),
      ],
    },
    callerConfig
  );

  console.log('\n--- Parent collectedUsage (CHAT_MODEL_END handler) ---');
  console.dir(collectedUsage, { depth: null });

  console.log('\n--- Subagent usage sink events ---');
  console.dir(sunkEvents, { depth: null });

  const failures: string[] = [];

  if (collectedUsage.length < 2) {
    failures.push(
      `expected >= 2 parent model calls in collectedUsage, got ${collectedUsage.length}`
    );
  }
  if (sunkEvents.length === 0) {
    failures.push('sink received NO child usage events');
  }
  for (const event of sunkEvents) {
    if (event.subagentType !== 'researcher') {
      failures.push(`unexpected subagentType: ${event.subagentType}`);
    }
    if (event.runId !== runId) {
      failures.push(`event.runId mismatch: ${event.runId}`);
    }
    if (!event.subagentRunId.startsWith(`${runId}_sub_`)) {
      failures.push(`event.subagentRunId mismatch: ${event.subagentRunId}`);
    }
    if (event.provider !== provider) {
      failures.push(`event.provider mismatch: ${event.provider}`);
    }
    if (event.model == null || event.model === '') {
      failures.push('event.model missing');
    }
    const input = Number(event.usage.input_tokens) || 0;
    const output = Number(event.usage.output_tokens) || 0;
    if (input <= 0 || output <= 0) {
      failures.push(
        `child usage has non-positive tokens: input=${input} output=${output}`
      );
    }
  }

  const childTotal = sunkEvents.reduce(
    (sum, e) =>
      sum +
      (Number(e.usage.input_tokens) || 0) +
      (Number(e.usage.output_tokens) || 0),
    0
  );
  console.log(
    `\nChild tokens that were previously invisible to billing: ${childTotal}`
  );

  if (failures.length > 0) {
    console.error('\nFAIL:');
    for (const failure of failures) {
      console.error(`  - ${failure}`);
    }
    process.exit(1);
  }
  console.log('\nPASS: subagent child usage reported through the sink.');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
