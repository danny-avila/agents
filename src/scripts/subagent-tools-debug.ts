import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents, Constants } from '@/common';
import { Run } from '@/run';

/**
 * Diagnostic: verify a self-spawned subagent can call parent's real tools.
 * Expected before-fix: parent delegates to self; child cannot invoke calculator.
 */
const apiKey = process.env.OPENAI_API_KEY!;
if (!apiKey) {
  console.error('Missing OPENAI_API_KEY');
  process.exit(1);
}

const calculator = tool(
  async ({ expression }) => {
    const result = eval(expression); // don't do this in prod
    return `${expression} = ${result}`;
  },
  {
    name: 'calculator',
    description: 'Evaluate a math expression. Use for any arithmetic.',
    schema: z.object({
      expression: z.string().describe("A JS math expression, e.g. '42 * 58'"),
    }),
  }
);

async function main() {
  console.log('=== Subagent Tool-Access Diagnostic ===\n');

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey },
    instructions: `You have calculator AND can spawn a "self" subagent in an isolated context.
For any arithmetic question that would bloat your context, spawn the "self" subagent with the math task.
The subagent must use the calculator tool — never estimate.`,
    maxContextTokens: 8000,
    tools: [calculator],
    subagentConfigs: [
      {
        type: 'self',
        self: true,
        name: 'supervisor',
        description:
          'Spawn a copy of this agent in an isolated context for a focused math subtask.',
      },
    ],
  };

  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event, data): void => {
        console.log(
          `[PARENT EVENT] ${event}`,
          JSON.stringify(data).slice(0, 200)
        );
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (event, data): void => {
        console.log(
          `[PARENT EVENT] ${event}`,
          JSON.stringify(data).slice(0, 200)
        );
      },
    },
  };

  const run = await Run.create<t.IState>({
    runId: `subagent-debug-${Date.now()}`,
    graphConfig: { type: 'standard', agents: [parentAgent] },
    customHandlers,
  });

  const question = new HumanMessage(
    'Compute (42 * 58) + (13 ^ 3). Spawn the self subagent to do this, and have IT use calculator.'
  );

  console.log('User:', question.content, '\n');

  await run.processStream(
    { messages: [question] },
    {
      configurable: { thread_id: `subagent-debug` },
      version: 'v2' as const,
    }
  );

  const msgs = (run.getRunMessages() ?? []) as BaseMessage[];
  console.log('\n--- Run messages ---\n');
  for (const msg of msgs) {
    const type = msg._getType();
    const name = 'name' in msg ? (msg as { name?: string }).name : undefined;
    const content =
      typeof msg.content === 'string'
        ? msg.content.slice(0, 400)
        : JSON.stringify(msg.content).slice(0, 400);
    const toolCalls =
      'tool_calls' in msg
        ? (msg as { tool_calls?: Array<{ name: string; args: unknown }> })
            .tool_calls
        : undefined;
    console.log(`[${type}]${name ? ` name=${name}` : ''}`);
    if (toolCalls?.length) {
      for (const tc of toolCalls) {
        console.log(
          `  tool_call: ${tc.name}(${JSON.stringify(tc.args).slice(0, 150)})`
        );
      }
    }
    console.log(`  content: ${content}\n`);
  }

  const subagentCalls = msgs.filter(
    (m) =>
      m._getType() === 'tool' &&
      'name' in m &&
      (m as { name?: string }).name === Constants.SUBAGENT
  );
  const calculatorCalls = msgs.filter(
    (m) =>
      m._getType() === 'tool' &&
      'name' in m &&
      (m as { name?: string }).name === 'calculator'
  );

  console.log('--- Verification ---');
  console.log(`subagent tool calls seen (parent): ${subagentCalls.length}`);
  console.log(
    `calculator tool calls seen (parent): ${calculatorCalls.length} (expected: 0 if subagent did the math)`
  );
  if (subagentCalls.length > 0) {
    const subResult = subagentCalls[0].content as string;
    console.log(`\nsubagent result snippet:\n${subResult.slice(0, 600)}\n`);
    if (/\berror\b/i.test(subResult) && /tool/i.test(subResult)) {
      console.log('⚠️  BUG CONFIRMED: subagent result mentions tool error');
    } else if (!/\d/.test(subResult)) {
      console.log('⚠️  POSSIBLY BUGGY: subagent result has no numbers');
    }
  }
}

main().catch((err) => {
  console.error('Script error:', err);
  process.exit(1);
});
