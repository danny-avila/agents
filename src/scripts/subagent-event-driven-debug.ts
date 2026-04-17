import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents, Constants } from '@/common';
import { Run } from '@/run';

/**
 * Repro for LibreChat's actual setup: event-driven tools via `toolDefinitions`
 * + an ON_TOOL_EXECUTE handler that runs the tool. Self-spawn subagent must
 * be able to drive the SAME tool pipeline.
 */
const apiKey = process.env.OPENAI_API_KEY!;
if (!apiKey) {
  console.error('Missing OPENAI_API_KEY');
  process.exit(1);
}

// Simulate LibreChat: tool definitions only, execution routed via event.
const calculatorDef: t.LCTool = {
  name: 'calculator',
  description: 'Evaluate a math expression. Use for any arithmetic.',
  parameters: {
    type: 'object',
    properties: {
      expression: {
        type: 'string',
        description: "A JS math expression, e.g. '42 * 58'",
      },
    },
    required: ['expression'],
  },
};

async function main() {
  console.log('=== Subagent Event-Driven Tool Diagnostic ===\n');

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey },
    instructions: `You have calculator AND can spawn a "self" subagent in an isolated context.
For any arithmetic question, spawn the "self" subagent with the math task.
The subagent MUST use the calculator tool — never estimate.`,
    maxContextTokens: 8000,
    toolDefinitions: [calculatorDef],
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

  let toolCallCount = 0;
  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: {
      handle: (_event, rawData): void => {
        const data = rawData as t.ToolExecuteBatchRequest;
        console.log(
          `[PARENT ON_TOOL_EXECUTE] agentId=${data.agentId} calls=${data.toolCalls
            .map((c) => c.name)
            .join(',')}`
        );
        const results: t.ToolExecuteResult[] = data.toolCalls.map((call) => {
          toolCallCount += 1;
          const args = call.args as { expression?: string };
          const expression = args.expression ?? '';
          let content: string;
          try {
            // eslint-disable-next-line no-eval
            const result = eval(expression);
            content = `${expression} = ${result}`;
          } catch (err) {
            content = `Error: ${String(err)}`;
          }
          return {
            toolCallId: call.id!,
            status: 'success',
            content,
          };
        });
        data.resolve(results);
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event, data): void => {
        const d = data as { type?: string; runId?: string; agentId?: string };
        console.log(
          `[PARENT ${event}] type=${d.type} agentId=${d.agentId ?? '-'} runId=${d.runId ?? '-'}`
        );
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (event, data): void => {
        const r = (
          data as { result: { type: string; tool_call?: { name?: string } } }
        ).result;
        console.log(
          `[PARENT ${event}] type=${r.type} tool=${r.tool_call?.name ?? '-'}`
        );
      },
    },
    [GraphEvents.ON_SUBAGENT_UPDATE]: {
      handle: (_event, rawData): void => {
        const d = rawData as t.SubagentUpdateEvent;
        console.log(
          `[SUBAGENT ${d.phase}] [${d.subagentType}] tool_call_id=${d.parentToolCallId ?? '-'} ${d.label ?? ''}`
        );
      },
    },
  };

  const run = await Run.create<t.IState>({
    runId: `sub-evt-${Date.now()}`,
    graphConfig: { type: 'standard', agents: [parentAgent] },
    customHandlers,
  });

  const question = new HumanMessage(
    'Compute (42 * 58) + (13 ** 3). Use the self subagent, and have it use the calculator.'
  );

  console.log('User:', question.content, '\n');

  await run.processStream(
    { messages: [question] },
    {
      configurable: { thread_id: `sub-evt` },
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

  const subagentMsgs = msgs.filter(
    (m) =>
      m._getType() === 'tool' &&
      (m as { name?: string }).name === Constants.SUBAGENT
  );

  console.log('--- Verification ---');
  console.log(`subagent tool calls seen (parent): ${subagentMsgs.length}`);
  console.log(
    `ON_TOOL_EXECUTE dispatched (parent saw): ${toolCallCount} (expected >= 1 if subagent used calculator)`
  );
  if (subagentMsgs[0]) {
    console.log(
      `\nsubagent result:\n${(subagentMsgs[0].content as string).slice(0, 600)}`
    );
  }
}

main().catch((err) => {
  console.error('Script error:', err);
  process.exit(1);
});
