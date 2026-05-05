import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { Run } from '@/run';

/**
 * Live verification that host-set fields on the parent's outer
 * `configurable` (e.g. `requestBody`, `user`, `userMCPAuthMap`)
 * propagate into the subagent's `ON_TOOL_EXECUTE` dispatches.
 *
 * Pass criteria: when the SUBAGENT calls the calculator tool, the
 * `data.configurable` arriving at the parent's ON_TOOL_EXECUTE
 * handler contains every key the parent put on its outer
 * configurable (with `thread_id` overridden to a child run id).
 */
const apiKey = process.env.OPENAI_API_KEY!;
if (!apiKey) {
  console.error('Missing OPENAI_API_KEY');
  process.exit(1);
}

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

type ConfigurableSnapshot = {
  agentId: string | undefined;
  configurable: Record<string, unknown> | undefined;
};

async function main() {
  console.log('=== Subagent parentConfigurable inheritance — live ===\n');

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey },
    instructions: `You can spawn a "self" subagent in an isolated context.
For any math task, spawn the "self" subagent and let it use the calculator.
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

  const parentSnapshots: ConfigurableSnapshot[] = [];
  const subagentSnapshots: ConfigurableSnapshot[] = [];

  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_TOOL_EXECUTE]: {
      handle: (_event, rawData): void => {
        const data = rawData as t.ToolExecuteBatchRequest;
        const snapshot: ConfigurableSnapshot = {
          agentId: data.agentId,
          configurable: data.configurable as Record<string, unknown> | undefined,
        };
        const callsLabel = data.toolCalls.map((c) => c.name).join(',');
        // For `self`-spawn the child inherits parent's agentId, so we can't
        // distinguish by agentId alone. The child's thread_id is set by the
        // SDK to `${parentRunId}_sub_<nanoid>` — that's a reliable marker.
        const threadId = (data.configurable as { thread_id?: string } | undefined)
          ?.thread_id;
        const isSubagent = typeof threadId === 'string' && threadId.includes('_sub_');
        if (isSubagent) {
          subagentSnapshots.push(snapshot);
        } else {
          parentSnapshots.push(snapshot);
        }
        console.log(
          `[ON_TOOL_EXECUTE] origin=${isSubagent ? 'SUBAGENT' : 'PARENT'} agentId=${data.agentId} calls=${callsLabel}`
        );
        const results: t.ToolExecuteResult[] = data.toolCalls.map((call) => {
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
  };

  const run = await Run.create<t.IState>({
    runId: `sub-cfg-inherit-${Date.now()}`,
    graphConfig: { type: 'standard', agents: [parentAgent] },
    customHandlers,
  });

  const question = new HumanMessage(
    'Compute (42 * 58) + (13 ** 3). Use the self subagent, and have it use the calculator.'
  );

  // Parent's outer configurable carries host-set fields. After the SDK
  // change, these should propagate into the subagent's tool dispatches.
  const outerConfigurable = {
    thread_id: 'parent-thread-conv-xyz',
    user_id: 'user_abc',
    user: { id: 'user_abc', email: 'a@b.c', role: 'USER' },
    requestBody: {
      messageId: 'msg-response-id-001',
      conversationId: 'parent-thread-conv-xyz',
      parentMessageId: 'user-message-id-000',
    },
    userMCPAuthMap: { 'mcp-github': { token: 'abc' } },
  };

  console.log('User:', question.content);
  console.log('Parent outer configurable keys:', Object.keys(outerConfigurable));
  console.log();

  await run.processStream(
    { messages: [question] },
    {
      configurable: outerConfigurable,
      version: 'v2' as const,
    }
  );

  console.log('\n=== Verification ===');
  console.log(
    `Parent ON_TOOL_EXECUTE dispatches captured: ${parentSnapshots.length}`
  );
  console.log(
    `Subagent ON_TOOL_EXECUTE dispatches captured: ${subagentSnapshots.length}`
  );

  if (subagentSnapshots.length === 0) {
    console.error(
      '\n❌ FAIL: subagent never invoked a tool — model may not have spawned the subagent.'
    );
    process.exit(2);
  }

  const expectedKeys = ['user_id', 'user', 'requestBody', 'userMCPAuthMap'];
  let allPassed = true;
  subagentSnapshots.forEach((snap, idx) => {
    const cfg = snap.configurable ?? {};
    console.log(`\nSubagent dispatch #${idx + 1} (agentId=${snap.agentId}):`);
    for (const key of expectedKeys) {
      const present = key in cfg;
      const value = cfg[key];
      console.log(
        `  ${present ? '✅' : '❌'} ${key} = ${JSON.stringify(value)}`
      );
      if (!present) allPassed = false;
    }
    const threadId = cfg.thread_id as string | undefined;
    const overridden =
      threadId !== undefined && threadId !== outerConfigurable.thread_id;
    console.log(
      `  ${overridden ? '✅' : '❌'} thread_id overridden (got "${threadId}", parent's was "${outerConfigurable.thread_id}")`
    );
    if (!overridden) allPassed = false;
  });

  if (allPassed) {
    console.log(
      '\n✅ PASS: subagent ON_TOOL_EXECUTE saw parent host-set fields with thread_id overridden.'
    );
    process.exit(0);
  } else {
    console.log('\n❌ FAIL: at least one expected key was missing.');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Script error:', err);
  process.exit(1);
});
