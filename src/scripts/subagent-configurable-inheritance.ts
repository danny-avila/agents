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
  metadata: Record<string, unknown> | undefined;
};

async function main() {
  console.log('=== Subagent parentConfigurable inheritance — live ===\n');

  // Parent has NO tools — it can only delegate via the math subagent.
  // The math subagent has the calculator. This forces the spawn-subagent
  // path so we can observe the subagent's `ON_TOOL_EXECUTE` dispatch.
  const mathSubagentInputs: t.AgentInputs = {
    agentId: 'math-worker',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o', apiKey },
    instructions:
      'You compute arithmetic. Always use the calculator tool — never estimate. Return the final numeric result as plain text.',
    maxContextTokens: 8000,
    toolDefinitions: [calculatorDef],
  };

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o', apiKey },
    instructions: `You delegate arithmetic to the "math" subagent. You have NO calculator yourself. For any math task, spawn the "math" subagent with the full task as its description, then echo the subagent's text result back to the user.`,
    maxContextTokens: 8000,
    // No toolDefinitions on the parent — only the subagent gets the calculator.
    subagentConfigs: [
      {
        type: 'math',
        name: 'math',
        description:
          'A focused arithmetic worker that uses the calculator tool to compute numerical results.',
        agentInputs: mathSubagentInputs,
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
          metadata: data.metadata as Record<string, unknown> | undefined,
        };
        const callsLabel = data.toolCalls.map((c) => c.name).join(',');
        // Parent and subagent have different agent IDs in this script
        // (parent: 'supervisor', subagent: 'math-worker'). With a self-spawn
        // subagent both would be the same; this script uses a non-self
        // subagent precisely so we can distinguish reliably.
        const isSubagent = data.agentId !== 'supervisor';
        const metadataRunId = (data.metadata as { run_id?: string } | undefined)
          ?.run_id;
        if (isSubagent) {
          subagentSnapshots.push(snapshot);
        } else {
          parentSnapshots.push(snapshot);
        }
        console.log(
          `[ON_TOOL_EXECUTE] origin=${isSubagent ? 'SUBAGENT' : 'PARENT'} agentId=${data.agentId} calls=${callsLabel}`
        );
        console.log(
          `  metadata keys: ${Object.keys(data.metadata ?? {}).join(',') || '<none>'}`
        );
        console.log(
          `  metadata.run_id="${metadataRunId ?? '<none>'}" configurable.run_id="${(data.configurable as { run_id?: string } | undefined)?.run_id ?? '<none>'}" configurable.thread_id="${(data.configurable as { thread_id?: string } | undefined)?.thread_id ?? '<none>'}"`
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

  // Parent's outer configurable carries host-set fields AND explicit
  // run-identity fields so we can verify whether LangGraph respects or
  // overwrites parent's `run_id` / `parent_run_id` when we forward them
  // into the child's `workflow.invoke`.
  const outerConfigurable = {
    thread_id: 'parent-thread-conv-xyz',
    run_id: 'parent-run-id-001',
    parent_run_id: 'grandparent-run-id-000',
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

  const expectedHostKeys = ['user_id', 'user', 'requestBody', 'userMCPAuthMap'];
  let allPassed = true;
  subagentSnapshots.forEach((snap, idx) => {
    const cfg = snap.configurable ?? {};
    const meta = snap.metadata ?? {};
    console.log(
      `\nSubagent dispatch #${idx + 1} (agentId=${snap.agentId}, metadata.run_id=${(meta as { run_id?: string }).run_id ?? '-'}):`
    );

    // Host-set fields must propagate.
    for (const key of expectedHostKeys) {
      const present = key in cfg;
      const value = cfg[key];
      console.log(
        `  ${present ? '✅' : '❌'} ${key} = ${JSON.stringify(value)}`
      );
      if (!present) allPassed = false;
    }

    // Run-identity fields: with full inheritance we expect parent's
    // values to flow through. LangGraph runtime MAY overwrite them at
    // child-invoke time — the script logs what actually arrived so we
    // can see empirically what propagates.
    console.log(`  ⓘ thread_id observed: "${cfg.thread_id as string}" (parent's: "${outerConfigurable.thread_id}")`);
    console.log(`  ⓘ run_id observed:    "${cfg.run_id as string}" (parent's: "${outerConfigurable.run_id}")`);
    console.log(`  ⓘ parent_run_id observed: "${cfg.parent_run_id as string}" (parent's: "${outerConfigurable.parent_run_id}")`);

    const threadInherited = cfg.thread_id === outerConfigurable.thread_id;
    const runInherited = cfg.run_id === outerConfigurable.run_id;
    const parentRunInherited =
      cfg.parent_run_id === outerConfigurable.parent_run_id;
    console.log(
      `  ${threadInherited ? '✅' : '⚠️ '} thread_id inherited from parent: ${threadInherited}`
    );
    console.log(
      `  ${runInherited ? '✅' : '⚠️ '} run_id inherited from parent: ${runInherited}`
    );
    console.log(
      `  ${parentRunInherited ? '✅' : '⚠️ '} parent_run_id inherited from parent: ${parentRunInherited}`
    );
  });

  if (allPassed) {
    console.log(
      '\n✅ Host-set fields propagate. (Run-identity inheritance is informational — see ⚠️ markers above for any LangGraph-runtime overwrites.)'
    );
    process.exit(0);
  } else {
    console.log('\n❌ FAIL: at least one expected host-set key was missing.');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Script error:', err);
  process.exit(1);
});
