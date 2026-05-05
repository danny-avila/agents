/**
 * src/scripts/local_engine_workspace.ts
 *
 * Live demo of the workspace boundary + HITL "ask the user when the
 * agent wants to leave the workspace" flow.
 *
 * Setup:
 *   - workspace.root  = a temp dir
 *   - additionalRoots = a sibling temp dir (sim. monorepo)
 *   - allowReadOutside = true (so the file tool's hard clamp lets the
 *     hook be the gate)
 *   - createWorkspacePolicyHook with outsideRead='ask'
 *   - humanInTheLoop.enabled = true
 *
 * The agent is then asked to read three files:
 *   1. one inside the workspace      → expect: PreToolUse 'allow', tool runs
 *   2. one inside the additional root → expect: PreToolUse 'allow', tool runs
 *   3. one truly outside              → expect: PreToolUse 'ask' interrupt
 *      → script auto-approves on resume → tool runs
 *
 * Asserts that the host received the interrupt for case (3) and that
 * the resume completed cleanly. Verifies the existing HITL machinery
 * (#134) does the lifting end-to-end with the new policy hook.
 *
 * Run with: `npm run local:workspace`
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { HookRegistry, createWorkspacePolicyHook } from '@/hooks';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

async function main(): Promise<void> {
  const { userName, provider } = await getArgs();
  const workspace = await mkdtemp(join(tmpdir(), 'lc-ws-demo-'));
  const sibling = await mkdtemp(join(tmpdir(), 'lc-ws-extra-'));
  const outside = await mkdtemp(join(tmpdir(), 'lc-ws-outside-'));
  console.log(`[ws] workspace root: ${workspace}`);
  console.log(`[ws] additional root: ${sibling}`);
  console.log(`[ws] outside root  : ${outside}`);

  await writeFile(join(workspace, 'inside.txt'), 'INSIDE\n', 'utf8');
  await writeFile(join(sibling, 'sibling.txt'), 'SIBLING\n', 'utf8');
  await writeFile(join(outside, 'secret.txt'), 'SECRET\n', 'utf8');

  const hookRegistry = new HookRegistry();
  hookRegistry.register('PreToolUse', {
    hooks: [
      createWorkspacePolicyHook({
        root: workspace,
        additionalRoots: [sibling],
        outsideRead: 'ask',
        outsideWrite: 'ask',
        reason: 'workspace-policy: {tool} wants outside paths {paths}',
      }),
    ],
  });

  const { aggregateContent } = createContentAggregator();
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    // Forward ON_RUN_STEP so the aggregator's stepMap is seeded
    // before ON_RUN_STEP_COMPLETED arrives (issue #142).
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData): void => {
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const obj = data as unknown as { name?: string; input?: unknown };
        console.log(
          `====== TOOL_START tool=${obj.name ?? '?'} input=${JSON.stringify(obj.input)} ======`
        );
      },
    },
  };

  const llmConfig = getLLMConfig(Providers.ANTHROPIC);
  const checkpointer = new MemorySaver();
  const conversation: BaseMessage[] = [];

  const buildRun = async (): Promise<Run<t.IState>> => {
    const runConfig: t.RunConfig = {
      runId: `local-engine-ws-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig: { ...llmConfig, promptCache: true },
        compileOptions: { checkpointer },
        instructions:
          `You are ${userName}'s assistant. The host has a workspace policy ` +
          'on `read_file`. If a read is outside the workspace, the host will ' +
          'ask the user to approve. After approval, retry the read.',
      },
      toolExecution: {
        engine: 'local',
        local: {
          workspace: {
            root: workspace,
            additionalRoots: [sibling],
            allowReadOutside: true,
          },
          timeoutMs: 30_000,
        },
      },
      hooks: hookRegistry,
      humanInTheLoop: { enabled: true },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    };
    return Run.create<t.IState>(runConfig);
  };

  const threadId = `ws-thread-${Date.now()}`;
  const streamConfig = {
    configurable: { provider, thread_id: threadId },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi ${userName}. Please call \`read_file\` on each of these in order:\n` +
      `  1. \`inside.txt\` (relative)\n` +
      `  2. \`${join(sibling, 'sibling.txt')}\` (sibling root, should be allowed)\n` +
      `  3. \`${join(outside, 'secret.txt')}\` (outside the workspace)\n\n` +
      `For each, tell me what's in it. If a call is blocked or asks for approval, just retry it once.`
  );
  conversation.push(userMessage);
  console.log('====== USER ======\n' + userMessage.content + '\n');

  // First run — likely to interrupt on the third read.
  let run = await buildRun();
  await run.processStream(
    { messages: conversation },
    streamConfig as Parameters<typeof run.processStream>[1]
  );
  let finalMessages = run.getRunMessages();
  if (finalMessages) conversation.push(...finalMessages);

  let interrupt = run.getInterrupt();
  let resumes = 0;
  while (interrupt != null && resumes < 4) {
    resumes++;
    const payload = interrupt.payload as t.ToolApprovalInterruptPayload;
    console.log(
      `\n====== INTERRUPT raised (resume #${resumes}) ======\n` +
        `payload.action_requests = ${JSON.stringify(
          payload.action_requests,
          null,
          2
        )}`
    );

    // Auto-approve every action request.
    const decisions: t.ToolApprovalDecision[] = payload.action_requests.map(
      (req) => ({ tool_call_id: req.tool_call_id, type: 'approve' })
    );
    console.log(
      `[ws] auto-approving ${decisions.length} tool call(s) and resuming`
    );

    run = await buildRun();
    await run.resume(
      decisions,
      streamConfig as Parameters<typeof run.processStream>[1]
    );
    finalMessages = run.getRunMessages();
    if (finalMessages) conversation.push(...finalMessages);
    interrupt = run.getInterrupt();
  }

  console.log('\n====== TOOL MESSAGES IN HISTORY ======');
  let askesObserved = 0;
  let successesObserved = 0;
  for (const msg of conversation) {
    if (msg instanceof ToolMessage) {
      const head =
        typeof msg.content === 'string'
          ? msg.content.slice(0, 200)
          : JSON.stringify(msg.content).slice(0, 200);
      console.log(`- ${msg.name} (${msg.status}): ${head.replace(/\n/g, ' ⏎ ')}`);
      if (msg.status === 'error') askesObserved++;
      if (msg.status === 'success') successesObserved++;
    }
  }

  console.log('\n====== ASSISTANT FINAL TEXT ======');
  const lastAssistant = [...conversation]
    .reverse()
    .find((m) => m._getType() === 'ai');
  if (lastAssistant) {
    const c = lastAssistant.content;
    console.log(
      typeof c === 'string'
        ? c
        : Array.isArray(c)
          ? c
            .map((b) => ('text' in b ? b.text : `<${b.type}>`))
            .join(' ')
          : JSON.stringify(c)
    );
  }

  console.log('\n====== ASSERTIONS ======');
  console.log(`HITL interrupts handled: ${resumes}`);
  console.log(`successful tool messages: ${successesObserved}`);
  if (resumes === 0) {
    console.error('[ws] expected at least one HITL interrupt; got 0');
    process.exitCode = 1;
  }
  if (successesObserved < 2) {
    console.error('[ws] expected at least 2 successful reads; got', successesObserved);
    process.exitCode = 1;
  } else {
    console.log(`[ws] workspace + HITL flow ✔`);
  }

  await Promise.all([
    rm(workspace, { recursive: true, force: true }),
    rm(sibling, { recursive: true, force: true }),
    rm(outside, { recursive: true, force: true }),
  ]);
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
