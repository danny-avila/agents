/**
 * src/scripts/local_engine_hooks.ts
 *
 * Live demonstration that PreToolUse / PostToolUse hooks now fire on
 * the direct-path tools the local engine registers. Wires up:
 *
 *   - a `createToolPolicyHook` that DENIES `write_file` and `edit_file`
 *     while allowing reads, `bash`, and the search tools, and
 *   - an explicit `PostToolUse` hook that prefixes every successful
 *     tool result with a "[reviewed]" tag so you can confirm it ran
 *     against the in-process tools.
 *
 * The script asks the model to (a) write a file, (b) read it back. Step
 * (a) should be blocked with a `Blocked: ...` ToolMessage and a
 * `PermissionDenied` hook fire; step (b) should succeed with the
 * "[reviewed]" prefix injected by `PostToolUse`.
 *
 * Run with: `npm run local:hooks`
 */
import { config } from 'dotenv';
config();
import { tmpdir } from 'os';
import { join } from 'path';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { HookRegistry, createToolPolicyHook } from '@/hooks';
import type {
  PostToolUseHookOutput,
  PermissionDeniedHookOutput,
} from '@/hooks';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

async function main(): Promise<void> {
  const { userName, provider } = await getArgs();
  const cwd = await mkdtemp(join(tmpdir(), 'lc-local-hooks-'));
  console.log(`[local_engine_hooks] workspace: ${cwd}`);

  // Seed a file so read_file has something interesting to return.
  await writeFile(
    join(cwd, 'allowed.txt'),
    'this is a pre-existing file the agent is allowed to read\n',
    'utf8'
  );

  // --- HOOK SETUP ---
  const denyEvents: { tool: string; reason?: string }[] = [];
  const reviewedTools = new Set<string>();

  const hookRegistry = new HookRegistry();

  // PreToolUse policy: deny mutations, allow everything else.
  hookRegistry.register('PreToolUse', {
    hooks: [
      createToolPolicyHook({
        mode: 'bypass',
        deny: ['write_file', 'edit_file'],
        reason: 'this script blocks {tool} to demonstrate direct-path hooks',
      }),
    ],
  });

  // PostToolUse: tag successful results so we can see the hook running.
  hookRegistry.register('PostToolUse', {
    hooks: [
      async ({
        toolName,
        toolOutput,
      }): Promise<PostToolUseHookOutput> => {
        reviewedTools.add(toolName);
        const text =
          typeof toolOutput === 'string'
            ? toolOutput
            : JSON.stringify(toolOutput);
        return { updatedOutput: `[reviewed] ${text}` };
      },
    ],
  });

  // PermissionDenied: observational sink so we can prove the hook fired.
  hookRegistry.register('PermissionDenied', {
    hooks: [
      async ({
        toolName,
        reason,
      }): Promise<PermissionDeniedHookOutput> => {
        denyEvents.push({ tool: toolName, reason });
        return {};
      },
    ],
  });

  // --- STREAM HANDLERS ---
  const { contentParts, aggregateContent } = createContentAggregator();
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
        const cast = data as unknown as { result: t.ToolEndEvent };
        const tc = (cast.result as { tool_call?: { name?: string } } | undefined)
          ?.tool_call;
        console.log(`====== ON_RUN_STEP_COMPLETED tool=${tc?.name ?? '?'} ======`);
        aggregateContent({ event, data: cast });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (_event: string, data: t.StreamEventData): void => {
        const obj = data as unknown as { name?: string; input?: unknown };
        console.log(
          `====== TOOL_START tool=${obj.name ?? '?'} input=${JSON.stringify(
            obj.input
          )} ======`
        );
      },
    },
  };

  const llmConfig = getLLMConfig(provider);

  const runConfig: t.RunConfig = {
    runId: 'local-engine-hooks-1',
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions:
        'You are a coding assistant. The host has loaded a permissions policy. ' +
        'When a tool call is denied you will see a ToolMessage starting with ' +
        '`Blocked:` — DO NOT retry the same denied tool. If a write is denied, ' +
        'just acknowledge the denial and continue with operations that are allowed.',
    },
    toolExecution: {
      engine: 'local',
      local: { cwd, timeoutMs: 30_000 },
    },
    hooks: hookRegistry,
    returnContent: true,
    skipCleanup: true,
    customHandlers,
  };
  const run = await Run.create<t.IState>(runConfig);

  const streamConfig = {
    configurable: { provider, thread_id: 'local-engine-hooks-thread-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = new HumanMessage(
    `Hi ${userName}. Please do these two things in order:\n\n` +
      '1. Use `write_file` to create `blocked.txt` with the content "should never land".\n' +
      '2. Use `read_file` on `allowed.txt` and tell me what is inside.\n\n' +
      'Then report whether each step succeeded.'
  );
  conversationHistory.push(userMessage);
  console.log('====== USER ======\n' + userMessage.content + '\n');

  await run.processStream({ messages: conversationHistory }, streamConfig);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log('\n====== FINAL CONTENT PARTS ======');
  console.dir(contentParts, { depth: null });

  console.log('\n====== HOOK OBSERVATIONS ======');
  console.log(
    `PermissionDenied fired ${denyEvents.length} time(s):`,
    denyEvents
  );
  console.log(
    `PostToolUse saw tools: ${[...reviewedTools].join(', ') || '<none>'}`
  );

  // The deny-target file must NOT exist.
  const fs = await import('fs/promises');
  const exists = await fs
    .stat(join(cwd, 'blocked.txt'))
    .then(() => true)
    .catch(() => false);
  console.log(
    `blocked.txt landed on disk? ${exists} ${
      exists ? '(BUG)' : '(expected: false)'
    }`
  );

  await rm(cwd, { recursive: true, force: true });
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
