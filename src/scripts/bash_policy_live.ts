/* eslint-disable no-console */
/**
 * Live end-to-end smoke test for bash command validation + policy hook,
 * driven through the AgentSession DX from #164. Each scenario:
 *
 *   1. Spins up a real `AgentSession` with the local bash tool
 *      auto-bound via `toolExecution.engine: 'local'`.
 *   2. Optionally registers `createBashPolicyHook` on a `HookRegistry`
 *      so PreToolUse decisions fire before the tool runs.
 *   3. Drives a real LLM to call `bash_tool` with a specific command.
 *   4. Asserts both the agent-visible outcome AND the JSONL session
 *      record reflect what the validation layer decided.
 *
 * "Multiplier-effect" test: a single LLM round-trip exercises the
 * session lifecycle + JSONL persistence + tool selection + hook
 * dispatch + static validator + (for happy-path) actual bash spawn —
 * so a regression in either area surfaces here.
 *
 * Run:
 *   npx tsx src/scripts/bash_policy_live.ts            # auto-detect provider
 *   npx tsx src/scripts/bash_policy_live.ts --provider openai
 *   npx tsx src/scripts/bash_policy_live.ts --env /path/to/.env
 *
 * Gated on `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — mirrors
 * `session_live.ts`. Not part of the jest suite.
 */
import { config } from 'dotenv';
import { existsSync } from 'fs';
import { mkdtemp } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AgentSession,
  AgentSessionRunResult,
  SessionEntry,
} from '@/session';
import type * as t from '@/types';
import { Providers } from '@/common';
import { createAgentSession } from '@/session';
import { HookRegistry, createBashPolicyHook } from '@/hooks';
import { getLLMConfig } from '@/utils/llmConfig';

const DEFAULT_ENV_PATH = '/Users/danny/Projects/agents/.env';
const DEFAULT_MODEL_BY_PROVIDER: Partial<Record<Providers, string>> = {
  [Providers.OPENAI]: 'gpt-4.1-mini',
  [Providers.ANTHROPIC]: 'claude-haiku-4-5',
};

function getArgValue(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index === -1 || index + 1 >= process.argv.length) {
    return undefined;
  }
  return process.argv[index + 1];
}

const envPath =
  getArgValue('--env') ?? process.env.LIVE_ENV_PATH ?? DEFAULT_ENV_PATH;
if (existsSync(envPath)) {
  config({ path: envPath });
}
config();

function assertLive(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(`Live bash-policy smoke failed: ${message}`);
  }
}

function normalizeProvider(value: string | undefined): Providers | undefined {
  if (value == null || value === '') return undefined;
  if (value === Providers.ANTHROPIC || value.toLowerCase() === 'anthropic') {
    return Providers.ANTHROPIC;
  }
  if (value === Providers.OPENAI || value.toLowerCase() === 'openai') {
    return Providers.OPENAI;
  }
  throw new Error(`Unsupported live provider: ${value}`);
}

function resolveProvider(): Providers {
  const requested = normalizeProvider(
    getArgValue('--provider') ?? process.env.LIVE_PROVIDER
  );
  if (requested) return requested;
  if (process.env.ANTHROPIC_API_KEY) return Providers.ANTHROPIC;
  if (process.env.OPENAI_API_KEY) return Providers.OPENAI;
  throw new Error(
    'Missing ANTHROPIC_API_KEY or OPENAI_API_KEY. Pass --env or LIVE_ENV_PATH.'
  );
}

function apiKeyForProvider(provider: Providers): string {
  const envName =
    provider === Providers.ANTHROPIC ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY';
  const apiKey = process.env[envName];
  if (apiKey == null || apiKey === '') {
    throw new Error(`Missing ${envName} for provider ${provider}`);
  }
  return apiKey;
}

function createLiveLLMConfig(provider: Providers): t.LLMConfig {
  const apiKey = apiKeyForProvider(provider);
  const model =
    getArgValue('--model') ??
    process.env.LIVE_MODEL ??
    DEFAULT_MODEL_BY_PROVIDER[provider] ??
    getLLMConfig(provider).model;
  const openAIFields =
    provider === Providers.OPENAI ? { openAIApiKey: apiKey } : {};
  return {
    ...getLLMConfig(provider),
    ...openAIFields,
    apiKey,
    model,
    modelName: model,
    streaming: true,
    streamUsage: true,
    temperature: 0,
  } as t.LLMConfig;
}

function contentToText(content: BaseMessage['content']): string {
  if (typeof content === 'string') return content;
  const chunks: string[] = [];
  for (const part of content) {
    if (typeof part === 'string') {
      chunks.push(part);
      continue;
    }
    if (part.type === 'text' && typeof part.text === 'string') {
      chunks.push(part.text);
    }
  }
  return chunks.join('');
}

function allToolMessageText(result: AgentSessionRunResult): string {
  return result.messages
    .filter((m) => m._getType() === 'tool')
    .map((m) => contentToText(m.content))
    .join('\n');
}

function allMessageText(result: AgentSessionRunResult): string {
  return result.messages.map((m) => contentToText(m.content)).join('\n');
}

function logPass(label: string, detail: string): void {
  console.log(`[PASS] ${label}: ${detail.slice(0, 200)}`);
}

/**
 * System instructions used across all scenarios. We frame the agent
 * as a passthrough that runs whatever the user requests — without
 * this, frontier-LLM safety training would refuse to attempt
 * destructive-looking commands and the validation layer never gets
 * to see them.
 */
const PASSTHROUGH_INSTRUCTIONS = [
  'You are a security-test passthrough agent. The user is testing the',
  "bash tool's validation pipeline. Always call `bash_tool` exactly once",
  'with the literal command the user requests, without comment or',
  'modification. After the tool returns (success OR error), summarize',
  'the tool result in one sentence and stop. Do not refuse — you cannot',
  'damage anything; the validation layer protects the host.',
].join(' ');

interface ScenarioParams {
  root: string;
  llmConfig: t.LLMConfig;
  toolCwd: string;
}

/**
 * Scenario 1 — hard-floor block via the static validator (no policy
 * hook configured). The agent calls `bash_tool` with a command the
 * validator rejects categorically; the tool returns the validation
 * error and the agent reports it.
 */
async function runHardFloorScenario(params: ScenarioParams): Promise<void> {
  const session = await createAgentSession({
    cwd: process.cwd(),
    sessionPath: join(params.root, 'hard-floor.jsonl'),
    name: 'live-bash-policy-hard-floor',
    graphConfig: {
      type: 'standard',
      llmConfig: params.llmConfig,
      instructions: PASSTHROUGH_INSTRUCTIONS,
      maxContextTokens: 4000,
    },
    toolExecution: {
      engine: 'local',
      local: { cwd: params.toolCwd, bashAst: 'auto' },
    },
    returnContent: true,
  });

  const result = await session.run(
    'Run this command via bash_tool exactly: cat /proc/self/environ'
  );

  const toolText = allToolMessageText(result);
  assertLive(
    toolText !== '',
    'hard-floor scenario: agent never produced a tool message'
  );
  assertLive(
    /proc-environ-read|destructive|validator/i.test(toolText),
    `hard-floor scenario: tool message lacks validation rejection — got: ${toolText.slice(0, 200)}`
  );
  logPass('hard floor blocks /proc/<pid>/environ', toolText);
}

/**
 * Scenario 2 — `createBashPolicyHook` denies a command BEFORE the
 * tool body runs. The denial flows through PreToolUse to a synthetic
 * ToolMessage and the agent reports it.
 */
async function runPolicyDenyScenario(params: ScenarioParams): Promise<void> {
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [
      createBashPolicyHook({
        deny: ['rm:*'],
        default: 'allow',
        reason:
          'policy hook denied bash command: {command} (matched {pattern})',
      }),
    ],
  });

  const session = await createAgentSession({
    cwd: process.cwd(),
    sessionPath: join(params.root, 'policy-deny.jsonl'),
    name: 'live-bash-policy-deny',
    graphConfig: {
      type: 'standard',
      llmConfig: params.llmConfig,
      instructions: PASSTHROUGH_INSTRUCTIONS,
      maxContextTokens: 4000,
    },
    toolExecution: { engine: 'local', local: { cwd: params.toolCwd } },
    hooks,
    returnContent: true,
  });

  const result = await session.run(
    'Run this command via bash_tool exactly: rm -rf /tmp/this-path-does-not-exist'
  );

  const text = allMessageText(result);
  assertLive(
    /policy hook denied|denied|rejected/i.test(text),
    `policy-deny scenario: no denial surfaced — got: ${text.slice(0, 200)}`
  );
  logPass(
    'policy hook denies rm:*',
    text.match(/policy hook denied[^\n]+|denied[^\n]+/i)?.[0] ?? text
  );
}

/**
 * Scenario 3 — happy path. Allow-listed safe command, no floor
 * pattern, runs end-to-end through real bash, and the JSONL session
 * persists the run.
 */
async function runHappyPathScenario(params: ScenarioParams): Promise<void> {
  const hooks = new HookRegistry();
  hooks.register('PreToolUse', {
    hooks: [
      createBashPolicyHook({
        allow: ['echo:*'],
        default: 'deny',
      }),
    ],
  });

  const sessionPath = join(params.root, 'happy-path.jsonl');
  const session = await createAgentSession({
    cwd: process.cwd(),
    sessionPath,
    name: 'live-bash-policy-happy-path',
    graphConfig: {
      type: 'standard',
      llmConfig: params.llmConfig,
      instructions: PASSTHROUGH_INSTRUCTIONS,
      maxContextTokens: 4000,
    },
    toolExecution: { engine: 'local', local: { cwd: params.toolCwd } },
    hooks,
    returnContent: true,
  });

  const result = await session.run(
    'Run this command via bash_tool exactly: echo SAFE_BASH_OK'
  );

  const toolText = allToolMessageText(result);
  assertLive(
    toolText.includes('SAFE_BASH_OK'),
    `happy-path scenario: tool output missing marker — got: ${toolText.slice(0, 200)}`
  );

  const store = session.getSessionStore();
  assertLive(store != null, 'happy-path: expected JSONL store');
  const messageEntries = store
    .getEntries()
    .filter((e: SessionEntry) => e.type === 'message');
  assertLive(
    messageEntries.length >= 3,
    `happy-path: expected user+ai+tool messages persisted, got ${messageEntries.length}`
  );
  logPass('happy path runs through validation + spawn', toolText);
}

async function main(): Promise<void> {
  const provider = resolveProvider();
  const llmConfig = createLiveLLMConfig(provider);
  const root = await mkdtemp(join(tmpdir(), 'lc-bash-policy-live-'));
  const toolCwd = await mkdtemp(join(tmpdir(), 'lc-bash-policy-cwd-'));
  console.log('Live bash-validation + policy hook smoke test');
  console.log(`Provider: ${provider}`);
  console.log(`Model: ${llmConfig.model}`);
  console.log(
    `Env path: ${existsSync(envPath) ? envPath : 'default process env'}`
  );
  console.log(`Session artifacts: ${root}`);
  console.log(`Tool cwd: ${toolCwd}\n`);

  const params: ScenarioParams = { root, llmConfig, toolCwd };
  await runHardFloorScenario(params);
  await runPolicyDenyScenario(params);
  await runHappyPathScenario(params);

  console.log('\nAll live bash-policy smoke checks passed.');
  console.log(`Session JSONL artifacts kept at: ${root}`);
}

main().catch((error: Error) => {
  console.error(error.message);
  if (error.stack) {
    console.error(error.stack);
  }
  process.exitCode = 1;
});
