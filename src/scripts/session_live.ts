import { config } from 'dotenv';
import { existsSync } from 'fs';
import { mkdtemp } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AgentSession,
  AgentSessionRunResult,
  SessionMessageEntry,
} from '@/session';
import type * as t from '@/types';
import {
  createOpenAIHandlers,
  createOpenAIStreamTracker,
  sendOpenAIFinalChunk,
} from '@/openai';
import {
  createResponseTracker,
  createResponsesEventHandlers,
  emitResponseCompleted,
} from '@/responses';
import { Constants, GraphEvents, Providers } from '@/common';
import { createAgentSession } from '@/session';
import { Calculator } from '@/tools/Calculator';
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
    throw new Error(`Live session smoke failed: ${message}`);
  }
}

function normalizeProvider(value: string | undefined): Providers | undefined {
  if (value == null || value === '') {
    return undefined;
  }
  if (value === Providers.ANTHROPIC || value.toLowerCase() === 'anthropic') {
    return Providers.ANTHROPIC;
  }
  if (
    value === Providers.OPENAI ||
    value.toLowerCase() === 'openai' ||
    value.toLowerCase() === 'openai'
  ) {
    return Providers.OPENAI;
  }
  throw new Error(`Unsupported live provider: ${value}`);
}

function resolveProvider(): Providers {
  const requested = normalizeProvider(
    getArgValue('--provider') ?? process.env.LIVE_PROVIDER
  );
  if (requested) {
    return requested;
  }
  if (process.env.ANTHROPIC_API_KEY) {
    return Providers.ANTHROPIC;
  }
  if (process.env.OPENAI_API_KEY) {
    return Providers.OPENAI;
  }
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

function createAgentInputs(params: {
  agentId: string;
  provider: Providers;
  llmConfig: t.LLMConfig;
  instructions: string;
}): t.AgentInputs {
  const { provider: _provider, ...clientOptions } = params.llmConfig;
  return {
    agentId: params.agentId,
    provider: params.provider,
    clientOptions: clientOptions as t.ClientOptions,
    instructions: params.instructions,
    maxContextTokens: 8000,
  };
}

function contentToText(content: BaseMessage['content']): string {
  if (typeof content === 'string') {
    return content;
  }
  const chunks: string[] = [];
  for (const part of content) {
    if (typeof part === 'string') {
      chunks.push(part);
      continue;
    }
    if (part.type === 'text' && typeof part.text === 'string') {
      chunks.push(part.text);
      continue;
    }
    if ('think' in part && typeof part.think === 'string') {
      chunks.push(part.think);
    }
  }
  return chunks.join('');
}

function resultText(result: AgentSessionRunResult): string {
  if (result.text.trim() !== '') {
    return result.text.trim();
  }
  const aiMessage = [...result.messages]
    .reverse()
    .find((message) => message._getType() === 'ai');
  return aiMessage ? contentToText(aiMessage.content).trim() : '';
}

function resultMessagesText(result: AgentSessionRunResult): string {
  return result.messages
    .map((message) => contentToText(message.content))
    .filter((text) => text.trim() !== '')
    .join('\n')
    .trim();
}

function preview(text: string): string {
  return text.replace(/\s+/g, ' ').slice(0, 180);
}

function logPass(label: string, detail: string): void {
  console.log(`[PASS] ${label}: ${detail}`);
}

function messageName(message: BaseMessage): string | undefined {
  return (message as BaseMessage & { name?: string }).name;
}

function hasSubagentToolMessage(messages: BaseMessage[]): boolean {
  return messages.some(
    (message) =>
      message._getType() === 'tool' &&
      messageName(message) === Constants.SUBAGENT
  );
}

function getStore(
  session: AgentSession
): NonNullable<ReturnType<AgentSession['getSessionStore']>> {
  const store = session.getSessionStore();
  assertLive(store != null, 'expected JSONL-backed session store');
  return store;
}

async function runAdapterSmoke(): Promise<void> {
  const openAIWrites: string[] = [];
  const openAITracker = createOpenAIStreamTracker();
  const openAIHandlers = createOpenAIHandlers({
    writer: { write: (data) => void openAIWrites.push(data) },
    context: { requestId: 'chatcmpl_live', model: 'agent-live', created: 1 },
    tracker: openAITracker,
  });
  await openAIHandlers[GraphEvents.ON_MESSAGE_DELTA].handle(
    GraphEvents.ON_MESSAGE_DELTA,
    {
      id: 'msg_live',
      delta: { content: [{ type: 'text', text: 'adapter text' }] },
    } satisfies t.MessageDeltaEvent
  );
  await sendOpenAIFinalChunk({
    writer: { write: (data) => void openAIWrites.push(data) },
    context: { requestId: 'chatcmpl_live', model: 'agent-live', created: 1 },
    tracker: openAITracker,
  });
  assertLive(
    openAIWrites.join('').includes('"content":"adapter text"'),
    'OpenAI-compatible adapter did not stream text'
  );

  const responseWrites: string[] = [];
  const responseTracker = createResponseTracker();
  const responseHandlers = createResponsesEventHandlers({
    writer: { write: (data) => void responseWrites.push(data) },
    context: { responseId: 'resp_live', model: 'agent-live', createdAt: 1 },
    tracker: responseTracker,
  });
  await responseHandlers[GraphEvents.ON_MESSAGE_DELTA].handle(
    GraphEvents.ON_MESSAGE_DELTA,
    {
      id: 'msg_live',
      delta: { content: [{ type: 'text', text: 'response text' }] },
    } satisfies t.MessageDeltaEvent
  );
  await emitResponseCompleted({
    writer: { write: (data) => void responseWrites.push(data) },
    context: { responseId: 'resp_live', model: 'agent-live', createdAt: 1 },
    tracker: responseTracker,
  });
  assertLive(
    responseWrites.join('').includes('response.completed'),
    'Responses-compatible adapter did not complete'
  );
  logPass('adapter smoke', 'OpenAI chat and Responses writers emitted events');
}

async function runSessionLifecycleSmoke(params: {
  root: string;
  provider: Providers;
  llmConfig: t.LLMConfig;
}): Promise<void> {
  const basePath = join(params.root, 'base-session.jsonl');
  const session = await createAgentSession({
    cwd: process.cwd(),
    sessionPath: basePath,
    name: 'live-session-base',
    checkpointing: true,
    graphConfig: {
      type: 'standard',
      llmConfig: params.llmConfig,
      instructions:
        'You are a concise SDK smoke-test assistant. Obey exact output-token requests.',
      tools: [new Calculator()],
      maxContextTokens: 8000,
    },
    returnContent: true,
  });

  const first = await session.run(
    'Use the calculator tool to compute (18 * 7) + 13. Include the token SESSION_BASE_OK.'
  );
  const firstText = resultText(first);
  assertLive(firstText.includes('SESSION_BASE_OK'), 'base run marker missing');
  assertLive(firstText.includes('139'), 'calculator result missing');
  logPass('standard session run', preview(firstText));

  const store = getStore(session);
  const firstCheckpoint = await session.getLatestCheckpoint();
  assertLive(
    firstCheckpoint?.threadId === session.threadId,
    'LangGraph checkpoint reference missing after run'
  );
  assertLive(
    store.getCheckpoints(session.threadId).length > 0,
    'JSONL checkpoint journal entry missing'
  );
  logPass(
    'LangGraph checkpoint state',
    firstCheckpoint.checkpointId ?? 'latest'
  );

  const forkPoint = store.getForkPoints()[0];
  assertLive(forkPoint != null, 'no user fork point recorded');
  await store.setLabel(forkPoint.id, 'first user turn');
  assertLive(
    store.getLabel(forkPoint.id) === 'first user turn',
    'label round-trip failed'
  );
  logPass(
    'JSONL persistence',
    `${store.getEntries().length} entries at ${store.path}`
  );

  const clone = await session.clone({
    cwd: params.root,
    name: 'live-session-clone',
  });
  const cloneResult = await clone.run(
    'Continue this cloned session and include the token SESSION_CLONE_OK.'
  );
  assertLive(
    resultText(cloneResult).includes('SESSION_CLONE_OK'),
    'clone run marker missing'
  );
  logPass('clone', getStore(clone).path);

  const forkBefore = await session.fork(forkPoint.id, {
    cwd: params.root,
    name: 'live-session-fork-before',
    position: 'before',
  });
  const forkResult = await forkBefore.run(
    'This replacement branch should not know the prior arithmetic. Include SESSION_FORK_OK.'
  );
  assertLive(
    resultText(forkResult).includes('SESSION_FORK_OK'),
    'fork-before run marker missing'
  );
  logPass('fork before entry', getStore(forkBefore).path);

  await session.branch(forkPoint.id, {
    position: 'before',
    summarizeAbandoned: {
      instructions: 'Summary of abandoned live branch before branch switch.',
    },
  });
  const branchResult = await session.run(
    'This is an in-place alternate branch. Include SESSION_BRANCH_OK.'
  );
  assertLive(
    resultText(branchResult).includes('SESSION_BRANCH_OK'),
    'branch run marker missing'
  );
  const branchStore = getStore(session);
  assertLive(
    branchStore.getEntries().some((entry) => entry.type === 'compaction'),
    'branch abandoned-summary compaction entry missing'
  );
  logPass('branch in place', `${branchStore.getTree().length} root branch(es)`);

  await session.compact({
    instructions:
      'Write a concise checkpoint summary of this live session branch.',
    retainRecentTurns: 0,
  });
  const compactedMessages = branchStore.getMessages();
  assertLive(
    compactedMessages[0]?._getType() === 'system' &&
      typeof compactedMessages[0].content === 'string' &&
      compactedMessages[0].content.trim() !== '',
    'manual compaction summary not active'
  );
  assertLive(
    branchStore
      .getCheckpoints(session.threadId)
      .some((entry) => entry.data.source === 'reset'),
    'checkpoint reset entry missing after compact'
  );
  logPass('manual compact', `${compactedMessages.length} active messages`);

  const resumed = await createAgentSession({
    cwd: process.cwd(),
    ephemeral: true,
    graphConfig: {
      type: 'standard',
      llmConfig: params.llmConfig,
      instructions:
        'You are a concise SDK smoke-test assistant. Obey exact output-token requests.',
      maxContextTokens: 8000,
    },
    returnContent: true,
  });
  await resumed.resumeSession(basePath);
  assertLive(
    getStore(resumed).getMessages().length === branchStore.getMessages().length,
    'resumeSession did not restore active message path'
  );

  const stream = resumed.stream(
    'Reply with the exact token SESSION_STREAM_OK and no extra words.'
  );
  let streamedText = '';
  for await (const chunk of stream.toTextStream()) {
    streamedText += chunk;
  }
  const streamResult = await stream.finalResult();
  const finalStreamText = streamedText || resultText(streamResult);
  assertLive(
    finalStreamText.includes('SESSION_STREAM_OK'),
    'stream helper marker missing'
  );
  logPass('resume and stream helpers', preview(finalStreamText));
}

async function runMultiAgentSmoke(params: {
  root: string;
  provider: Providers;
  llmConfig: t.LLMConfig;
}): Promise<void> {
  const agents: t.AgentInputs[] = [
    createAgentInputs({
      agentId: 'session_architect',
      provider: params.provider,
      llmConfig: params.llmConfig,
      instructions:
        'You are Agent A. Start with AGENT_A and give one JSONL session-tree DX benefit in one sentence.',
    }),
    createAgentInputs({
      agentId: 'session_reviewer',
      provider: params.provider,
      llmConfig: params.llmConfig,
      instructions:
        'You are Agent B. Start with AGENT_B and add one implementation caution. End with MULTI_AGENT_OK.',
    }),
  ];
  const result = await (
    await createAgentSession({
      cwd: process.cwd(),
      sessionPath: join(params.root, 'multi-agent-session.jsonl'),
      name: 'live-session-multi-agent',
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges: [
          {
            from: 'session_architect',
            to: 'session_reviewer',
            edgeType: 'direct',
            description: 'Review the architect output',
          },
        ],
      },
      returnContent: true,
    })
  ).run('Run the two-agent SDK session DX review.');

  const aiCount = result.messages.filter(
    (message) => message._getType() === 'ai'
  ).length;
  const text = resultText(result) || resultMessagesText(result);
  assertLive(
    aiCount >= 2,
    'multi-agent direct edge did not produce two AI turns'
  );
  assertLive(text !== '', 'multi-agent graph produced empty text');
  logPass('multi-agent direct graph', preview(text));
}

async function runSubagentSmoke(params: {
  root: string;
  provider: Providers;
  llmConfig: t.LLMConfig;
}): Promise<void> {
  const analystConfig = createAgentInputs({
    agentId: 'jsonl_analyst',
    provider: params.provider,
    llmConfig: params.llmConfig,
    instructions:
      'You are an isolated JSONL analyst. Give exactly two concise tradeoffs and include SUBAGENT_CHILD_OK.',
  });
  const supervisor = createAgentInputs({
    agentId: 'supervisor',
    provider: params.provider,
    llmConfig: params.llmConfig,
    instructions:
      'You are a supervisor. You must use the subagent tool exactly once for JSONL tradeoff analysis, then summarize it and include SUBAGENT_PARENT_OK.',
  });
  supervisor.subagentConfigs = [
    {
      type: 'jsonl_analyst',
      name: 'JSONL Analyst',
      description: 'Analyzes JSONL session-tree tradeoffs.',
      agentInputs: analystConfig,
    },
  ];

  const session = await createAgentSession({
    cwd: process.cwd(),
    sessionPath: join(params.root, 'subagent-session.jsonl'),
    name: 'live-session-subagent',
    graphConfig: {
      type: 'standard',
      agents: [supervisor],
    },
    returnContent: true,
  });
  const result = await session.run(
    'Delegate to jsonl_analyst using the subagent tool. Ask for two tradeoffs of JSONL session trees for CI replay, then summarize.'
  );
  const storedMessages = getStore(session).getMessages();
  assertLive(
    hasSubagentToolMessage(result.messages) ||
      hasSubagentToolMessage(storedMessages),
    'subagent tool message missing'
  );
  const text = resultText(result) || resultMessagesText(result);
  assertLive(text !== '', 'subagent delegation produced empty text');
  logPass('subagent delegation', preview(text));
}

async function main(): Promise<void> {
  const provider = resolveProvider();
  const llmConfig = createLiveLLMConfig(provider);
  const root = await mkdtemp(join(tmpdir(), 'lc-agent-session-live-'));
  console.log('Live AgentSession smoke test');
  console.log(`Provider: ${provider}`);
  console.log(`Model: ${llmConfig.model}`);
  console.log(
    `Env path: ${existsSync(envPath) ? envPath : 'default process env'}`
  );
  console.log(`Artifacts: ${root}\n`);

  await runAdapterSmoke();
  await runSessionLifecycleSmoke({ root, provider, llmConfig });
  await runMultiAgentSmoke({ root, provider, llmConfig });
  await runSubagentSmoke({ root, provider, llmConfig });

  console.log('\nAll live session smoke checks passed.');
  console.log(`Session JSONL artifacts kept at: ${root}`);
}

main().catch((error: Error) => {
  console.error(error.message);
  if (error.stack) {
    console.error(error.stack);
  }
  process.exitCode = 1;
});
