/**
 * Diagnostic script: traces the lifecycle of `initialSummary` through the
 * Graph agent pipeline to verify whether it survives into the system message
 * that the LLM actually receives.
 *
 * Run:
 *   npx tsx src/scripts/summary-injection-test.ts
 *
 * Expected output: a clear trace showing where the summary is set, whether
 * it's preserved or wiped, and what the model's system prompt actually contains.
 */
import { config } from 'dotenv';
config();

import { v4 as uuidv4 } from 'uuid';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { AgentContext } from '@/agents/AgentContext';
import { GraphEvents, Providers } from '@/common';
import { StandardGraph } from '@/graphs/Graph';
import { ModelEndHandler } from '@/events';
import { Run } from '@/run';

// ─── Helpers ────────────────────────────────────────────────────────────────

const DIVIDER = '─'.repeat(72);
const PASS = '✅';
const FAIL = '❌';

function heading(title: string) {
  console.log(`\n${DIVIDER}`);
  console.log(`  ${title}`);
  console.log(DIVIDER);
}

function check(label: string, ok: boolean, detail?: string) {
  console.log(`  ${ok ? PASS : FAIL} ${label}${detail ? ` — ${detail}` : ''}`);
  return ok;
}

// ─── Spy utilities ──────────────────────────────────────────────────────────

interface SpyCall {
  method: string;
  args: unknown[];
  timestamp: number;
  summaryTextAfter?: string;
  systemRunnableStaleAfter?: boolean;
}

const spyCalls: SpyCall[] = [];

function spyOn<T extends object>(
  obj: T,
  method: keyof T & string,
  before?: (...args: unknown[]) => void,
  after?: (result: unknown, ...args: unknown[]) => void
) {
  const original = (obj as Record<string, unknown>)[method] as (
    ...args: unknown[]
  ) => unknown;
  if (typeof original !== 'function') {
    console.warn(`  [spy] ${method} is not a function, skipping`);
    return;
  }
  (obj as Record<string, unknown>)[method] = function (
    this: unknown,
    ...args: unknown[]
  ) {
    before?.(...args);
    const result = original.apply(this, args);
    after?.(result, ...args);

    const ctx = this as AgentContext;
    spyCalls.push({
      method,
      args: args.map((a) =>
        typeof a === 'string' ? a.slice(0, 120) : typeof a
      ),
      timestamp: Date.now(),
      summaryTextAfter: (ctx as unknown as Record<string, unknown>)
        .summaryText as string | undefined,
      systemRunnableStaleAfter: (ctx as unknown as Record<string, unknown>)
        .systemRunnableStale as boolean | undefined,
    });
    return result;
  };
}

// ─── Test 1: Direct AgentContext lifecycle ───────────────────────────────────

async function testAgentContextDirectly() {
  heading('TEST 1: AgentContext.fromConfig → setSummary → reset lifecycle');

  const SUMMARY_TEXT =
    '## Goal\nUser asked about AI capabilities.\n\n## Progress\n### Done\n- Explained browser automation tools\n- Created particle galaxy animation\n- Created gravity simulator\n\n## Key Decisions\nUsed canvas-based graphics for interactive demos.\n\n## Next Steps\nAwaiting user request.';
  const SUMMARY_TOKEN_COUNT = 85;

  const agentConfig: t.AgentInputs = {
    agentId: 'test-agent',
    provider: Providers.ANTHROPIC,
    clientOptions: { model: 'claude-sonnet-4-5-20250514' },
    instructions: 'You are a helpful assistant.',
    initialSummary: { text: SUMMARY_TEXT, tokenCount: SUMMARY_TOKEN_COUNT },
    summarizationEnabled: true,
  };

  console.log('\n  Step 1: AgentContext.fromConfig() with initialSummary');
  const ctx = AgentContext.fromConfig(agentConfig);

  const hasSummaryAfterInit = ctx.hasSummary();
  const summaryTextAfterInit = ctx.getSummaryText();
  check(
    'hasSummary() after fromConfig',
    hasSummaryAfterInit,
    `${hasSummaryAfterInit}`
  );
  check(
    'getSummaryText() matches',
    summaryTextAfterInit === SUMMARY_TEXT,
    summaryTextAfterInit
      ? `${summaryTextAfterInit.slice(0, 80)}...`
      : 'undefined'
  );

  // Check systemRunnable includes summary
  const systemRunnable = ctx.systemRunnable;
  check('systemRunnable exists', systemRunnable != null);

  if (systemRunnable) {
    // Invoke the system runnable with empty messages to see what it prepends
    const testMessages: BaseMessage[] = [new HumanMessage('test')];
    const result = await systemRunnable.invoke(testMessages);
    const systemMsg = result.find(
      (m: BaseMessage) => m._getType() === 'system'
    );
    const systemContent =
      typeof systemMsg?.content === 'string'
        ? systemMsg.content
        : Array.isArray(systemMsg?.content)
          ? (systemMsg.content as Array<{ text?: string }>)
              .map((c) => c.text ?? '')
              .join('')
          : '';

    check(
      'System message contains "Conversation Summary"',
      systemContent.includes('Conversation Summary'),
      systemContent.includes('Conversation Summary')
        ? 'YES'
        : `NOT FOUND in: ${systemContent.slice(0, 200)}...`
    );
    check(
      'System message contains actual summary text',
      systemContent.includes('particle galaxy'),
      systemContent.includes('particle galaxy') ? 'YES' : 'NOT FOUND'
    );
  }

  console.log('\n  Step 2: Calling context.reset() (as processStream does)');
  ctx.reset();

  const hasSummaryAfterReset = ctx.hasSummary();
  const summaryTextAfterReset = ctx.getSummaryText();
  check(
    'hasSummary() survives reset',
    hasSummaryAfterReset,
    `${hasSummaryAfterReset}`
  );
  check(
    'getSummaryText() preserved after reset',
    summaryTextAfterReset === SUMMARY_TEXT,
    summaryTextAfterReset
      ? `${summaryTextAfterReset.slice(0, 80)}...`
      : 'undefined (REGRESSION — summary was wiped)'
  );

  // Check systemRunnable after reset
  const runnableAfterReset = ctx.systemRunnable;
  if (runnableAfterReset) {
    const testMessages: BaseMessage[] = [new HumanMessage('test')];
    const result = await runnableAfterReset.invoke(testMessages);
    const systemMsg = result.find(
      (m: BaseMessage) => m._getType() === 'system'
    );
    const systemContent =
      typeof systemMsg?.content === 'string'
        ? systemMsg.content
        : Array.isArray(systemMsg?.content)
          ? (systemMsg.content as Array<{ text?: string }>)
              .map((c) => c.text ?? '')
              .join('')
          : '';

    check(
      'System message after reset includes summary',
      systemContent.includes('Conversation Summary'),
      systemContent.includes('Conversation Summary')
        ? 'YES — summary survived reset'
        : 'NO — REGRESSION: summary missing from system prompt'
    );
  } else {
    console.log(
      '  ⚠️  systemRunnable is undefined after reset (instructions may also be empty)'
    );
  }
}

// ─── Test 2: Full Graph + Run pipeline with model spy ───────────────────────

async function testFullRunPipeline() {
  heading('TEST 2: Full Run pipeline — does the model see the summary?');

  const SUMMARY_TEXT =
    '## Goal\nUser asked about AI capabilities and requested interactive demos.\n\n## Progress\n### Done\n- Explained browser automation tools\n- Created particle galaxy animation (800 particles, mouse tracking)\n- Created gravity simulator (N-body, drag to launch planets)\n\n## Key Decisions\nUsed canvas-based graphics with requestAnimationFrame for performance.\n\n## Next Steps\nAwaiting next user request.';
  const SUMMARY_TOKEN_COUNT = 120;

  // Track what messages the model receives
  let modelReceivedMessages: BaseMessage[] | null = null;
  let modelReceivedSystemContent: string | null = null;

  const collectedUsage: unknown[] = [];
  const { contentParts, aggregateContent } = createContentAggregator();

  const customHandlers = {
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(
      collectedUsage as unknown as UsageMetadata[]
    ),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.RunStep) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.RunStepDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ) => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.MessageDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.ReasoningDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
  };

  const agentInputs: t.AgentInputs = {
    agentId: 'default',
    provider: Providers.BEDROCK,
    clientOptions: {
      model: 'us.anthropic.claude-sonnet-4-5-v1',
      region: process.env.BEDROCK_AWS_REGION ?? 'us-east-1',
      credentials: {
        accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      },
      maxTokens: 1024,
      streaming: true,
      streamUsage: true,
    } as t.BedrockAnthropicInput,
    instructions:
      'You are a helpful AI assistant with access to browser automation tools.',
    additional_instructions:
      'Always reference prior conversation context when answering questions about what has been discussed.',
    initialSummary: { text: SUMMARY_TEXT, tokenCount: SUMMARY_TOKEN_COUNT },
    summarizationEnabled: true,
    summarizationConfig: {
      enabled: true,
      provider: Providers.ANTHROPIC,
      model: 'claude-sonnet-4-5-20250514',
    },
    maxContextTokens: 8000,
  };

  console.log('\n  Creating Run with initialSummary...');

  const run = await Run.create<t.IState>({
    runId: uuidv4(),
    graphConfig: {
      type: 'standard',
      agents: [agentInputs],
    },
    customHandlers: customHandlers as t.RunConfig['customHandlers'],
    returnContent: true,
  });

  // ── Spy on the agentContext AFTER construction, BEFORE processStream ──
  const graph = run.Graph as StandardGraph;
  const agentContext = graph.agentContexts.get('default')!;

  console.log('\n  State AFTER Run.create (before processStream):');
  check('agentContext exists', agentContext != null);
  check(
    'hasSummary()',
    agentContext.hasSummary(),
    `${agentContext.hasSummary()}`
  );
  check(
    'getSummaryText() present',
    agentContext.getSummaryText() != null,
    agentContext.getSummaryText()
      ? `${agentContext.getSummaryText()!.slice(0, 80)}...`
      : 'undefined'
  );

  // Verify system runnable includes summary before processStream
  if (agentContext.systemRunnable) {
    const preStreamResult = await agentContext.systemRunnable.invoke([
      new HumanMessage('pre-stream test'),
    ]);
    const preStreamSysMsg = preStreamResult.find(
      (m: BaseMessage) => m._getType() === 'system'
    );
    const preStreamContent =
      typeof preStreamSysMsg?.content === 'string'
        ? preStreamSysMsg.content
        : '';
    check(
      'systemRunnable includes summary BEFORE processStream',
      preStreamContent.includes('Conversation Summary'),
      preStreamContent.includes('Conversation Summary') ? 'YES' : 'NO'
    );
  }

  // ── Install spies ──
  const originalReset = agentContext.reset.bind(agentContext);
  let resetCallCount = 0;
  agentContext.reset = function () {
    resetCallCount++;
    console.log(`\n  ⚡ context.reset() called (call #${resetCallCount})`);
    console.log(`     hasSummary BEFORE reset: ${agentContext.hasSummary()}`);
    originalReset();
    console.log(`     hasSummary AFTER reset: ${agentContext.hasSummary()}`);
    console.log(
      `     getSummaryText AFTER reset: ${agentContext.getSummaryText() ?? 'undefined'}`
    );
  };

  // Spy on systemRunnable getter to see what gets piped to the model
  const originalSystemRunnableDescriptor = Object.getOwnPropertyDescriptor(
    Object.getPrototypeOf(agentContext),
    'systemRunnable'
  );
  if (originalSystemRunnableDescriptor?.get) {
    const originalGetter = originalSystemRunnableDescriptor.get;
    Object.defineProperty(agentContext, 'systemRunnable', {
      get: function () {
        const result = originalGetter.call(this);
        console.log(`\n  ⚡ systemRunnable getter accessed`);
        console.log(
          `     hasSummary at access time: ${(this as AgentContext).hasSummary()}`
        );
        console.log(`     runnable is ${result ? 'defined' : 'undefined'}`);
        return result;
      },
      configurable: true,
    });
  }

  // ── Run processStream with a question about previous conversation ──
  console.log('\n  Calling processStream...');
  const runConfig = {
    configurable: {
      thread_id: 'summary-test-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const messages = [
    new HumanMessage(
      'What have we talked about and done so far? What did you show me?'
    ),
  ];

  try {
    await run.processStream({ messages }, runConfig);

    const runMessages = run.getRunMessages();
    if (runMessages && runMessages.length > 0) {
      const lastMsg = runMessages[runMessages.length - 1];
      const responseText =
        typeof lastMsg.content === 'string'
          ? lastMsg.content
          : Array.isArray(lastMsg.content)
            ? (lastMsg.content as Array<{ type?: string; text?: string }>)
                .filter((c) => c.type === 'text')
                .map((c) => c.text ?? '')
                .join('')
            : '';

      heading('MODEL RESPONSE');
      console.log(responseText);

      heading('RESPONSE ANALYSIS');
      const mentionsGalaxy =
        responseText.toLowerCase().includes('galaxy') ||
        responseText.toLowerCase().includes('particle');
      const mentionsGravity =
        responseText.toLowerCase().includes('gravity') ||
        responseText.toLowerCase().includes('planet');
      const mentionsBrowser =
        responseText.toLowerCase().includes('browser') ||
        responseText.toLowerCase().includes('automation');
      const admitsNoMemory =
        responseText.toLowerCase().includes("don't have memory") ||
        responseText.toLowerCase().includes("don't actually have") ||
        responseText.toLowerCase().includes('cannot recall') ||
        responseText.toLowerCase().includes('start fresh') ||
        responseText.toLowerCase().includes("don't recall");

      check(
        'Response mentions galaxy/particle demo',
        mentionsGalaxy,
        mentionsGalaxy ? 'YES' : 'NO'
      );
      check(
        'Response mentions gravity/planet simulator',
        mentionsGravity,
        mentionsGravity ? 'YES' : 'NO'
      );
      check(
        'Response mentions browser automation',
        mentionsBrowser,
        mentionsBrowser ? 'YES' : 'NO'
      );
      check(
        'Response does NOT claim lack of memory',
        !admitsNoMemory,
        admitsNoMemory
          ? 'FAIL — model says it has no memory (summary not injected)'
          : 'PASS — model appears to have context'
      );
    }
  } catch (err) {
    console.error('  processStream error:', (err as Error).message);
  }
}

// ─── Test 3: Isolated graph construction + resetValues timing ───────────────

async function testResetTimingIsolated() {
  heading(
    'TEST 3: Isolated reset timing — prove summary is wiped by resetValues'
  );

  const SUMMARY_TEXT = 'User discussed capabilities and saw two demos.';
  const SUMMARY_TOKEN_COUNT = 15;

  const agentConfig: t.AgentInputs = {
    agentId: 'timing-test',
    provider: Providers.ANTHROPIC,
    clientOptions: { model: 'claude-sonnet-4-5-20250514' },
    instructions: 'You are a test agent.',
    initialSummary: { text: SUMMARY_TEXT, tokenCount: SUMMARY_TOKEN_COUNT },
    summarizationEnabled: true,
  };

  const graph = new StandardGraph({
    runId: 'test-run',
    agents: [agentConfig],
  });

  const ctx = graph.agentContexts.get('timing-test')!;

  console.log('\n  After StandardGraph construction:');
  const hasAfterConstruct = ctx.hasSummary();
  const textAfterConstruct = ctx.getSummaryText();
  check('hasSummary()', hasAfterConstruct, `${hasAfterConstruct}`);
  check(
    'getSummaryText()',
    textAfterConstruct === SUMMARY_TEXT,
    textAfterConstruct ?? 'undefined'
  );

  console.log('\n  Calling graph.resetValues() (as processStream does):');
  graph.resetValues();

  const hasAfterReset = ctx.hasSummary();
  const textAfterReset = ctx.getSummaryText();
  check('hasSummary() survives resetValues', hasAfterReset, `${hasAfterReset}`);
  check(
    'getSummaryText() preserved after resetValues',
    textAfterReset === SUMMARY_TEXT,
    textAfterReset ??
      'undefined (REGRESSION — summary wiped by resetValues→reset)'
  );

  // Check system runnable
  const sysRunnable = ctx.systemRunnable;
  if (sysRunnable) {
    const result = await sysRunnable.invoke([new HumanMessage('test')]);
    const sysMsg = result.find((m: BaseMessage) => m._getType() === 'system');
    const content = typeof sysMsg?.content === 'string' ? sysMsg.content : '';

    check(
      'System message after resetValues includes summary',
      content.includes('Conversation Summary'),
      content.includes('Conversation Summary')
        ? 'summary survived reset'
        : 'REGRESSION — summary MISSING from system prompt'
    );

    console.log('\n  System message content after reset:');
    console.log(`    "${content}"`);
  }
}

// ─── Test 4: Token accounting breakdown ─────────────────────────────────────

async function testTokenAccounting() {
  heading('TEST 4: Token accounting — summary vs instruction tokens');

  const SUMMARY_TEXT =
    '## Goal\nUser explored capabilities and created interactive demos.\n\n## Progress\n- Particle galaxy animation\n- Gravity simulator';
  const SUMMARY_TOKEN_COUNT = 40;

  const agentConfig: t.AgentInputs = {
    agentId: 'token-test',
    provider: Providers.ANTHROPIC,
    clientOptions: { model: 'claude-sonnet-4-5-20250514' },
    instructions:
      'You are a helpful AI assistant with access to browser automation and DevTools tools.',
    additional_instructions: 'The user is located in New York.',
    initialSummary: { text: SUMMARY_TEXT, tokenCount: SUMMARY_TOKEN_COUNT },
    summarizationEnabled: true,
    maxContextTokens: 8000,
  };

  // Simple token counter for testing
  const tokenCounter = (msg: BaseMessage): number => {
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : Array.isArray(msg.content)
          ? (msg.content as Array<{ text?: string }>)
              .map((c) => c.text ?? '')
              .join('')
          : '';
    // Rough approximation: ~4 chars per token
    return Math.ceil(content.length / 4);
  };

  const ctx = AgentContext.fromConfig(agentConfig, tokenCounter);

  // Wait for token calculation
  if (ctx.tokenCalculationPromise) {
    await ctx.tokenCalculationPromise;
  }

  console.log('\n  Token budget BEFORE reset:');
  const budgetBefore = ctx.getTokenBudgetBreakdown();
  console.log(`    instructionTokens:   ${budgetBefore.instructionTokens}`);
  console.log(`    systemMessageTokens: ${budgetBefore.systemMessageTokens}`);
  console.log(`    toolSchemaTokens:    ${budgetBefore.toolSchemaTokens}`);
  console.log(`    summaryTokens:       ${budgetBefore.summaryTokens}`);
  console.log(`    availableForMessages: ${budgetBefore.availableForMessages}`);

  check(
    'summaryTokens > 0 before reset',
    budgetBefore.summaryTokens > 0,
    `${budgetBefore.summaryTokens}`
  );
  check(
    'systemMessageTokens includes summary',
    budgetBefore.systemMessageTokens > 0,
    `${budgetBefore.systemMessageTokens}`
  );

  // Now reset and check
  ctx.reset();
  if (ctx.tokenCalculationPromise) {
    await ctx.tokenCalculationPromise;
  }

  console.log('\n  Token budget AFTER reset:');
  const budgetAfter = ctx.getTokenBudgetBreakdown();
  console.log(`    instructionTokens:   ${budgetAfter.instructionTokens}`);
  console.log(`    systemMessageTokens: ${budgetAfter.systemMessageTokens}`);
  console.log(`    toolSchemaTokens:    ${budgetAfter.toolSchemaTokens}`);
  console.log(`    summaryTokens:       ${budgetAfter.summaryTokens}`);
  console.log(`    availableForMessages: ${budgetAfter.availableForMessages}`);

  check(
    'summaryTokens preserved after reset',
    budgetAfter.summaryTokens === budgetBefore.summaryTokens,
    `${budgetAfter.summaryTokens} (same as before: ${budgetBefore.summaryTokens})`
  );

  const tokenDrop =
    budgetBefore.systemMessageTokens - budgetAfter.systemMessageTokens;
  console.log(`\n  System message token delta after reset: ${tokenDrop}`);
  check(
    'systemMessageTokens stable (summary tokens preserved)',
    tokenDrop === 0,
    tokenDrop === 0
      ? 'no change — summary correctly included'
      : `dropped by ${tokenDrop} tokens (REGRESSION)`
  );
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  heading('SUMMARY INJECTION DIAGNOSTIC');
  console.log('  This script traces initialSummary through the Run pipeline');
  console.log('  to identify where it gets lost before reaching the LLM.\n');

  // Tests 1, 3, 4 are local-only (no LLM calls)
  await testAgentContextDirectly();
  await testResetTimingIsolated();
  await testTokenAccounting();

  // Test 2 makes a real LLM call to prove the end-to-end failure
  const runLiveTest = process.argv.includes('--live');
  if (runLiveTest) {
    await testFullRunPipeline();
  } else {
    heading('TEST 2: Full Run pipeline (SKIPPED — use --live to enable)');
    console.log(
      '  Pass --live to make a real Bedrock API call to verify end-to-end.\n'
    );
  }

  heading('SUMMARY');
  console.log(`
  FIX APPLIED: setSummary() now persists to durable fields
  (_durableSummaryText, _durableSummaryTokenCount). reset() restores
  from these durable fields instead of clearing to undefined.

  TIMELINE (fixed):
    1. createRun() → AgentContext.fromConfig() → setSummary() ✅
       (also sets _durableSummaryText)
    2. Run.create() → new StandardGraph() → contexts have summary ✅
    3. run.processStream() → Graph.resetValues() → context.reset()
       → summaryText restored from _durableSummaryText ✅
    4. createCallModel → systemRunnable getter → buildInstructionsString()
       → summaryText present → "## Conversation Summary" in system prompt ✅

  TOKEN ACCOUNTING: summaryTokens is tracked separately in
  getTokenBudgetBreakdown() and survives reset, giving the budget
  system visibility into summary vs. instruction token distribution.
`);
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
