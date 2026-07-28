// src/scripts/preempt-scenarios.ts
//
// Live scenario matrix for the seal paths the basic probe does not exercise:
//
//   npx tsx --env-file=/path/to/.env src/scripts/preempt-scenarios.ts \
//     --scenario multiseal|toolturn|budget|cache|ma-halt|ma-continue|websearch \
//     --provider anthropic|bedrock|...
//
// One scenario per invocation; prints SCENARIO_VERDICT as the last line.
import { config } from 'dotenv';
config();
import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents } from '@/common';
import { createTokenCounter } from '@/utils/tokens';
import { getLLMConfig } from '@/utils/llmConfig';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

const LONG_PROMPT =
  'Write a detailed history of the Byzantine Empire from Constantine to ' +
  '1453 in at least 800 words of flowing prose. No headings or bullets.';

function argOf(flag: string, fallback: string): string {
  const index = process.argv.indexOf(flag);
  return index !== -1 && process.argv[index + 1] != null
    ? process.argv[index + 1]
    : fallback;
}

function textOf(message: BaseMessage | undefined): string {
  if (message == null) return '';
  if (typeof message.content === 'string') return message.content;
  let text = '';
  for (const block of message.content) {
    if (block.type === 'text' && typeof block.text === 'string') {
      text += block.text;
    }
  }
  return text;
}

function kindOf(message: BaseMessage): string {
  if (message.getType() === 'human') {
    const source = message.additional_kwargs.source;
    return typeof source === 'string' ? `human(${source})` : 'human';
  }
  if (message.getType() === 'tool') return 'tool';
  return message.getType();
}

type Outcome = {
  scenario: string;
  provider: string;
  ok: boolean;
  detail: Record<string, unknown>;
  error: string | null;
};

async function main(): Promise<Outcome> {
  const scenario = argOf('--scenario', 'multiseal');
  const providerKey = argOf('--provider', 'anthropic');
  const llmConfig = getLLMConfig(providerKey);
  if (llmConfig == null) {
    throw new Error(`No llmConfig entry for provider "${providerKey}"`);
  }
  const tokenCounter = await createTokenCounter();

  let armed = false;
  let deltas = 0;
  let armAt = 15;
  let injections = 0;
  let maxInjections = 1;
  let chatModelStarts = 0;
  let llmEnds = 0;
  let toolExecutions = 0;
  let haltOnBoundary = false;

  const hooks = new HookRegistry();
  hooks.register('PreemptBoundary', {
    hooks: [
      async () => {
        armed = false;
        if (haltOnBoundary) {
          return { preventContinuation: true, stopReason: 'scenario_halt' };
        }
        if (injections >= maxInjections) return {};
        injections += 1;
        if (scenario === 'multiseal' && injections < maxInjections + 1) {
          /** Re-arm after the resumed stream produces more deltas. */
          armAt = deltas + 12;
        }
        return {
          injectedMessages: [
            {
              role: 'user' as const,
              content:
                injections === 1 && maxInjections > 1
                  ? 'Change of plan: write 300 words about the fall of Constantinople specifically.'
                  : 'Stop. In one short sentence: what year did Constantinople fall?',
              source: 'steer',
            },
          ],
        };
      },
    ],
  });

  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (): void => {
        deltas += 1;
        /** Never re-arm past the scenario's injection budget: a real host
         *  disarms when its queue drains; this models that. */
        if (deltas >= armAt && injections < maxInjections) armed = true;
      },
    },
  };

  const calculator = tool(
    async () => {
      toolExecutions += 1;
      return '345';
    },
    {
      name: 'calculator',
      description: 'Evaluates a basic arithmetic expression.',
      schema: z.object({ expression: z.string() }),
    }
  );

  const streamConfig = {
    runId: uuidv4(),
    configurable: { user_id: 'probe-user', thread_id: `scenario-${scenario}` },
    streamMode: 'values',
    version: 'v2' as const,
    callbacks: [
      {
        handleChatModelStart: (): void => {
          chatModelStarts += 1;
        },
        handleLLMEnd: (): void => {
          llmEnds += 1;
        },
      },
    ],
  };

  const base = {
    runId: uuidv4(),
    tokenCounter,
    customHandlers,
    hooks,
    returnContent: true,
    skipCleanup: true,
  };

  let run: Run<t.IState>;
  let prompt = LONG_PROMPT;

  switch (scenario) {
    case 'multiseal':
      maxInjections = 2;
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: { type: 'standard', llmConfig, instructions: 'You are a history assistant.' },
        preemption: { shouldPreempt: () => armed, maxSeals: 3 },
      });
      break;
    case 'toolturn':
      armAt = 0;
      armed = true;
      prompt =
        'Use the calculator tool to compute 15 * 23, then explain the historical significance of the number 345 in Roman history in at least 300 words.';
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [calculator],
          instructions:
            'Call the calculator immediately, before writing ANY text. After the tool result, write the explanation.',
        },
        preemption: { shouldPreempt: () => armed, maxSeals: 2 },
      });
      break;
    case 'budget':
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: { type: 'standard', llmConfig, instructions: 'You are a history assistant.' },
        preemption: {
          shouldPreempt: () => {
            /** Deliberately NEVER disarmed by the host: level-triggered forever. */
            return deltas >= 15;
          },
          maxSeals: 1,
        },
      });
      break;
    case 'cache':
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as typeof llmConfig,
          instructions:
            'You are a history assistant. '.repeat(80) +
            'Answer at length in flowing prose.',
        },
        preemption: { shouldPreempt: () => armed, maxSeals: 2 },
      });
      break;
    case 'ma-halt':
    case 'ma-continue':
    case 'ma-baseline':
      haltOnBoundary = scenario === 'ma-halt';
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: {
          type: 'multi-agent',
          agents: (() => {
            const { provider, ...clientOptions } = llmConfig as {
              provider: t.AgentInputs['provider'];
            } & Record<string, unknown>;
            return [
              {
                agentId: 'writer',
                provider,
                clientOptions,
                instructions: 'You are the writer. Write the requested essay.',
              },
              {
                agentId: 'critic',
                provider,
                clientOptions,
                instructions:
                  'You are the critic. In two sentences, critique what the writer produced.',
              },
            ] as t.AgentInputs[];
          })(),
          edges: [{ from: 'writer', to: 'critic', edgeType: 'direct' }],
        },
        ...(scenario === 'ma-baseline'
          ? {}
          : { preemption: { shouldPreempt: () => armed, maxSeals: 1 } }),
      });
      break;
    case 'websearch':
      armAt = Number(argOf('--arm-at', '1'));
      prompt =
        'Search the web for the current population of Istanbul, then write a 300-word reflection on the city\'s growth.';
      run = await Run.create<t.IState>({
        ...base,
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [
            { type: 'web_search_20250305', name: 'web_search', max_uses: 1 },
          ] as unknown as t.GraphTools,
          instructions: 'Use web search before answering.',
        },
        preemption: { shouldPreempt: () => armed, maxSeals: 2 },
      });
      break;
    default:
      throw new Error(`Unknown scenario: ${scenario}`);
  }

  await run.processStream({ messages: [new HumanMessage(prompt)] }, streamConfig);

  const messages = run.getRunMessages() ?? [];
  const sequence = messages.map(kindOf);
  const stats = run.getPreemptStats();
  const steerCount = sequence.filter((k) => k === 'human(steer)').length;
  const finalText = textOf(messages[messages.length - 1]);
  const detail: Record<string, unknown> = {
    sequence,
    seals: stats.seals,
    emptyBoundaries: stats.emptyBoundaries,
    steerCount,
    chatModelStarts,
    llmEnds,
    runsLeftOpen: chatModelStarts - llmEnds,
    toolExecutions,
    haltReason: run.getHaltReason() ?? null,
    finalChars: finalText.length,
    sealedBlockTypes: Array.isArray(messages[0]?.content)
      ? (messages[0].content as Array<{ type?: string }>).map((b) => b.type)
      : ['string'],
    lastTexts: messages.slice(-3).map((m) => ({
      kind: kindOf(m),
      agent: (m as { name?: string }).name ?? m.additional_kwargs.agentId ?? null,
      chars: textOf(m).length,
      head: textOf(m).slice(0, 80),
    })),
  };

  let ok = false;
  switch (scenario) {
    case 'multiseal':
      ok =
        stats.seals === 2 &&
        steerCount === 2 &&
        chatModelStarts === 3 &&
        llmEnds === 3 &&
        /1453/.test(finalText);
      break;
    case 'toolturn': {
      const toolIdx = sequence.indexOf('tool');
      const steerIdx = sequence.indexOf('human(steer)');
      ok =
        stats.seals === 1 &&
        toolExecutions === 1 &&
        toolIdx !== -1 &&
        steerIdx > toolIdx &&
        chatModelStarts === llmEnds &&
        finalText.trim().length > 0;
      break;
    }
    case 'budget':
      ok =
        stats.seals === 1 &&
        steerCount === 1 &&
        chatModelStarts === 2 &&
        llmEnds === 2 &&
        /1453/.test(finalText);
      break;
    case 'cache':
      ok =
        stats.seals === 1 &&
        steerCount === 1 &&
        chatModelStarts === llmEnds &&
        finalText.trim().length > 0;
      break;
    case 'ma-halt':
      ok =
        stats.seals === 1 &&
        run.getHaltReason() === 'scenario_halt' &&
        chatModelStarts === 1 &&
        llmEnds === 1;
      break;
    case 'ma-baseline':
      ok = chatModelStarts === 2 && llmEnds === 2;
      break;
    case 'ma-continue':
      ok =
        stats.seals === 1 &&
        steerCount === 1 &&
        chatModelStarts === 3 &&
        llmEnds === 3 &&
        finalText.trim().length > 0;
      break;
    case 'websearch':
      /**
       * Best-effort: the load-bearing claims are (a) no seal orphans an
       * unanswered server-tool call — the provider would 400 the resume —
       * and (b) the run completes with balanced lifecycle counters.
       */
      ok =
        chatModelStarts === llmEnds &&
        finalText.trim().length > 0 &&
        stats.seals <= 2;
      break;
  }

  return { scenario, provider: providerKey, ok, detail, error: null };
}

main()
  .then((outcome) => {
    console.log(`SCENARIO_VERDICT ${JSON.stringify(outcome)}`);
    process.exit(outcome.ok ? 0 : 1);
  })
  .catch((error: unknown) => {
    console.log(
      `SCENARIO_VERDICT ${JSON.stringify({
        scenario: argOf('--scenario', '?'),
        provider: argOf('--provider', '?'),
        ok: false,
        error: error instanceof Error ? error.message.slice(0, 400) : String(error),
      })}`
    );
    process.exit(1);
  });
