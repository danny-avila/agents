// src/scripts/preempt-probe.ts
//
// Live-provider probe for cooperative mid-generation preemption.
//
//   npx tsx --env-file=.env src/scripts/preempt-probe.ts --provider anthropic
//
// (tsx rather than the repo's node-loader `script` runner: the import chain
// reaches @mistralai/mistralai, which ships ESM-only and defeats the loader.
// Keep live credentials OUT of the worktree's .env — jest loads it — and
// point --env-file at wherever they actually live.)
//
// Asks for a long, tool-free answer, arms `shouldPreempt` once text has
// started streaming, and asserts the full seal -> PreemptBoundary -> inject ->
// self-loop path against a real provider: the stream stops mid-answer, the
// injected user turn lands between two assistant turns, generation resumes in
// the same run, and CHAT_MODEL_END fires for the sealed turn so usage is still
// recorded. Prints a JSON verdict block as its last line.
import { config } from 'dotenv';
config();
import { v4 as uuidv4 } from 'uuid';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { createTokenCounter } from '@/utils/tokens';
import { getLLMConfig } from '@/utils/llmConfig';
import { HookRegistry } from '@/hooks';
import { Run } from '@/run';

/** Deltas to let through before arming the seal. */
const ARM_AFTER_DELTAS = 15;

const PROMPT =
  'Write a detailed history of the Byzantine Empire from Constantine to ' +
  '1453. Cover the major emperors, the religious schisms, and the military ' +
  'campaigns. Aim for at least 800 words of flowing prose. Do not use ' +
  'headings or bullet points.';

const STEER =
  'Stop. Forget the essay. Answer only this, in one short sentence: what ' +
  'year did Constantinople fall?';

type Verdict = {
  provider: string;
  model: string;
  ok: boolean;
  sealed: boolean;
  seals: number;
  emptyBoundaries: number;
  boundaryFired: boolean;
  boundarySealCount: number | null;
  modelEndEvents: number;
  usageEventsWithTokens: number;
  chatModelStarts: number;
  llmEnds: number;
  llmErrors: number;
  runsLeftOpen: number;
  sealedTextChars: number;
  resumedTextChars: number;
  sequence: string[];
  injectedFound: boolean;
  resumeAnswersSteer: boolean;
  error: string | null;
};

/**
 * Streamed assistant turns arrive as `AIMessageChunk`, which extends
 * `BaseMessageChunk` rather than `AIMessage` — `instanceof AIMessage` is
 * false for them. Type discrimination has to go through `getType()`.
 */
function isAssistant(message: BaseMessage | undefined): boolean {
  return message?.getType() === 'ai';
}

function messageKind(message: BaseMessage): string {
  if (message.getType() === 'human') {
    const source = message.additional_kwargs.source;
    return typeof source === 'string' ? `human(${source})` : 'human';
  }
  return message.getType();
}

function textOf(message: BaseMessage | undefined): string {
  if (message == null) {
    return '';
  }
  if (typeof message.content === 'string') {
    return message.content;
  }
  let text = '';
  for (const block of message.content) {
    if (block.type === 'text' && typeof block.text === 'string') {
      text += block.text;
    }
  }
  return text;
}

function parseProvider(): string {
  const index = process.argv.indexOf('--provider');
  if (index !== -1 && process.argv[index + 1] != null) {
    return process.argv[index + 1];
  }
  return Providers.ANTHROPIC;
}

/** Control mode: identical run with no `preemption` config at all. */
function isControl(): boolean {
  return process.argv.includes('--control');
}

async function probe(): Promise<Verdict> {
  const providerKey = parseProvider();
  const llmConfig = getLLMConfig(providerKey);
  if (llmConfig == null) {
    throw new Error(`No llmConfig entry for provider "${providerKey}"`);
  }

  let armed = false;
  let armedOnce = false;
  let deltaCount = 0;
  let modelEndEvents = 0;
  let usageEventsWithTokens = 0;
  let chatModelStarts = 0;
  let llmEnds = 0;
  let llmErrors = 0;
  let boundaryFired = false;
  let boundarySealCount: number | null = null;
  let injectedOnce = false;
  const collectedUsage: UsageMetadata[] = [];

  /**
   * Deliberately NOT a `ChatModelStreamHandler` instance: that would make
   * `getRegisteredDefaultChatStreamHandler` return a handler and route the run
   * down the registered-handler loop, which never seals. Plain-object handlers
   * mirror how LibreChat registers, which is the sealable path.
   */
  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (): void => {
        deltaCount += 1;
        if (!armedOnce && deltaCount >= ARM_AFTER_DELTAS) {
          armed = true;
          armedOnce = true;
        }
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event: string, data: t.ModelEndData): void => {
        modelEndEvents += 1;
        const usage = data?.output?.usage_metadata;
        if (usage != null) {
          collectedUsage.push(usage);
          if ((usage.output_tokens ?? 0) > 0) {
            usageEventsWithTokens += 1;
          }
        }
      },
    },
  };

  const hooks = new HookRegistry();
  hooks.register('PreemptBoundary', {
    hooks: [
      async (input) => {
        boundaryFired = true;
        boundarySealCount = input.sealCount;
        /**
         * Disarm on drain. The SDK polls `shouldPreempt` on every chunk and
         * does not clear it for you — a host that leaves the request set
         * armed re-seals the resumed turn immediately. LibreChat's real drain
         * does this via `noteSteersRemoved`.
         */
        armed = false;
        if (injectedOnce) {
          return {};
        }
        injectedOnce = true;
        return {
          injectedMessages: [
            { role: 'user', content: STEER, source: 'steer' },
          ],
        };
      },
    ],
  });

  /**
   * A real host configures this. It is what `dispatchSealedModelEnd` falls
   * back to when the provider never got to send its usage chunk, so running
   * without one hides whether sealed-turn usage is recoverable. Pass
   * `--no-token-counter` to probe the unconfigured case deliberately.
   */
  const tokenCounter = process.argv.includes('--no-token-counter')
    ? undefined
    : await createTokenCounter();

  const run = await Run.create<t.IState>({
    runId: uuidv4(),
    graphConfig: {
      type: 'standard',
      llmConfig,
      instructions: 'You are a knowledgeable history assistant.',
    },
    tokenCounter,
    customHandlers,
    hooks,
    ...(isControl()
      ? {}
      : {
        preemption: {
          shouldPreempt: (): boolean => armed,
          maxSeals: 2,
        },
      }),
    returnContent: true,
    skipCleanup: true,
  });

  /**
   * Counts the RAW LangChain model-run lifecycle, which is what a tracer
   * (LangSmith, Langfuse) sees. A seal that fails to close its run shows up
   * here as starts > ends — an span that would hang open forever.
   */
  const streamConfig = {
    runId: uuidv4(),
    configurable: { user_id: 'probe-user', thread_id: 'preempt-probe' },
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
        handleLLMError: (): void => {
          llmErrors += 1;
        },
      },
    ],
  };

  await run.processStream(
    { messages: [new HumanMessage(PROMPT)] },
    streamConfig
  );

  const messages = run.getRunMessages() ?? [];
  const sequence = messages.map(messageKind);
  const injectedIndex = messages.findIndex(
    (message) =>
      message instanceof HumanMessage &&
      message.additional_kwargs.source === 'steer'
  );
  const stats = run.getPreemptStats();
  const sealedText = textOf(messages[injectedIndex - 1]);
  const resumedText = textOf(messages[injectedIndex + 1]);
  const resumeAnswersSteer = /1453/.test(resumedText);

  /**
   * Control mode is the no-preemption baseline: one uninterrupted model run,
   * nothing sealed, nothing injected. Its value is the contrast — the same
   * lifecycle counters must balance WITHOUT the seal machinery — so it gets
   * its own criteria instead of failing the seal-path ones by construction.
   */
  const ok = isControl()
    ? stats.seals === 0 &&
      !boundaryFired &&
      injectedIndex === -1 &&
      messages.length > 0 &&
      textOf(messages[messages.length - 1]).trim().length > 0 &&
      modelEndEvents >= 1 &&
      usageEventsWithTokens >= 1 &&
      llmErrors === 0 &&
      chatModelStarts === llmEnds
    : stats.seals === 1 &&
      stats.emptyBoundaries === 0 &&
      boundaryFired &&
      injectedIndex > 0 &&
      isAssistant(messages[injectedIndex - 1]) &&
      isAssistant(messages[injectedIndex + 1]) &&
      sealedText.trim().length > 0 &&
      resumedText.trim().length > 0 &&
      modelEndEvents >= 2 &&
      usageEventsWithTokens >= 2 &&
      chatModelStarts === llmEnds;

  return {
    provider: providerKey,
    model: String((llmConfig as { model?: unknown }).model ?? 'unknown'),
    ok,
    sealed: stats.seals > 0,
    seals: stats.seals,
    emptyBoundaries: stats.emptyBoundaries,
    boundaryFired,
    boundarySealCount,
    modelEndEvents,
    usageEventsWithTokens,
    chatModelStarts,
    llmEnds,
    llmErrors,
    runsLeftOpen: chatModelStarts - llmEnds,
    sealedTextChars: sealedText.length,
    resumedTextChars: resumedText.length,
    sequence,
    injectedFound: injectedIndex >= 0,
    resumeAnswersSteer,
    error: null,
  };
}

probe()
  .then((verdict) => {
    console.log('\n===== SEALED TAIL / RESUMED HEAD =====');
    console.log(JSON.stringify(verdict, null, 2));
    console.log(`\nVERDICT_JSON ${JSON.stringify(verdict)}`);
    process.exit(verdict.ok ? 0 : 1);
  })
  .catch((error: unknown) => {
    const verdict: Partial<Verdict> = {
      provider: parseProvider(),
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
    console.error(error);
    console.log(`\nVERDICT_JSON ${JSON.stringify(verdict)}`);
    process.exit(1);
  });
