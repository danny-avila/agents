import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type { UsageMetadata, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { StreamLimitState } from '@/llm/streamLimits';
import type { AgentContext } from '@/agents/AgentContext';
import type { HookRegistry } from '@/hooks';
import type { OnChunk } from '@/llm/invoke';
import type * as t from '@/types';
import {
  cloneToolMessageWithContent,
  compactToolContent,
  isComputerCallOutputMessage,
  serializeToolContentBounded,
} from '@/utils/toolContent';
import {
  addTailCacheControl,
  resolvePromptCacheTtl,
  type PromptCacheTtl,
} from '@/messages/cache';
import {
  DEFAULT_RETAIN_RECENT_TURNS,
  splitAtRecencyBoundary,
} from '@/messages/recency';
import {
  Constants,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
} from '@/common';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import {
  enforceStreamLimitsForWireChunk,
  StreamLimitExceededError,
  STREAM_LIMIT_EPOCH_KEY,
} from '@/llm/streamLimits';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { calculateMaxToolResultChars } from '@/utils/truncation';
import { createRemoveAllMessage } from '@/messages/reducer';
import { getMaxOutputTokensKey } from '@/llm/request';
import { initializeModel } from '@/llm/init';
import { getChunkContent } from '@/stream';
import { executeHooks } from '@/hooks';

const SUMMARIZATION_PARAM_KEYS = new Set(['maxSummaryTokens']);

/**
 * Default number of recent user-led turns preserved verbatim during
 * compaction.  A turn begins at a HumanMessage and includes every
 * following AIMessage and ToolMessage up to the next HumanMessage.
 * The most recent turn is always retained regardless of this value;
 * the default of `2` additionally keeps the prior exchange so the
 * model has fresh context on what just happened.  Setting
 * `retainRecent.turns` to `0` reverts to the legacy "summarize every
 * message" behavior.
 */
/**
 * Token overhead of the XML wrapper + instruction text added around the
 * summary at injection time in AgentContext.buildSystemRunnable:
 * `<summary>\n${text}\n</summary>\n\nYour context window was compacted...`
 * ~33 tokens on Anthropic, ~24-27 on OpenAI.  Using 33 as a safe ceiling.
 */
const SUMMARY_WRAPPER_OVERHEAD_TOKENS = 33;

/** Structured checkpoint prompt for fresh summarization (no prior summary). */
export const DEFAULT_SUMMARIZATION_PROMPT = `Hold on, before you continue I need you to write me a checkpoint of everything so far. Your context window is filling up and this checkpoint replaces the messages above, so capture everything you need to pick right back up.

Don't second-guess or fact-check anything you did, your tool results reflect exactly what happened. If a tool result appears truncated, that's just a display artifact from context management: the tool executed fully. Just record what you did and what you observed. Only the checkpoint, don't respond to me or continue the conversation.

## Checkpoint

## Goal
What I asked you to do and any sub-goals you identified.

## Constraints & Preferences
Any rules, preferences, or configuration I established.

## Progress
### Done
- What you completed and the outcomes

### In Progress
- What you're currently working on

## Key Decisions
Decisions you made and why.

## Next Steps
Concrete task actions remaining, in priority order.

## Critical Context
Exact identifiers, names, error messages, URLs, and details you need to preserve verbatim.

Rules:
- Record what you did and observed, don't judge or re-evaluate it
- For each tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Short declarative sentences
- Skip empty sections`;

/** Prompt for re-compaction when a prior summary exists. */
export const DEFAULT_UPDATE_SUMMARIZATION_PROMPT = `Hold on again, update your checkpoint. Merge the new messages into your existing checkpoint and give me a single consolidated replacement.

Keep it roughly the same length as your last checkpoint. Compress older details to make room for what's new, don't just append. Give recent actions more detail, compress older items to one-liners.

Don't fact-check or second-guess anything, your tool results are ground truth. If a tool result appears truncated, that's just a display artifact: the tool executed fully. Only the checkpoint, don't respond to me or continue the conversation.

Rules:
- Merge new progress into existing sections, don't duplicate headers
- Compress older completed items into one-line entries
- Move items from "In Progress" to "Done" when you completed them
- Update "Next Steps" to reflect current task priorities.
- For each new tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Skip empty sections`;

function separateParameters(parameters: Record<string, unknown>): {
  llmParams: Record<string, unknown>;
  maxSummaryTokens?: number;
} {
  const llmParams: Record<string, unknown> = {};
  let maxSummaryTokens: number | undefined;

  for (const [key, value] of Object.entries(parameters)) {
    if (SUMMARIZATION_PARAM_KEYS.has(key)) {
      if (
        key === 'maxSummaryTokens' &&
        typeof value === 'number' &&
        value > 0
      ) {
        maxSummaryTokens = value;
      }
    } else {
      llmParams[key] = value;
    }
  }

  return { llmParams, maxSummaryTokens };
}

/**
 * Generates a structural metadata summary without making an LLM call.
 * Used as a last-resort fallback when all summarization attempts fail.
 * Preserves tool names and message counts so the agent retains basic context.
 */
function generateMetadataStub(messages: BaseMessage[]): string {
  const counts: Record<string, number> = {};
  const toolNames = new Set<string>();

  for (const msg of messages) {
    const role = msg.getType();
    counts[role] = (counts[role] ?? 0) + 1;

    if (role === 'tool' && msg.name != null && msg.name !== '') {
      toolNames.add(msg.name);
    }

    if (
      role === 'ai' &&
      msg instanceof AIMessage &&
      msg.tool_calls &&
      msg.tool_calls.length > 0
    ) {
      for (const tc of msg.tool_calls) {
        toolNames.add(tc.name);
      }
    }
  }

  const countParts = Object.entries(counts)
    .map(([role, count]) => `${count} ${role}`)
    .join(', ');

  const lines = [
    `[Metadata summary: ${messages.length} messages (${countParts})]`,
  ];

  if (toolNames.size > 0) {
    lines.push(`[Tools used: ${Array.from(toolNames).join(', ')}]`);
  }

  return lines.join('\n');
}

/** Maximum number of tool failures to include in the enrichment section. */
const MAX_TOOL_FAILURES = 8;
/** Maximum chars per failure summary line. */
const MAX_TOOL_FAILURE_CHARS = 240;

/**
 * Extracts failed tool results from messages and formats them as a structured
 * section. LLMs often omit specific failure details (exit codes, error messages)
 * from their summaries, this mechanical enrichment guarantees they survive.
 */
function extractToolFailuresSection(messages: BaseMessage[]): string {
  const failures: Array<{ toolName: string; summary: string }> = [];
  const seen = new Set<string>();

  for (const msg of messages) {
    if (msg.getType() !== 'tool') {
      continue;
    }
    const toolMsg = msg as ToolMessage;
    if (toolMsg.status !== 'error') {
      continue;
    }
    // Deduplicate by tool_call_id
    const callId = toolMsg.tool_call_id;
    if (callId && seen.has(callId)) {
      continue;
    }
    if (callId) {
      seen.add(callId);
    }

    const toolName = toolMsg.name ?? 'tool';
    const content = serializeToolContentBounded(
      toolMsg.content,
      MAX_TOOL_FAILURE_CHARS * 4
    );
    const normalized = content.replace(/\s+/g, ' ').trim();
    const summary =
      normalized.length > MAX_TOOL_FAILURE_CHARS
        ? `${normalized.slice(0, MAX_TOOL_FAILURE_CHARS - 3)}...`
        : normalized;

    failures.push({ toolName, summary });
  }

  if (failures.length === 0) {
    return '';
  }

  const lines = failures
    .slice(0, MAX_TOOL_FAILURES)
    .map((f) => `- ${f.toolName}: ${f.summary}`);
  if (failures.length > MAX_TOOL_FAILURES) {
    lines.push(`- ...and ${failures.length - MAX_TOOL_FAILURES} more`);
  }

  return `\n\n## Tool Failures\n${lines.join('\n')}`;
}

/**
 * Appends mechanical enrichment sections to an LLM-generated summary.
 * Tool failures are appended verbatim because LLMs often omit specific
 * error details from their summaries.
 */
function enrichSummary(summaryText: string, messages: BaseMessage[]): string {
  return summaryText + extractToolFailuresSection(messages);
}

/**
 * Restores pre-masking tool content onto the messages array using
 * `pendingOriginalToolContent` stored on AgentContext. Only allocates
 * a new array when there are entries to restore; otherwise returns the
 * input reference unchanged.
 */
function restoreOriginalToolContent(
  messages: BaseMessage[],
  originalToolContent: Map<number, string> | undefined,
  maxContextTokens?: number
): BaseMessage[] {
  if (originalToolContent == null || originalToolContent.size === 0) {
    return messages;
  }

  const restorable: Array<{
    index: number;
    message: ToolMessage;
    content: string;
  }> = [];
  for (const [index, content] of originalToolContent) {
    const message = messages[index];
    if (
      message instanceof ToolMessage &&
      !isComputerCallOutputMessage(message)
    ) {
      restorable.push({ index, message, content });
    }
  }
  if (restorable.length === 0) {
    return messages;
  }

  /**
   * Restored originals improve checkpoint quality, but they still feed a
   * provider call. Share one tool-result budget across every restoration so
   * several previously masked results cannot overflow the summarizer.
   */
  let remainingChars = calculateMaxToolResultChars(maxContextTokens);
  const restored = [...messages];
  for (let i = 0; i < restorable.length; i++) {
    const { index, message, content } = restorable[i];
    const maxChars = Math.floor(remainingChars / (restorable.length - i));
    const compacted = compactToolContent(content, maxChars).content;
    restored[index] = cloneToolMessageWithContent(message, compacted);
    remainingChars -= serializeToolContentBounded(compacted, maxChars).length;
  }
  return restored;
}

// ---------------------------------------------------------------------------
// Extracted helpers for createSummarizeNode
// ---------------------------------------------------------------------------

interface SummarizationClientConfig {
  provider: string;
  modelName?: string;
  clientOptions: Record<string, unknown>;
  effectiveMaxSummaryTokens?: number;
  promptText: string;
  updatePromptText: string;
}

/** Assembles the summarization model's client options from agent and config. */
function buildSummarizationClientConfig(
  agentContext: AgentContext,
  summarizationConfig?: t.SummarizationConfig
): SummarizationClientConfig {
  const provider = (summarizationConfig?.provider ??
    agentContext.provider) as string;
  const modelName = summarizationConfig?.model;
  const parameters = summarizationConfig?.parameters ?? {};
  const promptText =
    summarizationConfig?.prompt ?? DEFAULT_SUMMARIZATION_PROMPT;
  const updatePromptText =
    summarizationConfig?.updatePrompt ?? DEFAULT_UPDATE_SUMMARIZATION_PROMPT;

  const { llmParams, maxSummaryTokens: paramMaxSummaryTokens } =
    separateParameters(parameters);

  const isSelfSummarize = provider === (agentContext.provider as string);
  const baseOptions =
    isSelfSummarize && agentContext.clientOptions
      ? { ...agentContext.clientOptions }
      : {};

  const clientOptions: Record<string, unknown> = {
    ...baseOptions,
    ...llmParams,
  };

  if (modelName != null && modelName !== '') {
    clientOptions.model = modelName;
    clientOptions.modelName = modelName;
  }

  const effectiveMaxSummaryTokens =
    paramMaxSummaryTokens ?? summarizationConfig?.maxSummaryTokens;

  if (effectiveMaxSummaryTokens != null) {
    clientOptions[getMaxOutputTokensKey(provider)] = effectiveMaxSummaryTokens;
  }

  return {
    provider,
    modelName,
    clientOptions,
    effectiveMaxSummaryTokens,
    promptText,
    updatePromptText,
  };
}

/** Computes the token count for a summary, preferring provider output tokens when available. */
function computeSummaryTokenCount(
  summaryText: string,
  summaryUsage: Partial<UsageMetadata> | undefined,
  tokenCounter?: (message: BaseMessage) => number
): number {
  const providerOutputTokens = Number(summaryUsage?.output_tokens) || 0;
  if (providerOutputTokens > 0) {
    return providerOutputTokens + SUMMARY_WRAPPER_OVERHEAD_TOKENS;
  }
  if (tokenCounter) {
    return (
      tokenCounter(new SystemMessage(summaryText)) +
      SUMMARY_WRAPPER_OVERHEAD_TOKENS
    );
  }
  return 0;
}

/**
 * Names the first retained message so the summary declares its own extent
 * rather than leaving the next run to infer coverage from where the block
 * happens to sit.
 *
 * Anchored to the retained side, not the covered side. One source message can
 * expand into several messages — a steer splits an assistant entry into
 * pre-steer, steer, and post-steer entries sharing its ID — and the recency
 * split lands on any human-type message, including the steer. Naming the last
 * *covered* message would then name a half-covered ID with no correct reading;
 * naming the first *retained* message makes that same message the anchor, so it
 * survives whole and everything before it is unambiguously covered.
 *
 * Synthetic entries are skipped, because they can sit *before* a resolvable
 * one. `formatAgentMessages` reconstructs skill bodies inside its payload loop
 * (see the `pendingSkillNames` block) and keeps processing payload entries
 * afterwards, so an unstamped skill body — which `messagesStateReducer` then
 * gives a UUID no payload entry carries — is followed by stamped messages.
 * Anchoring on the UUID would look resolvable at write time and degrade to
 * positional trimming on read, dropping the retained tail. Skipping it reaches
 * the stamped message behind it.
 *
 * `convertInjectedMessages` records `injected` on everything it builds, which is
 * what makes this decidable: `isMeta` and `source` are both optional on
 * `InjectedMessage`, so a bare entry carries no marker of its own, and an
 * injected `source: 'steer'` is otherwise indistinguishable from a replayed one.
 * The remaining `isMeta`/`source` checks cover the constructors that build
 * synthetic entries directly instead of going through that funnel — hook context
 * in `ToolNode` and `StandardGraph`, handoff cues, reconstructed skill bodies.
 *
 * `steer` is exempt from the `source` check because a replayed steer *is*
 * stamped from its payload entry; rejecting every marked `source` once dropped
 * exactly those retained steers. Injected steers are still caught, by `injected`.
 *
 * Known limitation: a payload entry that omits `messageId` is never stamped, so
 * the reducer's UUID is recorded and cannot resolve on the next run. There is no
 * write-time fix — such an entry has no stable ID to name in the next payload
 * either — and the reader's positional fallback is what `main` already does, so
 * the anchor degrades rather than misleads.
 */
function isSyntheticContext(message: BaseMessage): boolean {
  const { additional_kwargs: kwargs } = message;
  if (kwargs.injected === true || kwargs.isMeta === true) {
    return true;
  }
  return kwargs.source != null && kwargs.source !== 'steer';
}

function resolveSummaryCoverage(
  messagesToRetain: BaseMessage[]
): t.SummaryCoverage | undefined {
  for (let i = 0; i < messagesToRetain.length; i++) {
    const message = messagesToRetain[i];
    if (isSyntheticContext(message)) {
      continue;
    }
    const id = message.id?.trim();
    if (id != null && id !== '') {
      return { retainedFromMessageId: id };
    }
  }
  return undefined;
}

/** Constructs the SummaryContentBlock persisted in the run step and dispatched to events. */
function buildSummaryBlock(params: {
  summaryText: string;
  tokenCount: number;
  coverage?: t.SummaryCoverage;
  stepId: string;
  stepIndex: number;
  modelName?: string;
  provider: string;
  summaryVersion: number;
}): t.SummaryContentBlock {
  return {
    type: ContentTypes.SUMMARY,
    content: [
      {
        type: ContentTypes.TEXT,
        text: params.summaryText,
      } as t.MessageContentComplex,
    ],
    tokenCount: params.tokenCount,
    ...(params.coverage != null ? { coverage: params.coverage } : {}),
    summaryVersion: params.summaryVersion,
    boundary: {
      messageId: params.stepId,
      contentIndex: params.stepIndex,
    },
    model: params.modelName,
    provider: params.provider,
    createdAt: new Date().toISOString(),
  };
}

type LogFn = (
  level: 'debug' | 'info' | 'warn' | 'error',
  message: string,
  data?: Record<string, unknown>
) => void;

/**
 * Extracts an HTTP status code from a thrown LLM-provider error. Returns
 * `undefined` for non-object values (including `null` or `undefined`, both
 * valid `throw` targets in JS) so callers never dereference a nullish
 * value.
 */
function extractHttpStatus(err: unknown): number | undefined {
  if (err == null || typeof err !== 'object') {
    return undefined;
  }
  const errRecord = err as Record<string, unknown>;
  const direct = errRecord.status;
  if (typeof direct === 'number') {
    return direct;
  }
  const statusCode = errRecord.statusCode;
  if (typeof statusCode === 'number') {
    return statusCode;
  }
  const response = errRecord.response;
  if (response != null && typeof response === 'object') {
    const nested = (response as Record<string, unknown>).status;
    if (typeof nested === 'number') {
      return nested;
    }
  }
  return undefined;
}

/**
 * Formats a provider-level error for logging. Returns both a human-readable
 * suffix (safe to include in the message string so it survives any host-side
 * formatter) and a structured metadata bag for rich log backends.
 */
function describeProviderError(
  err: unknown,
  provider: string,
  modelName?: string
): { suffix: string; data: Record<string, unknown> } {
  const providerLabel = `${provider}/${modelName ?? '(no-model)'}`;
  const errMsg = err instanceof Error ? err.message : String(err);

  const data: Record<string, unknown> = {
    provider,
    model: modelName,
  };
  if (err instanceof Error) {
    data.errorName = err.name;
    data.errorStack = err.stack;
  }

  const status = extractHttpStatus(err);
  const statusSuffix = status != null ? ` (HTTP ${status})` : '';
  if (status != null) {
    data.status = status;
  }

  return {
    suffix: `[${providerLabel}]${statusSuffix}: ${errMsg}`,
    data,
  };
}

/**
 * Formats an exhausted-fallback error. `tryFallbackProviders` throws the
 * last fallback provider's error, which may be from any of the configured
 * fallbacks — not the primary — so we label the log with the list of
 * fallback providers attempted rather than mis-attributing to the primary.
 *
 * Entries in `fallbacks` are normally strongly typed, but we defend against
 * malformed runtime config (null/undefined entries, missing `provider`
 * field) so a recoverable summarization failure is never promoted to an
 * uncaught exception from inside the logging path.
 */
function describeFallbackError(
  err: unknown,
  fallbacks: unknown
): { suffix: string; data: Record<string, unknown> } {
  const errMsg = err instanceof Error ? err.message : String(err);
  const list: ReadonlyArray<unknown> = Array.isArray(fallbacks)
    ? fallbacks
    : [];
  const providerNames = list
    .map((f) => {
      if (f == null || typeof f !== 'object') {
        return undefined;
      }
      const raw = (f as { provider?: unknown }).provider;
      return raw != null ? String(raw) : undefined;
    })
    .filter((p): p is string => typeof p === 'string');
  const label =
    providerNames.length > 0
      ? `fallbacks=[${providerNames.join(',')}]`
      : 'no-fallbacks';

  const data: Record<string, unknown> = {
    fallbackProviders: providerNames,
    fallbackCount: list.length,
  };
  if (err instanceof Error) {
    data.errorName = err.name;
    data.errorStack = err.stack;
  }
  const status = extractHttpStatus(err);
  const statusSuffix = status != null ? ` (HTTP ${status})` : '';
  if (status != null) {
    data.status = status;
  }

  return {
    suffix: `[${label}]${statusSuffix}: ${errMsg}`,
    data,
  };
}

/**
 * Runs the summarization LLM call with primary + fallback providers,
 * falling back to a metadata stub when all calls fail.
 */
async function executeSummarizationWithFallback(params: {
  agentContext: AgentContext;
  messages: BaseMessage[];
  clientConfig: SummarizationClientConfig;
  summarizeConfig?: RunnableConfig;
  stepId: string;
  usePromptCache: boolean;
  log: LogFn;
  /** Carries the run's stream limits so the event cap covers summary streams. */
  graph?: StreamLimitState & { getBreakerSignal?: () => AbortSignal };
}): Promise<{
  text: string;
  usage?: Partial<UsageMetadata>;
  /**
   * True when every model call failed and `text` is the generated metadata
   * stub rather than a real summary. Callers that would replace history with
   * this need to know it carries none of the original content.
   */
  usedMetadataStub?: boolean;
}> {
  const {
    agentContext,
    messages,
    clientConfig,
    summarizeConfig,
    stepId,
    usePromptCache,
    log,
    graph,
  } = params;

  const priorSummaryText = agentContext.getSummaryText()?.trim() ?? '';

  let summaryText = '';
  let summaryUsage: Partial<UsageMetadata> | undefined;
  let usedMetadataStub = false;

  try {
    /**
     * Initialize inside the try so that a misconfigured provider
     * (e.g. an unrecognized summarization.provider) surfaces through the
     * `log('error', ...)` path below rather than bubbling up silently.
     */
    const summarizationModel = initializeModel({
      provider: clientConfig.provider as Providers,
      clientOptions: clientConfig.clientOptions as t.ClientOptions,
      tools: agentContext.getToolsForBinding(),
    }) as t.ChatModel;

    const result = await summarizeWithCacheHit({
      model: summarizationModel,
      messages,
      promptText: clientConfig.promptText,
      updatePromptText: clientConfig.updatePromptText,
      priorSummaryText,
      config: summarizeConfig,
      stepId,
      provider: clientConfig.provider as Providers,
      reasoningKey: agentContext.reasoningKey,
      graph,
      usePromptCache,
      promptCacheTtl:
        (clientConfig.provider as Providers) === Providers.ANTHROPIC ||
        (clientConfig.provider as Providers) === Providers.OPENROUTER
          ? resolvePromptCacheTtl(
            (
                clientConfig.clientOptions as {
                  promptCacheTtl?: PromptCacheTtl;
                }
            ).promptCacheTtl
          )
          : undefined,
      log,
    });
    summaryText = result.text;
    summaryUsage = result.usage;
  } catch (primaryError) {
    const primaryDescribed = describeProviderError(
      primaryError,
      clientConfig.provider,
      clientConfig.modelName
    );
    log('error', `Summarization LLM call failed ${primaryDescribed.suffix}`, {
      ...primaryDescribed.data,
      messagesToRefineCount: messages.length,
    });

    /**
     * A tripped stream circuit breaker is a safety abort, not a
     * summarization failure to recover from: retrying on fallback providers
     * would spend more provider work after the limit fired, and degrading to
     * the metadata stub would let the run continue (and periodic compaction
     * commit the stub) despite the `StreamLimits` contract that a breach
     * rejects the run. Rethrown so the summarize node fails the run with
     * the actionable limit error, consistent with agent turns and subagents.
     */
    if (primaryError instanceof StreamLimitExceededError) {
      throw primaryError;
    }
    /** A parallel branch's trip aborts the composed summarization signal,
     * and a provider can surface that as a generic abort error; entering
     * fallback recovery or degrading to the metadata stub would continue
     * work after the run-wide safety abort. Checked on BOTH channels: the
     * graph's own breaker, and the composed config signal that carries a
     * parent run's trip into subagent child graphs. */
    {
      const trippedReason = findStreamLimitAbortReason(
        graph?.getBreakerSignal?.(),
        summarizeConfig?.signal
      );
      if (trippedReason != null) {
        throw trippedReason;
      }
    }

    const rawFallbacks = (
      clientConfig.clientOptions as unknown as t.LLMConfig | undefined
    )?.fallbacks;
    const fallbacks = Array.isArray(rawFallbacks) ? rawFallbacks : [];
    if (fallbacks.length > 0) {
      try {
        const onChunk = createSummarizationChunkHandler({
          stepId,
          config: traceConfig(summarizeConfig, 'cache_hit_compaction'),
          provider: clientConfig.provider as Providers,
          reasoningKey: agentContext.reasoningKey,
          graph,
        });
        const fbResult = await tryFallbackProviders({
          fallbacks,
          tools: agentContext.getToolsForBinding(),
          messages: [
            ...messages,
            new HumanMessage(
              buildSummarizationInstruction(
                clientConfig.promptText,
                clientConfig.updatePromptText,
                priorSummaryText
              )
            ),
          ],
          config: withBreakerSignal(
            traceConfig(summarizeConfig, 'cache_hit_compaction'),
            graph
          ),
          primaryError,
          onChunk,
          streamLimitState: graph,
        });
        const fbMsg = fbResult?.messages?.[0];
        if (fbMsg) {
          summaryText = extractResponseText(
            fbMsg as { content: string | object }
          );
        }
      } catch (fbErr) {
        if (fbErr instanceof StreamLimitExceededError) {
          throw fbErr;
        }
        const fbDescribed = describeFallbackError(fbErr, fallbacks);
        log('warn', `Fallback providers also failed ${fbDescribed.suffix}`, {
          ...fbDescribed.data,
        });
      }
    }
    if (!summaryText) {
      log(
        'warn',
        `Summarization failed, falling back to metadata stub ${primaryDescribed.suffix}`,
        {
          ...primaryDescribed.data,
          messagesToRefineCount: messages.length,
        }
      );
      summaryText = generateMetadataStub(messages);
      usedMetadataStub = true;
    }
  }

  return { text: summaryText, usage: summaryUsage, usedMetadataStub };
}

/** Dispatches run step completion, ON_SUMMARIZE_COMPLETE, and rebuilds token map. */
async function dispatchCompletionEvents(params: {
  graph: CreateSummarizeNodeParams['graph'];
  runnableConfig?: RunnableConfig;
  stepId: string;
  summaryBlock: t.SummaryContentBlock;
  agentContext: AgentContext;
  runStep: t.RunStep;
  summaryUsage?: Partial<UsageMetadata>;
  agentId: string;
  /**
   * Number of messages preserved verbatim by the recency window after
   * compaction.  Reported via the PostCompact hook payload so observers
   * (metrics, cleanup) see the true post-compaction message count
   * instead of always-zero.
   */
  messagesAfterCount: number;
}): Promise<void> {
  const {
    graph,
    runnableConfig,
    stepId,
    summaryBlock,
    agentContext,
    runStep,
    summaryUsage,
    agentId,
    messagesAfterCount,
  } = params;

  runStep.summary = summaryBlock;
  if (summaryUsage) {
    runStep.usage = {
      prompt_tokens: Number(summaryUsage.input_tokens) || 0,
      completion_tokens: Number(summaryUsage.output_tokens) || 0,
      total_tokens:
        (Number(summaryUsage.input_tokens) || 0) +
        (Number(summaryUsage.output_tokens) || 0),
    };
  }

  await graph.dispatchRunStepCompleted(
    stepId,
    { type: 'summary', summary: summaryBlock } satisfies t.SummaryCompleted,
    runnableConfig
  );

  if (runnableConfig) {
    await safeDispatchCustomEvent(
      GraphEvents.ON_SUMMARIZE_COMPLETE,
      {
        id: stepId,
        agentId,
        summary: summaryBlock,
      } satisfies t.SummarizeCompleteEvent,
      runnableConfig
    );
  }

  const sessionId = graph.runId ?? '';
  if (graph.hookRegistry?.hasHookFor('PostCompact', sessionId) === true) {
    const threadId = (
      runnableConfig?.configurable as Record<string, unknown> | undefined
    )?.thread_id as string | undefined;
    const firstBlock = summaryBlock.content?.[0];
    const summaryText =
      firstBlock != null &&
      typeof firstBlock === 'object' &&
      'text' in firstBlock &&
      typeof firstBlock.text === 'string'
        ? firstBlock.text
        : '';
    await executeHooks({
      registry: graph.hookRegistry,
      input: {
        hook_event_name: 'PostCompact',
        runId: sessionId,
        threadId,
        agentId,
        summary: summaryText,
        messagesAfterCount,
      },
      sessionId,
    }).catch(() => {
      /* PostCompact is observational — swallow errors */
    });
  }

  agentContext.rebuildTokenMapAfterSummarization({});
}

// ---------------------------------------------------------------------------
// createSummarizeNode
// ---------------------------------------------------------------------------

interface CreateSummarizeNodeParams {
  agentContext: AgentContext;
  graph: {
    contentData: t.RunStep[];
    contentIndexMap: Map<string, number>;
    config?: RunnableConfig;
    runId?: string;
    isMultiAgent: boolean;
    hookRegistry?: HookRegistry;
    dispatchRunStep: (
      runStep: t.RunStep,
      config?: RunnableConfig
    ) => Promise<void>;
    dispatchRunStepCompleted: (
      stepId: string,
      result: t.StepCompleted,
      config?: RunnableConfig
    ) => Promise<void>;
    /** The run's shared breaker signal, composed into every summarization
     * model attempt so a sibling branch tripping a stream limit also
     * cancels in-flight summaries. */
    getBreakerSignal?: () => AbortSignal;
    /** The run's shared breaker controller. Preferred over the bare signal
     * accessor: the node captures it at entry so a breach detected by its
     * own chunk handler trips the run that STARTED the summarization. */
    getBreakerController?: () => AbortController;
    /** The controller's epoch, captured at entry and stamped into summary
     * attempt metadata so the wire consumer epoch-gates old-run summary
     * chunks like model-attempt chunks. */
    getBreakerEpoch?: () => number;
  } & StreamLimitState;
  generateStepId: (stepKey: string) => [string, number];
}

/** Composes the run's breaker signal (when the adapter provides one) into a
 * summarization attempt's config, so breaches in sibling branches cancel
 * summaries too. */
function withBreakerSignal<T extends { signal?: AbortSignal }>(
  config: T | undefined,
  graph?: { getBreakerSignal?: () => AbortSignal }
): T | undefined {
  const breakerSignal = graph?.getBreakerSignal?.();
  if (breakerSignal == null) {
    return config;
  }
  if (config == null) {
    return { signal: breakerSignal } as T;
  }
  const existing = config.signal;
  if (existing == null || existing === breakerSignal) {
    return { ...config, signal: breakerSignal };
  }
  return { ...config, signal: AbortSignal.any([existing, breakerSignal]) };
}

/** First stream-limit reason found on any of the given signals. Checked
 * alongside the graph's own breaker because a subagent child graph receives
 * a ROOT sibling's trip through its constructor/invocation signal — the
 * child's private breaker never fires for it. */
function findStreamLimitAbortReason(
  ...signals: Array<AbortSignal | undefined>
): StreamLimitExceededError | undefined {
  for (const signal of signals) {
    if (
      signal?.aborted === true &&
      signal.reason instanceof StreamLimitExceededError
    ) {
      return signal.reason;
    }
  }
  return undefined;
}

export function createSummarizeNode({
  agentContext,
  graph: adapterGraph,
  generateStepId,
}: CreateSummarizeNodeParams) {
  return async (
    state: {
      messages: BaseMessage[];
      summarizationRequest?: t.SummarizationNodeInput;
    },
    config?: RunnableConfig
  ): Promise<{ summarizationRequest: undefined; messages?: BaseMessage[] }> => {
    /** The breaker signal is captured at node ENTRY, before dispatchRunStep,
     * ON_SUMMARIZE_START, or PreCompact awaits: a sibling's trip plus a
     * prompt next run can reset the controller during one of those awaits,
     * and resolving the live accessor later would bind this summarization
     * to the fresh controller. `Object.create` freezes the signal while
     * DELEGATING every other adapter member — spreading would snapshot the
     * charge-credit accessor pair and detach it from graph resets. */
    const entryBreakerController = adapterGraph.getBreakerController?.();
    const entryBreakerSignal =
      entryBreakerController?.signal ?? adapterGraph.getBreakerSignal?.();
    const entryBreakerEpoch = adapterGraph.getBreakerEpoch?.();
    const graph =
      entryBreakerSignal == null
        ? adapterGraph
        : (Object.create(adapterGraph, {
          getBreakerSignal: { value: (): AbortSignal => entryBreakerSignal },
          ...(entryBreakerController != null && {
            getBreakerController: {
              value: (): AbortController => entryBreakerController,
            },
          }),
        }) as typeof adapterGraph);
    const request = state.summarizationRequest;
    if (request == null) {
      return { summarizationRequest: undefined };
    }
    /** Already-tripped-at-entry: a parallel sibling's breach failed the run
     * before this node was scheduled — via this graph's own breaker or, in
     * a subagent child graph, the parent trip on the invocation signal.
     * Rethrow before dispatching steps, hooks, or the model call — an
     * adapter that doesn't synchronously reject an aborted signal would
     * otherwise spend summarization provider work on a failed run. */
    const entryTripReason = findStreamLimitAbortReason(
      entryBreakerSignal,
      config?.signal
    );
    if (entryTripReason != null) {
      throw entryTripReason;
    }

    /**
     * Overflow recovery routes through this node purely to get back to the
     * agent node with a corrected budget, and deliberately spends no model
     * call on its first attempt: re-pruning under the raised context pressure
     * drives the pruner's tool-output compression and masking, which is
     * cheaper than a summary and cannot lose message content. Summarization
     * is also skipped outright when the caller never enabled it.
     */
    if (
      request.reason === 'overflow' &&
      (request.allowSummarization !== true ||
        agentContext.summarizationEnabled !== true)
    ) {
      emitAgentLog(
        config,
        'debug',
        'summarize',
        'Overflow recovery re-prune — compressing tool output without a summarization call',
        {
          maxContextTokens: agentContext.maxContextTokens,
          summarizationEnabled: agentContext.summarizationEnabled === true,
          allowSummarization: request.allowSummarization === true,
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      return { summarizationRequest: undefined };
    }

    const maxCtx = agentContext.maxContextTokens ?? 0;
    if (maxCtx > 0 && agentContext.instructionTokens >= maxCtx) {
      emitAgentLog(
        config,
        'warn',
        'summarize',
        'Summarization skipped, instructions exceed context budget. Reduce the number of tools or increase maxContextTokens.',
        {
          instructionTokens: agentContext.instructionTokens,
          maxContextTokens: maxCtx,
          breakdown: agentContext.formatTokenBudgetBreakdown(),
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      return { summarizationRequest: undefined };
    }

    /**
     * Capture the original-tool-content map locally before doing the
     * split.  We need it in three places: to restore the head for
     * summarizer quality, to leave intact on the skip path (state is
     * unchanged), and — critically — to carry forward the tail-relevant
     * entries on the summarize-fired path.  Clearing it eagerly here
     * would lose the originals for masked tool messages that the
     * recency window keeps in the tail; a future summarization could
     * then only summarize the masked stub instead of the full payload.
     */
    const originalPending = agentContext.pendingOriginalToolContent;

    const restoredMessages = restoreOriginalToolContent(
      state.messages,
      originalPending,
      agentContext.maxContextTokens
    );

    const runnableConfig = config ?? graph.config;

    const retainRecent = agentContext.summarizationConfig?.retainRecent;
    const { head: messagesToRefine, tailStartIndex } = splitAtRecencyBoundary(
      restoredMessages,
      {
        turns: retainRecent?.turns ?? DEFAULT_RETAIN_RECENT_TURNS,
        tokens: retainRecent?.tokens,
        tokenCounter: agentContext.tokenCounter,
      }
    );
    /**
     * Use the *masked* messages for the retained tail so that any
     * truncation prune applied to oversized ToolMessage content stays
     * truncated in live state.  The summarizer above reads the restored
     * (full-content) head for summary quality, but reinjecting restored
     * tool payloads into state would defeat masking and bloat the
     * checkpoint, forcing more expensive re-pruning on later turns.
     * `restoreOriginalToolContent` returns an array with identical
     * length and structure to `state.messages` (replacements only at
     * specific indices), so the same tailStartIndex slices both arrays
     * at the same turn boundary.
     */
    const messagesToRetain = state.messages.slice(tailStartIndex);

    if (messagesToRefine.length === 0) {
      /**
       * Recency window covers the entire conversation — there is no
       * older content to summarize.  Skipping prevents the model from
       * destroying the user's most recent message (e.g. a large pasted
       * payload on the first turn) by replacing it with a generic
       * checkpoint summary.  Mark the trigger so the same unchanged
       * state is not re-evaluated on the next prune cycle.
       */
      emitAgentLog(
        config,
        'debug',
        'summarize',
        'Summarization skipped — recency window retains all messages',
        {
          messagesRetained: messagesToRetain.length,
          retainTurns: retainRecent?.turns ?? DEFAULT_RETAIN_RECENT_TURNS,
        },
        { runId: graph.runId, agentId: request.agentId }
      );
      agentContext.markSummarizationTriggered(state.messages.length);
      return { summarizationRequest: undefined };
    }

    const clientConfig = buildSummarizationClientConfig(
      agentContext,
      agentContext.summarizationConfig
    );

    const stepKey = `summarize-${request.agentId}`;
    const [stepId, stepIndex] = generateStepId(stepKey);

    const placeholderSummary: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      model: clientConfig.modelName,
      provider: clientConfig.provider,
    };

    const runStep: t.RunStep = {
      stepIndex,
      id: stepId,
      type: StepTypes.MESSAGE_CREATION,
      index: graph.contentData.length,
      stepDetails: {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: { message_id: stepId },
      },
      summary: placeholderSummary,
      usage: null,
    };

    if (graph.runId != null && graph.runId !== '') {
      runStep.runId = graph.runId;
    }
    if (graph.isMultiAgent && agentContext.agentId) {
      runStep.agentId = agentContext.agentId;
    }

    await graph.dispatchRunStep(runStep, runnableConfig);

    if (runnableConfig) {
      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_START,
        {
          agentId: request.agentId,
          provider: clientConfig.provider,
          model: clientConfig.modelName,
          messagesToRefineCount: messagesToRefine.length,
          summaryVersion: agentContext.summaryVersion + 1,
        } satisfies t.SummarizeStartEvent,
        runnableConfig
      );
    }

    const sessionId = graph.runId ?? '';
    if (graph.hookRegistry?.hasHookFor('PreCompact', sessionId) === true) {
      const threadId = (
        runnableConfig?.configurable as Record<string, unknown> | undefined
      )?.thread_id as string | undefined;
      await executeHooks({
        registry: graph.hookRegistry,
        input: {
          hook_event_name: 'PreCompact',
          runId: sessionId,
          threadId,
          agentId: request.agentId,
          messagesBeforeCount: messagesToRefine.length,
          trigger: agentContext.summarizationConfig?.trigger?.type ?? 'default',
        },
        sessionId,
      }).catch(() => {
        /* PreCompact is observational — swallow errors */
      });
    }

    const isSelfSummarizeModel =
      clientConfig.provider === (agentContext.provider as string);
    const hasPromptCache =
      isSelfSummarizeModel &&
      (agentContext.clientOptions as Record<string, unknown> | undefined)
        ?.promptCache === true;

    const log: LogFn = (level, message, data) => {
      emitAgentLog(runnableConfig, level, 'summarize', message, data, {
        runId: graph.runId,
        agentId: request.agentId,
      });
    };

    log('debug', 'Summarization starting', {
      messagesToRefineCount: messagesToRefine.length,
      hasPriorSummary: (agentContext.getSummaryText()?.trim() ?? '') !== '',
      summaryVersion: agentContext.summaryVersion + 1,
      isSelfSummarize: isSelfSummarizeModel,
      hasPromptCache,
      provider: clientConfig.provider,
    });

    const summarizeConfig: RunnableConfig | undefined = config
      ? {
        ...config,
        metadata: {
          ...config.metadata,
          agent_id: request.agentId,
          /** Canonical agent-identity key, overwritten by every component
             *  that stamps identity (graph model path, ToolNode, here) so the
             *  closest stamper always wins via spread order — Langfuse scope
             *  trust relies on it (see `isForeignScope`). */
          agentId: request.agentId,
          summarization_provider: clientConfig.provider,
          summarization_model: clientConfig.modelName,
          /**
             * Per-call model attribution for usage consumers (the subagent
             * usage-capture handler): the summarizer's model can differ from
             * the agent's primary, and providers that emit no `ls_model_name`
             * would otherwise be billed against the primary config's model.
             * Omitted for self-summarize (no explicit model — the primary
             * config fallback is then correct). `tryFallbackProviders`
             * overrides this per fallback attempt; `INVOKED_PROVIDER` is
             * stamped by `attemptInvoke` itself.
             */
          ...(clientConfig.modelName != null && clientConfig.modelName !== ''
            ? { [Constants.INVOKED_MODEL]: clientConfig.modelName }
            : {}),
          /** Entry-captured breaker epoch: the wire consumer epoch-gates
           * old-run summary chunks exactly like model-attempt chunks —
           * without the stamp, a straggling summary from a failed run
           * reads as current and could abort the next run's controller. */
          ...(entryBreakerEpoch != null
            ? { [STREAM_LIMIT_EPOCH_KEY]: entryBreakerEpoch }
            : {}),
        },
      }
      : undefined;

    /** Rechecked after the pre-call awaits (dispatchRunStep,
     * ON_SUMMARIZE_START, PreCompact): a sibling can trip the breaker while
     * this node is paused in one of them, and a provider that doesn't
     * synchronously reject an aborted signal would still start the
     * summarization call. Mirrors the model node's pre-invoke recheck. */
    {
      const preCallTrip = findStreamLimitAbortReason(
        entryBreakerSignal,
        config?.signal
      );
      if (preCallTrip != null) {
        throw preCallTrip;
      }
    }
    const {
      text: rawText,
      usage: summaryUsage,
      usedMetadataStub,
    } = await executeSummarizationWithFallback({
      agentContext,
      messages: messagesToRefine,
      clientConfig,
      summarizeConfig,
      stepId,
      usePromptCache: isSelfSummarizeModel && hasPromptCache,
      log,
      graph,
    });

    /**
     * The metadata stub describes the history rather than summarizing it, so
     * committing it means removing the head and keeping nothing of what it
     * said. That trade is never worth making to paper over an overflow: the
     * recovery would "succeed" only by destroying the conversation it was
     * supposed to preserve. Leave state untouched and let the provider error
     * surface instead.
     */
    if (usedMetadataStub === true && request.reason === 'overflow') {
      log(
        'warn',
        'Overflow summarization failed; keeping history rather than replacing it with a metadata stub'
      );
      agentContext.markSummarizationTriggered(state.messages.length);
      /**
       * The run step was already dispatched, so it has to be resolved here or
       * consumers tracking step lifecycle keep an unfinished placeholder for
       * the rest of the run.
       */
      await graph.dispatchRunStepCompleted(
        stepId,
        {
          type: 'summary',
          summary: placeholderSummary,
        } satisfies t.SummaryCompleted,
        runnableConfig
      );
      if (runnableConfig) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            id: stepId,
            agentId: request.agentId,
            error:
              'Summarization failed during overflow recovery; conversation history was preserved',
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      return { summarizationRequest: undefined };
    }

    if (!rawText) {
      agentContext.markSummarizationTriggered(0);
      if (runnableConfig) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            id: stepId,
            agentId: request.agentId,
            error: 'Summarization produced empty output',
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      return { summarizationRequest: undefined };
    }

    const summaryText = enrichSummary(rawText, messagesToRefine);

    const tokenCount = computeSummaryTokenCount(
      summaryText,
      summaryUsage,
      agentContext.tokenCounter
    );

    agentContext.setSummary(summaryText, tokenCount);

    log('info', 'Summary persisted');
    log('debug', 'Summary details', {
      summaryTokens: tokenCount,
      textLength: summaryText.length,
      messagesCompacted: messagesToRefine.length,
      summaryVersion: agentContext.summaryVersion,
      ...(summaryUsage != null
        ? {
          input_tokens: summaryUsage.input_tokens,
          output_tokens: summaryUsage.output_tokens,
          cache_read: summaryUsage.input_token_details?.cache_read,
          cache_creation: summaryUsage.input_token_details?.cache_creation,
        }
        : {}),
    });

    const summaryBlock = buildSummaryBlock({
      summaryText,
      tokenCount,
      coverage: resolveSummaryCoverage(messagesToRetain),
      stepId,
      stepIndex: runStep.index,
      modelName: clientConfig.modelName,
      provider: clientConfig.provider,
      summaryVersion: agentContext.summaryVersion,
    });

    await dispatchCompletionEvents({
      graph,
      runnableConfig,
      stepId,
      summaryBlock,
      agentContext,
      runStep,
      summaryUsage,
      agentId: request.agentId,
      messagesAfterCount: messagesToRetain.length,
    });

    /**
     * `dispatchCompletionEvents` calls `rebuildTokenMapAfterSummarization({})`
     * which resets the dedupe baseline to 0 — correct under the legacy
     * "remove-all only" shape where no messages survived, but stale once
     * the recency window keeps a tail.  Realign the baseline to the
     * surviving tail length so a subsequent prune cycle on the unchanged
     * tail short-circuits via `shouldSkipSummarization` instead of
     * looping back into another summarize call.
     */
    agentContext.markSummarizationTriggered(messagesToRetain.length);

    /**
     * Carry forward the original-content entries that correspond to the
     * retained tail, reindexed for the post-removeAll state where tail
     * messages start at index 0.  Without this, a future summarization
     * that pulls these tail messages into its head would only see the
     * masked stubs (since `setSummary` clears `pruneMessages`, and the
     * fresh pruner at the next turn has no record of prior masks).
     * Entries for indices < `tailStartIndex` belong to messages we just
     * summarized — they are no longer reachable so they are dropped.
     */
    if (originalPending != null && originalPending.size > 0) {
      const tailPending = new Map<number, string>();
      for (const [idx, content] of originalPending) {
        if (idx >= tailStartIndex) {
          tailPending.set(idx - tailStartIndex, content);
        }
      }
      agentContext.pendingOriginalToolContent =
        tailPending.size > 0 ? tailPending : undefined;
    } else {
      agentContext.pendingOriginalToolContent = undefined;
    }

    return {
      summarizationRequest: undefined,
      messages:
        messagesToRetain.length > 0
          ? [createRemoveAllMessage(), ...messagesToRetain]
          : [createRemoveAllMessage()],
    };
  };
}

/** Extracts text from an LLM response, skipping reasoning/thinking blocks. */
function extractResponseText(response: { content: string | object }): string {
  const { content } = response;
  if (typeof content === 'string') {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return '';
  }
  const parts: string[] = [];
  for (const block of content) {
    if (typeof block === 'string') {
      parts.push(block);
      continue;
    }
    if (block == null || typeof block !== 'object') {
      continue;
    }
    const rec = block as Record<string, unknown>;
    if (
      rec.type === ContentTypes.THINKING ||
      rec.type === ContentTypes.REASONING_CONTENT ||
      rec.type === 'redacted_thinking'
    ) {
      continue;
    }
    if (rec.type === 'text' && typeof rec.text === 'string') {
      parts.push(rec.text);
    }
  }
  return parts.join('').trim();
}

function buildSummarizationInstruction(
  promptText: string,
  updatePromptText: string | undefined,
  priorSummaryText: string
): string {
  const effectivePrompt = priorSummaryText
    ? (updatePromptText ?? promptText)
    : promptText;
  const parts = [effectivePrompt];
  if (priorSummaryText) {
    parts.push(
      `\n\n<previous-summary>\n${priorSummaryText}\n</previous-summary>`
    );
  }
  return parts.join('');
}

/** Creates an `onChunk` callback that dispatches `ON_SUMMARIZE_DELTA` events for streaming. */
export function createSummarizationChunkHandler({
  stepId,
  config,
  provider,
  reasoningKey = 'reasoning_content',
  graph,
}: {
  stepId?: string;
  config?: RunnableConfig;
  provider?: Providers;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  graph?: StreamLimitState & { getBreakerController?: () => AbortController };
}): OnChunk | undefined {
  if (stepId == null || stepId === '' || !config) {
    return undefined;
  }
  const limitMetadata = config.metadata as Record<string, unknown> | undefined;
  return (chunk, attemptMetadata) => {
    /**
     * Charged as `producer` so it PAIRS with the run's registered stream
     * handler: inside a live run the provider callbacks route each chunk
     * through the graph's `on_chat_model_stream` consumer too, and two
     * same-side consumer claims would charge the chunk twice — halving an
     * opt-in event cap. Standalone callers (tests, custom hosts, runs with
     * no registered handler) still get single-sided enforcement from this
     * one claim. Summarization deliberately passes NO `context` to
     * `attemptInvoke`, whose onChunk branch would otherwise add a second
     * producer claim. A breach trips the run's shared breaker (the
     * ENTRY-captured controller when the adapter provides one) before
     * rethrowing: this producer claim wins the race, so the wire consumer's
     * breaker-aborting catch never fires for the same chunk — without the
     * trip here, sibling model calls, tools, and subagents would keep
     * consuming quota while the summary branch rejects.
     * `executeSummarizationWithFallback` then rethrows past fallbacks and
     * the metadata stub, failing the run like any other breach.
     */
    if (graph != null) {
      try {
        enforceStreamLimitsForWireChunk({
          graph,
          metadata: attemptMetadata ?? limitMetadata,
          chunk: chunk as Parameters<
            typeof enforceStreamLimitsForWireChunk
          >[0]['chunk'],
        });
      } catch (error) {
        if (error instanceof StreamLimitExceededError) {
          graph.getBreakerController?.().abort(error);
        }
        throw error;
      }
    }
    const chunkAny = chunk as Parameters<typeof getChunkContent>[0]['chunk'];
    const raw = getChunkContent({ chunk: chunkAny, provider, reasoningKey });
    if (raw == null || (typeof raw === 'string' && !raw)) {
      return;
    }
    const contentBlocks: t.MessageContentComplex[] =
      typeof raw === 'string'
        ? [{ type: ContentTypes.TEXT, text: raw } as t.MessageContentComplex]
        : raw;

    void safeDispatchCustomEvent(
      GraphEvents.ON_SUMMARIZE_DELTA,
      {
        id: stepId,
        delta: {
          summary: {
            type: ContentTypes.SUMMARY,
            content: contentBlocks,
            provider: String(config.metadata?.summarization_provider ?? ''),
            model: String(config.metadata?.summarization_model ?? ''),
          },
        },
      } satisfies t.SummarizeDeltaEvent,
      config
    );
  };
}

function traceConfig(
  config: RunnableConfig | undefined,
  stage: string
): RunnableConfig | undefined {
  if (!config) {
    return undefined;
  }
  return {
    ...config,
    runName: `summarization:${stage}`,
    metadata: { ...config.metadata, summarization: true, stage },
  };
}

/**
 * Cache-friendly compaction: sends raw conversation messages with the
 * summarization instruction appended as the final HumanMessage.
 * Providers with prompt caching get a cache hit on the system prompt +
 * tool definitions prefix.
 */
async function summarizeWithCacheHit({
  model,
  messages,
  promptText,
  updatePromptText,
  priorSummaryText,
  config,
  stepId,
  provider,
  reasoningKey,
  graph,
  usePromptCache,
  promptCacheTtl,
  log,
}: {
  model: t.ChatModel;
  messages: BaseMessage[];
  promptText: string;
  updatePromptText?: string;
  priorSummaryText: string;
  config?: RunnableConfig;
  stepId?: string;
  provider: Providers;
  reasoningKey?: 'reasoning_content' | 'reasoning';
  graph?: StreamLimitState & { getBreakerSignal?: () => AbortSignal };
  usePromptCache?: boolean;
  promptCacheTtl?: PromptCacheTtl;
  log?: LogFn;
}): Promise<{ text: string; usage?: Partial<UsageMetadata> }> {
  const instruction = buildSummarizationInstruction(
    promptText,
    updatePromptText,
    priorSummaryText
  );

  const fullMessages = [...messages, new HumanMessage(instruction)];
  const invokeMessages =
    usePromptCache === true
      ? addTailCacheControl(fullMessages, promptCacheTtl)
      : fullMessages;

  const result = await attemptInvoke(
    {
      model,
      messages: invokeMessages,
      provider,
      onChunk: createSummarizationChunkHandler({
        stepId,
        config: traceConfig(config, 'cache_hit_compaction'),
        provider,
        reasoningKey,
        graph,
      }),
      /** Accounting lease only (summarization passes no `context` so its
       * producer claim never stacks with the onChunk handler's). */
      streamLimitState: graph,
    },
    withBreakerSignal(traceConfig(config, 'cache_hit_compaction'), graph)
  );

  const responseMsg = result.messages?.[0];
  const text = responseMsg
    ? extractResponseText(responseMsg as { content: string | object })
    : '';
  let usage: Partial<UsageMetadata> | undefined;
  let usageSource = 'none';
  if (
    responseMsg != null &&
    'usage_metadata' in responseMsg &&
    responseMsg.usage_metadata != null
  ) {
    usage = responseMsg.usage_metadata as Partial<UsageMetadata>;
    usageSource = 'usage_metadata';
  } else if (responseMsg != null) {
    const respMeta = responseMsg.response_metadata as
      | Record<string, unknown>
      | undefined;
    const raw = (respMeta?.metadata as Record<string, unknown> | undefined)
      ?.usage as Record<string, unknown> | undefined;
    if (raw != null) {
      usage = {
        input_tokens: Number(raw.inputTokens) || undefined,
        output_tokens: Number(raw.outputTokens) || undefined,
      } as Partial<UsageMetadata>;
      usageSource = 'response_metadata';
    }
  }
  const cacheDetails = (
    usage as
      | {
          input_token_details?: {
            cache_read?: number;
            cache_creation?: number;
          };
        }
      | undefined
  )?.input_token_details;
  log?.('debug', 'Summarization LLM usage', {
    source: usageSource,
    input_tokens: usage?.input_tokens,
    output_tokens: usage?.output_tokens,
    ...(cacheDetails?.cache_read != null || cacheDetails?.cache_creation != null
      ? {
        'input_token_details.cache_read': cacheDetails.cache_read,
        'input_token_details.cache_creation': cacheDetails.cache_creation,
      }
      : {}),
  });
  return { text, usage };
}
