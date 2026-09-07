import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  ProviderToolCallIndex,
  ProviderToolResultPartDescriptor,
} from './toolResultTypes';
import type * as t from '@/types';
import {
  appendProviderMessageToolCalls,
  appendProviderToolCallDescriptor,
  consumeProviderToolResultPair,
  getProviderMessageRole,
  getProviderToolCallPartDescriptor,
  getProviderToolMessageResultDescriptor,
  getProviderToolResultPartDescriptor,
  isExecutableCodePart,
} from './toolResultTypes';
import { getProviderMessageProvenance } from './provenance';
import { apportionTokenCounts } from '@/utils/tokens';
import { isReasoningContentBlock } from './core';
import { emitAgentLog } from '@/utils/events';
import { ContentTypes } from '@/common';

const UNKNOWN_TOOL = 'unknown_tool';
const BLANK_TEXT = /^\s*$/u;

/** Guards an aggregate: sums of rounded counts must stay safe integers.
 *  Callers must degrade rather than propagate: this runs on the live pre-invoke path. */
function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError('Invalid tool context token count');
  }
  return value;
}

/** Accepts what the `TokenCounter` contract allows, an approximate `number`
 *  such as `length / 4`, by rounding it, and rejects only what no subset claim
 *  can be derived from: NaN, infinities, negatives and values past the safe range. */
function toCount(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError('Invalid tool context token count');
  }
  return safeCount(Math.round(value));
}

function safeRawCount(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > Number.MAX_SAFE_INTEGER) {
    throw new RangeError('Invalid tool context token count');
  }
  return value;
}

let warnedUnavailableToolShare = false;

/** Warns once per process: an unusable counter is a permanent host-integration
 *  fault, while the share is dropped on every affected call regardless. The
 *  latch is spent only when a config can carry the event, so a config-less
 *  caller (the pre-send projection) cannot swallow the live path's one warning. */
function warnUnavailableToolShare(
  config: RunnableConfig | undefined,
  error: unknown
): void {
  if (warnedUnavailableToolShare || config == null) {
    return;
  }
  warnedUnavailableToolShare = true;
  emitAgentLog(
    config,
    'warn',
    'budget',
    'Tool-message context share unavailable: the token counter must return finite, non-negative counts within the safe range',
    { error: error instanceof Error ? error.message : String(error) }
  );
}

/** `isReasoningContentBlock` matches every provider reasoning block by prefix
 *  but deliberately excludes LibreChat's `think`, which this gauge must treat
 *  the same way: reasoning attached to the turn, not visible conversation. */
function isReasoningPart(part: MessageContentComplex): boolean {
  return part.type === ContentTypes.THINK || isReasoningContentBlock(part);
}

function isBlankTextPart(part: string | MessageContentComplex): boolean {
  if (typeof part === 'string') {
    return BLANK_TEXT.test(part);
  }
  return (
    part.type === ContentTypes.TEXT &&
    typeof part.text === 'string' &&
    BLANK_TEXT.test(part.text)
  );
}

function readLegacyFunctionName(message: BaseMessage): string | undefined {
  const name = message.additional_kwargs.function_call?.name;
  return typeof name === 'string' && name.length > 0 ? name : undefined;
}

interface ResultPairing {
  readonly paired: boolean;
  readonly name?: string;
}

/** Pairs a result with its call the way the wire walkers do: the call is
 *  validated for kind and name and then consumed, so a provider that reuses a
 *  call id in a later turn is attributed to the current call rather than
 *  poisoning the id, and the bounded index never fills with answered calls. */
function pairResult(
  descriptor: ProviderToolResultPartDescriptor,
  calls: ProviderToolCallIndex,
  previousPart?: unknown
): ResultPairing {
  const name =
    descriptor.toolCallId == null
      ? undefined
      : calls.get(descriptor.toolCallId)?.descriptor.name;
  const paired = consumeProviderToolResultPair(descriptor, calls, previousPart);
  return paired ? { paired, name } : { paired };
}

/**
 * A user turn a provider transform built from tool history: a tool-less
 * destination inheriting tool turns folds each call and its results into one
 * synthetic `HumanMessage`, and compaction of that fold keeps the lineage. The
 * role is user on the wire, the bytes are retained tool output, and the fold
 * is what the provenance stamp records. The counter measures whole messages,
 * so per-tool attribution is lost with the fold, and any visible model text the
 * fold carried alongside its calls (a "let me search" preamble, the fold's own
 * scaffolding) is counted with it. That is a bounded over-count on a turn that
 * exists only because of tool content; the alternative is reporting zero.
 */
function isFoldedToolHistory(message: BaseMessage): boolean {
  const parts = getProviderMessageProvenance(message)?.parts;
  return parts != null && parts.some((part) => part.attribution === 'tool');
}

interface InvocationScan {
  readonly recognizedCalls: number;
  readonly toolOnly: boolean;
}

/**
 * Registers an assistant turn's calls and decides whether the turn is
 * tool-only. Tool-only means every part is blank text, a tool call, a provider
 * tool result returned inline (Anthropic server tools), or reasoning attached
 * to the turn. Visible text, media and unrecognized blocks keep the whole turn
 * in the conversation share. Shape recognition is the taxonomy's, so a call or
 * result representation this repo's converters accept is accepted here.
 */
function scanInvocation(
  message: BaseMessage,
  calls: ProviderToolCallIndex,
  provider?: t.ProviderName
): InvocationScan {
  let recognizedCalls = appendProviderMessageToolCalls(
    message,
    calls,
    provider
  );
  if (typeof message.content === 'string') {
    return { recognizedCalls, toolOnly: BLANK_TEXT.test(message.content) };
  }
  const parts: ReadonlyArray<string | MessageContentComplex> = message.content;
  let toolOnly = true;
  let previousPart: string | MessageContentComplex | undefined;
  for (const part of parts) {
    if (typeof part === 'string') {
      toolOnly &&= BLANK_TEXT.test(part);
      previousPart = part;
      continue;
    }
    const call = getProviderToolCallPartDescriptor(part);
    if (call != null) {
      appendProviderToolCallDescriptor(calls, call);
      recognizedCalls += 1;
      previousPart = part;
      continue;
    }
    if (isExecutableCodePart(part)) {
      recognizedCalls += 1;
      previousPart = part;
      continue;
    }
    const result = getProviderToolResultPartDescriptor(part);
    if (result != null) {
      pairResult(result, calls, previousPart);
      previousPart = part;
      continue;
    }
    previousPart = part;
    if (isReasoningPart(part)) {
      continue;
    }
    toolOnly &&= isBlankTextPart(part);
  }
  return { recognizedCalls, toolOnly };
}

/** Result attribution: the paired call's name, then the message's own name,
 *  then the legacy call still pending for a nameless `FunctionMessage`. */
function getToolMessageResultName(
  message: BaseMessage,
  calls: ProviderToolCallIndex,
  pendingLegacyName: string | undefined,
  provider?: t.ProviderName
): string {
  const descriptor = getProviderToolMessageResultDescriptor(message, provider);
  const pairing =
    descriptor == null ? undefined : pairResult(descriptor, calls);
  if (pairing?.name != null) {
    return pairing.name;
  }
  if (message.name != null && message.name.length > 0) {
    return message.name;
  }
  return pendingLegacyName ?? UNKNOWN_TOOL;
}

/**
 * A user turn made only of `tool_result` parts that each pair with a pending
 * call, the split `AIMessage(tool_call)` + `HumanMessage(tool_result)` history
 * the converters accept. Pairing runs against a candidate copy and commits only
 * when every part pairs, as the recency walker does; anything else is an
 * ordinary user turn. The counter measures whole messages, so the turn is
 * attributed to its one paired tool, or to `unknown_tool` when parts name
 * several. Returns undefined for any other user turn.
 */
function takeUserToolResultName(
  message: BaseMessage,
  calls: ProviderToolCallIndex
): string | undefined {
  if (typeof message.content === 'string' || message.content.length === 0) {
    return undefined;
  }
  const candidate: ProviderToolCallIndex = new Map(calls);
  let name: string | undefined;
  let mixed = false;
  let previousPart: unknown;
  for (const part of message.content) {
    const descriptor = getProviderToolResultPartDescriptor(part);
    if (descriptor?.allowHumanMessagePairing !== true) {
      return undefined;
    }
    const pairing = pairResult(descriptor, candidate, previousPart);
    if (!pairing.paired) {
      return undefined;
    }
    mixed ||= name != null && pairing.name !== name;
    name = pairing.name;
    previousPart = part;
  }
  calls.clear();
  for (const [callId, entry] of candidate) {
    calls.set(callId, entry);
  }
  return mixed || name == null ? UNKNOWN_TOOL : name;
}

/** Counts retained tool exchanges without serializing arguments or result content. */
export interface ToolMessageUsageAccumulator {
  add(message: BaseMessage, rawTokens: number): void;
  finish(
    calibrationRatio?: number
  ): Pick<
    t.TokenBudgetBreakdown,
    'toolMessageTokens' | 'toolMessageTokenCounts'
  >;
}

/** Accumulates tool-share attribution while its caller walks a provider payload. */
export function createToolMessageUsageAccumulator(
  provider?: t.ProviderName
): ToolMessageUsageAccumulator {
  const calls: ProviderToolCallIndex = new Map();
  const counts: Record<string, number> = Object.create(null);
  let pendingLegacyName: string | undefined;
  let total = 0;
  let resultTotal = 0;
  const addRaw = (current: number, increment: number): number =>
    safeRawCount(current + safeRawCount(increment));
  const countResult = (tokens: number, name: string): void => {
    total = addRaw(total, tokens);
    if (tokens === 0) {
      return;
    }
    counts[name] = addRaw(counts[name] ?? 0, tokens);
    resultTotal = addRaw(resultTotal, tokens);
  };

  const add = (message: BaseMessage, rawTokens: number): void => {
    const tokens = safeRawCount(rawTokens);
    const role = getProviderMessageRole(message, provider);
    if (role === 'assistant') {
      const scan = scanInvocation(message, calls, provider);
      pendingLegacyName = readLegacyFunctionName(message);
      if (scan.recognizedCalls > 0 && scan.toolOnly) {
        total = addRaw(total, tokens);
      }
      return;
    }
    if (role === 'tool' || role === 'function') {
      const legacyName = role === 'function' ? pendingLegacyName : undefined;
      countResult(
        tokens,
        getToolMessageResultName(message, calls, legacyName, provider)
      );
      if (role === 'function') {
        pendingLegacyName = undefined;
      }
      return;
    }
    if (role === 'user' && isFoldedToolHistory(message)) {
      total = addRaw(total, tokens);
      return;
    }
    const userResultName =
      role === 'user' ? takeUserToolResultName(message, calls) : undefined;
    if (userResultName != null) {
      countResult(tokens, userResultName);
      return;
    }
    if (role === 'user') {
      calls.clear();
    }
    pendingLegacyName = undefined;
  };

  const finish = (
    calibrationRatio = 1
  ): Pick<
    t.TokenBudgetBreakdown,
    'toolMessageTokens' | 'toolMessageTokenCounts'
  > => {
    const ratio =
      Number.isFinite(calibrationRatio) && calibrationRatio > 0
        ? calibrationRatio
        : 1;
    const toolMessageTokens = safeCount(Math.round(total * ratio));
    const resultTokens = Math.min(
      toolMessageTokens,
      safeCount(Math.round(resultTotal * ratio))
    );
    return {
      toolMessageTokens,
      toolMessageTokenCounts:
        resultTotal > 0 && resultTokens > 0
          ? apportionTokenCounts(counts, ratio, resultTokens)
          : undefined,
    };
  };

  return { add, finish };
}

function computeToolMessageUsage(
  context: readonly BaseMessage[],
  tokenCounter: t.TokenCounter,
  calibrationRatio = 1,
  provider?: t.ProviderName
): Pick<
  t.TokenBudgetBreakdown,
  'toolMessageTokens' | 'toolMessageTokenCounts'
> {
  const accumulator = createToolMessageUsageAccumulator(provider);
  for (const message of context) {
    accumulator.add(message, tokenCounter(message));
  }
  return accumulator.finish(calibrationRatio);
}

/** Applies the retained tool share and clamps it to the conversation total it
 * is a subset of. Throws when a count cannot support that claim. */
function syncToolMessageShare(
  usage: t.ContextUsageEvent,
  context?: readonly BaseMessage[],
  tokenCounter?: t.TokenCounter,
  provider?: t.ProviderName
): void {
  const { breakdown } = usage;
  if (context != null && tokenCounter != null) {
    Object.assign(
      breakdown,
      computeToolMessageUsage(
        context,
        tokenCounter,
        usage.calibrationRatio,
        provider
      )
    );
  }
  if (breakdown.toolMessageTokens == null) {
    return;
  }
  const total = toCount(breakdown.toolMessageTokens);
  const clamped = Math.min(total, toCount(breakdown.messageTokens));
  if (clamped !== total && breakdown.toolMessageTokenCounts != null) {
    let resultTotal = 0;
    for (const count of Object.values(breakdown.toolMessageTokenCounts)) {
      resultTotal = safeCount(resultTotal + toCount(count));
    }
    const factor = total > 0 ? clamped / total : 0;
    const target = Math.min(clamped, Math.round(resultTotal * factor));
    breakdown.toolMessageTokenCounts =
      target > 0
        ? apportionTokenCounts(breakdown.toolMessageTokenCounts, factor, target)
        : undefined;
  }
  breakdown.toolMessageTokens = clamped;
}

/** Reconciles derived budget fields and, when supplied, the retained tool share.
 * The index map is keyed before pruning, so it cannot index the retained context.
 * Callers may pass the existing exact token cache's counter, never a stale index lookup.
 * Runs on the live pre-invoke path, so an unusable count drops the tool share
 * (warned once) instead of failing the model call it only measures. */
export function syncBudgetDerivedFields(
  usage: t.ContextUsageEvent,
  context?: readonly BaseMessage[],
  tokenCounter?: t.TokenCounter,
  config?: RunnableConfig,
  provider?: t.ProviderName,
  toolMessageUsageError?: Error
): void {
  const { breakdown, contextBudget, effectiveInstructionTokens } = usage;
  if (effectiveInstructionTokens != null) {
    breakdown.instructionTokens = effectiveInstructionTokens;
    if (contextBudget != null) {
      breakdown.availableForMessages = Math.max(
        0,
        contextBudget - effectiveInstructionTokens
      );
      if (usage.remainingContextTokens != null) {
        breakdown.messageTokens = Math.max(
          0,
          contextBudget -
            effectiveInstructionTokens -
            usage.remainingContextTokens
        );
      }
    }
  }
  try {
    if (toolMessageUsageError != null) {
      throw toolMessageUsageError;
    }
    syncToolMessageShare(usage, context, tokenCounter, provider);
  } catch (error) {
    warnUnavailableToolShare(config, error);
    delete breakdown.toolMessageTokens;
    delete breakdown.toolMessageTokenCounts;
  }
}
