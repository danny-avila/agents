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
  getProviderMessageRole,
  getProviderToolCallPartDescriptor,
  getProviderToolMessageResultDescriptor,
  getProviderToolResultPartDescriptor,
} from './toolResultTypes';
import { apportionTokenCounts } from '@/utils/tokens';
import { isReasoningContentBlock } from './core';
import { emitAgentLog } from '@/utils/events';
import { ContentTypes } from '@/common';

const UNKNOWN_TOOL = 'unknown_tool';
const BLANK_TEXT = /^\s*$/u;

/** Rejects counts a subset claim cannot be derived from. Callers must degrade
 *  rather than propagate: this runs on the live pre-invoke path. */
function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
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
    'Tool-message context share unavailable: the token counter must return safe, non-negative integers',
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

function getPairedToolName(
  descriptor: ProviderToolResultPartDescriptor,
  calls: ProviderToolCallIndex
): string | undefined {
  return descriptor.toolCallId == null
    ? undefined
    : calls.get(descriptor.toolCallId)?.descriptor.name;
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
  calls: ProviderToolCallIndex
): InvocationScan {
  let recognizedCalls = appendProviderMessageToolCalls(message, calls);
  if (typeof message.content === 'string') {
    return { recognizedCalls, toolOnly: BLANK_TEXT.test(message.content) };
  }
  const parts: ReadonlyArray<string | MessageContentComplex> = message.content;
  let toolOnly = true;
  for (const part of parts) {
    if (typeof part === 'string') {
      toolOnly &&= BLANK_TEXT.test(part);
      continue;
    }
    const call = getProviderToolCallPartDescriptor(part);
    if (call != null) {
      appendProviderToolCallDescriptor(calls, call);
      recognizedCalls += 1;
      continue;
    }
    if (getProviderToolResultPartDescriptor(part) != null) {
      continue;
    }
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
  pendingLegacyName: string | undefined
): string {
  const descriptor = getProviderToolMessageResultDescriptor(message);
  const paired =
    descriptor == null ? undefined : getPairedToolName(descriptor, calls);
  if (paired != null) {
    return paired;
  }
  if (message.name != null && message.name.length > 0) {
    return message.name;
  }
  return pendingLegacyName ?? UNKNOWN_TOOL;
}

/**
 * A user turn made only of `tool_result` parts the converters accept in that
 * position: the split `AIMessage(tool_call)` + `HumanMessage(tool_result)`
 * history. The counter measures whole messages, so the turn is attributed to
 * its one paired tool, or to `unknown_tool` when its parts name several.
 * Returns undefined for any other user turn.
 */
function getUserToolResultName(
  message: BaseMessage,
  calls: ProviderToolCallIndex
): string | undefined {
  if (typeof message.content === 'string' || message.content.length === 0) {
    return undefined;
  }
  let name: string | undefined;
  let mixed = false;
  for (const part of message.content) {
    const descriptor = getProviderToolResultPartDescriptor(part);
    if (descriptor?.allowHumanMessagePairing !== true) {
      return undefined;
    }
    const paired = getPairedToolName(descriptor, calls);
    mixed ||= name != null && paired !== name;
    name = paired;
  }
  return mixed || name == null ? UNKNOWN_TOOL : name;
}

/** Counts retained tool exchanges without serializing arguments or result content. */
function computeToolMessageUsage(
  context: readonly BaseMessage[],
  tokenCounter: t.TokenCounter,
  calibrationRatio = 1
): Pick<
  t.TokenBudgetBreakdown,
  'toolMessageTokens' | 'toolMessageTokenCounts'
> {
  const calls: ProviderToolCallIndex = new Map();
  const counts: Record<string, number> = Object.create(null);
  let pendingLegacyName: string | undefined;
  let total = 0;
  let resultTotal = 0;
  const countResult = (message: BaseMessage, name: string): void => {
    const tokens = safeCount(tokenCounter(message));
    total = safeCount(total + tokens);
    if (tokens === 0) {
      return;
    }
    counts[name] = safeCount((counts[name] ?? 0) + tokens);
    resultTotal = safeCount(resultTotal + tokens);
  };

  for (const message of context) {
    const role = getProviderMessageRole(message);
    if (role === 'assistant') {
      const scan = scanInvocation(message, calls);
      pendingLegacyName = readLegacyFunctionName(message);
      if (scan.recognizedCalls > 0 && scan.toolOnly) {
        total = safeCount(total + safeCount(tokenCounter(message)));
      }
      continue;
    }
    if (role === 'tool' || role === 'function') {
      const legacyName = role === 'function' ? pendingLegacyName : undefined;
      countResult(message, getToolMessageResultName(message, calls, legacyName));
      if (role === 'function') {
        pendingLegacyName = undefined;
      }
      continue;
    }
    const userResultName =
      role === 'user' ? getUserToolResultName(message, calls) : undefined;
    if (userResultName != null) {
      countResult(message, userResultName);
      continue;
    }
    pendingLegacyName = undefined;
  }

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
}

/** Applies the retained tool share and clamps it to the conversation total it
 * is a subset of. Throws when a count cannot support that claim. */
function syncToolMessageShare(
  usage: t.ContextUsageEvent,
  context?: readonly BaseMessage[],
  tokenCounter?: t.TokenCounter
): void {
  const { breakdown } = usage;
  if (context != null && tokenCounter != null) {
    Object.assign(
      breakdown,
      computeToolMessageUsage(context, tokenCounter, usage.calibrationRatio)
    );
  }
  if (breakdown.toolMessageTokens == null) {
    return;
  }
  const total = safeCount(breakdown.toolMessageTokens);
  const clamped = Math.min(total, safeCount(breakdown.messageTokens));
  if (clamped !== total && breakdown.toolMessageTokenCounts != null) {
    let resultTotal = 0;
    for (const count of Object.values(breakdown.toolMessageTokenCounts)) {
      resultTotal = safeCount(resultTotal + safeCount(count));
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
  config?: RunnableConfig
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
    syncToolMessageShare(usage, context, tokenCounter);
  } catch (error) {
    warnUnavailableToolShare(config, error);
    delete breakdown.toolMessageTokens;
    delete breakdown.toolMessageTokenCounts;
  }
}
