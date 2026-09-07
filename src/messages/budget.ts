import type {
  AIMessage,
  BaseMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { apportionTokenCounts } from '@/utils/tokens';

type NamedToolCall = {
  id?: string;
  name?: string;
  function?: { name?: string };
};

function safeCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError('Invalid tool context token count');
  }
  return value;
}

/** Counts retained tool exchanges without serializing arguments or result content.
 * Mixed text/reasoning/media assistant messages remain in the conversation share. */
function computeToolMessageUsage(
  context: readonly BaseMessage[],
  tokenCounter: t.TokenCounter,
  calibrationRatio = 1
): Pick<
  t.TokenBudgetBreakdown,
  'toolMessageTokens' | 'toolMessageTokenCounts'
> {
  const names = new Map<string, string>();
  const counts: Record<string, number> = Object.create(null);
  let legacyName: string | undefined;
  let total = 0;
  let resultTotal = 0;
  const rememberCall = (call: NamedToolCall): void => {
    const name =
      call.name != null && call.name.length > 0
        ? call.name
        : call.function?.name;
    if (
      typeof call.id === 'string' &&
      typeof name === 'string' &&
      name.length > 0
    ) {
      names.set(call.id, name);
    }
  };

  for (const message of context) {
    const type = message.getType();
    const isResult = type === 'tool' || type === 'function';
    if (type === 'ai') {
      const ai = message as AIMessage;
      const calls = ai.tool_calls;
      const rawCalls = ai.additional_kwargs.tool_calls;
      if (rawCalls != null) {
        for (const call of rawCalls) {
          rememberCall(call);
        }
      }
      if (calls != null) {
        for (const call of calls) {
          rememberCall(call);
        }
      }
      legacyName = ai.additional_kwargs.function_call?.name;
      let hasCalls =
        (calls?.length ?? 0) > 0 ||
        (rawCalls?.length ?? 0) > 0 ||
        (legacyName != null && legacyName.length > 0);
      let toolOnly = true;
      if (typeof ai.content === 'string') {
        toolOnly = !/\S/u.test(ai.content);
      } else {
        const content: ReadonlyArray<
          string | Exclude<AIMessage['content'], string>[number]
        > = ai.content;
        for (const part of content) {
          if (typeof part === 'string') {
            toolOnly &&= !/\S/u.test(part);
          } else if (part.type === 'tool_use') {
            hasCalls = true;
            if (
              typeof part.id === 'string' &&
              typeof part.name === 'string' &&
              !names.has(part.id)
            ) {
              rememberCall({ id: part.id, name: part.name });
            }
          } else {
            toolOnly &&=
              part.type === 'text' &&
              typeof part.text === 'string' &&
              !/\S/u.test(part.text);
          }
        }
      }
      if (!hasCalls || !toolOnly) {
        continue;
      }
    } else if (!isResult) {
      legacyName = undefined;
      continue;
    }

    const tokens = safeCount(tokenCounter(message));
    total = safeCount(total + tokens);
    if (!isResult) {
      continue;
    }
    const id =
      type === 'tool' ? (message as ToolMessage).tool_call_id : undefined;
    const pendingName =
      type === 'function' && legacyName != null && legacyName.length > 0
        ? legacyName
        : undefined;
    const name =
      (id != null ? names.get(id) : undefined) ??
      (message.name != null && message.name.length > 0
        ? message.name
        : pendingName) ??
      'unknown_tool';
    if (type === 'function') {
      legacyName = undefined;
    }
    if (tokens > 0) {
      counts[name] = safeCount((counts[name] ?? 0) + tokens);
      resultTotal = safeCount(resultTotal + tokens);
    }
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

/** Reconciles derived budget fields and, when supplied, the retained tool share.
 * The index map is keyed before pruning, so it cannot index the retained context.
 * Callers may pass the existing exact token cache's counter, never a stale index lookup. */
export function syncBudgetDerivedFields(
  usage: t.ContextUsageEvent,
  context?: readonly BaseMessage[],
  tokenCounter?: t.TokenCounter
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
