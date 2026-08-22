import { ToolMessage } from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderPayloadMeasurement } from '@/llm/prepareProviderRequest';
import type * as t from '@/types';

import {
  REPLY_PRIMER_TOKENS,
  isSyntheticProviderContextMessage,
} from '@/messages';
import { apportionTokenCounts } from '@/utils';

interface ContextPressureUsage {
  contextBudget?: number;
  effectiveInstructionTokens?: number;
  remainingContextTokens?: number;
  calibrationRatio?: number;
}

interface ContextPressureMeterParams {
  tokenCounter?: t.TokenCounter;
  sourceMessages: BaseMessage[];
  retainedMessages: BaseMessage[];
  indexTokenCountMap: Record<string, number | undefined>;
  contextUsage?: ContextPressureUsage | null;
  instructionTokens: number;
  calibrationRatio: number;
}

interface ProviderPayloadMeasureOptions {
  contextBudget?: number;
  forceRawRecount?: boolean;
}

interface ProviderMessageBaseline {
  rawTokens: number;
  accountingWeight: number;
}

interface ProviderBaselineAttribution {
  attributedByOrigin: number[];
  projectedBaseTokens: number;
}

export interface ContextPressureMeter {
  trackProjection(before: BaseMessage[], after: BaseMessage[]): BaseMessage[];
  trackClone(source: BaseMessage, clone: BaseMessage): void;
  measure(
    messages: BaseMessage[],
    options?: ProviderPayloadMeasureOptions
  ): ProviderPayloadMeasurement;
}

function getProviderMessageOriginKey(message: BaseMessage): string | undefined {
  const type = message.getType();
  if (
    message instanceof ToolMessage &&
    typeof message.tool_call_id === 'string' &&
    message.tool_call_id.length > 0
  ) {
    return `tool:call:${message.tool_call_id}`;
  }
  if (typeof message.id === 'string' && message.id.length > 0) {
    return `${type}:id:${message.id}`;
  }
  return undefined;
}

/** Measures repeated provider projections while tokenizing each message object once. */
export function createContextPressureMeter({
  tokenCounter,
  sourceMessages,
  retainedMessages,
  indexTokenCountMap,
  contextUsage,
  instructionTokens,
  calibrationRatio,
}: ContextPressureMeterParams): ContextPressureMeter {
  const tokenCounts = new WeakMap<BaseMessage, number>();
  const origins = new WeakMap<BaseMessage, number>();
  const baselineAttributions = new Map<number, ProviderBaselineAttribution>();
  let baseline: ProviderMessageBaseline[] | undefined;
  let baselineWeights: Record<string, number> | undefined;
  let totalBaselineWeight = 0;

  const count = (message: BaseMessage): number => {
    const cached = tokenCounts.get(message);
    if (cached != null) {
      return cached;
    }
    const tokens = tokenCounter?.(message) ?? 0;
    tokenCounts.set(message, tokens);
    return tokens;
  };

  if (contextUsage != null && tokenCounter != null) {
    const sourceIndices = new WeakMap<BaseMessage, number>();
    for (let i = 0; i < sourceMessages.length; i++) {
      sourceIndices.set(sourceMessages[i], i);
    }
    baseline = retainedMessages.map((message, index) => {
      const rawTokens = count(message);
      const sourceIndex = sourceIndices.get(message);
      const indexedTokens =
        sourceIndex != null ? indexTokenCountMap[sourceIndex] : undefined;
      const accountingWeight =
        indexedTokens != null &&
        Number.isFinite(indexedTokens) &&
        indexedTokens >= 0
          ? indexedTokens
          : rawTokens;
      if (!origins.has(message)) {
        origins.set(message, index);
      }
      return { rawTokens, accountingWeight };
    });
    baselineWeights = {};
    for (let i = 0; i < baseline.length; i++) {
      baselineWeights[i] = baseline[i].accountingWeight;
      totalBaselineWeight += baseline[i].accountingWeight;
    }
  }

  const trackClone = (source: BaseMessage, clone: BaseMessage): void => {
    const origin = origins.get(source);
    if (origin != null && !origins.has(clone)) {
      origins.set(clone, origin);
    }
  };

  const trackProjection = (
    before: BaseMessage[],
    after: BaseMessage[]
  ): BaseMessage[] => {
    if (baseline == null || before === after) {
      return after;
    }
    if (before.length === after.length) {
      for (let i = 0; i < after.length; i++) {
        if (
          before[i].getType() === after[i].getType() &&
          !isSyntheticProviderContextMessage(after[i])
        ) {
          trackClone(before[i], after[i]);
        }
      }
      return after;
    }

    const keyedOrigins = new Map<string, number | null>();
    for (const message of before) {
      const origin = origins.get(message);
      const key = getProviderMessageOriginKey(message);
      if (origin == null || key == null) {
        continue;
      }
      keyedOrigins.set(key, keyedOrigins.has(key) ? null : origin);
    }
    for (const message of after) {
      if (origins.has(message) || isSyntheticProviderContextMessage(message)) {
        continue;
      }
      const key = getProviderMessageOriginKey(message);
      const origin = key != null ? keyedOrigins.get(key) : undefined;
      if (origin != null) {
        origins.set(message, origin);
      }
    }
    return after;
  };

  const measure = (
    messages: BaseMessage[],
    options?: ProviderPayloadMeasureOptions
  ): ProviderPayloadMeasurement => {
    const contextBudget = options?.contextBudget ?? contextUsage?.contextBudget;
    const forceRawRecount = options?.forceRawRecount === true;
    const effectiveInstructionTokens =
      contextUsage?.effectiveInstructionTokens ??
      (forceRawRecount ? instructionTokens : undefined);
    if (
      tokenCounter == null ||
      contextBudget == null ||
      effectiveInstructionTokens == null
    ) {
      return { fits: true };
    }
    const availableMessageTokens = Math.max(
      0,
      contextBudget - effectiveInstructionTokens
    );
    let usageRatio = calibrationRatio > 0 ? calibrationRatio : 1;
    if (
      contextUsage?.calibrationRatio != null &&
      contextUsage.calibrationRatio > 0
    ) {
      usageRatio = contextUsage.calibrationRatio;
    }
    if (forceRawRecount) {
      usageRatio = Math.max(1, usageRatio);
    }
    const baselineRemaining = contextUsage?.remainingContextTokens;
    const accountedMessageTokens =
      !forceRawRecount &&
      baseline != null &&
      baselineRemaining != null &&
      Number.isFinite(baselineRemaining)
        ? availableMessageTokens -
          Math.min(availableMessageTokens, Math.max(0, baselineRemaining))
        : undefined;

    let projectedMessageTokens: number;
    if (
      accountedMessageTokens != null &&
      baseline != null &&
      baselineWeights != null
    ) {
      let attribution = baselineAttributions.get(availableMessageTokens);
      if (attribution == null) {
        const replyPrimerTokens = Math.round(
          REPLY_PRIMER_TOKENS * usageRatio
        );
        const attributableTokens =
          totalBaselineWeight > 0
            ? Math.min(
              Math.max(0, accountedMessageTokens - replyPrimerTokens),
              Math.round(totalBaselineWeight * usageRatio)
            )
            : 0;
        const apportionedTokens =
          totalBaselineWeight > 0
            ? apportionTokenCounts(
              baselineWeights,
              attributableTokens / totalBaselineWeight,
              attributableTokens
            )
            : {};
        attribution = {
          attributedByOrigin: baseline.map(
            (_, origin) => apportionedTokens[origin] || 0
          ),
          projectedBaseTokens: Math.max(
            replyPrimerTokens,
            accountedMessageTokens - attributableTokens
          ),
        };
        baselineAttributions.set(availableMessageTokens, attribution);
      }
      projectedMessageTokens = attribution.projectedBaseTokens;
      let newRawTokens = 0;
      const usedOrigins = new Set<number>();
      for (const message of messages) {
        const rawTokens = count(message);
        const origin = origins.get(message);
        if (origin == null || usedOrigins.has(origin)) {
          newRawTokens += rawTokens;
          continue;
        }
        usedOrigins.add(origin);
        projectedMessageTokens += Math.max(
          0,
          attribution.attributedByOrigin[origin] +
            Math.round((rawTokens - baseline[origin].rawTokens) * usageRatio)
        );
      }
      projectedMessageTokens += Math.round(newRawTokens * usageRatio);
    } else {
      let rawTokens = REPLY_PRIMER_TOKENS;
      for (const message of messages) {
        rawTokens += count(message);
      }
      projectedMessageTokens = Math.round(rawTokens * usageRatio);
    }
    return {
      fits: projectedMessageTokens <= availableMessageTokens,
      projectedMessageTokens,
      availableMessageTokens,
      contextBudget,
      effectiveInstructionTokens,
    };
  };

  return { trackProjection, trackClone, measure };
}
