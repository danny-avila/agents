import {
  SystemMessage,
  ToolMessage,
  isAIMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { TokenCounter } from '@/types';
import { Constants } from '@/common';

export const DEFAULT_SUMMARIZATION_CONTEXT_TOKENS = 32_000;
export const MAX_SUMMARIZATION_CHUNKS = 8;

const DEFAULT_OUTPUT_RESERVE_RATIO = 0.05;
const MESSAGE_ENVELOPE_TOKENS = 8;

export interface SummarizationInputBudget {
  contextTokens: number;
  fixedOverheadTokens: number;
  outputReserveTokens: number;
  messageBudgetTokens: number;
}

export interface SummarizationChunkPlan {
  chunks: BaseMessage[][];
  estimatedMessageTokens: number;
  chunkTokenEstimates: number[];
}

export type SummarizationChunkResult =
  | { plan: SummarizationChunkPlan; error?: never }
  | { plan?: never; error: string; estimatedMessageTokens: number };

interface SerializedToolCall {
  id?: unknown;
  name?: unknown;
  function?: { name?: unknown };
}

interface ToolCallMetadata {
  tool_calls?: SerializedToolCall[];
}

interface ClientToolCall {
  id: string;
  name: string;
}

function estimateContentLength(message: BaseMessage): number {
  let serialized = '';
  try {
    serialized = JSON.stringify({
      type: message.getType(),
      content: message.content,
      additional_kwargs: message.additional_kwargs,
      tool_calls: isAIMessage(message) ? message.tool_calls : undefined,
      tool_call_id:
        message instanceof ToolMessage ? message.tool_call_id : undefined,
    });
  } catch {
    serialized = String(message.content);
  }
  return serialized.length;
}

function serializeMetadata(message: BaseMessage): string {
  try {
    return JSON.stringify({
      additional_kwargs:
        Object.keys(message.additional_kwargs).length > 0
          ? message.additional_kwargs
          : undefined,
      tool_calls: isAIMessage(message) ? message.tool_calls : undefined,
      tool_call_id:
        message instanceof ToolMessage ? message.tool_call_id : undefined,
      name: message.name || undefined,
    });
  } catch {
    return String(message.additional_kwargs);
  }
}

export function estimateMessageTokens(
  message: BaseMessage,
  tokenCounter?: TokenCounter
): number {
  let counted = 0;
  if (tokenCounter) {
    try {
      const value = tokenCounter(message);
      if (Number.isFinite(value) && value > 0) {
        counted = Math.ceil(value);
      }
    } catch {
      counted = 0;
    }
  }

  let contentTokens = estimateContentLength(message);
  if (counted > 0) {
    const serializedMetadata = serializeMetadata(message);
    let metadataTokens = serializedMetadata.length;
    if (tokenCounter) {
      try {
        const value = tokenCounter(new SystemMessage(serializedMetadata));
        if (Number.isFinite(value) && value >= 0) {
          metadataTokens = Math.ceil(value);
        }
      } catch {
        // Keep the conservative character-count fallback.
      }
    }
    contentTokens = counted + metadataTokens;
  }
  return contentTokens + MESSAGE_ENVELOPE_TOKENS;
}

export function estimateMessagesTokens(
  messages: BaseMessage[],
  tokenCounter?: TokenCounter
): number {
  let total = 0;
  for (const message of messages) {
    total += estimateMessageTokens(message, tokenCounter);
  }
  return total;
}

export function createSummarizationInputBudget(params: {
  maxContextTokens?: number;
  fixedOverheadTokens: number;
  maxSummaryTokens?: number;
  reserveRatio?: number;
  calibrationRatio?: number;
}): SummarizationInputBudget {
  const contextTokens =
    params.maxContextTokens != null &&
    Number.isFinite(params.maxContextTokens) &&
    params.maxContextTokens > 0
      ? Math.floor(params.maxContextTokens)
      : DEFAULT_SUMMARIZATION_CONTEXT_TOKENS;
  const configuredReserve =
    params.maxSummaryTokens != null &&
    Number.isFinite(params.maxSummaryTokens) &&
    params.maxSummaryTokens > 0
      ? Math.ceil(params.maxSummaryTokens)
      : 0;
  const reserveRatio =
    params.reserveRatio != null &&
    Number.isFinite(params.reserveRatio) &&
    params.reserveRatio >= 0 &&
    params.reserveRatio < 1
      ? params.reserveRatio
      : DEFAULT_OUTPUT_RESERVE_RATIO;
  const ratioReserve = Math.max(1, Math.ceil(contextTokens * reserveRatio));
  const outputReserveTokens = Math.max(configuredReserve, ratioReserve);
  const fixedOverheadTokens = Math.max(
    0,
    Math.ceil(params.fixedOverheadTokens)
  );
  const calibrationRatio =
    params.calibrationRatio != null &&
    Number.isFinite(params.calibrationRatio) &&
    params.calibrationRatio > 0
      ? params.calibrationRatio
      : 1;

  return {
    contextTokens,
    fixedOverheadTokens,
    outputReserveTokens,
    messageBudgetTokens: Math.max(
      0,
      Math.floor(
        (contextTokens - outputReserveTokens - fixedOverheadTokens) /
          calibrationRatio
      )
    ),
  };
}

function getClientToolCalls(message: BaseMessage): ClientToolCall[] {
  if (!isAIMessage(message)) {
    return [];
  }
  const calls = new Map<string, ClientToolCall>();
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (
        typeof call.id === 'string' &&
        call.id !== '' &&
        !call.id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
      ) {
        calls.set(call.id, { id: call.id, name: call.name });
      }
    }
  }
  const serializedCalls = (message.additional_kwargs as ToolCallMetadata)
    .tool_calls;
  if (Array.isArray(serializedCalls)) {
    for (const call of serializedCalls) {
      if (call == null || typeof call !== 'object') {
        continue;
      }
      const id = (call as { id?: unknown }).id;
      if (
        typeof id === 'string' &&
        id !== '' &&
        !id.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX)
      ) {
        const serializedName =
          typeof call.name === 'string'
            ? call.name
            : typeof call.function?.name === 'string'
              ? call.function.name
              : 'interrupted_tool';
        calls.set(id, { id, name: serializedName });
      }
    }
  }
  return [...calls.values()];
}

export function repairInterruptedToolCalls(
  messages: BaseMessage[]
): BaseMessage[] {
  const toolResultIds = new Set<string>();
  for (const message of messages) {
    if (
      message instanceof ToolMessage &&
      typeof message.tool_call_id === 'string' &&
      message.tool_call_id !== ''
    ) {
      toolResultIds.add(message.tool_call_id);
    }
  }

  const repaired: BaseMessage[] = [];
  for (const message of messages) {
    repaired.push(message);
    for (const call of getClientToolCalls(message)) {
      if (toolResultIds.has(call.id)) {
        continue;
      }
      repaired.push(
        new ToolMessage({
          content: 'Tool execution was interrupted before producing a result.',
          tool_call_id: call.id,
          name: call.name,
        })
      );
      toolResultIds.add(call.id);
    }
  }
  return repaired;
}

function buildAtomicMessageUnits(messages: BaseMessage[]): BaseMessage[][] {
  const callStarts = new Map<string, number>();
  const callResults = new Map<string, number>();

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i] as BaseMessage;
    for (const call of getClientToolCalls(message)) {
      if (!callStarts.has(call.id)) {
        callStarts.set(call.id, i);
      }
    }
    if (
      message instanceof ToolMessage &&
      typeof message.tool_call_id === 'string' &&
      message.tool_call_id !== '' &&
      !callResults.has(message.tool_call_id)
    ) {
      callResults.set(message.tool_call_id, i);
    }
  }

  const boundaryDelta = new Int32Array(messages.length + 1);
  for (const [id, start] of callStarts) {
    const result = callResults.get(id);
    const end =
      result != null && result >= start ? result : messages.length - 1;
    if (end <= start) {
      continue;
    }
    boundaryDelta[start] += 1;
    boundaryDelta[end] -= 1;
  }

  const units: BaseMessage[][] = [];
  let current: BaseMessage[] = [];
  let protectedBoundaries = 0;
  for (let i = 0; i < messages.length; i++) {
    protectedBoundaries += boundaryDelta[i] as number;
    current.push(messages[i] as BaseMessage);
    if (protectedBoundaries === 0) {
      units.push(current);
      current = [];
    }
  }
  if (current.length > 0) {
    units.push(current);
  }
  return units;
}

export function planSummarizationChunks(params: {
  messages: BaseMessage[];
  messageBudgetTokens: number;
  tokenCounter?: TokenCounter;
  maxChunks?: number;
}): SummarizationChunkResult {
  if (params.messageBudgetTokens <= 0) {
    return {
      error:
        'Summarization instructions and output reserve exhaust the model context window',
      estimatedMessageTokens: 0,
    };
  }

  const units = buildAtomicMessageUnits(
    repairInterruptedToolCalls(params.messages)
  );
  const unitTokenEstimates: number[] = [];
  let estimatedMessageTokens = 0;
  let oversizedUnitTokens = 0;
  for (const unit of units) {
    const unitTokens = estimateMessagesTokens(unit, params.tokenCounter);
    unitTokenEstimates.push(unitTokens);
    estimatedMessageTokens += unitTokens;
    oversizedUnitTokens = Math.max(oversizedUnitTokens, unitTokens);
  }
  if (oversizedUnitTokens > params.messageBudgetTokens) {
    return {
      error: `A complete message/tool-call unit requires ${oversizedUnitTokens} tokens but the summarizer input budget is ${params.messageBudgetTokens}`,
      estimatedMessageTokens,
    };
  }

  const chunks: BaseMessage[][] = [];
  const chunkTokenEstimates: number[] = [];
  let current: BaseMessage[] = [];
  let currentTokens = 0;

  for (let i = 0; i < units.length; i++) {
    const unit = units[i] as BaseMessage[];
    const unitTokens = unitTokenEstimates[i] as number;
    if (
      current.length > 0 &&
      currentTokens + unitTokens > params.messageBudgetTokens
    ) {
      chunks.push(current);
      chunkTokenEstimates.push(currentTokens);
      current = [];
      currentTokens = 0;
    }
    current.push(...unit);
    currentTokens += unitTokens;
  }

  if (current.length > 0) {
    chunks.push(current);
    chunkTokenEstimates.push(currentTokens);
  }

  const maxChunks = params.maxChunks ?? MAX_SUMMARIZATION_CHUNKS;
  if (chunks.length > maxChunks) {
    return {
      error: `Summarization requires ${chunks.length} chunks, exceeding the bounded limit of ${maxChunks}`,
      estimatedMessageTokens,
    };
  }

  return {
    plan: { chunks, estimatedMessageTokens, chunkTokenEstimates },
  };
}
