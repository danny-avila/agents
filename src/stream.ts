// src/stream.ts
import type { ChatOpenAIReasoningSummary } from '@langchain/openai';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  ToolCallTypes,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
  Constants,
  CODE_EXECUTION_TOOLS,
  LOCAL_CODING_BUNDLE_NAMES,
} from '@/common';
import {
  handleServerToolResult,
  handleToolCallChunks,
  handleToolCalls,
} from '@/tools/handlers';
import { getMessageId } from '@/messages';
import { safeDispatchCustomEvent } from '@/utils/events';
import {
  buildToolExecutionRequestPlan,
  coerceRecordArgs,
  normalizeError,
} from '@/tools/eagerEventExecution';
import {
  calculateMaxToolResultChars,
  truncateToolResultContent,
} from '@/utils/truncation';
import {
  getStreamedToolCallSeal,
  getStreamedToolCallAdapter,
  type StreamedToolCallSeal,
} from '@/tools/streamedToolCallSeals';
import { TOOL_OUTPUT_REF_PATTERN } from '@/tools/toolOutputReferences';

const LOCAL_CODING_BUNDLE_NAME_SET: ReadonlySet<string> = new Set(
  LOCAL_CODING_BUNDLE_NAMES
);

/**
 * Parses content to extract thinking sections enclosed in <think> tags using string operations
 * @param content The content to parse
 * @returns An object with separated text and thinking content
 */
function parseThinkingContent(content: string): {
  text: string;
  thinking: string;
} {
  // If no think tags, return the original content as text
  if (!content.includes('<think>')) {
    return { text: content, thinking: '' };
  }

  let textResult = '';
  const thinkingResult: string[] = [];
  let position = 0;

  while (position < content.length) {
    const thinkStart = content.indexOf('<think>', position);

    if (thinkStart === -1) {
      // No more think tags, add the rest and break
      textResult += content.slice(position);
      break;
    }

    // Add text before the think tag
    textResult += content.slice(position, thinkStart);

    const thinkEnd = content.indexOf('</think>', thinkStart);
    if (thinkEnd === -1) {
      // Malformed input, no closing tag
      textResult += content.slice(thinkStart);
      break;
    }

    // Add the thinking content
    const thinkContent = content.slice(thinkStart + 7, thinkEnd);
    thinkingResult.push(thinkContent);

    // Move position to after the think tag
    position = thinkEnd + 8; // 8 is the length of '</think>'
  }

  return {
    text: textResult.trim(),
    thinking: thinkingResult.join('\n').trim(),
  };
}

function getNonEmptyValue(possibleValues: string[]): string | undefined {
  for (const value of possibleValues) {
    if (value && value.trim() !== '') {
      return value;
    }
  }
  return undefined;
}

function isBatchSensitiveToolExecution(graph: StandardGraph): boolean {
  return graph.hookRegistry != null || graph.humanInTheLoop?.enabled === true;
}

function hasToolOutputReference(value: unknown): boolean {
  if (typeof value === 'string') {
    return TOOL_OUTPUT_REF_PATTERN.test(value);
  }
  if (Array.isArray(value)) {
    return value.some((item) => hasToolOutputReference(item));
  }
  if (value !== null && typeof value === 'object') {
    return Object.values(value as Record<string, unknown>).some((item) =>
      hasToolOutputReference(item)
    );
  }
  return false;
}

function isDirectGraphTool(
  name: string,
  agentContext: AgentContext | undefined
): boolean {
  if (name.startsWith(Constants.LC_TRANSFER_TO_)) {
    return true;
  }
  return (
    (agentContext?.graphTools as t.GenericTool[] | undefined)?.some(
      (tool) => 'name' in tool && tool.name === name
    ) === true
  );
}

function isDirectLocalTool(name: string, graph: StandardGraph): boolean {
  const toolExecution = graph.toolExecution;
  const engine = toolExecution?.engine;
  if (
    toolExecution == null ||
    (engine !== 'local' && engine !== 'cloudflare-sandbox')
  ) {
    return false;
  }
  const includeCodingTools =
    engine === 'cloudflare-sandbox'
      ? toolExecution.cloudflare?.includeCodingTools
      : toolExecution.local?.includeCodingTools;
  if (includeCodingTools === false) {
    return CODE_EXECUTION_TOOLS.has(name);
  }
  return LOCAL_CODING_BUNDLE_NAME_SET.has(name);
}

function toCodeEnvFile(file: t.FileRef, execSessionId: string): t.CodeEnvFile {
  const base = {
    id: file.id,
    resource_id: file.resource_id ?? file.id,
    name: file.name,
    storage_session_id: file.storage_session_id ?? execSessionId,
  };
  const kind = file.kind ?? 'user';
  if (kind === 'skill' && file.version != null) {
    return { ...base, kind: 'skill', version: file.version };
  }
  if (kind === 'agent') {
    return { ...base, kind: 'agent' };
  }
  return { ...base, kind: 'user' };
}

function getCodeSessionContext(
  graph: StandardGraph,
  name: string
): t.ToolCallRequest['codeSessionContext'] | undefined {
  if (
    !CODE_EXECUTION_TOOLS.has(name) &&
    name !== Constants.SKILL_TOOL &&
    name !== Constants.READ_FILE
  ) {
    return undefined;
  }

  const codeSession = graph.sessions.get(Constants.EXECUTE_CODE) as
    | t.CodeSessionContext
    | undefined;
  if (codeSession?.session_id == null || codeSession.session_id === '') {
    return undefined;
  }

  return {
    session_id: codeSession.session_id,
    files: codeSession.files?.map((file) =>
      toCodeEnvFile(file, codeSession.session_id)
    ),
  };
}

function isEagerToolExecutionEnabledForBatch(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
}): boolean {
  const { graph, metadata, agentContext } = args;
  if (graph.eagerEventToolExecution?.enabled !== true) {
    return false;
  }
  if ((agentContext?.toolDefinitions?.length ?? 0) === 0) {
    return false;
  }
  if (isBatchSensitiveToolExecution(graph)) {
    return false;
  }
  if (
    metadata?.[Constants.PROGRAMMATIC_TOOL_CALLING] === true ||
    metadata?.[Constants.BASH_PROGRAMMATIC_TOOL_CALLING] === true
  ) {
    return false;
  }
  if (
    graph.handlerRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) == null &&
    graph.eventToolExecutionAvailable !== true
  ) {
    return false;
  }
  return true;
}

function hasFinalToolCallSignal(chunk: Partial<AIMessageChunk>): boolean {
  const metadata = chunk.response_metadata as
    | Record<string, unknown>
    | undefined;
  const finishReason =
    metadata?.finish_reason ??
    metadata?.finishReason ??
    metadata?.stop_reason ??
    metadata?.stopReason;
  return finishReason === 'tool_calls' || finishReason === 'tool_use';
}

function canPrestartSequentialStreamedToolChunks(
  agentContext: AgentContext | undefined
): boolean {
  // Anthropic seals each prior streamed tool-use block when the next indexed
  // tool-use block begins. Live Kimi/Moonshot streams can still revise prior
  // args after advancing to the next index, so keep those on the final
  // tool-call path unless they grow an explicit adapter seal.
  return agentContext?.provider === Providers.ANTHROPIC;
}

function hasExplicitStreamedToolCallSeals(
  chunk: Partial<AIMessageChunk>
): boolean {
  return (
    getStreamedToolCallAdapter(
      chunk.response_metadata as Record<string, unknown> | undefined
    ) != null
  );
}

function hasDirectToolCallInBatch(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
}): boolean {
  const { graph, agentContext, toolCalls } = args;
  return toolCalls.some(
    (toolCall) =>
      toolCall.name !== '' &&
      (isDirectGraphTool(toolCall.name, agentContext) ||
        isDirectLocalTool(toolCall.name, graph))
  );
}

function hasPotentialDirectToolInStreamContext(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
}): boolean {
  const { graph, agentContext } = args;
  const engine = graph.toolExecution?.engine;
  if (engine === 'local' || engine === 'cloudflare-sandbox') {
    return true;
  }
  if ((agentContext?.graphTools?.length ?? 0) > 0) {
    return true;
  }
  return false;
}

function hasDirectToolCallChunkInBatch(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  toolCallChunks?: ToolCallChunk[];
}): boolean {
  const { graph, agentContext, toolCallChunks } = args;
  return (
    toolCallChunks?.some(
      (toolCallChunk) =>
        toolCallChunk.name != null &&
        toolCallChunk.name !== '' &&
        (isDirectGraphTool(toolCallChunk.name, agentContext) ||
          isDirectLocalTool(toolCallChunk.name, graph))
    ) === true
  );
}

function hasDirectToolCallChunkStateInStep(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  stepKey: string;
}): boolean {
  const { graph, agentContext, stepKey } = args;
  const prefix = `${stepKey}\u0000`;
  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    const name = state.name;
    if (
      name != null &&
      name !== '' &&
      (isDirectGraphTool(name, agentContext) || isDirectLocalTool(name, graph))
    ) {
      return true;
    }
  }
  return false;
}

type EagerToolExecutionEntry = {
  id: string;
  toolName: string;
  coercedArgs: Record<string, unknown>;
  request: t.ToolCallRequest;
};

function createEagerToolExecutionPlan(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
  skipExisting?: boolean;
}): EagerToolExecutionEntry[] | undefined {
  const {
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting = false,
  } = args;
  if (
    !isEagerToolExecutionEnabledForBatch({
      graph,
      metadata,
      agentContext,
    })
  ) {
    return undefined;
  }

  if (hasDirectToolCallInBatch({ graph, agentContext, toolCalls })) {
    return undefined;
  }
  if (
    graph.toolOutputReferences?.enabled === true &&
    toolCalls.some((toolCall) => hasToolOutputReference(toolCall.args))
  ) {
    return undefined;
  }

  const candidateToolCalls = skipExisting
    ? toolCalls.filter((toolCall) => {
      if (toolCall.id == null || toolCall.id === '') {
        return true;
      }
      return !graph.eagerEventToolExecutions.has(toolCall.id);
    })
    : toolCalls;
  if (candidateToolCalls.length === 0) {
    return [];
  }

  // Eager execution must preserve ToolNode batch semantics exactly for every
  // unstarted call. If any candidate cannot be planned, fall back for that
  // candidate set.
  if (
    candidateToolCalls.some(
      (toolCall) =>
        toolCall.id == null ||
        toolCall.id === '' ||
        toolCall.name === '' ||
        (!skipExisting && graph.eagerEventToolExecutions.has(toolCall.id))
    )
  ) {
    return undefined;
  }

  const plan = buildToolExecutionRequestPlan({
    toolCalls: candidateToolCalls.map((toolCall) => ({
      id: toolCall.id,
      name: toolCall.name,
      args: toolCall.args,
      stepId: graph.toolCallStepIds.get(toolCall.id!) ?? '',
      codeSessionContext: getCodeSessionContext(graph, toolCall.name),
    })),
    usageCount: graph.getEagerEventToolUsageCount(agentContext?.agentId),
  });
  if (plan == null) {
    return undefined;
  }

  return plan.requests.map(
    (request): EagerToolExecutionEntry => ({
      id: request.id,
      toolName: request.name,
      coercedArgs: request.args,
      request,
    })
  );
}

function startEagerToolExecutions(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  toolCalls: ToolCall[];
  skipExisting?: boolean;
}): void {
  const { graph, metadata, agentContext, toolCalls, skipExisting } = args;
  const entries = createEagerToolExecutionPlan({
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting,
  });
  if (entries == null || entries.length === 0) {
    return;
  }

  const records: t.EagerEventToolExecution[] = [];
  const promise: Promise<t.EagerEventToolExecutionOutcome> = new Promise<
    t.ToolExecuteResult[]
  >((resolve, reject) => {
    let dispatchSettled = false;
    let resultSettled = false;
    let settledResults: t.ToolExecuteResult[] | undefined;
    const maybeResolve = (): void => {
      if (dispatchSettled && resultSettled) {
        resolve(settledResults ?? []);
      }
    };
    const batchRequest: t.ToolExecuteBatchRequest = {
      toolCalls: entries.map((entry) => entry.request),
      userId: graph.config?.configurable?.user_id as string | undefined,
      agentId: agentContext?.agentId,
      configurable: graph.config?.configurable as
        | Record<string, unknown>
        | undefined,
      metadata,
      resolve: (results): void => {
        resultSettled = true;
        settledResults = results;
        maybeResolve();
      },
      reject,
    };

    void safeDispatchCustomEvent(
      GraphEvents.ON_TOOL_EXECUTE,
      batchRequest,
      graph.config
    )
      .then(() => {
        dispatchSettled = true;
        maybeResolve();
      })
      .catch(reject);
  }).then(
    async (results): Promise<t.EagerEventToolExecutionOutcome> => {
      await dispatchEagerToolCompletions({
        graph,
        agentContext,
        records,
        results,
      });
      return { results };
    },
    (error): t.EagerEventToolExecutionOutcome => ({
      error: normalizeError(error),
    })
  );

  for (const entry of entries) {
    const record: t.EagerEventToolExecution = {
      toolCallId: entry.id,
      toolName: entry.toolName,
      args: entry.coercedArgs,
      request: entry.request,
      promise,
    };
    records.push(record);
    graph.eagerEventToolExecutions.set(entry.id, record);
  }
}

async function dispatchEagerToolCompletions(args: {
  graph: StandardGraph;
  agentContext?: AgentContext;
  records: t.EagerEventToolExecution[];
  results: t.ToolExecuteResult[];
}): Promise<void> {
  const { graph, agentContext, records, results } = args;
  const recordById = new Map(
    records.map((record) => [record.toolCallId, record])
  );
  const maxToolResultChars =
    agentContext?.maxToolResultChars ??
    calculateMaxToolResultChars(agentContext?.maxContextTokens);

  for (const result of results) {
    const record = recordById.get(result.toolCallId);
    if (record == null) {
      continue;
    }
    if (graph.eagerEventToolExecutions.get(result.toolCallId) !== record) {
      continue;
    }
    const stepId =
      record.request.stepId ??
      graph.toolCallStepIds.get(result.toolCallId) ??
      '';
    if (stepId === '') {
      continue;
    }
    const output =
      result.status === 'error'
        ? `Error: ${result.errorMessage ?? 'Unknown error'}\n Please fix your mistakes.`
        : truncateToolResultContent(
          typeof result.content === 'string'
            ? result.content
            : JSON.stringify(result.content),
          maxToolResultChars
        );

    try {
      const dispatched = await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_COMPLETED,
        {
          result: {
            id: stepId,
            index: record.request.turn ?? 0,
            type: 'tool_call' as const,
            eager: true,
            tool_call: {
              args: JSON.stringify(record.request.args),
              name: record.toolName,
              id: result.toolCallId,
              output,
              progress: 1,
            } as t.ProcessedToolCall,
          },
        },
        graph.config
      );
      if (dispatched === false) {
        continue;
      }
      record.completionDispatched = true;
    } catch (error) {
      // Let ToolNode dispatch the completion through the normal path later.

      console.warn(
        `[stream] eager completion dispatch failed for toolCallId=${result.toolCallId}:`,
        error instanceof Error ? error.message : error
      );
    }
  }
}

function getEagerToolChunkKey(
  stepKey: string,
  toolCallChunk: ToolCallChunk
): string | undefined {
  let chunkKey: string | undefined;
  if (typeof toolCallChunk.index === 'number') {
    chunkKey = String(toolCallChunk.index);
  } else if (toolCallChunk.id != null && toolCallChunk.id !== '') {
    chunkKey = toolCallChunk.id;
  }
  if (chunkKey == null) {
    return undefined;
  }
  return `${stepKey}\u0000${chunkKey}`;
}

function getEagerToolChunkIndex(
  toolCallChunk: ToolCallChunk
): number | undefined {
  return typeof toolCallChunk.index === 'number'
    ? toolCallChunk.index
    : undefined;
}

function pruneEagerToolCallChunkStates(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallIds?: ReadonlySet<string>;
  clearStep?: boolean;
}): void {
  const { graph, stepKey, toolCallIds, clearStep = false } = args;
  const prefix = `${stepKey}\u0000`;
  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    if (
      clearStep ||
      (state.id != null && toolCallIds?.has(state.id) === true)
    ) {
      graph.eagerEventToolCallChunks.delete(key);
    }
  }
}

function isEagerToolChunkStateComplete(
  state: t.EagerEventToolCallChunkState
): boolean {
  return (
    state.id != null &&
    state.id !== '' &&
    state.name != null &&
    state.name !== '' &&
    coerceRecordArgs(state.argsText) != null
  );
}

function mergeToolCallArgsText(existing: string, incoming: string): string {
  if (incoming === '') {
    return existing;
  }
  if (existing === '') {
    return incoming;
  }
  if (incoming === existing) {
    try {
      JSON.parse(incoming);
      return incoming;
    } catch {
      return `${existing}${incoming}`;
    }
  }
  if (incoming.startsWith(existing)) {
    return incoming;
  }
  if (existing.startsWith(incoming)) {
    return existing;
  }
  try {
    JSON.parse(existing);
    JSON.parse(incoming);
    return incoming;
  } catch {
    // Fall through to delta concatenation.
  }
  for (
    let overlap = Math.min(existing.length, incoming.length);
    overlap >= 8;
    overlap -= 1
  ) {
    if (existing.endsWith(incoming.slice(0, overlap))) {
      return `${existing}${incoming.slice(overlap)}`;
    }
  }
  return `${existing}${incoming}`;
}

function recordEagerToolCallChunks(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
}): void {
  const { graph, stepKey, toolCallChunks } = args;
  if (toolCallChunks == null || toolCallChunks.length === 0) {
    return;
  }

  // Streamed args can be cumulative and parseable before the provider has
  // sealed the call. Recording stays separate from dispatch so the boundary
  // logic can wait for either a later tool index or the final tool-call signal.
  for (const toolCallChunk of toolCallChunks) {
    const key = getEagerToolChunkKey(stepKey, toolCallChunk);
    if (key == null) {
      continue;
    }

    const incomingId =
      toolCallChunk.id != null && toolCallChunk.id !== ''
        ? toolCallChunk.id
        : undefined;
    const incomingName =
      toolCallChunk.name != null && toolCallChunk.name !== ''
        ? toolCallChunk.name
        : undefined;
    const previous = graph.eagerEventToolCallChunks.get(key);
    const shouldReset =
      previous != null &&
      ((incomingId != null &&
        previous.id != null &&
        incomingId !== previous.id) ||
        (incomingName != null &&
          previous.name != null &&
          incomingName !== previous.name));
    const existing =
      previous == null || shouldReset
        ? {
          argsText: '',
        }
        : previous;
    const id = incomingId ?? existing.id;
    const name = incomingName ?? existing.name;
    const incomingArgs = toolCallChunk.args ?? '';
    const isRepeatedObservedFragment =
      incomingArgs !== '' &&
      incomingArgs.length > 1 &&
      incomingArgs === existing.lastArgsFragment;
    const argsText = isRepeatedObservedFragment
      ? existing.argsText
      : mergeToolCallArgsText(existing.argsText, incomingArgs);
    const next = {
      id,
      name,
      argsText,
      index: getEagerToolChunkIndex(toolCallChunk) ?? existing.index,
      lastArgsFragment:
        incomingArgs !== '' ? incomingArgs : existing.lastArgsFragment,
    };
    graph.eagerEventToolCallChunks.set(key, next);
  }
}

function getStreamedReadyToolCalls(args: {
  graph: StandardGraph;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
  seal?: StreamedToolCallSeal;
  allowSequentialSeal?: boolean;
  sealAll?: boolean;
}): ToolCall[] {
  const {
    graph,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal = false,
    sealAll = false,
  } = args;
  const currentIndices = new Set<number>();
  for (const toolCallChunk of toolCallChunks ?? []) {
    const index = getEagerToolChunkIndex(toolCallChunk);
    if (index != null) {
      currentIndices.add(index);
    }
  }
  const highestCurrentIndex =
    currentIndices.size > 0 ? Math.max(...currentIndices) : undefined;
  const prefix = `${stepKey}\u0000`;
  const readyEntries: Array<{
    key: string;
    state: t.EagerEventToolCallChunkState;
  }> = [];

  for (const [key, state] of graph.eagerEventToolCallChunks) {
    if (!key.startsWith(prefix)) {
      continue;
    }
    if (state.id != null && graph.eagerEventToolExecutions.has(state.id)) {
      graph.eagerEventToolCallChunks.delete(key);
      continue;
    }
    if (!isEagerToolChunkStateComplete(state)) {
      continue;
    }
    const isSealedByLaterChunk =
      allowSequentialSeal &&
      highestCurrentIndex != null &&
      state.index != null &&
      state.index < highestCurrentIndex &&
      !currentIndices.has(state.index);
    const isSealedExplicitly =
      seal?.kind === 'single' &&
      ((seal.id != null && state.id === seal.id) ||
        (seal.index != null && state.index === seal.index));
    if (
      sealAll ||
      seal?.kind === 'all' ||
      isSealedByLaterChunk ||
      isSealedExplicitly
    ) {
      readyEntries.push({ key, state });
    }
  }

  pruneEagerToolCallChunkStates({
    graph,
    stepKey,
    toolCallIds: new Set(
      readyEntries
        .map(({ state }) => state.id)
        .filter((id): id is string => id != null && id !== '')
    ),
  });
  if (sealAll) {
    pruneEagerToolCallChunkStates({ graph, stepKey, clearStep: true });
  }

  return readyEntries
    .sort((left, right) => (left.state.index ?? 0) - (right.state.index ?? 0))
    .flatMap(({ state }) => {
      const args = coerceRecordArgs(state.argsText);
      if (args == null) {
        return [];
      }
      return [
        {
          id: state.id,
          name: state.name ?? '',
          args,
        },
      ];
    });
}

function startReadyStreamedEagerToolExecutions(args: {
  graph: StandardGraph;
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
  stepKey: string;
  toolCallChunks?: ToolCallChunk[];
  seal?: StreamedToolCallSeal;
  allowSequentialSeal?: boolean;
  sealAll?: boolean;
}): void {
  const {
    graph,
    metadata,
    agentContext,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal,
    sealAll,
  } = args;
  if (
    hasPotentialDirectToolInStreamContext({ graph, agentContext }) ||
    hasDirectToolCallChunkInBatch({ graph, agentContext, toolCallChunks }) ||
    hasDirectToolCallChunkStateInStep({ graph, agentContext, stepKey }) ||
    !isEagerToolExecutionEnabledForBatch({ graph, metadata, agentContext })
  ) {
    return;
  }
  const toolCalls = getStreamedReadyToolCalls({
    graph,
    stepKey,
    toolCallChunks,
    seal,
    allowSequentialSeal,
    sealAll,
  });
  if (toolCalls.length === 0) {
    return;
  }
  startEagerToolExecutions({
    graph,
    metadata,
    agentContext,
    toolCalls,
    skipExisting: true,
  });
}

export function getChunkContent({
  chunk,
  provider,
  reasoningKey,
}: {
  chunk?: Partial<AIMessageChunk>;
  provider?: Providers;
  reasoningKey: 'reasoning_content' | 'reasoning';
}): string | t.MessageContentComplex[] | undefined {
  if (
    (provider === Providers.OPENAI || provider === Providers.AZURE) &&
    (
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text != null &&
    ((
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text?.length ?? 0) > 0
  ) {
    return (
      chunk?.additional_kwargs?.reasoning as
        | Partial<ChatOpenAIReasoningSummary>
        | undefined
    )?.summary?.[0]?.text;
  }
  /**
   * For OpenRouter, reasoning is stored in additional_kwargs.reasoning (not reasoning_content).
   * NOTE: We intentionally do NOT extract text from reasoning_details here.
   * The reasoning_details array contains the FULL accumulated reasoning text (set only on final chunk),
   * but individual reasoning tokens are already streamed via additional_kwargs.reasoning.
   * Extracting from reasoning_details would cause duplication.
   * The reasoning_details is only used for:
   * 1. Detecting reasoning mode in handleReasoning()
   * 2. Final message storage (for thought signatures)
   */
  if (provider === Providers.OPENROUTER) {
    // Content presence signals end of reasoning phase - prefer content over reasoning
    // This handles transitional chunks that may have both reasoning and content
    if (typeof chunk?.content === 'string' && chunk.content !== '') {
      return chunk.content;
    }
    const reasoning = chunk?.additional_kwargs?.reasoning as string | undefined;
    if (reasoning != null && reasoning !== '') {
      return reasoning;
    }
    return chunk?.content;
  }
  return (
    ((chunk?.additional_kwargs?.[reasoningKey] as string | undefined) ?? '') ||
    chunk?.content
  );
}

export class ChatModelStreamHandler implements t.EventHandler {
  async handle(
    event: string,
    data: t.StreamEventData,
    metadata?: Record<string, unknown>,
    graph?: StandardGraph
  ): Promise<void> {
    if (!graph) {
      throw new Error('Graph not found');
    }
    if (!graph.config) {
      throw new Error('Config not found in graph');
    }

    if (!data.chunk) {
      console.warn(`No chunk found in ${event} event`);
      return;
    }

    const agentContext = graph.getAgentContext(metadata);

    const chunk = data.chunk as Partial<AIMessageChunk>;

    const content = getChunkContent({
      chunk,
      reasoningKey: agentContext.reasoningKey,
      provider: agentContext.provider,
    });
    const skipHandling = await handleServerToolResult({
      graph,
      content,
      metadata,
      agentContext,
    });
    if (skipHandling) {
      return;
    }
    this.handleReasoning(chunk, agentContext);
    const stepKey = graph.getStepKey(metadata);
    let hasToolCalls = false;
    const hasToolCallChunks =
      (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) ?? false;
    if (
      chunk.tool_calls &&
      chunk.tool_calls.length > 0 &&
      chunk.tool_calls.every(
        (tc) =>
          tc.id != null &&
          tc.id !== '' &&
          (tc as Partial<ToolCall>).name != null &&
          tc.name !== ''
      )
    ) {
      hasToolCalls = true;
      await handleToolCalls(chunk.tool_calls, metadata, graph);
      if (hasFinalToolCallSignal(chunk)) {
        startEagerToolExecutions({
          graph,
          metadata,
          agentContext,
          toolCalls: chunk.tool_calls,
          skipExisting: true,
        });
        if (!hasToolCallChunks) {
          pruneEagerToolCallChunkStates({ graph, stepKey, clearStep: true });
        }
      }
    }

    const isEmptyContent =
      typeof content === 'undefined' ||
      !content.length ||
      (typeof content === 'string' && !content);

    /** Set a preliminary message ID if found in empty chunk */
    const isEmptyChunk = isEmptyContent && !hasToolCallChunks;
    if (
      isEmptyChunk &&
      (chunk.id ?? '') !== '' &&
      !graph.prelimMessageIdsByStepKey.has(chunk.id ?? '')
    ) {
      graph.prelimMessageIdsByStepKey.set(stepKey, chunk.id ?? '');
    } else if (isEmptyChunk) {
      return;
    }

    if (
      hasToolCallChunks &&
      chunk.tool_call_chunks &&
      chunk.tool_call_chunks.length &&
      typeof chunk.tool_call_chunks[0]?.index === 'number'
    ) {
      const streamedToolCallSeal = getStreamedToolCallSeal(
        chunk.response_metadata as Record<string, unknown> | undefined
      );
      const allowSequentialSeal =
        canPrestartSequentialStreamedToolChunks(agentContext);
      const canStreamEager =
        (allowSequentialSeal || hasExplicitStreamedToolCallSeals(chunk)) &&
        !hasPotentialDirectToolInStreamContext({ graph, agentContext }) &&
        isEagerToolExecutionEnabledForBatch({ graph, metadata, agentContext });
      if (canStreamEager) {
        recordEagerToolCallChunks({
          graph,
          stepKey,
          toolCallChunks: chunk.tool_call_chunks,
        });
      }
      await handleToolCallChunks({
        graph,
        stepKey,
        toolCallChunks: chunk.tool_call_chunks,
        metadata,
      });
      if (canStreamEager) {
        startReadyStreamedEagerToolExecutions({
          graph,
          metadata,
          agentContext,
          stepKey,
          toolCallChunks: chunk.tool_call_chunks,
          seal: streamedToolCallSeal,
          allowSequentialSeal,
          sealAll: hasFinalToolCallSignal(chunk),
        });
      }
    }

    if (isEmptyContent) {
      return;
    }

    const message_id = getMessageId(stepKey, graph) ?? '';
    if (message_id) {
      await graph.dispatchRunStep(
        stepKey,
        {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: {
            message_id,
          },
        },
        metadata
      );
    }

    const stepId = graph.getStepIdByKey(stepKey);
    const runStep = graph.getRunStep(stepId);
    if (!runStep) {
      console.warn(`\n
==============================================================


Run step for ${stepId} does not exist, cannot dispatch delta event.

event: ${event}
stepId: ${stepId}
stepKey: ${stepKey}
message_id: ${message_id}
hasToolCalls: ${hasToolCalls}
hasToolCallChunks: ${hasToolCallChunks}

==============================================================
\n`);
      return;
    }

    /* Note: tool call chunks may have non-empty content that matches the current tool chunk generation */
    if (typeof content === 'string' && runStep.type === StepTypes.TOOL_CALLS) {
      return;
    } else if (
      hasToolCallChunks &&
      (chunk.tool_call_chunks?.some((tc) => tc.args === content) ?? false)
    ) {
      return;
    } else if (typeof content === 'string') {
      if (agentContext.currentTokenType === ContentTypes.TEXT) {
        await graph.dispatchMessageDelta(
          stepId,
          {
            content: [
              {
                type: ContentTypes.TEXT,
                text: content,
              },
            ],
          },
          metadata
        );
      } else if (agentContext.currentTokenType === 'think_and_text') {
        const { text, thinking } = parseThinkingContent(content);
        if (thinking) {
          await graph.dispatchReasoningDelta(
            stepId,
            {
              content: [
                {
                  type: ContentTypes.THINK,
                  think: thinking,
                },
              ],
            },
            metadata
          );
        }
        if (text) {
          agentContext.currentTokenType = ContentTypes.TEXT;
          agentContext.tokenTypeSwitch = 'content';
          const newStepKey = graph.getStepKey(metadata);
          const message_id = getMessageId(newStepKey, graph) ?? '';
          await graph.dispatchRunStep(
            newStepKey,
            {
              type: StepTypes.MESSAGE_CREATION,
              message_creation: {
                message_id,
              },
            },
            metadata
          );

          const newStepId = graph.getStepIdByKey(newStepKey);
          await graph.dispatchMessageDelta(
            newStepId,
            {
              content: [
                {
                  type: ContentTypes.TEXT,
                  text: text,
                },
              ],
            },
            metadata
          );
        }
      } else {
        await graph.dispatchReasoningDelta(
          stepId,
          {
            content: [
              {
                type: ContentTypes.THINK,
                think: content,
              },
            ],
          },
          metadata
        );
      }
    } else if (
      content.every((c) => c.type?.startsWith(ContentTypes.TEXT) ?? false)
    ) {
      await graph.dispatchMessageDelta(
        stepId,
        {
          content,
        },
        metadata
      );
    } else if (
      content.every(
        (c) =>
          (c.type?.startsWith(ContentTypes.THINKING) ?? false) ||
          (c.type?.startsWith(ContentTypes.REASONING) ?? false) ||
          (c.type?.startsWith(ContentTypes.REASONING_CONTENT) ?? false) ||
          c.type === 'redacted_thinking'
      )
    ) {
      await graph.dispatchReasoningDelta(
        stepId,
        {
          content: content.map((c) => ({
            type: ContentTypes.THINK,
            think:
              (c as t.ThinkingContentText).thinking ??
              (c as Partial<t.GoogleReasoningContentText>).reasoning ??
              (c as Partial<t.BedrockReasoningContentText>).reasoningText
                ?.text ??
              '',
          })),
        },
        metadata
      );
    }
  }
  handleReasoning(
    chunk: Partial<AIMessageChunk>,
    agentContext: AgentContext
  ): void {
    let reasoning_content = chunk.additional_kwargs?.[
      agentContext.reasoningKey
    ] as string | Partial<ChatOpenAIReasoningSummary> | undefined;
    if (
      Array.isArray(chunk.content) &&
      (chunk.content[0]?.type === ContentTypes.THINKING ||
        chunk.content[0]?.type === ContentTypes.REASONING ||
        chunk.content[0]?.type === ContentTypes.REASONING_CONTENT ||
        chunk.content[0]?.type === 'redacted_thinking')
    ) {
      reasoning_content = 'valid';
    } else if (
      (agentContext.provider === Providers.OPENAI ||
        agentContext.provider === Providers.AZURE) &&
      reasoning_content != null &&
      typeof reasoning_content !== 'string' &&
      reasoning_content.summary?.[0]?.text != null &&
      reasoning_content.summary[0].text
    ) {
      reasoning_content = 'valid';
    } else if (
      agentContext.provider === Providers.OPENROUTER &&
      // Only set reasoning as valid if content is NOT present (content signals end of reasoning)
      (chunk.content == null || chunk.content === '') &&
      // Check for reasoning_details (final chunk) OR reasoning string (intermediate chunks)
      ((chunk.additional_kwargs?.reasoning_details != null &&
        Array.isArray(chunk.additional_kwargs.reasoning_details) &&
        chunk.additional_kwargs.reasoning_details.length > 0) ||
        (typeof chunk.additional_kwargs?.reasoning === 'string' &&
          chunk.additional_kwargs.reasoning !== ''))
    ) {
      reasoning_content = 'valid';
    }
    if (
      reasoning_content != null &&
      reasoning_content !== '' &&
      (chunk.content == null ||
        chunk.content === '' ||
        reasoning_content === 'valid')
    ) {
      agentContext.currentTokenType = ContentTypes.THINK;
      agentContext.tokenTypeSwitch = 'reasoning';
      return;
    } else if (
      agentContext.tokenTypeSwitch === 'reasoning' &&
      agentContext.currentTokenType !== ContentTypes.TEXT &&
      ((chunk.content != null && chunk.content !== '') ||
        (chunk.tool_calls?.length ?? 0) > 0 ||
        (chunk.tool_call_chunks?.length ?? 0) > 0)
    ) {
      agentContext.currentTokenType = ContentTypes.TEXT;
      agentContext.tokenTypeSwitch = 'content';
      agentContext.reasoningTransitionCount++;
    } else if (
      chunk.content != null &&
      typeof chunk.content === 'string' &&
      chunk.content.includes('<think>') &&
      chunk.content.includes('</think>')
    ) {
      agentContext.currentTokenType = 'think_and_text';
      agentContext.tokenTypeSwitch = 'content';
    } else if (
      chunk.content != null &&
      typeof chunk.content === 'string' &&
      chunk.content.includes('<think>')
    ) {
      agentContext.currentTokenType = ContentTypes.THINK;
      agentContext.tokenTypeSwitch = 'content';
    } else if (
      agentContext.lastToken != null &&
      agentContext.lastToken.includes('</think>')
    ) {
      agentContext.currentTokenType = ContentTypes.TEXT;
      agentContext.tokenTypeSwitch = 'content';
    }
    if (typeof chunk.content !== 'string') {
      return;
    }
    agentContext.lastToken = chunk.content;
  }
}

export function createContentAggregator(): t.ContentAggregatorResult {
  const contentParts: Array<t.MessageContentComplex | undefined> = [];
  const stepMap = new Map<string, t.RunStep>();
  const toolCallIdMap = new Map<string, string>();
  // Track agentId and groupId for each content index (applied to content parts)
  const contentMetaMap = new Map<
    number,
    { agentId?: string; groupId?: number }
  >();
  const getFirstContentPart = (
    content?: t.MessageDelta['content'] | t.MessageContentComplex
  ): t.MessageContentComplex | undefined => {
    if (content == null) {
      return undefined;
    }
    return Array.isArray(content) ? content[0] : content;
  };

  const updateContent = (
    index: number,
    contentPart?: t.MessageContentComplex,
    finalUpdate = false
  ): void => {
    if (!contentPart) {
      console.warn('No content part found in \'updateContent\'');
      return;
    }
    const partType = contentPart.type ?? '';
    if (!partType) {
      console.warn('No content type found in content part');
      return;
    }

    if (!contentParts[index] && partType !== ContentTypes.TOOL_CALL) {
      contentParts[index] = { type: partType };
    }

    if (!partType.startsWith(contentParts[index]?.type ?? '')) {
      console.warn('Content type mismatch');
      return;
    }

    if (
      partType.startsWith(ContentTypes.TEXT) &&
      ContentTypes.TEXT in contentPart &&
      typeof contentPart.text === 'string'
    ) {
      // TODO: update this!!
      const currentContent = contentParts[index] as t.MessageDeltaUpdate;
      const update: t.MessageDeltaUpdate = {
        type: ContentTypes.TEXT,
        text: (currentContent.text || '') + contentPart.text,
      };

      if (contentPart.tool_call_ids) {
        update.tool_call_ids = contentPart.tool_call_ids;
      }
      contentParts[index] = update;
    } else if (
      partType.startsWith(ContentTypes.THINK) &&
      ContentTypes.THINK in contentPart &&
      typeof contentPart.think === 'string'
    ) {
      const currentContent = contentParts[index] as t.ReasoningDeltaUpdate;
      const update: t.ReasoningDeltaUpdate = {
        type: ContentTypes.THINK,
        think: (currentContent.think || '') + contentPart.think,
      };
      contentParts[index] = update;
    } else if (
      partType.startsWith(ContentTypes.AGENT_UPDATE) &&
      ContentTypes.AGENT_UPDATE in contentPart &&
      contentPart.agent_update != null
    ) {
      const update: t.AgentUpdate = {
        type: ContentTypes.AGENT_UPDATE,
        agent_update: contentPart.agent_update,
      };

      contentParts[index] = update;
    } else if (partType === ContentTypes.SUMMARY) {
      const currentSummary = contentParts[index] as
        | t.SummaryContentBlock
        | undefined;
      const incoming = contentPart as t.SummaryContentBlock;
      contentParts[index] = {
        ...incoming,
        content: [
          ...(currentSummary?.content ?? []),
          ...(incoming.content ?? []),
        ],
      };
    } else if (
      partType === ContentTypes.IMAGE_URL &&
      'image_url' in contentPart
    ) {
      const currentContent = contentParts[index] as {
        type: 'image_url';
        image_url: string;
      };
      contentParts[index] = {
        ...currentContent,
      };
    } else if (
      partType === ContentTypes.TOOL_CALL &&
      'tool_call' in contentPart
    ) {
      const incomingName = contentPart.tool_call.name;
      const incomingId = contentPart.tool_call.id;
      const toolCallArgs = (contentPart.tool_call as t.ToolCallPart).args;

      // When we receive a tool call with a name, it's the complete tool call
      // Consolidate with any previously accumulated args from chunks
      const hasValidName = incomingName != null && incomingName !== '';

      // Only process if incoming has a valid name (complete tool call)
      // or if we're doing a final update with complete data
      if (!hasValidName && !finalUpdate) {
        return;
      }

      const existingContent = contentParts[index] as
        | (Omit<t.ToolCallContent, 'tool_call'> & {
            tool_call?: t.ToolCallPart & t.PartMetadata;
          })
        | undefined;
      if (!finalUpdate && existingContent?.tool_call?.progress === 1) {
        return;
      }

      /** When args are a valid object, they are likely already invoked */
      let args =
        finalUpdate ||
        typeof existingContent?.tool_call?.args === 'object' ||
        typeof toolCallArgs === 'object'
          ? contentPart.tool_call.args
          : (existingContent?.tool_call?.args ?? '') + (toolCallArgs ?? '');
      if (
        finalUpdate &&
        args == null &&
        existingContent?.tool_call?.args != null
      ) {
        args = existingContent.tool_call.args;
      }

      const id =
        getNonEmptyValue([incomingId, existingContent?.tool_call?.id]) ?? '';
      const name =
        getNonEmptyValue([incomingName, existingContent?.tool_call?.name]) ??
        '';

      const newToolCall: ToolCall & t.PartMetadata = {
        id,
        name,
        args,
        type: ToolCallTypes.TOOL_CALL,
      };

      const auth =
        contentPart.tool_call.auth ?? existingContent?.tool_call?.auth;
      const expiresAt =
        contentPart.tool_call.expires_at ??
        existingContent?.tool_call?.expires_at;
      if (auth != null) {
        newToolCall.auth = auth;
        newToolCall.expires_at = expiresAt;
      }

      if (finalUpdate) {
        newToolCall.progress = 1;
        newToolCall.output = contentPart.tool_call.output;
      }

      contentParts[index] = {
        type: ContentTypes.TOOL_CALL,
        tool_call: newToolCall,
      };
    }

    // Apply agentId (for MultiAgentGraph) and groupId (for parallel execution) to content parts
    // - agentId present → MultiAgentGraph (show agent labels)
    // - groupId present → parallel execution (render columns)
    const meta = contentMetaMap.get(index);
    if (meta?.agentId != null) {
      (contentParts[index] as t.MessageContentComplex).agentId = meta.agentId;
    }
    if (meta?.groupId != null) {
      (contentParts[index] as t.MessageContentComplex).groupId = meta.groupId;
    }
  };

  const aggregateContent = ({
    event,
    data,
  }: {
    event: GraphEvents;
    data:
      | t.RunStep
      | t.AgentUpdate
      | t.MessageDeltaEvent
      | t.ReasoningDeltaEvent
      | t.RunStepDeltaEvent
      | t.SummarizeDeltaData
      | t.SummarizeCompleteEvent
      | { result: t.ToolEndEvent };
  }): void => {
    if (event === GraphEvents.ON_SUMMARIZE_DELTA) {
      const deltaData = data as t.SummarizeDeltaData;
      const runStep = stepMap.get(deltaData.id);
      if (!runStep) {
        console.warn('No run step found for summarize delta event');
        return;
      }
      updateContent(runStep.index, deltaData.delta.summary);
      return;
    }

    if (event === GraphEvents.ON_SUMMARIZE_COMPLETE) {
      const completeData = data as t.SummarizeCompleteEvent;
      const summary = completeData.summary;
      if (!summary?.boundary) {
        return;
      }
      const runStep = stepMap.get(summary.boundary.messageId);
      if (!runStep) {
        return;
      }
      // Replace accumulated delta text with the authoritative final summary.
      // Multi-stage summarization streams deltas from each chunk, which
      // concatenate in updateContent.  This event carries only the correct
      // final text from the last stage.
      contentParts[runStep.index] = summary;
      return;
    }

    if (event === GraphEvents.ON_RUN_STEP) {
      const runStep = data as t.RunStep;
      stepMap.set(runStep.id, runStep);

      // Track agentId (MultiAgentGraph) and groupId (parallel execution) separately
      // - agentId: present for all MultiAgentGraph runs (enables agent labels in UI)
      // - groupId: present only for parallel execution (enables column rendering)
      const hasAgentId = runStep.agentId != null && runStep.agentId !== '';
      const hasGroupId = runStep.groupId != null;
      if (hasAgentId || hasGroupId) {
        const existingMeta = contentMetaMap.get(runStep.index) ?? {};
        if (hasAgentId) {
          existingMeta.agentId = runStep.agentId;
        }
        if (hasGroupId) {
          existingMeta.groupId = runStep.groupId;
        }
        contentMetaMap.set(runStep.index, existingMeta);
      }

      if (runStep.summary != null) {
        updateContent(runStep.index, runStep.summary);
      }

      if (
        runStep.stepDetails.type === StepTypes.TOOL_CALLS &&
        runStep.stepDetails.tool_calls
      ) {
        (runStep.stepDetails.tool_calls as ToolCall[]).forEach((toolCall) => {
          const toolCallId = toolCall.id ?? '';
          if ('id' in toolCall && toolCallId) {
            toolCallIdMap.set(runStep.id, toolCallId);
          }
          const contentPart: t.MessageContentComplex = {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              args: toolCall.args,
              name: toolCall.name,
              id: toolCallId,
            },
          };

          updateContent(runStep.index, contentPart);
        });
      }
    } else if (event === GraphEvents.ON_MESSAGE_DELTA) {
      const messageDelta = data as t.MessageDeltaEvent;
      const runStep = stepMap.get(messageDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for message delta event');
        return;
      }

      const contentPart = getFirstContentPart(messageDelta.delta.content);
      if (contentPart != null) {
        updateContent(runStep.index, contentPart);
      }
    } else if (
      event === GraphEvents.ON_AGENT_UPDATE &&
      (data as t.AgentUpdate | undefined)?.agent_update
    ) {
      const contentPart = data as t.AgentUpdate | undefined;
      if (!contentPart) {
        return;
      }
      updateContent(contentPart.agent_update.index, contentPart);
    } else if (event === GraphEvents.ON_REASONING_DELTA) {
      const reasoningDelta = data as t.ReasoningDeltaEvent;
      const runStep = stepMap.get(reasoningDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for reasoning delta event');
        return;
      }

      const contentPart = getFirstContentPart(reasoningDelta.delta.content);
      if (contentPart != null) {
        updateContent(runStep.index, contentPart);
      }
    } else if (event === GraphEvents.ON_RUN_STEP_DELTA) {
      const runStepDelta = data as t.RunStepDeltaEvent;
      const runStep = stepMap.get(runStepDelta.id);
      if (!runStep) {
        console.warn('No run step or runId found for run step delta event');
        return;
      }

      if (
        runStepDelta.delta.type === StepTypes.TOOL_CALLS &&
        runStepDelta.delta.tool_calls
      ) {
        runStepDelta.delta.tool_calls.forEach((toolCallDelta) => {
          const toolCallId = toolCallIdMap.get(runStepDelta.id);

          const contentPart: t.MessageContentComplex = {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              args: toolCallDelta.args ?? '',
              name: toolCallDelta.name,
              id: toolCallId,
              auth: runStepDelta.delta.auth,
              expires_at: runStepDelta.delta.expires_at,
            },
          };

          updateContent(runStep.index, contentPart);
        });
      }
    } else if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
      const { result } = data as unknown as {
        result:
          | t.ToolEndEvent
          | (t.SummaryCompleted & { id: string; index: number });
      };

      const { id: stepId } = result;

      const runStep = stepMap.get(stepId);
      if (!runStep) {
        console.warn('No run step or runId found for completed step event');
        return;
      }

      if (result.type === ContentTypes.SUMMARY && 'summary' in result) {
        contentParts[runStep.index] = result.summary as t.MessageContentComplex;
      } else if ('tool_call' in result) {
        const contentPart: t.MessageContentComplex = {
          type: ContentTypes.TOOL_CALL,
          tool_call: (result as t.ToolEndEvent).tool_call,
        };
        updateContent(runStep.index, contentPart, true);
      }
    }
  };

  return { contentParts, aggregateContent, stepMap };
}
