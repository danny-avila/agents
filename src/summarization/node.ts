import { createHash } from 'crypto';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { AgentContext } from '@/agents/AgentContext';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, StepTypes, Providers } from '@/common';
import { safeDispatchCustomEvent } from '@/utils/events';
import { getChatModelClass } from '@/llm/providers';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Per-field character limits, differentiated by signal value for summarization.
 *
 * Tool args (code, config, JSON payloads) are low-signal — the summarizer
 * needs to know WHAT was called and roughly WHY, not the exact code.
 * Tool results and message content are high-signal — they carry the actual
 * information the conversation is about.
 */
const LIMITS = {
  /** Tool call argument values (per key) — usually code/config, low signal */
  toolArg: 200,
  /** Tool result content — actual data returned, high signal */
  toolResult: 800,
  /** User/assistant message content — conversational substance */
  messageContent: 600,
} as const;

/**
 * Total character budget for a single summarization chunk.
 * ~20K chars ≈ ~5K tokens, leaving room for the system prompt + prior summary
 * within a typical summarization model's context.
 */
const CHUNK_BUDGET_CHARS = 20_000;

/** Default number of parts for multi-stage summarization. */
const DEFAULT_PARTS = 2;

/** Minimum messages before multi-stage split is attempted. */
const DEFAULT_MIN_MESSAGES_FOR_SPLIT = 4;

/**
 * Known summarization-specific parameter keys that must NOT be passed to the
 * LLM constructor. These control summarization behavior, not model behavior.
 */
const SUMMARIZATION_PARAM_KEYS = new Set([
  'parts',
  'minMessagesForSplit',
  'maxInputTokensForSinglePass',
  'maxSummaryTokens',
]);

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

/**
 * Computes a short hash of the messages being summarized.
 * Used for deduplication and debugging — if the same rangeHash appears on consecutive
 * summaries, the same messages were re-summarized (likely a no-op).
 */
function computeRangeHash(messages: BaseMessage[]): string {
  const hash = createHash('sha256');
  for (const msg of messages) {
    hash.update(msg._getType());
    hash.update(
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content)
    );
  }
  return hash.digest('hex').slice(0, 16);
}

// ---------------------------------------------------------------------------
// Message formatting for summarization input
// ---------------------------------------------------------------------------

/**
 * Truncates a string to `maxLen` characters, appending an ellipsis indicator
 * with the original length so the summarizer knows content was cut.
 */
function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) {
    return text;
  }
  return `${text.slice(0, maxLen)}… [truncated, ${text.length} chars total]`;
}

/**
 * Head+tail truncation: keeps the beginning and end of the text so the
 * summarizer sees both the opening context and final outcome.
 * Falls back to head-only `truncate()` when `maxLen` is too small for
 * a meaningful head+tail split.
 */
function smartTruncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) {
    return text;
  }
  const indicator = `… [truncated, ${text.length} chars total] …`;
  const available = maxLen - indicator.length;
  if (available <= 0) {
    return truncate(text, maxLen);
  }
  const head = Math.ceil(available / 2);
  const tail = Math.floor(available / 2);
  return text.slice(0, head) + indicator + text.slice(text.length - tail);
}

/**
 * Formats tool call args for summarization. Produces a compact representation
 * instead of dumping raw JSON (which can be thousands of chars for code-heavy
 * tool calls like script evaluation or file writes).
 */
function formatToolArgs(args: Record<string, unknown> | undefined): string {
  if (!args || typeof args !== 'object') {
    return '';
  }
  const parts: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    if (value == null) {
      continue;
    }
    const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
    parts.push(`${key}: ${truncate(valueStr, LIMITS.toolArg)}`);
  }
  return parts.join(', ');
}

function formatMessageForSummary(msg: BaseMessage): string {
  const role = msg._getType();

  // Tool result messages: format as "[tool_result: name] → content"
  if (role === 'tool') {
    const name = msg.name ?? 'unknown';
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
    return `[tool_result: ${name}] → ${smartTruncate(content, LIMITS.toolResult)}`;
  }

  // AI messages with tool calls: extract text content and format tool calls readably
  if (
    role === 'ai' &&
    msg instanceof AIMessage &&
    msg.tool_calls &&
    msg.tool_calls.length > 0
  ) {
    const parts: string[] = [];

    // Extract any text content
    if (typeof msg.content === 'string') {
      if (msg.content.trim()) {
        parts.push(smartTruncate(msg.content.trim(), LIMITS.messageContent));
      }
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (
          typeof block === 'object' &&
          'type' in block &&
          block.type === 'text' &&
          'text' in block &&
          typeof block.text === 'string' &&
          block.text.trim()
        ) {
          parts.push(smartTruncate(block.text.trim(), LIMITS.messageContent));
        }
      }
    }

    // Format each tool call with truncated args
    for (const tc of msg.tool_calls) {
      const argsStr = formatToolArgs(tc.args as Record<string, unknown>);
      parts.push(`[tool_call: ${tc.name}(${argsStr})]`);
    }

    return `[${role}]: ${parts.join('\n')}`;
  }

  // All other messages: use content directly, truncated
  const content =
    typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
  return `[${role}]: ${smartTruncate(content, LIMITS.messageContent)}`;
}

/**
 * Formats messages for the summarization prompt, applying a total character
 * budget. Individual messages are first formatted with per-field limits, then
 * if the combined text exceeds the budget, the longest formatted messages are
 * proportionally trimmed so every message retains representation.
 */
function formatMessagesForSummarization(
  messages: BaseMessage[],
  budgetChars: number = CHUNK_BUDGET_CHARS
): string {
  const formatted = messages.map(formatMessageForSummary);

  const totalChars = formatted.reduce((sum, s) => sum + s.length, 0);
  if (totalChars <= budgetChars) {
    return formatted.join('\n');
  }

  // Over budget — trim the longest messages proportionally.
  const ratio = budgetChars / totalChars;
  const trimmed = formatted.map((text) => {
    const allowance = Math.max(80, Math.floor(text.length * ratio));
    return truncate(text, allowance);
  });

  return trimmed.join('\n');
}

// ---------------------------------------------------------------------------
// Adaptive chunk sizing
// ---------------------------------------------------------------------------

/** Base chunk ratio — at normal message density, use 40% of budget per chunk. */
const BASE_CHUNK_RATIO = 0.4;
/** Minimum chunk ratio — at high message density, shrink to 15% of budget. */
const MIN_CHUNK_RATIO = 0.15;
/** Safety margin applied to estimated chunk sizes to account for token estimation inaccuracy. */
const SAFETY_MARGIN = 1.2;

/**
 * Computes the minimum number of chunks needed based on content density.
 * When average message size is large relative to the chunk budget, the chunk
 * count increases proportionally. This prevents oversized chunks that would
 * overflow the summarization model's context.
 *
 * @param messages - Messages to be chunked.
 * @param totalCharWeight - Total character weight across all messages.
 * @param chunkBudgetChars - Character budget per chunk (e.g., CHUNK_BUDGET_CHARS).
 * @returns Minimum number of chunks to use.
 */
function computeAdaptiveChunkCount(params: {
  messages: BaseMessage[];
  totalCharWeight: number;
  chunkBudgetChars: number;
}): number {
  const { messages, totalCharWeight, chunkBudgetChars } = params;
  if (messages.length === 0 || chunkBudgetChars <= 0) {
    return 1;
  }

  const avgMessageChars = totalCharWeight / messages.length;
  const budgetFraction = avgMessageChars / chunkBudgetChars;

  // Scale the per-chunk ratio down when messages are large.
  // At budgetFraction >= 0.25, ratio starts decreasing from BASE toward MIN.
  const effectiveRatio =
    budgetFraction >= 0.25
      ? Math.max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO * (1 - budgetFraction))
      : BASE_CHUNK_RATIO;

  const effectiveBudget = chunkBudgetChars * effectiveRatio;
  const estimatedChunks = Math.ceil(
    (totalCharWeight * SAFETY_MARGIN) / effectiveBudget
  );

  return Math.max(1, estimatedChunks);
}

// ---------------------------------------------------------------------------
// Multi-stage summarization
// ---------------------------------------------------------------------------

/**
 * Splits messages into `parts` groups of roughly equal token weight.
 * Uses character length as a proxy for tokens (sufficient for splitting).
 */
function splitMessagesByCharShare(
  messages: BaseMessage[],
  parts: number
): BaseMessage[][] {
  if (messages.length === 0 || parts <= 1) {
    return [messages];
  }
  const normalizedParts = Math.min(parts, messages.length);

  // Estimate character weight per message
  const weights = messages.map((msg) => {
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
    return content.length;
  });
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  const targetWeight = totalWeight / normalizedParts;

  const chunks: BaseMessage[][] = [];
  let currentChunk: BaseMessage[] = [];
  let currentWeight = 0;

  for (let i = 0; i < messages.length; i++) {
    currentChunk.push(messages[i]);
    currentWeight += weights[i];

    // Break when we've reached target weight for this chunk,
    // unless we're filling the last chunk (which gets the remainder).
    if (
      currentWeight >= targetWeight &&
      chunks.length < normalizedParts - 1 &&
      currentChunk.length > 0
    ) {
      chunks.push(currentChunk);
      currentChunk = [];
      currentWeight = 0;
    }
  }

  // Push the last chunk (may be empty if messages divided evenly)
  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

// ---------------------------------------------------------------------------
// Prompt & config helpers
// ---------------------------------------------------------------------------

/**
 * Structured checkpoint format for fresh summarization (no prior summary).
 * Adapted from pi-coding-agent's SUMMARIZATION_PROMPT pattern.
 * The explicit section structure ensures consistent, machine-parseable output
 * that survives multiple summarization cycles without information loss.
 */
export const DEFAULT_SUMMARIZATION_PROMPT = `Create a structured context checkpoint summary of the conversation. This summary will replace the conversation messages, so it must capture everything needed to continue the work.

Use this exact format:

## Goal
The user's primary objective and any sub-goals identified during the conversation.

## Constraints & Preferences
Configuration choices, style preferences, architectural decisions, technology choices, or rules the user has established.

## Progress
### Done
- Completed items with their outcomes

### In Progress
- Current work items and their state

### Blocked
- Items that cannot proceed and why

## Key Decisions
Decisions made and their rationale. Include alternatives that were rejected and why.

## Next Steps
What should happen next, in priority order.

## Critical Context
Specific identifiers, names, error messages, URLs, and other details that must be preserved verbatim.

Rules:
- Preserve exact identifiers, names, error messages, and key references — do NOT paraphrase them
- Record each tool call outcome (success or failure with reason), not raw output
- Do NOT reproduce tool output or long content verbatim
- Use short declarative sentences
- Omit empty sections
- Output only the summary`;

/**
 * Prompt for incremental summary updates when a prior summary exists.
 * Adapted from pi-coding-agent's UPDATE_SUMMARIZATION_PROMPT pattern.
 * The rules ensure existing information is preserved while new information
 * is integrated into the correct sections.
 */
export const DEFAULT_UPDATE_SUMMARIZATION_PROMPT = `Update the existing summary checkpoint with information from new conversation messages. Maintain the same structured format.

Rules:
- PRESERVE all existing information from the previous summary unless explicitly contradicted by new messages
- ADD new progress, decisions, and context from the new messages
- Move items from "In Progress" to "Done" when completed
- Move items from "Blocked" to "In Progress" or "Done" as appropriate
- Add new items to the appropriate sections
- Update "Next Steps" to reflect current priorities
- Preserve exact identifiers, names, error messages, and key references — do NOT paraphrase them
- Record each tool call outcome (success or failure with reason), not raw output
- Do NOT reproduce tool output or long content verbatim
- Omit empty sections
- Output only the updated summary using the same format`;

/**
 * Separates summarization-specific parameters (parts, minMessagesForSplit, etc.)
 * from LLM constructor parameters (temperature, maxTokens, etc.).
 * Prevents summarization config from leaking into the model constructor.
 */
function separateParameters(parameters: Record<string, unknown>): {
  llmParams: Record<string, unknown>;
  summarizationParams: {
    parts: number;
    minMessagesForSplit: number;
    maxSummaryTokens?: number;
  };
} {
  const llmParams: Record<string, unknown> = {};
  let parts = DEFAULT_PARTS;
  let minMessagesForSplit = DEFAULT_MIN_MESSAGES_FOR_SPLIT;
  let maxSummaryTokens: number | undefined;

  for (const [key, value] of Object.entries(parameters)) {
    if (SUMMARIZATION_PARAM_KEYS.has(key)) {
      if (key === 'parts' && typeof value === 'number' && value >= 1) {
        parts = value;
      } else if (
        key === 'minMessagesForSplit' &&
        typeof value === 'number' &&
        value >= 2
      ) {
        minMessagesForSplit = value;
      } else if (
        key === 'maxSummaryTokens' &&
        typeof value === 'number' &&
        value > 0
      ) {
        maxSummaryTokens = value;
      }
      // maxInputTokensForSinglePass: reserved for future use
    } else {
      llmParams[key] = value;
    }
  }

  return {
    llmParams,
    summarizationParams: { parts, minMessagesForSplit, maxSummaryTokens },
  };
}

// ---------------------------------------------------------------------------
// Graceful degradation
// ---------------------------------------------------------------------------

/**
 * Generates a structural metadata summary without making an LLM call.
 * Used as a last-resort fallback when all summarization attempts fail.
 * Preserves tool names and message counts so the agent retains basic context.
 */
function generateMetadataStub(messages: BaseMessage[]): string {
  const counts: Record<string, number> = {};
  const toolNames = new Set<string>();

  for (const msg of messages) {
    const role = msg._getType();
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

// ---------------------------------------------------------------------------
// Post-LLM summary enrichment
// ---------------------------------------------------------------------------
// Adapted from openclaw's compaction-safeguard pattern. After the LLM produces
// its summary, mechanically append structured data that the LLM might miss:
// tool failures and file operations. This guarantees these survive compaction
// regardless of LLM quality.

/** Maximum number of tool failures to include in the enrichment section. */
const MAX_TOOL_FAILURES = 8;
/** Maximum chars per failure summary line. */
const MAX_TOOL_FAILURE_CHARS = 240;

/**
 * Extracts failed tool results from messages and formats them as a structured
 * section. LLMs often omit specific failure details (exit codes, error messages)
 * from their summaries — this mechanical enrichment guarantees they survive.
 */
function extractToolFailuresSection(messages: BaseMessage[]): string {
  const failures: Array<{ toolName: string; summary: string }> = [];
  const seen = new Set<string>();

  for (const msg of messages) {
    if (msg._getType() !== 'tool') {
      continue;
    }
    const toolMsg = msg as import('@langchain/core/messages').ToolMessage;
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
    const content =
      typeof toolMsg.content === 'string'
        ? toolMsg.content
        : JSON.stringify(toolMsg.content);
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

// ---------------------------------------------------------------------------
// Summarize node
// ---------------------------------------------------------------------------

interface CreateSummarizeNodeParams {
  agentContext: AgentContext;
  graph: {
    contentData: t.RunStep[];
    contentIndexMap: Map<string, number>;
    config?: RunnableConfig;
    runId?: string;
    isMultiAgent: boolean;
  };
  generateStepId: (stepKey: string) => [string, number];
}

export function createSummarizeNode({
  agentContext,
  graph,
  generateStepId,
}: CreateSummarizeNodeParams) {
  return async (
    state: {
      messages: BaseMessage[];
      summarizationRequest?: t.SummarizationNodeInput;
    },
    config?: RunnableConfig
  ): Promise<{ summarizationRequest: undefined }> => {
    const request = state.summarizationRequest;
    if (request == null) {
      return { summarizationRequest: undefined };
    }

    // ----- Resolve summarization config -----
    const summarizationConfig = agentContext.summarizationConfig;
    const provider = summarizationConfig?.provider ?? agentContext.provider;
    const modelName = summarizationConfig?.model;
    const parameters = summarizationConfig?.parameters ?? {};
    const promptText =
      summarizationConfig?.prompt ?? DEFAULT_SUMMARIZATION_PROMPT;
    // Use a dedicated update prompt when integrating into an existing summary.
    // Falls back to the main promptText if no update prompt is configured
    // (which is fine for custom prompts that handle both cases).
    const updatePromptText =
      summarizationConfig?.updatePrompt ?? DEFAULT_UPDATE_SUMMARIZATION_PROMPT;
    const streaming = summarizationConfig?.stream !== false;

    // Separate summarization-specific params from LLM constructor params.
    // This prevents keys like `parts`, `minMessagesForSplit` from being
    // passed to the ChatModel constructor where they don't belong.
    const { llmParams, summarizationParams } = separateParameters(parameters);

    // Build LLM constructor options — clean of any main agent context.
    // The summarizer is its own agent: no tools, no system instructions
    // from the parent, just the summarization prompt.
    const clientOptions: Record<string, unknown> = {
      ...llmParams,
    };
    if (modelName != null && modelName !== '') {
      clientOptions.model = modelName;
      clientOptions.modelName = modelName;
    }

    // Apply maxSummaryTokens: parameter takes priority, then config, then default.
    const DEFAULT_MAX_SUMMARY_TOKENS = 2048;
    const effectiveMaxSummaryTokens =
      summarizationParams.maxSummaryTokens ??
      summarizationConfig?.maxSummaryTokens ??
      DEFAULT_MAX_SUMMARY_TOKENS;

    if (provider === Providers.GOOGLE || provider === Providers.VERTEXAI) {
      clientOptions.maxOutputTokens = effectiveMaxSummaryTokens;
    } else {
      clientOptions.maxTokens = effectiveMaxSummaryTokens;
    }

    // The graph's RunnableConfig carries callbacks for tracing/observability.
    // It does NOT carry system instructions, tools, or agent persona — those
    // are bound to the model instance or in the message array respectively.
    // We pass this to both event dispatch and model.invoke() for traceability.
    const runnableConfig = config ?? graph.config;

    // ----- Set up run step -----
    const stepKey = `summarize-${request.agentId}`;
    const [stepId, stepIndex] = generateStepId(stepKey);

    const placeholderSummary: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      text: '',
      tokenCount: 0,
      model: modelName,
      provider: provider as string,
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

    graph.contentData.push(runStep);
    graph.contentIndexMap.set(stepId, runStep.index);

    if (runnableConfig) {
      await safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP,
        runStep,
        runnableConfig
      );

      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_START,
        {
          agentId: request.agentId,
          provider: provider as string,
          model: modelName,
          messagesToRefineCount: request.messagesToRefine.length,
          summaryVersion: agentContext.summaryVersion + 1,
        } satisfies t.SummarizeStartEvent,
        runnableConfig
      );
    }

    // ----- Create the summarizer model (fresh, isolated instance) -----
    const ChatModelClass = getChatModelClass(provider as Providers);
    const model = new ChatModelClass(clientOptions as never);

    // Check for a prior summary — either from a previous summarization within
    // this run or from a cross-run summary forwarded via initialSummary.
    const priorSummaryText = agentContext.getSummaryText()?.trim() ?? '';

    // ----- Enrich config with summarization metadata for Langfuse tracing -----
    // The strategies further add stage-specific metadata (single_pass, chunk, merge).
    const summarizeConfig: RunnableConfig | undefined = config
      ? {
        ...config,
        metadata: {
          ...config.metadata,
          agent_id: request.agentId,
          summarization_provider: provider,
          summarization_model: modelName,
        },
      }
      : undefined;

    // ----- Invoke: single-pass or multi-stage with graceful degradation -----
    let summaryText = '';
    const { parts, minMessagesForSplit } = summarizationParams;
    const messagesToRefine = request.messagesToRefine;

    // Tier 1: Full input, normal budget
    try {
      if (
        parts > 1 &&
        messagesToRefine.length >= Math.max(2, minMessagesForSplit)
      ) {
        summaryText = await summarizeInStages({
          model,
          messages: messagesToRefine,
          parts,
          promptText,
          updatePromptText,
          priorSummaryText,
          config: summarizeConfig,
          streaming,
          maxSummaryTokens: effectiveMaxSummaryTokens,
        });
      } else {
        summaryText = await summarizeSinglePass({
          model,
          messages: messagesToRefine,
          promptText,
          updatePromptText,
          priorSummaryText,
          config: summarizeConfig,
          streaming,
        });
      }
    } catch {
      // Tier 2: Retry with single-pass, half character budget
      try {
        summaryText = await summarizeSinglePass({
          model,
          messages: messagesToRefine,
          promptText,
          updatePromptText,
          priorSummaryText,
          config: summarizeConfig,
          streaming,
          budgetChars: Math.floor(CHUNK_BUDGET_CHARS / 2),
        });
      } catch {
        // Tier 3: Metadata stub — no LLM call, preserves tool names and message counts
        summaryText = generateMetadataStub(messagesToRefine);
      }
    }

    if (!summaryText) {
      if (runnableConfig) {
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            agentId: request.agentId,
            summary: placeholderSummary,
            error: 'Summarization produced empty output',
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      return { summarizationRequest: undefined };
    }

    // ----- Enrich with structured data the LLM may have omitted -----
    summaryText = enrichSummary(summaryText, messagesToRefine);

    // ----- Finalize -----
    let tokenCount = 0;
    if (agentContext.tokenCounter) {
      tokenCount = agentContext.tokenCounter(new SystemMessage(summaryText));
    }

    agentContext.setSummary(summaryText, tokenCount);

    const summaryBlock: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      text: summaryText,
      tokenCount,
      summaryVersion: agentContext.summaryVersion,
      rangeHash: computeRangeHash(request.messagesToRefine),
      boundary: {
        messageId: stepId,
        contentIndex: runStep.index,
      },
      model: modelName,
      provider: provider as string,
      createdAt: new Date().toISOString(),
    };

    runStep.summary = summaryBlock;

    if (runnableConfig) {
      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_COMPLETE,
        {
          agentId: request.agentId,
          summary: summaryBlock,
        } satisfies t.SummarizeCompleteEvent,
        runnableConfig
      );
    }

    return { summarizationRequest: undefined };
  };
}

// ---------------------------------------------------------------------------
// Summarization strategies
// ---------------------------------------------------------------------------

/** Minimal model interface used by summarization strategies. */
interface SummarizeModel {
  invoke: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<{ content: string | object }>;
  stream?: (
    messages: BaseMessage[],
    config?: RunnableConfig
  ) => Promise<AsyncIterable<{ content: string | object }>>;
}

/**
 * Wraps a RunnableConfig with summarization-specific metadata so that
 * Langfuse (or other tracing) can distinguish summarization LLM calls
 * from the main agent's calls.
 */
function traceConfig(
  config: RunnableConfig | undefined,
  stage: string,
  extra?: Record<string, unknown>
): RunnableConfig | undefined {
  if (!config) {
    return undefined;
  }
  return {
    ...config,
    runName: `summarization:${stage}`,
    metadata: {
      ...config.metadata,
      ...extra,
      summarization: true,
      stage,
    },
  };
}

interface SummarizeParams {
  model: SummarizeModel;
  messages: BaseMessage[];
  promptText: string;
  /** Prompt to use when a prior summary exists (incremental update). Falls back to promptText if not provided. */
  updatePromptText?: string;
  priorSummaryText: string;
  /** RunnableConfig for callback/tracing propagation (NOT for system instructions). */
  config?: RunnableConfig;
  /** When false, disables streaming (uses invoke instead of stream). Defaults to true. */
  streaming?: boolean;
}

/**
 * Extracts the text from an LLM response, handling both string and
 * structured content formats.
 */
function extractResponseText(response: { content: string | object }): string {
  return (
    typeof response.content === 'string'
      ? response.content
      : JSON.stringify(response.content)
  ).trim();
}

/**
 * Streams an LLM call and collects the full response text.
 * Uses `model.stream()` when available so that native `on_chat_model_stream`
 * events flow through the graph to `ChatModelStreamHandler`, which handles
 * dispatching `ON_SUMMARIZE_DELTA` events to the client.
 * Falls back to `model.invoke()` when streaming isn't supported or is disabled.
 *
 * @param streaming - When false, forces invoke() even if model supports stream().
 *   Defaults to true.
 */
async function streamAndCollect(
  model: SummarizeModel,
  messages: BaseMessage[],
  config?: RunnableConfig,
  streaming = true
): Promise<string> {
  if (!streaming || typeof model.stream !== 'function') {
    const response = await model.invoke(messages, config);
    return extractResponseText(response);
  }

  let fullText = '';
  const stream = await model.stream(messages, config);
  for await (const chunk of stream) {
    const text =
      typeof chunk.content === 'string'
        ? chunk.content
        : JSON.stringify(chunk.content);
    if (text) {
      fullText += text;
    }
  }
  return fullText.trim();
}

/**
 * Single-pass summarization: formats all messages → one LLM call.
 * Used when `parts <= 1` or when the message count is below `minMessagesForSplit`.
 */
async function summarizeSinglePass({
  model,
  messages,
  promptText,
  updatePromptText,
  priorSummaryText,
  config,
  streaming,
  budgetChars,
}: SummarizeParams & { budgetChars?: number }): Promise<string> {
  const formattedText = formatMessagesForSummarization(messages, budgetChars);

  // Select the appropriate prompt: use the update variant when integrating
  // new messages into an existing summary, fresh prompt otherwise.
  const effectivePrompt = priorSummaryText
    ? (updatePromptText ?? promptText)
    : promptText;

  const humanMessageText = priorSummaryText
    ? `<previous-summary>\n${priorSummaryText}\n</previous-summary>\n\n<conversation>\n${formattedText}\n</conversation>`
    : `<conversation>\n${formattedText}\n</conversation>`;

  return streamAndCollect(
    model,
    [new SystemMessage(effectivePrompt), new HumanMessage(humanMessageText)],
    traceConfig(config, 'single_pass'),
    streaming
  );
}

/**
 * Multi-stage summarization via progressive chaining:
 *   1. Split messages into chunks by character weight (adaptive sizing)
 *   2. Summarize chunks sequentially — each chunk's summary becomes
 *      context for the next, preserving temporal ordering
 *
 * Uses the update prompt (if available) when a running summary exists,
 * falling back to the fresh prompt for the first chunk with no prior context.
 */
async function summarizeInStages({
  model,
  messages,
  parts,
  promptText,
  updatePromptText,
  priorSummaryText,
  config,
  streaming,
  maxSummaryTokens,
}: SummarizeParams & {
  parts: number;
  maxSummaryTokens?: number;
}): Promise<string> {
  // Compute total character weight for adaptive sizing.
  const totalCharWeight = messages.reduce((sum, msg) => {
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
    return sum + content.length;
  }, 0);

  // The user-configured `parts` acts as a minimum; adaptive sizing may increase it.
  const adaptiveParts = computeAdaptiveChunkCount({
    messages,
    totalCharWeight,
    chunkBudgetChars: CHUNK_BUDGET_CHARS,
  });
  const effectiveParts = Math.max(parts, adaptiveParts);

  const chunks = splitMessagesByCharShare(messages, effectiveParts);

  // If splitting didn't produce multiple chunks, fall back to single-pass
  if (chunks.length <= 1) {
    return summarizeSinglePass({
      model,
      messages,
      promptText,
      updatePromptText,
      priorSummaryText,
      config,
      streaming,
    });
  }

  // Progressive summary chaining: chunk N's summary becomes context for chunk N+1.
  // This preserves temporal ordering better than independent summarization + merge,
  // and eliminates the merge LLM call entirely.
  // Trade-off: chunks must be processed sequentially (no parallelism).
  let runningSummary = priorSummaryText || '';
  // Cap running summary so it doesn't consume too much of the next chunk's input.
  // Use 4 chars/token (standard ratio) and also cap at 40% of chunk budget so the
  // actual messages still get the majority of the summarizer's attention.
  const maxSummaryChars = Math.min(
    (maxSummaryTokens ?? 2048) * 4,
    Math.floor(CHUNK_BUDGET_CHARS * 0.4)
  );

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const formattedText = formatMessagesForSummarization(chunk);

    // Use update prompt when we have a running summary (either from prior
    // summary or from a previous chunk), fresh prompt for the first chunk
    // with no prior context.
    const effectivePrompt = runningSummary
      ? (updatePromptText ?? promptText)
      : promptText;

    let humanText: string;
    if (runningSummary) {
      humanText = `<previous-summary>\n${runningSummary}\n</previous-summary>\n\n<conversation>\nPart ${i + 1} of ${chunks.length}:\n\n${formattedText}\n</conversation>`;
    } else {
      humanText = `<conversation>\nPart ${i + 1} of ${chunks.length}:\n\n${formattedText}\n</conversation>`;
    }

    const text = await streamAndCollect(
      model,
      [new SystemMessage(effectivePrompt), new HumanMessage(humanText)],
      traceConfig(config, 'multi_stage_chunk', {
        chunk_index: i,
        total_chunks: chunks.length,
      }),
      streaming
    );

    if (text) {
      runningSummary = text;
    }

    // Guard: if running summary exceeds ~1.5× maxSummaryTokens (estimated via chars/4),
    // truncate it before passing to the next chunk to prevent input growth.
    if (runningSummary.length > maxSummaryChars) {
      runningSummary =
        runningSummary.slice(0, maxSummaryChars) +
        '\n\n… [summary truncated to fit token budget]';
    }
  }

  return runningSummary;
}
