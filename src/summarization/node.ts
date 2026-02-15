import {
  HumanMessage,
  SystemMessage,
  AIMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import { createHash } from 'crypto';
import type * as t from '@/types';
import { getChatModelClass } from '@/llm/providers';
import { safeDispatchCustomEvent } from '@/utils/events';
import { ContentTypes, GraphEvents, StepTypes, Providers } from '@/common';
import type { AgentContext } from '@/agents/AgentContext';

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
]);

const MERGE_PROMPT =
  'Merge these partial summaries into one cohesive summary. De-duplicate overlapping information, preserve all unique details and identifiers, maintain chronological order. Output only the merged summary.';

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
    return `[tool_result: ${name}] → ${truncate(content, LIMITS.toolResult)}`;
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
        parts.push(truncate(msg.content.trim(), LIMITS.messageContent));
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
          parts.push(truncate(block.text.trim(), LIMITS.messageContent));
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
  return `[${role}]: ${truncate(content, LIMITS.messageContent)}`;
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

export const DEFAULT_SUMMARIZATION_PROMPT =
  'Summarize the conversation concisely. Preserve key facts, decisions, identifiers, file paths, tool results, and open questions. Use short sentences. If a prior summary is provided, integrate new information into it. Output only the summary.';

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
  };
} {
  const llmParams: Record<string, unknown> = {};
  let parts = DEFAULT_PARTS;
  let minMessagesForSplit = DEFAULT_MIN_MESSAGES_FOR_SPLIT;

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
      }
      // maxInputTokensForSinglePass: reserved for future use
    } else {
      llmParams[key] = value;
    }
  }

  return {
    llmParams,
    summarizationParams: { parts, minMessagesForSplit },
  };
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

    // ----- Invoke: single-pass or multi-stage -----
    let summaryText = '';
    try {
      const { parts, minMessagesForSplit } = summarizationParams;
      const messagesToRefine = request.messagesToRefine;

      if (
        parts > 1 &&
        messagesToRefine.length >= Math.max(2, minMessagesForSplit)
      ) {
        summaryText = await summarizeInStages({
          model,
          messages: messagesToRefine,
          parts,
          promptText,
          priorSummaryText,
          config: summarizeConfig,
          streaming,
        });
      } else {
        summaryText = await summarizeSinglePass({
          model,
          messages: messagesToRefine,
          promptText,
          priorSummaryText,
          config: summarizeConfig,
          streaming,
        });
      }
    } catch (err) {
      if (runnableConfig) {
        const errorMessage =
          err instanceof Error ? err.message : 'Summarization failed';
        await safeDispatchCustomEvent(
          GraphEvents.ON_SUMMARIZE_COMPLETE,
          {
            agentId: request.agentId,
            summary: placeholderSummary,
            error: errorMessage,
          } satisfies t.SummarizeCompleteEvent,
          runnableConfig
        );
      }
      return { summarizationRequest: undefined };
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
  priorSummaryText,
  config,
  streaming,
}: SummarizeParams): Promise<string> {
  const formattedText = formatMessagesForSummarization(messages);

  const humanMessageText = priorSummaryText
    ? `## Prior Summary\n\n${priorSummaryText}\n\n## New Messages to Incorporate\n\n${formattedText}`
    : formattedText;

  return streamAndCollect(
    model,
    [new SystemMessage(promptText), new HumanMessage(humanMessageText)],
    traceConfig(config, 'single_pass'),
    streaming
  );
}

/**
 * Multi-stage summarization:
 *   1. Split messages into `parts` groups by character weight
 *   2. Summarize each chunk independently
 *   3. Merge partial summaries with a dedicated merge prompt
 *
 * This produces better summaries for large message histories because each
 * LLM call processes a focused subset instead of a truncated overview.
 */
async function summarizeInStages({
  model,
  messages,
  parts,
  promptText,
  priorSummaryText,
  config,
  streaming,
}: SummarizeParams & { parts: number }): Promise<string> {
  const chunks = splitMessagesByCharShare(messages, parts);

  // If splitting didn't produce multiple chunks, fall back to single-pass
  if (chunks.length <= 1) {
    return summarizeSinglePass({
      model,
      messages,
      promptText,
      priorSummaryText,
      config,
      streaming,
    });
  }

  // Stage 1: Summarize each chunk independently.
  // The prior summary is only included in the first chunk so the summarizer
  // has continuity context without duplicating it across every chunk.
  // When streaming is enabled, chunks produce native on_chat_model_stream events
  // that flow through ChatModelStreamHandler → ON_SUMMARIZE_DELTA for real-time UX.
  const partialSummaries: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const formattedText = formatMessagesForSummarization(chunk);

    let humanText: string;
    if (i === 0 && priorSummaryText) {
      humanText = `## Prior Summary\n\n${priorSummaryText}\n\n## Messages (Part ${i + 1} of ${chunks.length})\n\n${formattedText}`;
    } else {
      humanText = `## Messages (Part ${i + 1} of ${chunks.length})\n\n${formattedText}`;
    }

    const text = await streamAndCollect(
      model,
      [new SystemMessage(promptText), new HumanMessage(humanText)],
      traceConfig(config, 'multi_stage_chunk', {
        chunk_index: i,
        total_chunks: chunks.length,
      }),
      streaming
    );

    if (text) {
      partialSummaries.push(text);
    }
  }

  // If only one chunk produced output, return it directly
  if (partialSummaries.length <= 1) {
    return partialSummaries[0] ?? '';
  }

  // Stage 2: Merge partial summaries — this is the final call, so stream it.
  const mergeInput = partialSummaries
    .map((summary, i) => `## Part ${i + 1}\n\n${summary}`)
    .join('\n\n');

  return streamAndCollect(
    model,
    [new SystemMessage(MERGE_PROMPT), new HumanMessage(mergeInput)],
    traceConfig(config, 'multi_stage_merge', { total_chunks: chunks.length }),
    streaming
  );
}
