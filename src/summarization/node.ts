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
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { createRemoveAllMessage } from '@/messages/reducer';
import { getChatModelClass } from '@/llm/providers';
import { getChunkContent } from '@/stream';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Minimum per-field character limits. The adaptive budget system scales these
 * up based on available budget and message count, but never below these floors.
 */
const MIN_LIMITS = {
  toolArg: 200,
  toolResult: 400,
  messageContent: 300,
} as const;

/**
 * Computes adaptive per-field character limits based on the total budget
 * and the number of messages. More budget per message = higher limits.
 * Recent messages (higher index) get a larger share via recency weighting.
 */
function computeAdaptiveLimits(
  budgetChars: number,
  messageCount: number,
  messageIndex: number
): { toolArg: number; toolResult: number; messageContent: number } {
  if (messageCount <= 0) {
    return { ...MIN_LIMITS };
  }

  // Recency weight: messages in the last third get 2x, middle third 1x, first third 0.5x
  const position = messageCount > 1 ? messageIndex / (messageCount - 1) : 1;
  const recencyMultiplier = position > 0.66 ? 2.0 : position > 0.33 ? 1.0 : 0.5;

  // Base per-message budget: divide total budget by weighted message count
  // Average weight across all messages ≈ 1.17, so use messageCount directly
  const perMessageBudget = Math.floor(
    (budgetChars / messageCount) * recencyMultiplier
  );

  // Allocate: 60% to tool results (high signal), 30% to message content, 10% to tool args
  return {
    toolArg: Math.max(MIN_LIMITS.toolArg, Math.floor(perMessageBudget * 0.1)),
    toolResult: Math.max(
      MIN_LIMITS.toolResult,
      Math.floor(perMessageBudget * 0.6)
    ),
    messageContent: Math.max(
      MIN_LIMITS.messageContent,
      Math.floor(perMessageBudget * 0.3)
    ),
  };
}

/**
 * Token budget for a single summarization chunk.
 * When tokenCounter is available, actual token counts are used for
 * chunking/splitting decisions. The char-based formatting budget is
 * derived from this via a ~4:1 chars-per-token ratio.
 */
const DEFAULT_MAX_INPUT_TOKENS = 10_000;

/** Default number of parts for multi-stage summarization (1 = single-pass). */
const DEFAULT_PARTS = 1;

/** Minimum messages before multi-stage split is attempted. */
const DEFAULT_MIN_MESSAGES_FOR_SPLIT = 4;

/**
 * Known summarization-specific parameter keys that must NOT be passed to the
 * LLM constructor. These control summarization behavior, not model behavior.
 */
const SUMMARIZATION_PARAM_KEYS = new Set([
  'parts',
  'minMessagesForSplit',
  'maxSummaryTokens',
]);

// ---------------------------------------------------------------------------
// Message formatting for summarization input
// ---------------------------------------------------------------------------

/** Remove unpaired surrogates from the edges of a sliced string. */
function stripBrokenSurrogates(s: string): string {
  let start = 0;
  let end = s.length;
  // Leading low surrogate (orphaned tail of a pair)
  if (
    end > 0 &&
    s.charCodeAt(start) >= 0xdc00 &&
    s.charCodeAt(start) <= 0xdfff
  ) {
    start++;
  }
  // Trailing high surrogate (orphaned head of a pair)
  if (
    end > start &&
    s.charCodeAt(end - 1) >= 0xd800 &&
    s.charCodeAt(end - 1) <= 0xdbff
  ) {
    end--;
  }
  return start === 0 && end === s.length ? s : s.slice(start, end);
}

/**
 * Truncates a string to `maxLen` characters, appending an ellipsis indicator
 * with the original length so the summarizer knows content was cut.
 */
function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) {
    return text;
  }
  const sliced = stripBrokenSurrogates(text.slice(0, maxLen));
  return `${sliced}… [truncated, ${text.length} chars total]`;
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
  const headStr = stripBrokenSurrogates(text.slice(0, head));
  const tailStr = stripBrokenSurrogates(text.slice(text.length - tail));
  return headStr + indicator + tailStr;
}

/**
 * Formats tool call args for summarization. Produces a compact representation
 * instead of dumping raw JSON (which can be thousands of chars for code-heavy
 * tool calls like script evaluation or file writes).
 */
function formatToolArgs(
  args: Record<string, unknown> | undefined,
  toolArgLimit: number = MIN_LIMITS.toolArg
): string {
  if (!args || typeof args !== 'object') {
    return '';
  }
  const parts: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    if (value == null) {
      continue;
    }
    const valueStr = typeof value === 'string' ? value : JSON.stringify(value);
    parts.push(`${key}: ${truncate(valueStr, toolArgLimit)}`);
  }
  return parts.join(', ');
}

function formatMessageForSummary(
  msg: BaseMessage,
  limits: ReturnType<typeof computeAdaptiveLimits>
): string {
  const role = msg.getType();

  // Tool result messages: format as "[tool_result: name] → content"
  if (role === 'tool') {
    const name = msg.name ?? 'unknown';
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
    return `[tool_result: ${name}] → ${smartTruncate(content, limits.toolResult)}`;
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
        parts.push(smartTruncate(msg.content.trim(), limits.messageContent));
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
          parts.push(smartTruncate(block.text.trim(), limits.messageContent));
        }
      }
    }

    // Format each tool call with truncated args
    for (const tc of msg.tool_calls) {
      const argsStr = formatToolArgs(
        tc.args as Record<string, unknown>,
        limits.toolArg
      );
      parts.push(`[tool_call: ${tc.name}(${argsStr})]`);
    }

    return `[${role}]: ${parts.join('\n')}`;
  }

  // All other messages: extract text content, skip reasoning/thinking blocks
  let content: string;
  if (typeof msg.content === 'string') {
    content = msg.content;
  } else if (Array.isArray(msg.content)) {
    const textParts: string[] = [];
    for (const block of msg.content) {
      if (typeof block === 'string') {
        textParts.push(block);
      } else if (
        typeof block === 'object' &&
        'type' in block &&
        block.type === 'text' &&
        'text' in block &&
        typeof block.text === 'string'
      ) {
        textParts.push(block.text);
      }
    }
    content = textParts.join('\n');
  } else {
    content = '';
  }
  return `[${role}]: ${smartTruncate(content, limits.messageContent)}`;
}

/**
 * Formats messages for the summarization prompt, applying a total character
 * budget. Individual messages are first formatted with per-field limits, then
 * if the combined text exceeds the budget, the longest formatted messages are
 * proportionally trimmed so every message retains representation.
 */
function formatMessagesForSummarization(
  messages: BaseMessage[],
  budgetChars: number = DEFAULT_MAX_INPUT_TOKENS * 4
): string {
  const formatted = messages.map((msg, i) => {
    const limits = computeAdaptiveLimits(budgetChars, messages.length, i);
    return formatMessageForSummary(msg, limits);
  });

  const totalChars = formatted.reduce((sum, s) => sum + s.length, 0);
  if (totalChars <= budgetChars) {
    return formatted.join('\n');
  }

  // Over budget, trim the longest messages proportionally,
  // but weight recent messages higher to preserve their content.
  const weights = formatted.map((_, i) => {
    const position = messages.length > 1 ? i / (messages.length - 1) : 1;
    return position > 0.66 ? 2.0 : position > 0.33 ? 1.0 : 0.5;
  });
  const totalWeight = weights.reduce((a, b) => a + b, 0);

  const trimmed = formatted.map((text, i) => {
    const share = (weights[i] / totalWeight) * budgetChars;
    const allowance = Math.max(80, Math.floor(share));
    return text.length <= allowance ? text : truncate(text, allowance);
  });

  return trimmed.join('\n');
}

// ---------------------------------------------------------------------------
// Adaptive chunk sizing
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Multi-stage summarization
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Prompt & config helpers
// ---------------------------------------------------------------------------

/**
 * Structured checkpoint format for fresh summarization (no prior summary).
 * Adapted from pi-coding-agent's SUMMARIZATION_PROMPT pattern.
 * The explicit section structure ensures consistent, machine-parseable output
 * that survives multiple summarization cycles without information loss.
 */
export const DEFAULT_SUMMARIZATION_PROMPT = `Hold on, before you continue I need you to write me a checkpoint of everything so far. Your context window is filling up and this checkpoint replaces the messages above, so capture everything you need to pick right back up.

Don't second-guess or fact-check anything you did, your tool results reflect exactly what happened. Just record what you did and what you observed. Only the checkpoint, don't respond to me or continue the conversation.

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
What you need to do next, in priority order.

## Critical Context
Exact identifiers, names, error messages, URLs, and details you need to preserve verbatim.

Rules:
- Record what you did and observed, don't judge or re-evaluate it
- For each tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Short declarative sentences
- Skip empty sections`;

/**
 * Prompt for incremental summary updates when a prior summary exists.
 * Adapted from pi-coding-agent's UPDATE_SUMMARIZATION_PROMPT pattern.
 * The rules ensure existing information is preserved while new information
 * is integrated into the correct sections.
 */
export const DEFAULT_UPDATE_SUMMARIZATION_PROMPT = `Hold on again, update your checkpoint. Merge the new messages into your existing checkpoint and give me a single consolidated replacement.

Keep it roughly the same length as your last checkpoint. Compress older details to make room for what's new, don't just append. Give recent actions more detail, compress older items to one-liners.

Don't fact-check or second-guess anything, your tool results are ground truth. Only the checkpoint, don't respond to me or continue the conversation.

Rules:
- Merge new progress into existing sections, don't duplicate headers
- Compress older completed items into one-line entries
- Move items from "In Progress" to "Done" when you completed them
- Update "Next Steps" to reflect current priorities
- For each new tool call: the tool name, key inputs, and the outcome
- Preserve exact identifiers, names, errors, and references verbatim
- Skip empty sections`;

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

// ---------------------------------------------------------------------------
// Post-LLM summary enrichment
// ---------------------------------------------------------------------------
// After the LLM produces
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
 * from their summaries, this mechanical enrichment guarantees they survive.
 */
function extractToolFailuresSection(messages: BaseMessage[]): string {
  const failures: Array<{ toolName: string; summary: string }> = [];
  const seen = new Set<string>();

  for (const msg of messages) {
    if (msg.getType() !== 'tool') {
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
    dispatchRunStep: (
      runStep: t.RunStep,
      config?: RunnableConfig
    ) => Promise<void>;
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
  ): Promise<{ summarizationRequest: undefined; messages?: BaseMessage[] }> => {
    const request = state.summarizationRequest;
    if (request == null) {
      return { summarizationRequest: undefined };
    }

    // ----- Budget check: skip if instructions alone exceed the context budget -----
    // When this happens, summarization would only make things worse, the summary
    // gets added to the system message, further increasing instruction overhead.
    // Log clearly so the issue is visible in debug logs.
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
    const maxInputTokens =
      summarizationConfig?.maxInputTokens ?? DEFAULT_MAX_INPUT_TOKENS;

    // Separate summarization-specific params from LLM constructor params.
    // This prevents keys like `parts`, `minMessagesForSplit` from being
    // passed to the ChatModel constructor where they don't belong.
    const { llmParams, summarizationParams } = separateParameters(parameters);

    // When the summarization provider matches the agent's provider (self-summarize),
    // reuse the agent's clientOptions as the base. This ensures provider-specific
    // settings (region, credentials, endpoint, proxy, etc.) are inherited without
    // requiring the caller to forward them explicitly.
    const isSelfSummarize = provider === agentContext.provider;
    const baseOptions =
      isSelfSummarize && agentContext.clientOptions
        ? { ...agentContext.clientOptions }
        : {};

    // Build LLM constructor options, clean of any main agent context.
    // The summarizer is its own agent: no tools, no system instructions
    // from the parent, just the summarization prompt.
    const clientOptions: Record<string, unknown> = {
      ...baseOptions,
      ...llmParams,
    };

    // Strip agent-specific fields that don't apply to summarization.

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
    // It does NOT carry system instructions, tools, or agent persona, those
    // are bound to the model instance or in the message array respectively.
    // We pass this to both event dispatch and model.invoke() for traceability.
    const runnableConfig = config ?? graph.config;

    // ----- Set up run step -----
    const stepKey = `summarize-${request.agentId}`;
    const [stepId, stepIndex] = generateStepId(stepKey);

    const placeholderSummary: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
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

    await graph.dispatchRunStep(runStep, runnableConfig);

    if (runnableConfig) {
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

    // Check for a prior summary, either from a previous summarization within
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

    // S1: Log before summarization LLM call
    const strategy =
      parts > 1 && messagesToRefine.length >= Math.max(2, minMessagesForSplit)
        ? 'multi_stage'
        : 'single_pass';
    emitAgentLog(
      runnableConfig,
      'info',
      'summarize',
      'Summarization starting',
      {
        strategy,
        parts,
        messagesToRefineCount: messagesToRefine.length,
        hasPriorSummary: priorSummaryText !== '',
        summaryVersion: agentContext.summaryVersion + 1,
      },
      { runId: graph.runId, agentId: request.agentId }
    );

    // Tier 1: Send raw conversation messages with the summarization instruction
    // appended as the final HumanMessage. This preserves the original message
    // format and enables cache hits on the system prompt + tool definitions.
    // Bind the agent's tools so providers that require tool definitions when
    // tool_use/tool_result blocks are present (e.g. Bedrock) accept the messages.
    const tools = agentContext.getToolsForBinding();
    const tier1Model =
      tools != null &&
      tools.length > 0 &&
      typeof (model as t.ModelWithTools).bindTools === 'function'
        ? ((model as t.ModelWithTools).bindTools(
          tools
        ) as unknown as SummarizeModel)
        : model;
    try {
      summaryText = await summarizeWithCacheHit({
        model: tier1Model,
        messages: messagesToRefine,
        promptText,
        updatePromptText,
        priorSummaryText,
        config: summarizeConfig,
        streaming,
        stepId,
        provider: provider as Providers,
        reasoningKey: agentContext.reasoningKey,
      });
    } catch (err) {
      // S2a: Log tier 1 failure, falling back to tier 2
      emitAgentLog(
        runnableConfig,
        'warn',
        'summarize',
        'Tier 1 failed, falling back',
        {
          tier: 1,
          fallback: 'single_pass_half_budget',
          error: err instanceof Error ? err.message : String(err),
        },
        { runId: graph.runId, agentId: request.agentId }
      );
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
          stepId,
          budgetChars: Math.floor((maxInputTokens * 4) / 2),
          tokenCounter: agentContext.tokenCounter,
          provider: provider as Providers,
        });
      } catch (err2) {
        // S2b: Log tier 2 failure, falling back to tier 3
        emitAgentLog(
          runnableConfig,
          'warn',
          'summarize',
          'Tier 2 failed, falling back',
          {
            tier: 2,
            fallback: 'metadata_stub',
            error: err2 instanceof Error ? err2.message : String(err2),
          },
          { runId: graph.runId, agentId: request.agentId }
        );
        // Tier 3: Metadata stub, no LLM call, preserves tool names and message counts
        summaryText = generateMetadataStub(messagesToRefine);
      }
    }

    if (!summaryText) {
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

    // ----- Enrich with structured data the LLM may have omitted -----
    summaryText = enrichSummary(summaryText, messagesToRefine);

    // ----- Finalize -----
    let tokenCount = 0;
    if (agentContext.tokenCounter) {
      tokenCount = agentContext.tokenCounter(new SystemMessage(summaryText));
    }

    agentContext.setSummary(summaryText, tokenCount);

    // S3: Log after summary persisted
    emitAgentLog(
      runnableConfig,
      'info',
      'summarize',
      'Summary persisted',
      {
        summaryTokens: tokenCount,
        textLength: summaryText.length,
        contextLength: request.context.length,
        survivingMessages: request.context.length,
        summaryVersion: agentContext.summaryVersion,
      },
      { runId: graph.runId, agentId: request.agentId }
    );

    const summaryBlock: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      content: [
        {
          type: ContentTypes.TEXT,
          text: summaryText,
        } as t.MessageContentComplex,
      ],
      tokenCount,
      summaryVersion: agentContext.summaryVersion,
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
          id: stepId,
          agentId: request.agentId,
          summary: summaryBlock,
        } satisfies t.SummarizeCompleteEvent,
        runnableConfig
      );
    }

    // ----- Clean slate: remove all messages from graph state -----
    // Full compaction: the entire conversation has been captured in the
    // summary.  The model restarts with only the summary (injected as a
    // user message by the agent node) — no surviving messages.
    agentContext.rebuildTokenMapAfterSummarization({});

    return {
      summarizationRequest: undefined,
      messages: [createRemoveAllMessage()],
    };
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
  /** Step ID for dispatching ON_SUMMARIZE_DELTA events during streaming. */
  stepId?: string;
  /** Token counter from the graph's agent context. When provided, enables accurate token-based budgeting for chunking decisions. */
  tokenCounter?: (msg: BaseMessage) => number;
  /** Provider for chunk content extraction (determines how reasoning tokens are handled). */
  provider?: Providers;
  /** Key used for reasoning content in additional_kwargs. */
  reasoningKey?: 'reasoning_content' | 'reasoning';
  /** Maximum input tokens per summarization pass. */
  maxInputTokens?: number;
}

/**
 * Extracts the text from an LLM response, handling both string and
 * structured content formats.
 */
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
    // Skip reasoning/thinking blocks, only collect text output.
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

/**
 * Streams an LLM call and collects the full response text.
 * Dispatches `ON_SUMMARIZE_DELTA` events directly for each chunk so the
 * client can render streaming summarization output in real time.
 * Falls back to `model.invoke()` when streaming isn't supported or is disabled.
 *
 * @param streaming - When false, forces invoke() even if model supports stream().
 *   Defaults to true.
 * @param stepId - Step ID for dispatching ON_SUMMARIZE_DELTA events.
 */
async function streamAndCollect(
  model: SummarizeModel,
  messages: BaseMessage[],
  config?: RunnableConfig,
  streaming = true,
  stepId?: string,
  provider?: Providers,
  reasoningKey: 'reasoning_content' | 'reasoning' = 'reasoning_content'
): Promise<string> {
  if (!streaming || typeof model.stream !== 'function') {
    const response = await model.invoke(messages, config);
    return extractResponseText(response);
  }

  const textBlocks: t.MessageContentComplex[] = [];
  const reasoningBlocks: t.MessageContentComplex[] = [];
  const stream = await model.stream(messages, config);
  for await (const chunk of stream) {
    const chunkAny = chunk as Parameters<typeof getChunkContent>[0]['chunk'];
    const raw = getChunkContent({
      chunk: chunkAny,
      provider,
      reasoningKey,
    });
    if (raw == null || (typeof raw === 'string' && !raw)) {
      continue;
    }

    // Normalize to MessageContentComplex[], getChunkContent may return a string.
    const contentBlocks: t.MessageContentComplex[] =
      typeof raw === 'string'
        ? [{ type: ContentTypes.TEXT, text: raw } as t.MessageContentComplex]
        : raw;

    // Detect reasoning: chunk has reasoning_content in additional_kwargs
    // or content is an array with reasoning-type blocks.
    const hasReasoningKwarg =
      chunkAny?.additional_kwargs?.[reasoningKey] != null &&
      chunkAny.additional_kwargs[reasoningKey] !== '';
    const hasReasoningContent =
      Array.isArray(chunkAny?.content) &&
      chunkAny.content.length > 0 &&
      (chunkAny.content[0]?.type === ContentTypes.THINKING ||
        chunkAny.content[0]?.type === ContentTypes.REASONING_CONTENT);
    const isReasoning = hasReasoningKwarg || hasReasoningContent;

    if (isReasoning) {
      for (const block of contentBlocks) {
        reasoningBlocks.push(block);
      }
    } else {
      for (const block of contentBlocks) {
        textBlocks.push(block);
      }
    }

    // Stream both reasoning and text to the client.
    if (stepId != null && stepId !== '' && config) {
      safeDispatchCustomEvent(
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
    }
  }

  let textResult = '';
  for (let i = 0; i < textBlocks.length; i++) {
    if ('text' in textBlocks[i]) {
      textResult += (textBlocks[i] as { text: string }).text;
    }
  }
  if (textResult.trim()) {
    return textResult.trim();
  }
  let reasoningResult = '';
  for (let i = 0; i < reasoningBlocks.length; i++) {
    if ('text' in reasoningBlocks[i]) {
      reasoningResult += (reasoningBlocks[i] as { text: string }).text;
    }
  }
  return reasoningResult.trim();
}

/**
 * Cache-friendly compaction: sends the raw conversation messages to the LLM
 * with the summarization instruction appended as the final HumanMessage.
 *
 * Because the messages are in the same format (and same prefix) as the agent's
 * last call, providers with prompt caching (Anthropic, Bedrock) get a cache hit
 * on the system prompt + tool definitions. Only the instruction is new tokens.
 *
 * The messages have already been through the pruner's progressive truncation,
 * so oversized tool results are already trimmed.
 */
async function summarizeWithCacheHit({
  model,
  messages,
  promptText,
  updatePromptText,
  priorSummaryText,
  config,
  streaming,
  stepId,
  provider,
  reasoningKey,
}: Omit<SummarizeParams, 'maxInputTokens'>): Promise<string> {
  const effectivePrompt = priorSummaryText
    ? (updatePromptText ?? promptText)
    : promptText;

  // Build the instruction as the final human message
  const instructionParts = [effectivePrompt];
  if (priorSummaryText) {
    instructionParts.push(
      `\n\n<previous-summary>\n${priorSummaryText}\n</previous-summary>`
    );
  }

  // Pass the raw conversation messages with instruction appended
  const messagesForLLM = [
    ...messages,
    new HumanMessage(instructionParts.join('')),
  ];

  return streamAndCollect(
    model,
    messagesForLLM,
    traceConfig(config, 'cache_hit_compaction'),
    streaming,
    stepId,
    provider,
    reasoningKey
  );
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
  stepId,
  budgetChars,
  provider,
  reasoningKey,
  maxInputTokens: passedMaxInputTokens,
}: SummarizeParams & { budgetChars?: number }): Promise<string> {
  const effectiveBudgetChars =
    budgetChars ?? (passedMaxInputTokens ?? DEFAULT_MAX_INPUT_TOKENS) * 4;
  const formattedText = formatMessagesForSummarization(
    messages,
    effectiveBudgetChars
  );

  // S-SP: Log what the summarizer will see
  const msgTypes = messages.map((m) => {
    const t = m.getType();
    return t === 'tool' ? `tool(${m.name ?? '?'})` : t;
  });
  emitAgentLog(
    config,
    'debug',
    'summarize',
    'Single-pass input',
    {
      messageCount: messages.length,
      messageTypes: msgTypes.join(', '),
      formattedChars: formattedText.length,
      hasPriorSummary: priorSummaryText !== '',
      priorSummaryChars: priorSummaryText.length,
    },
    {
      runId: config?.metadata?.runId as string | undefined,
      agentId: config?.metadata?.agent_id as string | undefined,
    }
  );

  /** Selected as the effective prompt: updated variant when integrating
   * new messages into an existing summary, main prompt otherwise.
   */
  const effectivePrompt = priorSummaryText
    ? (updatePromptText ?? promptText)
    : promptText;

  const summarySection = priorSummaryText
    ? `<previous-summary>\n${priorSummaryText}\n</previous-summary>\n\n`
    : '';
  const humanMessageText = `${summarySection}<conversation>\n${formattedText}\n</conversation>`;

  return streamAndCollect(
    model,
    [new SystemMessage(effectivePrompt), new HumanMessage(humanMessageText)],
    traceConfig(config, 'single_pass'),
    streaming,
    stepId,
    provider,
    reasoningKey
  );
}
