import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { AgentContext } from '@/agents/AgentContext';
import type { OnChunk } from '@/llm/invoke';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, StepTypes, Providers } from '@/common';
import { safeDispatchCustomEvent, emitAgentLog } from '@/utils/events';
import { attemptInvoke, tryFallbackProviders } from '@/llm/invoke';
import { createRemoveAllMessage } from '@/messages/reducer';
import { getMaxOutputTokensKey } from '@/llm/request';
import { addCacheControl } from '@/messages/cache';
import { initializeModel } from '@/llm/init';
import { getChunkContent } from '@/stream';

const SUMMARIZATION_PARAM_KEYS = new Set(['maxSummaryTokens']);

/**
 * Token overhead of the XML wrapper + instruction text added around the
 * summary at injection time in AgentContext.buildSystemRunnable:
 * `<summary>\n${text}\n</summary>\n\nYour context window was compacted...`
 * ~33 tokens on Anthropic, ~24-27 on OpenAI.  Using 33 as a safe ceiling.
 */
const SUMMARY_WRAPPER_OVERHEAD_TOKENS = 33;

/** Structured checkpoint prompt for fresh summarization (no prior summary). */
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

/** Prompt for re-compaction when a prior summary exists. */
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
 * Extracts usage metadata from an LLM response message, handling provider differences:
 * - Standard: `usage_metadata` directly on the message
 * - Bedrock: `response_metadata.metadata.usage` with camelCase keys
 * - Other: `response_metadata.usage`
 */
function extractUsageFromMessage(
  message:
    | Partial<{ usage_metadata: unknown; response_metadata: unknown }>
    | undefined
): Partial<UsageMetadata> | undefined {
  if (message == null) {
    return undefined;
  }

  const usageMeta = message.usage_metadata as
    | Partial<UsageMetadata>
    | undefined;
  if (usageMeta?.input_tokens != null || usageMeta?.output_tokens != null) {
    return usageMeta;
  }

  const respMeta = message.response_metadata as
    | Record<string, unknown>
    | undefined;
  if (respMeta == null) {
    return undefined;
  }

  // Bedrock nests usage under response_metadata.metadata.usage with camelCase keys
  const nested =
    (respMeta.metadata as Record<string, unknown> | undefined)?.usage ??
    respMeta.usage;
  if (nested == null || typeof nested !== 'object') {
    return undefined;
  }

  const raw = nested as Record<string, unknown>;
  const inputTokens = Number(raw.inputTokens ?? raw.input_tokens) || undefined;
  const outputTokens =
    Number(raw.outputTokens ?? raw.output_tokens) || undefined;
  if (inputTokens == null && outputTokens == null) {
    return undefined;
  }

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    ...(raw.cacheReadInputTokens != null || raw.cacheWriteInputTokens != null
      ? {
        input_token_details: {
          cache_read: Number(raw.cacheReadInputTokens) || undefined,
          cache_creation: Number(raw.cacheWriteInputTokens) || undefined,
        },
      }
      : {}),
  } as Partial<UsageMetadata>;
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
    dispatchRunStepCompleted: (
      stepId: string,
      result: t.StepCompleted,
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

    const summarizationConfig = agentContext.summarizationConfig;
    const provider = summarizationConfig?.provider ?? agentContext.provider;
    const modelName = summarizationConfig?.model;
    const parameters = summarizationConfig?.parameters ?? {};
    const promptText =
      summarizationConfig?.prompt ?? DEFAULT_SUMMARIZATION_PROMPT;
    const updatePromptText =
      summarizationConfig?.updatePrompt ?? DEFAULT_UPDATE_SUMMARIZATION_PROMPT;

    const { llmParams, maxSummaryTokens: paramMaxSummaryTokens } =
      separateParameters(parameters);

    const isSelfSummarize = provider === agentContext.provider;
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
      clientOptions[getMaxOutputTokensKey(provider as string)] =
        effectiveMaxSummaryTokens;
    }

    const runnableConfig = config ?? graph.config;

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

    const priorSummaryText = agentContext.getSummaryText()?.trim() ?? '';
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

    let summaryText = '';
    let summaryUsage: Partial<UsageMetadata> | undefined;
    const messagesToRefine = request.messagesToRefine;

    const isSelfSummarizeModel = provider === agentContext.provider;
    const hasPromptCache =
      isSelfSummarizeModel &&
      (agentContext.clientOptions as Record<string, unknown> | undefined)
        ?.promptCache === true;

    emitAgentLog(
      runnableConfig,
      'debug',
      'summarize',
      'Summarization starting',
      {
        messagesToRefineCount: messagesToRefine.length,
        hasPriorSummary: priorSummaryText !== '',
        summaryVersion: agentContext.summaryVersion + 1,
        isSelfSummarize: isSelfSummarizeModel,
        hasPromptCache,
        provider: provider as string,
      },
      { runId: graph.runId, agentId: request.agentId }
    );

    const summarizationModel = initializeModel({
      provider: provider as Providers,
      clientOptions: clientOptions as t.ClientOptions,
      tools: agentContext.getToolsForBinding(),
    }) as t.ChatModel;

    try {
      const summarizeResult = await summarizeWithCacheHit({
        model: summarizationModel,
        messages: messagesToRefine,
        promptText,
        updatePromptText,
        priorSummaryText,
        config: summarizeConfig,
        stepId,
        provider: provider as Providers,
        reasoningKey: agentContext.reasoningKey,
        usePromptCache: isSelfSummarizeModel && hasPromptCache,
      });
      summaryText = summarizeResult.text;
      summaryUsage = summarizeResult.usage;
    } catch (primaryError) {
      emitAgentLog(
        runnableConfig,
        'error',
        'summarize',
        'Summarization LLM call failed',
        {
          error:
            primaryError instanceof Error
              ? primaryError.message
              : String(primaryError),
          provider: provider as string,
          model: modelName,
          messagesToRefineCount: messagesToRefine.length,
        },
        { runId: graph.runId, agentId: request.agentId }
      );

      const fallbacks =
        (clientOptions as unknown as t.LLMConfig | undefined)?.fallbacks ?? [];
      if (fallbacks.length > 0) {
        try {
          const onChunk = createSummarizationChunkHandler({
            stepId,
            config: traceConfig(summarizeConfig, 'cache_hit_compaction'),
            provider: provider as Providers,
            reasoningKey: agentContext.reasoningKey,
          });
          const fbResult = await tryFallbackProviders({
            fallbacks,
            tools: agentContext.getToolsForBinding(),
            messages: [
              ...messagesToRefine,
              new HumanMessage(
                buildSummarizationInstruction(
                  promptText,
                  updatePromptText,
                  priorSummaryText
                )
              ),
            ],
            config: traceConfig(summarizeConfig, 'cache_hit_compaction'),
            primaryError,
            onChunk,
          });
          const fbMsg = fbResult?.messages?.[0];
          if (fbMsg) {
            summaryText = extractResponseText(
              fbMsg as { content: string | object }
            );
          }
        } catch {
          // Fallbacks exhausted
        }
      }
      if (!summaryText) {
        emitAgentLog(
          runnableConfig,
          'warn',
          'summarize',
          'Summarization failed, falling back to metadata stub',
          {
            error:
              primaryError instanceof Error
                ? primaryError.message
                : String(primaryError),
          },
          { runId: graph.runId, agentId: request.agentId }
        );
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

    summaryText = enrichSummary(summaryText, messagesToRefine);

    let tokenCount = 0;
    const providerOutputTokens = Number(summaryUsage?.output_tokens) || 0;
    if (providerOutputTokens > 0) {
      tokenCount = providerOutputTokens + SUMMARY_WRAPPER_OVERHEAD_TOKENS;
    } else if (agentContext.tokenCounter) {
      tokenCount =
        agentContext.tokenCounter(new SystemMessage(summaryText)) +
        SUMMARY_WRAPPER_OVERHEAD_TOKENS;
    }

    agentContext.setSummary(summaryText, tokenCount);

    if (summaryUsage) {
      agentContext.updateLastCallUsage(summaryUsage);
    }

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
        ...(summaryUsage != null
          ? {
            input_tokens: summaryUsage.input_tokens,
            output_tokens: summaryUsage.output_tokens,
            cache_read: summaryUsage.input_token_details?.cache_read,
            cache_creation: summaryUsage.input_token_details?.cache_creation,
          }
          : {}),
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
          agentId: request.agentId,
          summary: summaryBlock,
        } satisfies t.SummarizeCompleteEvent,
        runnableConfig
      );
    }

    agentContext.rebuildTokenMapAfterSummarization({});

    return {
      summarizationRequest: undefined,
      messages: [createRemoveAllMessage()],
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
function createSummarizationChunkHandler({
  stepId,
  config,
  provider,
  reasoningKey = 'reasoning_content',
}: {
  stepId?: string;
  config?: RunnableConfig;
  provider?: Providers;
  reasoningKey?: 'reasoning_content' | 'reasoning';
}): OnChunk | undefined {
  if (stepId == null || stepId === '' || !config) {
    return undefined;
  }
  return (chunk) => {
    const chunkAny = chunk as Parameters<typeof getChunkContent>[0]['chunk'];
    const raw = getChunkContent({ chunk: chunkAny, provider, reasoningKey });
    if (raw == null || (typeof raw === 'string' && !raw)) {
      return;
    }
    const contentBlocks: t.MessageContentComplex[] =
      typeof raw === 'string'
        ? [{ type: ContentTypes.TEXT, text: raw } as t.MessageContentComplex]
        : raw;

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
  usePromptCache,
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
  usePromptCache?: boolean;
}): Promise<{ text: string; usage?: Partial<UsageMetadata> }> {
  const instruction = buildSummarizationInstruction(
    promptText,
    updatePromptText,
    priorSummaryText
  );

  // Apply cache markers to the full message array (conversation + instruction)
  // so the summarizer benefits from the cached prefix established by the
  // agent's prior calls.  The instruction HumanMessage becomes the last user
  // message and gets a cache breakpoint, while earlier conversation messages
  // form the cached prefix.
  const fullMessages = [...messages, new HumanMessage(instruction)];
  const invokeMessages =
    usePromptCache === true ? addCacheControl(fullMessages) : fullMessages;

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
      }),
    },
    traceConfig(config, 'cache_hit_compaction')
  );

  const responseMsg = result.messages?.[0];
  const text = responseMsg
    ? extractResponseText(responseMsg as { content: string | object })
    : '';
  const usage = extractUsageFromMessage(responseMsg);
  return { text, usage };
}
