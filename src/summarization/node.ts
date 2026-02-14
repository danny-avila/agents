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

/**
 * Formats a message for summarization input. Produces human-readable text instead of
 * raw JSON for structured content like tool calls and tool results.
 */
function formatMessageForSummary(msg: BaseMessage): string {
  const role = msg._getType();

  // Tool result messages: format as "[tool_result: name] → content"
  if (role === 'tool') {
    const name = msg.name ?? 'unknown';
    const content =
      typeof msg.content === 'string'
        ? msg.content
        : JSON.stringify(msg.content);
    return `[tool_result: ${name}] → ${content}`;
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
        parts.push(msg.content.trim());
      }
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (
          block != null &&
          typeof block === 'object' &&
          'type' in block &&
          block.type === 'text' &&
          'text' in block &&
          typeof block.text === 'string' &&
          block.text.trim()
        ) {
          parts.push(block.text.trim());
        }
      }
    }

    // Format each tool call readably
    for (const tc of msg.tool_calls) {
      const argsStr = tc.args ? JSON.stringify(tc.args) : '';
      parts.push(`[tool_call: ${tc.name}(${argsStr})]`);
    }

    return `[${role}]: ${parts.join('\n')}`;
  }

  // All other messages: use content directly
  const content =
    typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
  return `[${role}]: ${content}`;
}

export const DEFAULT_SUMMARIZATION_PROMPT =
  'You are a summarization assistant. Summarize the following conversation messages concisely, preserving key facts, decisions, and context needed to continue the conversation. Do not include preamble -- output only the summary.';

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

    const summarizationConfig = agentContext.summarizationConfig;
    const provider = summarizationConfig?.provider ?? agentContext.provider;
    const modelName = summarizationConfig?.model;
    const parameters = summarizationConfig?.parameters ?? {};
    const promptText =
      summarizationConfig?.prompt ?? DEFAULT_SUMMARIZATION_PROMPT;

    const clientOptions: Record<string, unknown> = {
      disableStreaming: true,
      ...parameters,
    };
    if (modelName != null && modelName !== '') {
      clientOptions.model = modelName;
      clientOptions.modelName = modelName;
    }

    const runnableConfig = config ?? graph.config;

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

    const ChatModelClass = getChatModelClass(provider as Providers);
    const model = new ChatModelClass(clientOptions as never);

    // Check for a prior summary in the kept context (injected by formatAgentMessages on cross-run).
    // Without this, re-summarization would only cover newly pruned messages, losing all
    // information from the previous summary.
    let priorSummaryText = '';
    if (request.context.length > 0) {
      const firstContext = request.context[0];
      if (firstContext._getType() === 'system') {
        const text =
          typeof firstContext.content === 'string'
            ? firstContext.content
            : JSON.stringify(firstContext.content);
        if (text.trim().length > 0) {
          priorSummaryText = text.trim();
        }
      }
    }

    const messagesToRefineText = request.messagesToRefine
      .map(formatMessageForSummary)
      .join('\n');

    const humanMessageText = priorSummaryText
      ? `## Prior Summary\n\n${priorSummaryText}\n\n## New Messages to Incorporate\n\n${messagesToRefineText}`
      : messagesToRefineText;

    const messages: BaseMessage[] = [
      new SystemMessage(promptText),
      new HumanMessage(humanMessageText),
    ];

    let summaryText = '';
    let tokenCount = 0;
    try {
      const response = await model.invoke(messages, config);
      summaryText = (
        typeof response.content === 'string'
          ? response.content
          : JSON.stringify(response.content)
      ).trim();

      if (agentContext.tokenCounter) {
        tokenCount = agentContext.tokenCounter(new SystemMessage(summaryText));
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
      const deltaEvent: t.SummarizeDeltaEvent = {
        id: stepId,
        delta: { summary: summaryBlock },
      };
      await safeDispatchCustomEvent(
        GraphEvents.ON_SUMMARIZE_DELTA,
        deltaEvent,
        runnableConfig
      );

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
