import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { getChatModelClass } from '@/llm/providers';
import { safeDispatchCustomEvent } from '@/utils/events';
import { ContentTypes, GraphEvents, StepTypes, Providers } from '@/common';
import type { AgentContext } from '@/agents/AgentContext';

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
        } satisfies t.SummarizeStartEvent,
        runnableConfig
      );
    }

    const ChatModelClass = getChatModelClass(provider as Providers);
    const model = new ChatModelClass(clientOptions as never);

    const messagesToRefineText = request.messagesToRefine
      .map((msg) => {
        const role = msg._getType();
        const content =
          typeof msg.content === 'string'
            ? msg.content
            : JSON.stringify(msg.content);
        return `[${role}]: ${content}`;
      })
      .join('\n');

    const messages: BaseMessage[] = [
      new SystemMessage(promptText),
      new HumanMessage(messagesToRefineText),
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
    } catch {
      return { summarizationRequest: undefined };
    }

    if (!summaryText) {
      return { summarizationRequest: undefined };
    }

    agentContext.setSummary(summaryText, tokenCount);

    const summaryBlock: t.SummaryContentBlock = {
      type: ContentTypes.SUMMARY,
      text: summaryText,
      tokenCount,
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
