/* eslint-disable no-console */
// src/tools/handlers.ts
import { nanoid } from 'nanoid';
import { ToolMessage } from '@langchain/core/messages';
import type { AnthropicWebSearchResultBlockParam } from '@/llm/anthropic/types';
import type { ToolCall, ToolCallChunk } from '@langchain/core/messages/tool';
import type { MultiAgentGraph, StandardGraph } from '@/graphs';
import type { AgentContext } from '@/agents/AgentContext';
import type * as t from '@/types';
import {
  ToolCallTypes,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
  Constants,
} from '@/common';
import {
  coerceAnthropicSearchResults,
  isAnthropicWebSearchResult,
} from '@/tools/search/anthropic';
import { formatResultsForLLM } from '@/tools/search/format';
import { getMessageId } from '@/messages';

export async function handleToolCallChunks({
  graph,
  stepKey,
  toolCallChunks,
}: {
  graph: StandardGraph | MultiAgentGraph;
  stepKey: string;
  toolCallChunks: ToolCallChunk[];
}): Promise<void> {
  let prevStepId: string;
  let prevRunStep: t.RunStep | undefined;
  try {
    prevStepId = graph.getStepIdByKey(stepKey, graph.contentData.length - 1);
    prevRunStep = graph.getRunStep(prevStepId);
  } catch {
    /** Edge Case: If no previous step exists, create a new message creation step */
    const message_id = getMessageId(stepKey, graph, true) ?? '';
    prevStepId = await graph.dispatchRunStep(stepKey, {
      type: StepTypes.MESSAGE_CREATION,
      message_creation: {
        message_id,
      },
    });
    prevRunStep = graph.getRunStep(prevStepId);
  }

  const _stepId = graph.getStepIdByKey(stepKey, prevRunStep?.index);

  /** Edge Case: Tool Call Run Step or `tool_call_ids` never dispatched */
  const tool_calls: ToolCall[] | undefined =
    prevStepId && prevRunStep && prevRunStep.type === StepTypes.MESSAGE_CREATION
      ? []
      : undefined;

  /** Edge Case: `id` and `name` fields cannot be empty strings */
  for (const toolCallChunk of toolCallChunks) {
    if (toolCallChunk.name === '') {
      toolCallChunk.name = undefined;
    }
    if (toolCallChunk.id === '') {
      toolCallChunk.id = undefined;
    } else if (
      tool_calls != null &&
      toolCallChunk.id != null &&
      toolCallChunk.name != null
    ) {
      tool_calls.push({
        args: {},
        id: toolCallChunk.id,
        name: toolCallChunk.name,
        type: ToolCallTypes.TOOL_CALL,
      });
    }
  }

  let stepId: string = _stepId;
  const alreadyDispatched =
    prevRunStep?.type === StepTypes.MESSAGE_CREATION &&
    graph.messageStepHasToolCalls.has(prevStepId);
  if (!alreadyDispatched && tool_calls?.length === toolCallChunks.length) {
    await graph.dispatchMessageDelta(prevStepId, {
      content: [
        {
          type: ContentTypes.TEXT,
          text: '',
          tool_call_ids: tool_calls.map((tc) => tc.id ?? ''),
        },
      ],
    });
    graph.messageStepHasToolCalls.set(prevStepId, true);
    stepId = await graph.dispatchRunStep(stepKey, {
      type: StepTypes.TOOL_CALLS,
      tool_calls,
    });
  }
  await graph.dispatchRunStepDelta(stepId, {
    type: StepTypes.TOOL_CALLS,
    tool_calls: toolCallChunks,
  });
}

export const handleToolCalls = async (
  toolCalls?: ToolCall[],
  metadata?: Record<string, unknown>,
  graph?: StandardGraph | MultiAgentGraph
): Promise<void> => {
  if (!graph || !metadata) {
    console.warn(`Graph or metadata not found in ${event} event`);
    return;
  }

  if (!toolCalls) {
    return;
  }

  if (toolCalls.length === 0) {
    return;
  }

  const stepKey = graph.getStepKey(metadata);

  for (const tool_call of toolCalls) {
    const toolCallId = tool_call.id ?? `toolu_${nanoid()}`;
    tool_call.id = toolCallId;
    if (!toolCallId || graph.toolCallStepIds.has(toolCallId)) {
      continue;
    }

    let prevStepId = '';
    let prevRunStep: t.RunStep | undefined;
    try {
      prevStepId = graph.getStepIdByKey(stepKey, graph.contentData.length - 1);
      prevRunStep = graph.getRunStep(prevStepId);
    } catch {
      // no previous step
    }

    const dispatchToolCallIds = async (
      lastMessageStepId: string
    ): Promise<void> => {
      await graph.dispatchMessageDelta(lastMessageStepId, {
        content: [
          {
            type: 'text',
            text: '',
            tool_call_ids: [toolCallId],
          },
        ],
      });
    };
    /* If the previous step exists and is a message creation */
    if (
      prevStepId &&
      prevRunStep &&
      prevRunStep.type === StepTypes.MESSAGE_CREATION
    ) {
      await dispatchToolCallIds(prevStepId);
      graph.messageStepHasToolCalls.set(prevStepId, true);
      /* If the previous step doesn't exist or is not a message creation */
    } else if (
      !prevRunStep ||
      prevRunStep.type !== StepTypes.MESSAGE_CREATION
    ) {
      const messageId = getMessageId(stepKey, graph, true) ?? '';
      const stepId = await graph.dispatchRunStep(stepKey, {
        type: StepTypes.MESSAGE_CREATION,
        message_creation: {
          message_id: messageId,
        },
      });
      await dispatchToolCallIds(stepId);
      graph.messageStepHasToolCalls.set(prevStepId, true);
    }

    await graph.dispatchRunStep(stepKey, {
      type: StepTypes.TOOL_CALLS,
      tool_calls: [tool_call],
    });
  }
};

export const toolResultTypes = new Set([
  // 'tool_use',
  // 'server_tool_use',
  // 'input_json_delta',
  'tool_result',
  'web_search_result',
  'web_search_tool_result',
]);

/**
 * Handles the result of a server tool call; in other words, a provider's built-in tool.
 * As of 2025-07-06, only Anthropic handles server tool calls with this pattern.
 */
export async function handleServerToolResult({
  graph,
  content,
  metadata,
  agentContext,
}: {
  graph: StandardGraph | MultiAgentGraph;
  content?: string | t.MessageContentComplex[];
  metadata?: Record<string, unknown>;
  agentContext?: AgentContext;
}): Promise<boolean> {
  let skipHandling = false;
  if (agentContext?.provider !== Providers.ANTHROPIC) {
    return skipHandling;
  }
  if (
    typeof content === 'string' ||
    content == null ||
    content.length === 0 ||
    (content.length === 1 &&
      (content[0] as t.ToolResultContent).tool_use_id == null)
  ) {
    return skipHandling;
  }

  for (const contentPart of content) {
    const toolUseId = (contentPart as t.ToolResultContent).tool_use_id;
    if (toolUseId == null || toolUseId === '') {
      continue;
    }
    const stepId = graph.toolCallStepIds.get(toolUseId);
    if (stepId == null || stepId === '') {
      console.warn(
        `Tool use ID ${toolUseId} not found in graph, cannot dispatch tool result.`
      );
      continue;
    }
    const runStep = graph.getRunStep(stepId);
    if (!runStep) {
      console.warn(
        `Run step for ${stepId} does not exist, cannot dispatch tool result.`
      );
      continue;
    } else if (runStep.type !== StepTypes.TOOL_CALLS) {
      console.warn(
        `Run step for ${stepId} is not a tool call step, cannot dispatch tool result.`
      );
      continue;
    }

    const toolCall =
      runStep.stepDetails.type === StepTypes.TOOL_CALLS
        ? (runStep.stepDetails.tool_calls?.find(
          (toolCall) => toolCall.id === toolUseId
        ) as ToolCall)
        : undefined;

    if (!toolCall) {
      continue;
    }

    if (
      contentPart.type === 'web_search_result' ||
      contentPart.type === 'web_search_tool_result'
    ) {
      await handleAnthropicSearchResults({
        contentPart: contentPart as t.ToolResultContent,
        toolCall,
        metadata,
        graph,
      });
    }

    if (!skipHandling) {
      skipHandling = true;
    }
  }

  return skipHandling;
}

async function handleAnthropicSearchResults({
  contentPart,
  toolCall,
  metadata,
  graph,
}: {
  contentPart: t.ToolResultContent;
  toolCall: Partial<ToolCall>;
  metadata?: Record<string, unknown>;
  graph: StandardGraph | MultiAgentGraph;
}): Promise<void> {
  if (!Array.isArray(contentPart.content)) {
    console.warn(
      `Expected content to be an array, got ${typeof contentPart.content}`
    );
    return;
  }

  if (!isAnthropicWebSearchResult(contentPart.content[0])) {
    console.warn(
      `Expected content to be an Anthropic web search result, got ${JSON.stringify(
        contentPart.content
      )}`
    );
    return;
  }

  const turn = graph.invokedToolIds?.size ?? 0;
  const searchResultData = coerceAnthropicSearchResults({
    turn,
    results: contentPart.content as AnthropicWebSearchResultBlockParam[],
  });

  const name = toolCall.name;
  const input = toolCall.args ?? {};
  const artifact = {
    [Constants.WEB_SEARCH]: searchResultData,
  };
  const { output: formattedOutput } = formatResultsForLLM(
    turn,
    searchResultData
  );
  const output = new ToolMessage({
    name,
    artifact,
    content: formattedOutput,
    tool_call_id: toolCall.id!,
  });
  const toolEndData: t.ToolEndData = {
    input,
    output,
  };
  await graph.handlerRegistry
    ?.getHandler(GraphEvents.TOOL_END)
    ?.handle(GraphEvents.TOOL_END, toolEndData, metadata, graph);

  if (graph.invokedToolIds == null) {
    graph.invokedToolIds = new Set<string>();
  }

  graph.invokedToolIds.add(toolCall.id!);
}
