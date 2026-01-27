// src/stream.ts
import type { ChatOpenAIReasoningSummary } from '@langchain/openai';
import type { AIMessageChunk } from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  ToolCallTypes,
  ContentTypes,
  GraphEvents,
  StepTypes,
  Providers,
} from '@/common';
import {
  handleServerToolResult,
  handleToolCallChunks,
  handleToolCalls,
} from '@/tools/handlers';
import { getMessageId } from '@/messages';

function getNonEmptyValue(possibleValues: string[]): string | undefined {
  for (const value of possibleValues) {
    if (value && value.trim() !== '') {
      return value;
    }
  }
  return undefined;
}

/**
 * Flush any remaining buffered content from the thinking tag state machine.
 * Called when the stream ends to ensure partial tags aren't lost.
 */
export async function flushThinkingBuffer(
  agentContext: AgentContext,
  graph: StandardGraph
): Promise<void> {
  if (!agentContext.tagBuffer || !agentContext.currentStepId) {
    return;
  }

  const stepId = agentContext.currentStepId;

  if (
    agentContext.thinkingState === 'buffering_open' ||
    agentContext.thinkingState === 'normal'
  ) {
    // Incomplete opening tag or normal state - flush as TEXT
    await graph.dispatchMessageDelta(stepId, {
      content: [{ type: ContentTypes.TEXT, text: agentContext.tagBuffer }],
    });
  } else if (
    agentContext.thinkingState === 'buffering_close' ||
    agentContext.thinkingState === 'thinking'
  ) {
    // Incomplete closing tag or thinking state - flush as THINK
    await graph.dispatchReasoningDelta(stepId, {
      content: [{ type: ContentTypes.THINK, think: agentContext.tagBuffer }],
    });
  }

  // Reset buffer state
  agentContext.tagBuffer = '';
  agentContext.thinkingState = 'normal';
  agentContext.currentStepId = undefined;
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
    let hasToolCalls = false;
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
    }

    const hasToolCallChunks =
      (chunk.tool_call_chunks && chunk.tool_call_chunks.length > 0) ?? false;
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
      const stepKey = graph.getStepKey(metadata);
      graph.prelimMessageIdsByStepKey.set(stepKey, chunk.id ?? '');
    } else if (isEmptyChunk) {
      return;
    }

    const stepKey = graph.getStepKey(metadata);

    if (
      hasToolCallChunks &&
      chunk.tool_call_chunks &&
      chunk.tool_call_chunks.length &&
      typeof chunk.tool_call_chunks[0]?.index === 'number'
    ) {
      await handleToolCallChunks({
        graph,
        stepKey,
        toolCallChunks: chunk.tool_call_chunks,
        metadata,
      });
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
      // State machine for detecting <thinking>/<think> tags in streamed fragments
      // States: 'normal', 'buffering_open', 'thinking', 'buffering_close'
      if (!agentContext.thinkingState) {
        agentContext.thinkingState = 'normal';
        agentContext.tagBuffer = '';
      }

      // Store stepId for potential buffer flush at stream end
      agentContext.currentStepId = stepId;

      let remaining = content;

      while (remaining.length > 0) {
        if (agentContext.thinkingState === 'normal') {
          // Look for '<' to start buffering potential opening tag
          const ltPos = remaining.indexOf('<');
          if (ltPos === -1) {
            // No '<', yield all as TEXT
            await graph.dispatchMessageDelta(stepId, {
              content: [{ type: ContentTypes.TEXT, text: remaining }],
            });
            remaining = '';
          } else if (ltPos > 0) {
            // Yield text before '<'
            await graph.dispatchMessageDelta(stepId, {
              content: [
                { type: ContentTypes.TEXT, text: remaining.slice(0, ltPos) },
              ],
            });
            remaining = remaining.slice(ltPos);
          } else {
            // Starts with '<', start buffering
            agentContext.thinkingState = 'buffering_open';
            agentContext.tagBuffer = remaining;
            remaining = '';

            // Check immediately if we have enough to decide
            const bufLower = agentContext.tagBuffer.toLowerCase();
            if (bufLower.startsWith('<thinking>')) {
              agentContext.thinkingState = 'thinking';
              agentContext.currentTokenType = ContentTypes.THINK;
              remaining = agentContext.tagBuffer.slice(10);
              agentContext.tagBuffer = '';
            } else if (bufLower.startsWith('<think>')) {
              agentContext.thinkingState = 'thinking';
              agentContext.currentTokenType = ContentTypes.THINK;
              remaining = agentContext.tagBuffer.slice(7);
              agentContext.tagBuffer = '';
            } else if (
              !(
                '<thinking>'.startsWith(bufLower) ||
                '<think>'.startsWith(bufLower)
              )
            ) {
              // Definitely not a thinking tag
              await graph.dispatchMessageDelta(stepId, {
                content: [
                  { type: ContentTypes.TEXT, text: agentContext.tagBuffer },
                ],
              });
              agentContext.tagBuffer = '';
              agentContext.thinkingState = 'normal';
            }
            // else: partial match, keep buffering
          }
        } else if (agentContext.thinkingState === 'buffering_open') {
          // Accumulating potential <thinking> tag
          agentContext.tagBuffer += remaining;
          remaining = '';

          const bufLower = agentContext.tagBuffer.toLowerCase();
          if (bufLower.startsWith('<thinking>')) {
            agentContext.thinkingState = 'thinking';
            agentContext.currentTokenType = ContentTypes.THINK;
            remaining = agentContext.tagBuffer.slice(10);
            agentContext.tagBuffer = '';
          } else if (bufLower.startsWith('<think>')) {
            agentContext.thinkingState = 'thinking';
            agentContext.currentTokenType = ContentTypes.THINK;
            remaining = agentContext.tagBuffer.slice(7);
            agentContext.tagBuffer = '';
          } else if (
            '<thinking>'.startsWith(bufLower) ||
            '<think>'.startsWith(bufLower)
          ) {
            // Still could be a tag, keep buffering
          } else {
            // Not a thinking tag, yield buffer and return to normal
            await graph.dispatchMessageDelta(stepId, {
              content: [
                { type: ContentTypes.TEXT, text: agentContext.tagBuffer },
              ],
            });
            agentContext.tagBuffer = '';
            agentContext.thinkingState = 'normal';
          }
        } else if (agentContext.thinkingState === 'thinking') {
          // In thinking mode, look for '<' to start buffering potential closing tag
          const ltPos = remaining.indexOf('<');
          if (ltPos === -1) {
            // No '<', yield all as THINK
            await graph.dispatchReasoningDelta(stepId, {
              content: [{ type: ContentTypes.THINK, think: remaining }],
            });
            remaining = '';
          } else if (ltPos > 0) {
            // Yield thinking text before '<'
            await graph.dispatchReasoningDelta(stepId, {
              content: [
                { type: ContentTypes.THINK, think: remaining.slice(0, ltPos) },
              ],
            });
            remaining = remaining.slice(ltPos);
          } else {
            // Starts with '<', start buffering for closing tag
            agentContext.thinkingState = 'buffering_close';
            agentContext.tagBuffer = remaining;
            remaining = '';

            // Check immediately if we have enough to decide
            const bufLower = agentContext.tagBuffer.toLowerCase();
            if (bufLower.startsWith('</thinking>')) {
              agentContext.thinkingState = 'normal';
              agentContext.currentTokenType = ContentTypes.TEXT;
              remaining = agentContext.tagBuffer.slice(11);
              agentContext.tagBuffer = '';
            } else if (bufLower.startsWith('</think>')) {
              agentContext.thinkingState = 'normal';
              agentContext.currentTokenType = ContentTypes.TEXT;
              remaining = agentContext.tagBuffer.slice(8);
              agentContext.tagBuffer = '';
            } else if (
              !(
                '</thinking>'.startsWith(bufLower) ||
                '</think>'.startsWith(bufLower)
              )
            ) {
              // Definitely not a closing tag, yield as thinking
              await graph.dispatchReasoningDelta(stepId, {
                content: [
                  { type: ContentTypes.THINK, think: agentContext.tagBuffer },
                ],
              });
              agentContext.tagBuffer = '';
              agentContext.thinkingState = 'thinking';
            }
            // else: partial match, keep buffering
          }
        } else {
          // buffering_close: Accumulating potential </thinking> tag
          agentContext.tagBuffer += remaining;
          remaining = '';

          const bufLower = agentContext.tagBuffer.toLowerCase();
          if (bufLower.startsWith('</thinking>')) {
            agentContext.thinkingState = 'normal';
            agentContext.currentTokenType = ContentTypes.TEXT;
            remaining = agentContext.tagBuffer.slice(11);
            agentContext.tagBuffer = '';
          } else if (bufLower.startsWith('</think>')) {
            agentContext.thinkingState = 'normal';
            agentContext.currentTokenType = ContentTypes.TEXT;
            remaining = agentContext.tagBuffer.slice(8);
            agentContext.tagBuffer = '';
          } else if (
            '</thinking>'.startsWith(bufLower) ||
            '</think>'.startsWith(bufLower)
          ) {
            // Still could be a closing tag, keep buffering
          } else {
            // Not a closing tag, yield buffer as thinking and return to thinking mode
            await graph.dispatchReasoningDelta(stepId, {
              content: [
                { type: ContentTypes.THINK, think: agentContext.tagBuffer },
              ],
            });
            agentContext.tagBuffer = '';
            agentContext.thinkingState = 'thinking';
          }
        }
      }
    } else if (
      content.every((c) => c.type?.startsWith(ContentTypes.TEXT) ?? false)
    ) {
      await graph.dispatchMessageDelta(stepId, {
        content,
      });
    } else if (
      content.every(
        (c) =>
          (c.type?.startsWith(ContentTypes.THINKING) ?? false) ||
          (c.type?.startsWith(ContentTypes.REASONING) ?? false) ||
          (c.type?.startsWith(ContentTypes.REASONING_CONTENT) ?? false) ||
          c.type === 'redacted_thinking'
      )
    ) {
      await graph.dispatchReasoningDelta(stepId, {
        content: content.map((c) => ({
          type: ContentTypes.THINK,
          think:
            (c as t.ThinkingContentText).thinking ??
            (c as Partial<t.GoogleReasoningContentText>).reasoning ??
            (c as Partial<t.BedrockReasoningContentText>).reasoningText?.text ??
            '',
        })),
      });
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

    if (!contentParts[index]) {
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
            tool_call?: t.ToolCallPart;
          })
        | undefined;

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
      | t.RunStepDeltaEvent
      | { result: t.ToolEndEvent };
  }): void => {
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

      // Store tool call IDs if present
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

      if (messageDelta.delta.content) {
        const contentPart = Array.isArray(messageDelta.delta.content)
          ? messageDelta.delta.content[0]
          : messageDelta.delta.content;

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

      if (reasoningDelta.delta.content) {
        const contentPart = Array.isArray(reasoningDelta.delta.content)
          ? reasoningDelta.delta.content[0]
          : reasoningDelta.delta.content;

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
            },
          };

          updateContent(runStep.index, contentPart);
        });
      }
    } else if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
      const { result } = data as unknown as { result: t.ToolEndEvent };

      const { id: stepId } = result;

      const runStep = stepMap.get(stepId);
      if (!runStep) {
        console.warn(
          'No run step or runId found for completed tool call event'
        );
        return;
      }

      const contentPart: t.MessageContentComplex = {
        type: ContentTypes.TOOL_CALL,
        tool_call: result.tool_call,
      };

      updateContent(runStep.index, contentPart, true);
    }
  };

  return { contentParts, aggregateContent, stepMap };
}
