import type { AIMessageChunk } from '@langchain/core/messages';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, GraphNodeKeys } from '@/common';
import { safeDispatchCustomEvent } from '@/utils/events';

/**
 * Handles a `on_chat_model_stream` event for a summarize node.
 *
 * Extracts the text delta from the chunk and dispatches an
 * `ON_SUMMARIZE_DELTA` event so the client can render streaming
 * summarization output in real time.
 *
 * @returns `true` if the event was handled (caller should return early),
 *          `false` if this is not a summarize-node event.
 */
export async function handleSummarizeStream(
  data: t.StreamEventData,
  metadata: Record<string, unknown> | undefined,
  graph: StandardGraph
): Promise<boolean> {
  const currentNode = metadata?.langgraph_node as string | undefined;
  if (currentNode == null || !currentNode.startsWith(GraphNodeKeys.SUMMARIZE)) {
    return false;
  }

  if (!data.chunk) {
    return true;
  }

  const chunk = data.chunk as Partial<AIMessageChunk>;
  const raw = chunk.content;
  if (raw == null || (typeof raw === 'string' && !raw)) {
    return true;
  }

  const contentBlocks: t.MessageContentComplex[] =
    typeof raw === 'string'
      ? [{ type: ContentTypes.TEXT, text: raw } as t.MessageContentComplex]
      : Array.isArray(raw)
        ? (raw as t.MessageContentComplex[])
        : [];

  if (contentBlocks.length === 0) {
    return true;
  }

  // The summarize node creates its step before calling the model,
  // so it should be registered by the time chunks arrive.
  const agentId = currentNode.substring(GraphNodeKeys.SUMMARIZE.length);
  const stepKey = `summarize-${agentId}`;
  let stepId: string;
  try {
    stepId = graph.getStepIdByKey(stepKey);
  } catch {
    return true;
  }

  await safeDispatchCustomEvent(
    GraphEvents.ON_SUMMARIZE_DELTA,
    {
      id: stepId,
      delta: {
        summary: {
          type: ContentTypes.SUMMARY,
          content: contentBlocks,
          provider: String(metadata?.summarization_provider ?? ''),
          model: String(metadata?.summarization_model ?? ''),
        },
      },
    } satisfies t.SummarizeDeltaEvent,
    graph.config
  );

  return true;
}
