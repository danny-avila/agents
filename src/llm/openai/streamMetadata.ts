import type { ChatGenerationChunk } from '@langchain/core/outputs';

/**
 * `@langchain/openai` stamps these scalar fields together onto every chunk
 * whose `choice.finish_reason` is set. Providers that emit `finish_reason` on
 * more than one streamed chunk (e.g. OpenRouter) otherwise make core's
 * `_mergeDicts` concatenate them into `stopstop` / duplicated model names — in
 * both the aggregated graph message and the Langfuse trace. Core keeps
 * `id`/`name`/`model_provider` last but not these, so keep the first occurrence
 * and drop later repeats at the source, before either aggregation runs.
 */
const REPEATED_SCALAR_METADATA_FIELDS = [
  'model_name',
  'finish_reason',
  'service_tier',
  'system_fingerprint',
] as const;

/**
 * Strip scalar `response_metadata`/`generationInfo` fields that have already
 * appeared on an earlier chunk of the same stream, tracked via `seen`. Keeps
 * the first occurrence so the aggregated message still carries the value once.
 */
export function dropRepeatedScalarMetadata(
  chunk: ChatGenerationChunk,
  seen: Set<string>
): void {
  const generationInfo = chunk.generationInfo as
    | Record<string, unknown>
    | undefined;
  const responseMetadata = chunk.message.response_metadata as Record<
    string,
    unknown
  >;
  for (const field of REPEATED_SCALAR_METADATA_FIELDS) {
    const inGenerationInfo =
      generationInfo != null && generationInfo[field] != null;
    const inResponseMetadata = responseMetadata[field] != null;
    if (!inGenerationInfo && !inResponseMetadata) {
      continue;
    }
    if (!seen.has(field)) {
      seen.add(field);
      continue;
    }
    if (inGenerationInfo) {
      delete generationInfo[field];
    }
    if (inResponseMetadata) {
      delete responseMetadata[field];
    }
  }
}
