import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import type { LLMResult, Generation } from '@langchain/core/outputs';
import type { UsageMetadata } from '@langchain/core/messages';
import {
  PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY,
  type PromptCacheTtl,
} from '@/messages/cache';

type PromptCacheInputTokenDetails = Record<string, unknown> & {
  cache_creation?: number;
  cache_creation_5m?: number;
  cache_creation_1h?: number;
};

type PromptCacheCreationBuckets = {
  cache_creation_5m: number;
  cache_creation_1h: number;
};

type MessageConstructorParams = {
  lc_kwargs?: Record<string, unknown>;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value != null && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : undefined;
}

function getFiniteTokenCount(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.max(0, value)
    : 0;
}

function getPromptCacheTtl(value: unknown): PromptCacheTtl | undefined {
  return value === '5m' || value === '1h' ? value : undefined;
}

function getAnthropicPromptCacheCreationBuckets(
  responseMetadata: Record<string, unknown> | undefined
): PromptCacheCreationBuckets {
  const usage = asRecord(responseMetadata?.usage);
  const cacheCreation = asRecord(usage?.cache_creation);
  return {
    cache_creation_5m: getFiniteTokenCount(
      cacheCreation?.ephemeral_5m_input_tokens
    ),
    cache_creation_1h: getFiniteTokenCount(
      cacheCreation?.ephemeral_1h_input_tokens
    ),
  };
}

function getExistingPromptCacheCreationBuckets(
  inputTokenDetails: PromptCacheInputTokenDetails
): PromptCacheCreationBuckets {
  return {
    cache_creation_5m: getFiniteTokenCount(inputTokenDetails.cache_creation_5m),
    cache_creation_1h: getFiniteTokenCount(inputTokenDetails.cache_creation_1h),
  };
}

function getInferredPromptCacheCreationBuckets(
  inputTokenDetails: PromptCacheInputTokenDetails,
  responseMetadata: Record<string, unknown> | undefined
): PromptCacheCreationBuckets {
  const cacheCreation = getFiniteTokenCount(inputTokenDetails.cache_creation);
  const ttl = getPromptCacheTtl(
    responseMetadata?.[PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY]
  );

  if (cacheCreation <= 0 || ttl == null) {
    return { cache_creation_5m: 0, cache_creation_1h: 0 };
  }

  return ttl === '5m'
    ? { cache_creation_5m: cacheCreation, cache_creation_1h: 0 }
    : { cache_creation_5m: 0, cache_creation_1h: cacheCreation };
}

function getPromptCacheCreationBuckets(
  inputTokenDetails: PromptCacheInputTokenDetails,
  responseMetadata: Record<string, unknown> | undefined
): PromptCacheCreationBuckets {
  const anthropicBuckets =
    getAnthropicPromptCacheCreationBuckets(responseMetadata);
  if (
    anthropicBuckets.cache_creation_5m > 0 ||
    anthropicBuckets.cache_creation_1h > 0
  ) {
    return anthropicBuckets;
  }

  // Keep this path for providers that already emit Langfuse-ready bucket keys
  // in LangChain usage metadata. The normalizer remains idempotent if it sees
  // an already-normalized message.
  const existingBuckets =
    getExistingPromptCacheCreationBuckets(inputTokenDetails);
  if (
    existingBuckets.cache_creation_5m > 0 ||
    existingBuckets.cache_creation_1h > 0
  ) {
    return existingBuckets;
  }

  // Bedrock only reports generic cache write tokens; the TTL comes from the
  // cache point we sent, not a provider-confirmed usage field.
  return getInferredPromptCacheCreationBuckets(
    inputTokenDetails,
    responseMetadata
  );
}

function normalizeUsageMetadataForLangfuse(
  usageMetadata: UsageMetadata | undefined,
  responseMetadata: Record<string, unknown> | undefined
): UsageMetadata | undefined {
  const inputTokenDetails = asRecord(usageMetadata?.input_token_details) as
    | PromptCacheInputTokenDetails
    | undefined;
  if (usageMetadata == null || inputTokenDetails == null) {
    return usageMetadata;
  }

  const buckets = getPromptCacheCreationBuckets(
    inputTokenDetails,
    responseMetadata
  );
  const bucketTotal = buckets.cache_creation_5m + buckets.cache_creation_1h;
  if (bucketTotal <= 0) {
    return usageMetadata;
  }

  const genericCacheCreation = getFiniteTokenCount(
    inputTokenDetails.cache_creation
  );
  const normalizedInputTokenDetails: PromptCacheInputTokenDetails = {
    ...inputTokenDetails,
    cache_creation: Math.max(0, genericCacheCreation - bucketTotal),
  };

  if (buckets.cache_creation_5m > 0) {
    normalizedInputTokenDetails.cache_creation_5m = buckets.cache_creation_5m;
  }
  if (buckets.cache_creation_1h > 0) {
    normalizedInputTokenDetails.cache_creation_1h = buckets.cache_creation_1h;
  }

  return {
    ...usageMetadata,
    input_token_details:
      normalizedInputTokenDetails as UsageMetadata['input_token_details'],
  };
}

function cloneAIMessageWithUsageMetadata(
  message: AIMessage | AIMessageChunk,
  usageMetadata: UsageMetadata
): AIMessage | AIMessageChunk {
  const constructorParams = (message as MessageConstructorParams).lc_kwargs;
  const baseParams =
    constructorParams != null
      ? { ...constructorParams }
      : {
        content: message.content,
        additional_kwargs: message.additional_kwargs,
        response_metadata: message.response_metadata,
        id: message.id,
        name: message.name,
        tool_calls: message.tool_calls,
        invalid_tool_calls: message.invalid_tool_calls,
        ...(message instanceof AIMessageChunk
          ? { tool_call_chunks: message.tool_call_chunks }
          : {}),
      };

  return message instanceof AIMessageChunk
    ? new AIMessageChunk({
      ...baseParams,
      usage_metadata: usageMetadata,
    })
    : new AIMessage({
      ...baseParams,
      usage_metadata: usageMetadata,
    });
}

function normalizePromptCacheUsageGenerationForLangfuse(
  generation: Generation
): Generation {
  if (!('message' in generation)) {
    return generation;
  }

  const message = generation.message;
  if (!(message instanceof AIMessage || message instanceof AIMessageChunk)) {
    return generation;
  }

  const usageMetadata = normalizeUsageMetadataForLangfuse(
    message.usage_metadata as UsageMetadata | undefined,
    message.response_metadata as Record<string, unknown> | undefined
  );
  if (usageMetadata == null || usageMetadata === message.usage_metadata) {
    return generation;
  }

  return {
    ...generation,
    message: cloneAIMessageWithUsageMetadata(message, usageMetadata),
  } as Generation;
}

export function normalizePromptCacheUsageForLangfuse(
  output: LLMResult
): LLMResult {
  let changed = false;
  const generations = output.generations.map((generationList) =>
    generationList.map((generation) => {
      const normalized =
        normalizePromptCacheUsageGenerationForLangfuse(generation);
      if (normalized !== generation) {
        changed = true;
      }
      return normalized;
    })
  );

  return changed ? { ...output, generations } : output;
}
