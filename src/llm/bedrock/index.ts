/**
 * Optimized ChatBedrockConverse wrapper that fixes content block merging for
 * streaming responses and adds support for latest @langchain/aws features:
 *
 * - Application Inference Profiles (PR #9129)
 * - Service Tiers (Priority/Standard/Flex) (PR #9785) - requires AWS SDK 3.966.0+
 *
 * Bedrock's `@langchain/aws` library does not include an `index` property on content
 * blocks (unlike Anthropic/OpenAI), which causes LangChain's `_mergeLists` to append
 * each streaming chunk as a separate array entry instead of merging by index.
 *
 * This wrapper takes full ownership of the stream by directly interfacing with the
 * AWS SDK client (`this.client`) and using custom handlers from `./utils/` that
 * include `contentBlockIndex` in response_metadata for every delta type. It then
 * promotes `contentBlockIndex` to an `index` property on each content block
 * (mirroring Anthropic's pattern) and strips it from metadata to avoid
 * `_mergeDicts` conflicts.
 *
 * When multiple content block types are present (e.g. reasoning + text), text deltas
 * are promoted from strings to array form with `index` so they merge correctly once
 * the accumulated content is already an array.
 */

import { ChatBedrockConverse } from '@langchain/aws';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';
import {
  ConverseStreamCommand,
  type GuardrailConfiguration,
  type GuardrailStreamConfiguration,
} from '@aws-sdk/client-bedrock-runtime';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, ResponseMetadata } from '@langchain/core/messages';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import {
  convertToConverseMessages,
  createConverseToolUseStopChunk,
  handleConverseStreamContentBlockStart,
  handleConverseStreamContentBlockDelta,
  handleConverseStreamMetadata,
} from './utils';
import {
  resolveBedrockPromptCacheTtl,
  supportsBedrockToolCache,
  type PromptCacheTtl,
} from '@/messages/cache';
import { applyCachePointsToConversePayload } from './cachePoints';
import { insertBedrockToolCachePoint } from './toolCache';

/**
 * Service tier type for Bedrock invocations.
 * Requires AWS SDK >= 3.966.0 to actually work.
 * @see https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
 */
export type ServiceTierType = 'priority' | 'default' | 'flex' | 'reserved';

export type CustomGuardrailConfiguration = GuardrailConfiguration &
  Pick<GuardrailStreamConfiguration, 'streamProcessingMode'>;

function getCadencedStreamDelay({
  targetDelay,
  lastVisibleTextAt,
  now,
}: {
  targetDelay: number;
  lastVisibleTextAt?: number;
  now: number;
}): number {
  if (targetDelay <= 0 || lastVisibleTextAt == null) {
    return 0;
  }
  return Math.max(0, targetDelay - (now - lastVisibleTextAt));
}

async function waitForStreamDelay(
  delay: number,
  signal?: AbortSignal
): Promise<void> {
  if (delay <= 0 || isSignalAborted(signal)) {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeoutRef: { current?: ReturnType<typeof setTimeout> } = {};
    const onAbort = (): void => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      signal?.removeEventListener('abort', onAbort);
      resolve();
    };
    timeoutRef.current = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, delay);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (isSignalAborted(signal)) {
      onAbort();
    }
  });
}

function isSignalAborted(signal?: AbortSignal): boolean {
  return signal?.aborted === true;
}

/**
 * Extended input interface with additional features:
 * - applicationInferenceProfile: Use an inference profile ARN instead of model ID
 * - serviceTier: Specify service tier (Priority, Standard, Flex, Reserved)
 */
export interface CustomChatBedrockConverseInput
  extends ChatBedrockConverseInput {
  /**
   * Enables Bedrock prompt cache checkpoints for message and tool prefixes.
   */
  promptCache?: boolean;

  /**
   * Prompt-cache checkpoint TTL. Defaults to `'1h'` (extended cache) when
   * `promptCache` is enabled; set `'5m'` for the legacy 5-minute behavior.
   * Bedrock models that don't support the 1-hour TTL downgrade to 5m
   * server-side (verified on Sonnet/Opus 4.6), so the default is safe to leave
   * on; use `'5m'` for any model that rejects it.
   */
  promptCacheTtl?: PromptCacheTtl;

  /**
   * Minimum delay in milliseconds between visible streamed text deltas.
   */
  _lc_stream_delay?: number;

  /**
   * Guardrail configuration for Converse and ConverseStream invocations.
   * `streamProcessingMode` is only used by ConverseStream.
   */
  guardrailConfig?: CustomGuardrailConfiguration;

  /**
   * Application Inference Profile ARN to use for the model.
   * For example, "arn:aws:bedrock:eu-west-1:123456789102:application-inference-profile/fm16bt65tzgx"
   * When provided, this ARN will be used for the actual inference calls instead of the model ID.
   * Must still provide `model` as normal modelId to benefit from all the metadata.
   * @see https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html
   */
  applicationInferenceProfile?: string;

  /**
   * Service tier for model invocation.
   * Specifies the processing tier type used for serving the request.
   * Supported values are 'priority', 'default', 'flex', and 'reserved'.
   *
   * - 'priority': Prioritized processing for lower latency
   * - 'default': Standard processing tier
   * - 'flex': Flexible processing tier with lower cost
   * - 'reserved': Reserved capacity for consistent performance
   *
   * If not provided, AWS uses the default tier.
   * Note: Requires AWS SDK >= 3.966.0 to work.
   * @see https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
   */
  serviceTier?: ServiceTierType;
}

/**
 * Extended call options with serviceTier override support.
 */
export interface CustomChatBedrockConverseCallOptions {
  serviceTier?: ServiceTierType;
  guardrailConfig?: CustomGuardrailConfiguration;
}

export class CustomChatBedrockConverse extends ChatBedrockConverse {
  _lc_stream_delay: number;

  /**
   * Whether to insert Bedrock prompt cache checkpoints when available.
   */
  promptCache?: boolean;

  /**
   * Prompt-cache checkpoint TTL (`'5m'` legacy or `'1h'` extended cache).
   */
  promptCacheTtl?: PromptCacheTtl;

  /**
   * Application Inference Profile ARN to use instead of model ID.
   */
  applicationInferenceProfile?: string;

  /**
   * Service tier for model invocation.
   */
  serviceTier?: ServiceTierType;

  /**
   * The configured model id, captured at construction so it survives the
   * temporary `this.model` swap to an application-inference-profile ARN during
   * generation. Used to gate the Bedrock tool cache point to Claude models
   * (see {@link supportsBedrockToolCache}).
   */
  private readonly cacheModelId: string;

  constructor(fields?: CustomChatBedrockConverseInput) {
    super(fields);
    this.promptCache = fields?.promptCache;
    this.promptCacheTtl = fields?.promptCacheTtl;
    this._lc_stream_delay = Math.max(0, fields?._lc_stream_delay ?? 0);
    this.applicationInferenceProfile = fields?.applicationInferenceProfile;
    this.serviceTier = fields?.serviceTier;
    // `super(fields)` initializes `this.model` to LangChain's default Claude
    // model when `fields.model` is omitted, so fall back to it rather than ''
    // (which would treat the default Claude model as tool-cache-unsupported).
    this.cacheModelId = fields?.model ?? this.model;
  }

  static lc_name(): string {
    return 'LibreChatBedrockConverse';
  }

  /**
   * Get the model ID to use for API calls.
   * Returns applicationInferenceProfile if set, otherwise returns this.model.
   */
  protected getModelId(): string {
    return this.applicationInferenceProfile ?? this.model;
  }

  /**
   * Override invocationParams to add serviceTier support.
   */
  override invocationParams(
    options?: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions
  ): ReturnType<ChatBedrockConverse['invocationParams']> & {
    serviceTier?: { type: ServiceTierType };
  } {
    const baseParams = super.invocationParams(options);
    const toolConfig =
      this.promptCache === true && supportsBedrockToolCache(this.cacheModelId)
        ? insertBedrockToolCachePoint(
          baseParams.toolConfig,
          true,
          resolveBedrockPromptCacheTtl(this.promptCacheTtl, this.cacheModelId)
        )
        : baseParams.toolConfig;

    /** Service tier from options or fall back to class-level setting */
    const serviceTierType = options?.serviceTier ?? this.serviceTier;

    return {
      ...baseParams,
      toolConfig,
      serviceTier: serviceTierType ? { type: serviceTierType } : undefined,
    };
  }

  /**
   * Override _generateNonStreaming to use applicationInferenceProfile as modelId.
   * Uses the same model-swapping pattern as streaming for consistency.
   */
  override async _generateNonStreaming(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): Promise<ChatResult> {
    const originalModel = this.model;
    if (
      this.applicationInferenceProfile != null &&
      this.applicationInferenceProfile !== ''
    ) {
      this.model = this.applicationInferenceProfile;
    }

    try {
      return await super._generateNonStreaming(messages, options, runManager);
    } finally {
      this.model = originalModel;
    }
  }

  /**
   * Own the stream end-to-end so we have direct access to every
   * `contentBlockDelta.contentBlockIndex` from the AWS SDK.
   *
   * This replaces the parent's implementation which strips contentBlockIndex
   * from text and reasoning deltas, making it impossible to merge correctly.
   */
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const { converseMessages, converseSystem } =
      convertToConverseMessages(messages);
    const params = this.invocationParams(options);

    let { streamUsage } = this;
    if ((options as Record<string, unknown>).streamUsage !== undefined) {
      streamUsage = (options as Record<string, unknown>).streamUsage as boolean;
    }

    const modelId = this.getModelId();

    applyCachePointsToConversePayload({
      cacheControl: options.cache_control,
      system: converseSystem,
      messages: converseMessages,
      params,
      modelId,
    });

    const command = new ConverseStreamCommand({
      modelId,
      messages: converseMessages,
      system: converseSystem,
      ...(params as Record<string, unknown>),
    });

    const response = await this.client.send(command, {
      abortSignal: options.signal,
    });

    if (!response.stream) {
      return;
    }

    const seenBlockIndices = new Set<number>();
    const toolUseBlockIndices = new Set<number>();
    let hasEmittedText = false;
    let lastVisibleTextAt: number | undefined;
    /**
     * Guardrails can reject an already-streamed toolUse block at
     * `messageStop` (`guardrail_intervened`), after `contentBlockStop` has
     * passed. Only emit eager-execution seals when no guardrails are
     * configured, so a later intervention can't race an eagerly started tool.
     */
    const sealToolUseOnStop =
      options.guardrailConfig == null && this.guardrailConfig == null;

    for await (const event of response.stream) {
      if (event.contentBlockStart != null) {
        const startChunk = handleConverseStreamContentBlockStart(
          event.contentBlockStart
        );
        if (startChunk != null) {
          const idx = event.contentBlockStart.contentBlockIndex;
          if (idx != null) {
            seenBlockIndices.add(idx);
            if (event.contentBlockStart.start?.toolUse != null) {
              toolUseBlockIndices.add(idx);
            }
          }
          yield this.enrichChunk(startChunk, seenBlockIndices);

          // Registered stream handlers receive chunks through callback
          // events, not the yielded generator — dispatch the start chunk so
          // they see the tool call's id/name (eager chunk state needs both).
          await runManager?.handleLLMNewToken(
            startChunk.text,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: startChunk }
          );
        }
      } else if (event.contentBlockDelta != null) {
        const deltaChunk = handleConverseStreamContentBlockDelta(
          event.contentBlockDelta
        );

        const idx = event.contentBlockDelta.contentBlockIndex;
        if (idx != null) {
          seenBlockIndices.add(idx);
        }

        if (deltaChunk.text !== '') {
          await waitForStreamDelay(
            getCadencedStreamDelay({
              targetDelay: hasEmittedText ? this._lc_stream_delay : 0,
              lastVisibleTextAt,
              now: Date.now(),
            }),
            options.signal
          );
          if (isSignalAborted(options.signal)) {
            throw new Error('AbortError: User aborted the request.');
          }
          hasEmittedText = true;
          lastVisibleTextAt = Date.now();
        }

        yield this.enrichChunk(deltaChunk, seenBlockIndices);

        await runManager?.handleLLMNewToken(
          deltaChunk.text,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: deltaChunk }
        );
      } else if (event.metadata != null) {
        yield handleConverseStreamMetadata(event.metadata, { streamUsage });
      } else if (event.contentBlockStop != null) {
        const stopIdx = event.contentBlockStop.contentBlockIndex;
        if (stopIdx != null) {
          seenBlockIndices.add(stopIdx);
          if (sealToolUseOnStop && toolUseBlockIndices.has(stopIdx)) {
            // Converse guarantees the block's input is complete at stop, so
            // emit an explicit seal chunk for eager tool execution — through
            // the callback path too, for registered stream handlers.
            const sealChunk = createConverseToolUseStopChunk(stopIdx);
            yield sealChunk;
            await runManager?.handleLLMNewToken(
              sealChunk.text,
              undefined,
              undefined,
              undefined,
              undefined,
              { chunk: sealChunk }
            );
          }
        }
      } else {
        yield new ChatGenerationChunk({
          text: '',
          message: new AIMessageChunk({
            content: '',
            response_metadata: { ...event } as ResponseMetadata,
          }),
        });
      }
    }
  }

  /**
   * Inject `index` on content blocks for proper merge behaviour, then strip
   * `contentBlockIndex` from response_metadata to prevent `_mergeDicts` conflicts.
   *
   * Text string content is promoted to array form only when the stream contains
   * multiple content block indices (e.g. reasoning at index 0, text at index 1),
   * ensuring text merges correctly with the already-array accumulated content.
   */
  private enrichChunk(
    chunk: ChatGenerationChunk,
    seenBlockIndices: Set<number>
  ): ChatGenerationChunk {
    const message = chunk.message;
    if (!(message instanceof AIMessageChunk)) {
      return chunk;
    }

    const metadata = message.response_metadata as Record<string, unknown>;
    const blockIndex = this.extractContentBlockIndex(metadata);
    const hasMetadataIndex = blockIndex != null;

    let content: AIMessageChunk['content'] = message.content;
    let contentModified = false;

    if (Array.isArray(content) && blockIndex != null) {
      content = content.map((block) =>
        typeof block === 'object' && !('index' in block)
          ? { ...block, index: blockIndex }
          : block
      );
      contentModified = true;
    } else if (
      typeof content === 'string' &&
      content !== '' &&
      blockIndex != null &&
      seenBlockIndices.size > 1
    ) {
      content = [{ type: 'text', text: content, index: blockIndex }];
      contentModified = true;
    }

    if (!contentModified && !hasMetadataIndex) {
      return chunk;
    }

    const cleanedMetadata = hasMetadataIndex
      ? (this.removeContentBlockIndex(metadata) as Record<string, unknown>)
      : metadata;

    return new ChatGenerationChunk({
      text: chunk.text,
      message: new AIMessageChunk({
        ...message,
        content,
        response_metadata: cleanedMetadata,
      }),
      generationInfo: chunk.generationInfo,
    });
  }

  /**
   * Extract `contentBlockIndex` from the top level of response_metadata.
   * Our custom handlers always place it at the top level.
   */
  private extractContentBlockIndex(
    metadata: Record<string, unknown>
  ): number | undefined {
    if (
      'contentBlockIndex' in metadata &&
      typeof metadata.contentBlockIndex === 'number'
    ) {
      return metadata.contentBlockIndex;
    }
    return undefined;
  }

  private removeContentBlockIndex(obj: unknown): unknown {
    if (obj === null || obj === undefined) {
      return obj;
    }

    if (Array.isArray(obj)) {
      return obj.map((item) => this.removeContentBlockIndex(item));
    }

    if (typeof obj === 'object') {
      const cleaned: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        if (key !== 'contentBlockIndex') {
          cleaned[key] = this.removeContentBlockIndex(value);
        }
      }
      return cleaned;
    }

    return obj;
  }
}

export type { ChatBedrockConverseInput };
