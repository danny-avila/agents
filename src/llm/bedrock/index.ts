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
import { ConverseStreamCommand } from '@aws-sdk/client-bedrock-runtime';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import type { BaseMessage } from '@langchain/core/messages';
import {
  convertToConverseMessages,
  handleConverseStreamContentBlockStart,
  handleConverseStreamContentBlockDelta,
  handleConverseStreamMetadata,
} from './utils';

/**
 * Service tier type for Bedrock invocations.
 * Requires AWS SDK >= 3.966.0 to actually work.
 * @see https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
 */
export type ServiceTierType = 'priority' | 'default' | 'flex' | 'reserved';

/**
 * Extended input interface with additional features:
 * - applicationInferenceProfile: Use an inference profile ARN instead of model ID
 * - serviceTier: Specify service tier (Priority, Standard, Flex, Reserved)
 */
export interface CustomChatBedrockConverseInput
  extends ChatBedrockConverseInput {
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
}

export class CustomChatBedrockConverse extends ChatBedrockConverse {
  /**
   * Application Inference Profile ARN to use instead of model ID.
   */
  applicationInferenceProfile?: string;

  /**
   * Service tier for model invocation.
   */
  serviceTier?: ServiceTierType;

  constructor(fields?: CustomChatBedrockConverseInput) {
    super(fields);
    this.applicationInferenceProfile = fields?.applicationInferenceProfile;
    this.serviceTier = fields?.serviceTier;
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

    /** Service tier from options or fall back to class-level setting */
    const serviceTierType = options?.serviceTier ?? this.serviceTier;

    return {
      ...baseParams,
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

    for await (const event of response.stream) {
      if (event.contentBlockStart != null) {
        const startChunk = handleConverseStreamContentBlockStart(
          event.contentBlockStart
        );
        if (startChunk != null) {
          const idx = event.contentBlockStart.contentBlockIndex;
          if (idx != null) {
            seenBlockIndices.add(idx);
          }
          yield this.enrichChunk(startChunk, seenBlockIndices);
        }
      } else if (event.contentBlockDelta != null) {
        const deltaChunk = handleConverseStreamContentBlockDelta(
          event.contentBlockDelta
        );

        const idx = event.contentBlockDelta.contentBlockIndex;
        if (idx != null) {
          seenBlockIndices.add(idx);
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
        }
      } else {
        yield new ChatGenerationChunk({
          text: '',
          message: new AIMessageChunk({
            content: '',
            response_metadata: event,
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
