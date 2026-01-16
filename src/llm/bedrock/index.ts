/**
 * Optimized ChatBedrockConverse wrapper that fixes contentBlockIndex conflicts
 * and adds support for latest @langchain/aws features:
 *
 * - Application Inference Profiles (PR #9129)
 * - Service Tiers (Priority/Standard/Flex) (PR #9785) - requires AWS SDK 3.966.0+
 *
 * Bedrock sends the same contentBlockIndex for both text and tool_use content blocks,
 * causing LangChain's merge logic to fail with "field[contentBlockIndex] already exists"
 * errors. This wrapper simply strips contentBlockIndex from response_metadata to avoid
 * the conflict.
 *
 * The contentBlockIndex field is only used internally by Bedrock's streaming protocol
 * and isn't needed by application logic - the index field on tool_call_chunks serves
 * the purpose of tracking tool call ordering.
 */

import { ChatBedrockConverse } from '@langchain/aws';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk, ChatResult } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { ChatBedrockConverseInput } from '@langchain/aws';
import type { BaseMessage } from '@langchain/core/messages';

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
    // Temporarily swap model for applicationInferenceProfile support
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
      // Restore original model
      this.model = originalModel;
    }
  }

  /**
   * Override _streamResponseChunks to:
   * 1. Use applicationInferenceProfile as modelId (by temporarily swapping this.model)
   * 2. Strip contentBlockIndex from response_metadata to prevent merge conflicts
   *
   * Note: We delegate to super._streamResponseChunks() to preserve @langchain/aws's
   * internal chunk handling which correctly preserves array content for reasoning blocks.
   */
  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'] & CustomChatBedrockConverseCallOptions,
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    // Temporarily swap model for applicationInferenceProfile support
    const originalModel = this.model;
    if (
      this.applicationInferenceProfile != null &&
      this.applicationInferenceProfile !== ''
    ) {
      this.model = this.applicationInferenceProfile;
    }

    try {
      // Use parent's streaming logic which correctly handles reasoning content
      const baseStream = super._streamResponseChunks(
        messages,
        options,
        runManager
      );

      for await (const chunk of baseStream) {
        // Clean contentBlockIndex from response_metadata to prevent merge conflicts
        yield this.cleanChunk(chunk);
      }
    } finally {
      // Restore original model
      this.model = originalModel;
    }
  }

  /**
   * Clean a chunk by removing contentBlockIndex from response_metadata.
   */
  private cleanChunk(chunk: ChatGenerationChunk): ChatGenerationChunk {
    const message = chunk.message;
    if (!(message instanceof AIMessageChunk)) {
      return chunk;
    }

    const metadata = message.response_metadata as Record<string, unknown>;
    const hasContentBlockIndex = this.hasContentBlockIndex(metadata);
    if (!hasContentBlockIndex) {
      return chunk;
    }

    const cleanedMetadata = this.removeContentBlockIndex(metadata) as Record<
      string,
      unknown
    >;

    return new ChatGenerationChunk({
      text: chunk.text,
      message: new AIMessageChunk({
        ...message,
        response_metadata: cleanedMetadata,
      }),
      generationInfo: chunk.generationInfo,
    });
  }

  /**
   * Check if contentBlockIndex exists at any level in the object
   */
  private hasContentBlockIndex(obj: unknown): boolean {
    if (obj === null || obj === undefined || typeof obj !== 'object') {
      return false;
    }

    if ('contentBlockIndex' in obj) {
      return true;
    }

    for (const value of Object.values(obj)) {
      if (typeof value === 'object' && value !== null) {
        if (this.hasContentBlockIndex(value)) {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Recursively remove contentBlockIndex from all levels of an object
   */
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
