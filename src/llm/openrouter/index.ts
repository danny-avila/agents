import { ChatOpenAI } from '@/llm/openai';
import type { BaseMessage } from '@langchain/core/messages';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type {
  ChatOpenAICallOptions,
  OpenAIChatInput,
  OpenAIClient,
} from '@langchain/openai';

export type OpenRouterReasoningEffort =
  | 'xhigh'
  | 'high'
  | 'medium'
  | 'low'
  | 'minimal'
  | 'none';

export interface OpenRouterReasoning {
  effort?: OpenRouterReasoningEffort;
  max_tokens?: number;
  exclude?: boolean;
  enabled?: boolean;
}

export interface ChatOpenRouterCallOptions
  extends Omit<ChatOpenAICallOptions, 'reasoning'> {
  /** @deprecated Use `reasoning` object instead */
  include_reasoning?: boolean;
  reasoning?: OpenRouterReasoning;
  modelKwargs?: OpenAIChatInput['modelKwargs'];
}

export type ChatOpenRouterInput = Partial<
  ChatOpenRouterCallOptions & OpenAIChatInput
>;

/** invocationParams return type extended with OpenRouter reasoning */
export type OpenRouterInvocationParams = Omit<
  OpenAIClient.Chat.ChatCompletionCreateParams,
  'messages'
> & {
  reasoning?: OpenRouterReasoning;
};

interface OpenRouterReasoningTextDetail {
  type: 'reasoning.text';
  text?: string;
  format?: string;
  index?: number;
}

interface OpenRouterReasoningEncryptedDetail {
  type: 'reasoning.encrypted';
  id?: string;
  data?: string;
  format?: string;
  index?: number;
}

type OpenRouterReasoningDetail =
  | OpenRouterReasoningTextDetail
  | OpenRouterReasoningEncryptedDetail;

function isReasoningTextDetail(
  value: unknown
): value is OpenRouterReasoningTextDetail {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    value.type === 'reasoning.text'
  );
}

function isReasoningEncryptedDetail(
  value: unknown
): value is OpenRouterReasoningEncryptedDetail {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    value.type === 'reasoning.encrypted'
  );
}

function getReasoningDetails(value: unknown): OpenRouterReasoningDetail[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (detail): detail is OpenRouterReasoningDetail =>
      isReasoningTextDetail(detail) || isReasoningEncryptedDetail(detail)
  );
}

export class ChatOpenRouter extends ChatOpenAI {
  private openRouterReasoning?: OpenRouterReasoning;
  /** @deprecated Use `reasoning` object instead */
  private includeReasoning?: boolean;

  constructor(_fields: ChatOpenRouterInput) {
    const {
      include_reasoning,
      reasoning: openRouterReasoning,
      modelKwargs = {},
      ...fields
    } = _fields;

    // Extract reasoning from modelKwargs if provided there (e.g., from LLMConfig)
    const { reasoning: mkReasoning, ...restModelKwargs } = modelKwargs as {
      reasoning?: OpenRouterReasoning;
    } & Record<string, unknown>;

    super({
      ...fields,
      modelKwargs: restModelKwargs,
      includeReasoningDetails: true,
      convertReasoningDetailsToContent: true,
    });

    // Merge reasoning config: modelKwargs.reasoning < constructor reasoning
    if (mkReasoning != null || openRouterReasoning != null) {
      this.openRouterReasoning = {
        ...mkReasoning,
        ...openRouterReasoning,
      };
    }

    this.includeReasoning = include_reasoning;
  }
  static lc_name(): 'LibreChatOpenRouter' {
    return 'LibreChatOpenRouter';
  }

  // @ts-expect-error - OpenRouter reasoning extends OpenAI Reasoning with additional
  // effort levels ('xhigh' | 'none' | 'minimal') not in ReasoningEffort.
  // The parent's generic conditional return type cannot be widened in an override.
  override invocationParams(
    options?: this['ParsedCallOptions']
  ): OpenRouterInvocationParams {
    type MutableParams = Omit<
      OpenAIClient.Chat.ChatCompletionCreateParams,
      'messages'
    > & { reasoning_effort?: string; reasoning?: OpenRouterReasoning };

    const params = super.invocationParams(options) as MutableParams;

    // Remove the OpenAI-native reasoning_effort that the parent sets;
    // OpenRouter uses a `reasoning` object instead
    delete params.reasoning_effort;

    // Build the OpenRouter reasoning config
    const reasoning = this.buildOpenRouterReasoning(options);
    if (reasoning != null) {
      params.reasoning = reasoning;
    } else {
      delete params.reasoning;
    }

    return params;
  }

  private buildOpenRouterReasoning(
    options?: this['ParsedCallOptions']
  ): OpenRouterReasoning | undefined {
    let reasoning: OpenRouterReasoning | undefined;

    // 1. Instance-level reasoning config (from constructor)
    if (this.openRouterReasoning != null) {
      reasoning = { ...this.openRouterReasoning };
    }

    // 2. LangChain-style reasoning params (from parent's `this.reasoning`)
    const lcReasoning = this.getReasoningParams(options);
    if (lcReasoning?.effort != null) {
      reasoning = {
        ...reasoning,
        effort: lcReasoning.effort as OpenRouterReasoningEffort,
      };
    }

    // 3. Call-level reasoning override
    const callReasoning = (options as ChatOpenRouterCallOptions | undefined)
      ?.reasoning;
    if (callReasoning != null) {
      reasoning = { ...reasoning, ...callReasoning };
    }

    // 4. Legacy include_reasoning backward compatibility
    if (reasoning == null && this.includeReasoning === true) {
      reasoning = { enabled: true };
    }

    return reasoning;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const reasoningTextByIndex = new Map<
      number,
      OpenRouterReasoningTextDetail
    >();
    const reasoningEncryptedById = new Map<
      string,
      OpenRouterReasoningEncryptedDetail
    >();

    for await (const generationChunk of super._streamResponseChunks(
      messages,
      options,
      runManager
    )) {
      let currentReasoningText = '';
      const reasoningDetails = getReasoningDetails(
        generationChunk.message.additional_kwargs.reasoning_details
      );

      for (const detail of reasoningDetails) {
        if (detail.type === 'reasoning.text') {
          currentReasoningText += detail.text ?? '';
          const index = detail.index ?? 0;
          const existing = reasoningTextByIndex.get(index);
          if (existing != null) {
            existing.text = `${existing.text ?? ''}${detail.text ?? ''}`;
            continue;
          }
          reasoningTextByIndex.set(index, {
            ...detail,
            text: detail.text ?? '',
          });
          continue;
        }
        if (detail.id != null) {
          reasoningEncryptedById.set(detail.id, { ...detail });
        }
      }

      if (
        currentReasoningText.length > 0 &&
        generationChunk.message.additional_kwargs.reasoning == null
      ) {
        generationChunk.message.additional_kwargs.reasoning =
          currentReasoningText;
      }

      if (generationChunk.generationInfo?.finish_reason != null) {
        const finalReasoningDetails = [
          ...reasoningTextByIndex.values(),
          ...reasoningEncryptedById.values(),
        ];
        if (finalReasoningDetails.length > 0) {
          generationChunk.message.additional_kwargs.reasoning_details =
            finalReasoningDetails;
        } else {
          delete generationChunk.message.additional_kwargs.reasoning_details;
        }
        yield generationChunk;
        continue;
      }

      delete generationChunk.message.additional_kwargs.reasoning_details;
      yield generationChunk;
    }
  }
}
