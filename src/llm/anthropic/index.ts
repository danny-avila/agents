import { AIMessageChunk } from '@langchain/core/messages';
import { ChatAnthropicMessages } from '@langchain/anthropic';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { BaseChatModelParams } from '@langchain/core/language_models/chat_models';
import type {
  BaseMessage,
  UsageMetadata,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { AnthropicInput } from '@langchain/anthropic';
import type { Anthropic } from '@anthropic-ai/sdk';
import type {
  AnthropicMessageCreateParams,
  AnthropicStreamingMessageCreateParams,
  AnthropicStreamUsage,
  AnthropicMessageStartEvent,
  AnthropicMessageDeltaEvent,
  AnthropicOutputConfig,
  AnthropicBeta,
  ChatAnthropicToolType,
  AnthropicMCPServerURLDefinition,
  AnthropicContextManagementConfigParam,
} from '@/llm/anthropic/types';
import { _makeMessageChunkFromAnthropicEvent } from './utils/message_outputs';
import { _convertMessagesToAnthropicPayload } from './utils/message_inputs';
import { handleToolChoice } from './utils/tools';
import { TextStream } from '@/llm/text';

const ANTHROPIC_TOOL_BETAS: Partial<Record<string, AnthropicBeta>> = {
  tool_search_tool_regex_20251119: 'advanced-tool-use-2025-11-20',
  tool_search_tool_bm25_20251119: 'advanced-tool-use-2025-11-20',
  memory_20250818: 'context-management-2025-06-27',
  web_fetch_20250910: 'web-fetch-2025-09-10',
  code_execution_20250825: 'code-execution-2025-08-25',
  computer_20251124: 'computer-use-2025-11-24',
  computer_20250124: 'computer-use-2025-01-24',
  mcp_toolset: 'mcp-client-2025-11-20',
};

function _toolsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(params.tools && params.tools.length > 0);
}
function _documentsInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  for (const message of params.messages) {
    if (typeof message.content === 'string') {
      continue;
    }
    for (const block of message.content) {
      if (
        typeof block === 'object' &&
        block.type === 'document' &&
        block.citations != null &&
        typeof block.citations === 'object' &&
        block.citations.enabled === true
      ) {
        return true;
      }
    }
  }
  return false;
}

function _thinkingInParams(
  params: AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams
): boolean {
  return !!(
    params.thinking &&
    (params.thinking.type === 'enabled' || params.thinking.type === 'adaptive')
  );
}

function _compactionInParams(
  params: (
    | AnthropicMessageCreateParams
    | AnthropicStreamingMessageCreateParams
  ) & {
    context_management?: AnthropicContextManagementConfigParam;
  }
): boolean {
  return (
    params.context_management?.edits?.some(
      (edit) => edit.type === 'compact_20260112'
    ) === true
  );
}

function isThinkingEnabled(thinking: Anthropic.ThinkingConfigParam): boolean {
  return thinking.type === 'enabled' || thinking.type === 'adaptive';
}

function isOpus47Model(model?: string): boolean {
  return model?.startsWith('claude-opus-4-7') === true;
}

function combineBetas(
  ...betaGroups: (AnthropicBeta[] | undefined)[]
): AnthropicBeta[] {
  const betas = new Set<AnthropicBeta>();
  for (const betaGroup of betaGroups) {
    for (const beta of betaGroup ?? []) {
      betas.add(beta);
    }
  }
  return [...betas];
}

function getToolBetas(tools?: ChatAnthropicToolType[]): AnthropicBeta[] {
  const betas = new Set<AnthropicBeta>();
  for (const tool of tools ?? []) {
    if (typeof tool !== 'object' || !('type' in tool)) {
      continue;
    }
    const beta = ANTHROPIC_TOOL_BETAS[String(tool.type)];
    if (beta != null) {
      betas.add(beta);
    }
  }
  return [...betas];
}

function getCompactionBetas(
  contextManagement?: AnthropicContextManagementConfigParam
): AnthropicBeta[] {
  return contextManagement?.edits?.some(
    (edit) => edit.type === 'compact_20260112'
  ) === true
    ? ['compact-2026-01-12']
    : [];
}

function getTaskBudgetBetas(
  model: string,
  outputConfig?: AnthropicOutputConfig
): AnthropicBeta[] {
  return isOpus47Model(model) &&
    outputConfig != null &&
    'task_budget' in outputConfig &&
    outputConfig.task_budget != null
    ? ['task-budgets-2026-03-13']
    : [];
}

function validateInvocationParamCompatibility({
  model,
  thinking,
  topK,
  topP,
  temperature,
}: {
  model: string;
  thinking: Anthropic.ThinkingConfigParam;
  topK?: number;
  topP?: number | null;
  temperature?: number;
}): void {
  const opus47 = isOpus47Model(model);
  if (opus47 && thinking.type === 'enabled') {
    throw new Error(
      'thinking.type="enabled" is not supported for claude-opus-4-7; use thinking.type="adaptive" instead'
    );
  }
  if (opus47 && 'budget_tokens' in thinking) {
    throw new Error(
      'thinking.budget_tokens is not supported for claude-opus-4-7; use outputConfig.effort instead'
    );
  }
  if (opus47) {
    if (topK !== undefined) {
      throw new Error(
        'topK is not supported for claude-opus-4-7; omit topK/topP/temperature or use model prompting instead'
      );
    }
    if (topP != null && topP !== 1) {
      throw new Error(
        'topP is not supported for claude-opus-4-7 when set to non-default values'
      );
    }
    if (temperature !== undefined && temperature !== 1) {
      throw new Error(
        'temperature is not supported for claude-opus-4-7 when set to non-default values'
      );
    }
  }
  if (!isThinkingEnabled(thinking)) {
    return;
  }
  if (topK !== undefined) {
    throw new Error('topK is not supported when thinking is enabled');
  }
  if (topP != null) {
    throw new Error('topP is not supported when thinking is enabled');
  }
  if (temperature !== undefined && temperature !== 1) {
    throw new Error('temperature is not supported when thinking is enabled');
  }
}

function getSamplingParams({
  model,
  thinking,
  topK,
  topP,
  temperature,
}: {
  model: string;
  thinking: Anthropic.ThinkingConfigParam;
  topK?: number;
  topP?: number | null;
  temperature?: number;
}): {
  temperature?: number;
  top_k?: number;
  top_p?: number;
} {
  if (isThinkingEnabled(thinking) || isOpus47Model(model)) {
    return {};
  }
  return {
    ...(temperature !== undefined ? { temperature } : {}),
    ...(topK !== undefined ? { top_k: topK } : {}),
    ...(topP != null ? { top_p: topP } : {}),
  };
}

function extractToken(
  chunk: AIMessageChunk
): [string, 'string' | 'input' | 'content'] | [undefined] {
  if (typeof chunk.content === 'string') {
    return [chunk.content, 'string'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'input' in chunk.content[0]
  ) {
    return typeof chunk.content[0].input === 'string'
      ? [chunk.content[0].input, 'input']
      : [JSON.stringify(chunk.content[0].input), 'input'];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'text' in chunk.content[0]
  ) {
    const text = chunk.content[0].text;
    return typeof text === 'string' ? [text, 'content'] : [undefined];
  } else if (
    Array.isArray(chunk.content) &&
    chunk.content.length >= 1 &&
    'thinking' in chunk.content[0]
  ) {
    const thinking = chunk.content[0].thinking;
    return typeof thinking === 'string' ? [thinking, 'content'] : [undefined];
  }
  return [undefined];
}

function cloneChunk(
  text: string,
  tokenType: string,
  chunk: AIMessageChunk
): AIMessageChunk {
  if (tokenType === 'string') {
    return new AIMessageChunk(Object.assign({}, chunk, { content: text }));
  } else if (tokenType === 'input') {
    return chunk;
  }
  const content = chunk.content[0] as MessageContentComplex;
  if (tokenType === 'content' && content.type === 'text') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (tokenType === 'content' && content.type === 'text_delta') {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { text })],
      })
    );
  } else if (
    tokenType === 'content' &&
    typeof content.type === 'string' &&
    content.type.startsWith('thinking')
  ) {
    return new AIMessageChunk(
      Object.assign({}, chunk, {
        content: [Object.assign({}, content, { thinking: text })],
      })
    );
  }

  return chunk;
}

export type CustomAnthropicInput = AnthropicInput & {
  _lc_stream_delay?: number;
  outputConfig?: AnthropicOutputConfig;
  inferenceGeo?: string;
  contextManagement?: AnthropicContextManagementConfigParam;
} & BaseChatModelParams;

type CustomAnthropicCallOptions = {
  outputConfig?: AnthropicOutputConfig;
  outputFormat?: Anthropic.Messages.JSONOutputFormat;
  inferenceGeo?: string;
  betas?: AnthropicBeta[];
  container?: string;
  mcp_servers?: AnthropicMCPServerURLDefinition[];
};

/**
 * A type representing additional parameters that can be passed to the
 * Anthropic API.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Kwargs = Record<string, any>;

export class CustomAnthropic extends ChatAnthropicMessages {
  _lc_stream_delay: number;
  private message_start: AnthropicMessageStartEvent | undefined;
  private message_delta: AnthropicMessageDeltaEvent | undefined;
  private tools_in_params?: boolean;
  private emitted_usage?: boolean;
  top_k: number | undefined;
  outputConfig?: AnthropicOutputConfig;
  inferenceGeo?: string;
  contextManagement?: AnthropicContextManagementConfigParam;
  constructor(fields?: CustomAnthropicInput) {
    super(fields);
    this.resetTokenEvents();
    this.setDirectFields(fields);
    this._lc_stream_delay = fields?._lc_stream_delay ?? 25;
    this.outputConfig = fields?.outputConfig;
    this.inferenceGeo = fields?.inferenceGeo;
    this.contextManagement = fields?.contextManagement;
  }

  static lc_name(): 'LibreChatAnthropic' {
    return 'LibreChatAnthropic';
  }

  /**
   * Get the parameters used to invoke the model
   */
  override invocationParams(
    options?: this['ParsedCallOptions']
  ): Omit<
    AnthropicMessageCreateParams | AnthropicStreamingMessageCreateParams,
    'messages'
  > &
    Kwargs {
    const tool_choice:
      | Anthropic.Messages.ToolChoiceAuto
      | Anthropic.Messages.ToolChoiceAny
      | Anthropic.Messages.ToolChoiceTool
      | undefined = handleToolChoice(options?.tool_choice);

    const callOptions = options as CustomAnthropicCallOptions | undefined;
    const mergedOutputConfig: AnthropicOutputConfig | undefined = (():
      | AnthropicOutputConfig
      | undefined => {
      const base = {
        ...this.outputConfig,
        ...callOptions?.outputConfig,
      };
      if (callOptions?.outputFormat && !base.format) {
        base.format = callOptions.outputFormat;
      }
      return Object.keys(base).length > 0 ? base : undefined;
    })();

    const inferenceGeo = callOptions?.inferenceGeo ?? this.inferenceGeo;

    const contextManagement = this.contextManagement;
    const toolBetas = getToolBetas(options?.tools);
    const compactionBetas = getCompactionBetas(contextManagement);
    const taskBudgetBetas = getTaskBudgetBetas(this.model, mergedOutputConfig);
    const formattedTools = this.formatStructuredToolToAnthropic(options?.tools);

    const sharedParams = {
      tools: formattedTools,
      tool_choice,
      thinking: this.thinking,
      context_management: contextManagement,
      ...this.invocationKwargs,
      container: callOptions?.container,
      betas: combineBetas(
        this.betas,
        callOptions?.betas,
        toolBetas,
        compactionBetas,
        taskBudgetBetas
      ),
      output_config: mergedOutputConfig,
      inference_geo: inferenceGeo,
      mcp_servers: callOptions?.mcp_servers,
    };
    validateInvocationParamCompatibility({
      model: this.model,
      thinking: this.thinking,
      topK: this.top_k,
      topP: this.topP,
      temperature: this.temperature,
    });

    return {
      model: this.model,
      stop_sequences: options?.stop ?? this.stopSequences,
      stream: this.streaming,
      max_tokens: this.maxTokens,
      ...getSamplingParams({
        model: this.model,
        thinking: this.thinking,
        topK: this.top_k,
        topP: this.topP,
        temperature: this.temperature,
      }),
      ...sharedParams,
    };
  }

  /**
   * Get stream usage as returned by this client's API response.
   * @returns The stream usage object.
   */
  getStreamUsage(): UsageMetadata | undefined {
    if (this.emitted_usage === true) {
      return;
    }
    const inputUsage = this.message_start?.message.usage as
      | undefined
      | AnthropicStreamUsage;
    const outputUsage = this.message_delta?.usage as
      | undefined
      | Partial<AnthropicStreamUsage>;
    if (!outputUsage) {
      return;
    }
    const cacheCreationInputTokens =
      inputUsage?.cache_creation_input_tokens ?? 0;
    const cacheReadInputTokens = inputUsage?.cache_read_input_tokens ?? 0;
    const inputTokens =
      (inputUsage?.input_tokens ?? 0) +
      cacheCreationInputTokens +
      cacheReadInputTokens;
    const totalUsage: UsageMetadata = {
      input_tokens: inputTokens,
      output_tokens: outputUsage.output_tokens ?? 0,
      total_tokens: inputTokens + (outputUsage.output_tokens ?? 0),
    };

    if (
      inputUsage?.cache_creation_input_tokens != null ||
      inputUsage?.cache_read_input_tokens != null
    ) {
      totalUsage.input_token_details = {
        cache_creation: cacheCreationInputTokens,
        cache_read: cacheReadInputTokens,
      };
    }

    this.emitted_usage = true;
    return totalUsage;
  }

  resetTokenEvents(): void {
    this.message_start = undefined;
    this.message_delta = undefined;
    this.emitted_usage = undefined;
    this.tools_in_params = undefined;
  }

  setDirectFields(fields?: CustomAnthropicInput): void {
    this.temperature = fields?.temperature ?? undefined;
    this.topP = fields?.topP ?? undefined;
    this.top_k = fields?.topK;
    if (this.temperature === -1 || this.temperature === 1) {
      this.temperature = undefined;
    }
    if (this.topP === -1) {
      this.topP = undefined;
    }
    if (this.top_k === -1) {
      this.top_k = undefined;
    }
  }

  private createGenerationChunk({
    token,
    chunk,
    usageMetadata,
    shouldStreamUsage,
  }: {
    token?: string;
    chunk: AIMessageChunk;
    shouldStreamUsage: boolean;
    usageMetadata?: UsageMetadata;
  }): ChatGenerationChunk {
    const usage_metadata = shouldStreamUsage
      ? (usageMetadata ?? chunk.usage_metadata)
      : undefined;
    return new ChatGenerationChunk({
      message: new AIMessageChunk({
        // Just yield chunk as it is and tool_use will be concat by BaseChatModel._generateUncached().
        content: chunk.content,
        additional_kwargs: chunk.additional_kwargs,
        tool_call_chunks: chunk.tool_call_chunks,
        response_metadata: chunk.response_metadata,
        usage_metadata,
        id: chunk.id,
      }),
      text: token ?? '',
    });
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    this.resetTokenEvents();
    const params = this.invocationParams(options);
    const formattedMessages = _convertMessagesToAnthropicPayload(messages);
    const payload = {
      ...params,
      ...formattedMessages,
      stream: true,
    } as const;
    const coerceContentToString =
      !_toolsInParams(payload) &&
      !_documentsInParams(payload) &&
      !_thinkingInParams(payload) &&
      !_compactionInParams(payload);

    const stream = await this.createStreamWithRetry(payload, {
      headers: options.headers,
      signal: options.signal,
    });

    const shouldStreamUsage = options.streamUsage ?? this.streamUsage;

    for await (const data of stream) {
      if (options.signal?.aborted === true) {
        stream.controller.abort();
        throw new Error('AbortError: User aborted the request.');
      }

      if (data.type === 'message_start') {
        this.message_start = data as AnthropicMessageStartEvent;
      } else if (data.type === 'message_delta') {
        this.message_delta = data as AnthropicMessageDeltaEvent;
      }

      let usageMetadata: UsageMetadata | undefined;
      if (this.tools_in_params !== true && this.emitted_usage !== true) {
        usageMetadata = this.getStreamUsage();
      }

      const result = _makeMessageChunkFromAnthropicEvent(
        data as Anthropic.Beta.Messages.BetaRawMessageStreamEvent,
        {
          streamUsage: shouldStreamUsage,
          coerceContentToString,
        }
      );
      if (!result) continue;

      const { chunk } = result;
      const [token = '', tokenType] = extractToken(chunk);

      if (
        !tokenType ||
        tokenType === 'input' ||
        (token === '' && (usageMetadata != null || chunk.id != null))
      ) {
        const generationChunk = this.createGenerationChunk({
          token,
          chunk,
          usageMetadata,
          shouldStreamUsage,
        });
        yield generationChunk;
        await runManager?.handleLLMNewToken(
          token,
          undefined,
          undefined,
          undefined,
          undefined,
          { chunk: generationChunk }
        );
        continue;
      }

      const textStream = new TextStream(token, {
        delay: this._lc_stream_delay,
        firstWordChunk: true,
        minChunkSize: 4,
        maxChunkSize: 8,
      });

      const generator = textStream.generateText(options.signal);
      try {
        let emittedUsage = false;
        for await (const currentToken of generator) {
          if ((options.signal as AbortSignal | undefined)?.aborted === true) {
            break;
          }
          const newChunk = cloneChunk(currentToken, tokenType, chunk);

          const generationChunk = this.createGenerationChunk({
            token: currentToken,
            chunk: newChunk,
            usageMetadata: emittedUsage ? undefined : usageMetadata,
            shouldStreamUsage,
          });

          if (usageMetadata && !emittedUsage) {
            emittedUsage = true;
          }
          yield generationChunk;

          await runManager?.handleLLMNewToken(
            currentToken,
            undefined,
            undefined,
            undefined,
            undefined,
            { chunk: generationChunk }
          );
        }
      } finally {
        await generator.return();
      }
    }

    this.resetTokenEvents();
  }
}
