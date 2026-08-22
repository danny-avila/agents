import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type * as t from '@/types';
import {
  projectCacheControlledToolOutputsToText,
  projectComputerCallOutputsToText,
  projectOpenAIChatToolMessageContent,
  projectOpenAIResponsesToolMessageContent,
  projectOpenRouterToolMessageContent,
  projectSingleTextToolOutputsToText,
  projectStructuredToolOutputsToText,
  projectToolStreamContentForProvider,
} from '@/messages/core';
import {
  coalesceAdjacentUserTurns,
  strictAlternationProviders,
  appendPredecessorHandoffCue,
  removePredecessorHandoffCue,
} from '@/messages';
import {
  stripAnthropicCacheControl,
  stripBedrockCacheControl,
} from '@/messages/cache';
import { annotateMessagesForLLM } from '@/tools/toolOutputReferences';
import { Providers } from '@/common';
import { isAnthropicLike, isOpenAILike } from '@/utils/llm';

const preparedProviderRequestBrand = Symbol('PreparedProviderRequest');

export type ProviderMessageProjectionMode =
  | 'chat-messages'
  | 'openai-responses';

export interface ProviderPayloadMeasurement {
  readonly fits: boolean;
  readonly projectedMessageTokens?: number;
  readonly availableMessageTokens?: number;
  readonly contextBudget?: number;
  readonly effectiveInstructionTokens?: number;
}

export interface PreparedProviderRequest {
  readonly model: t.ChatModel;
  readonly modelId?: string;
  readonly provider: Providers;
  readonly projectionMode: ProviderMessageProjectionMode;
  readonly messages: BaseMessage[];
  readonly measurement?: ProviderPayloadMeasurement;
  readonly [preparedProviderRequestBrand]: true;
}

type PreparedProviderRequestData = Omit<
  PreparedProviderRequest,
  typeof preparedProviderRequestBrand
>;

export interface ProviderRequestContext {
  getOrCreateToolOutputRegistry?(): ToolOutputReferenceRegistry | undefined;
  isRunProducedMessage?(message: BaseMessage): boolean;
}

export interface PrepareProviderRequestParams {
  model: t.ChatModel;
  messages: BaseMessage[];
  provider: Providers;
  context?: ProviderRequestContext;
  config?: RunnableConfig;
  maxToolResultChars?: number;
  measure?: (messages: BaseMessage[]) => ProviderPayloadMeasurement;
}

export function usesNativeOpenAIResponses(
  model: t.ChatModel,
  provider: Providers,
  callOptions?: unknown
): boolean {
  if (!isOpenAILike(provider)) {
    return false;
  }
  let candidate: unknown = model;
  let effectiveCallOptions = callOptions;
  const seen = new Set<object>();
  for (let depth = 0; depth < 20; depth++) {
    if (candidate == null || typeof candidate !== 'object') {
      return false;
    }
    if (seen.has(candidate)) {
      return false;
    }
    seen.add(candidate);
    const runnable = candidate as {
      _useResponsesApi?: (options?: unknown) => boolean;
      bound?: unknown;
      defaultOptions?: unknown;
      last?: unknown;
      constructor?: { name?: unknown };
    };
    try {
      if (
        runnable.defaultOptions != null &&
        typeof runnable.defaultOptions === 'object' &&
        !Array.isArray(runnable.defaultOptions) &&
        effectiveCallOptions != null &&
        typeof effectiveCallOptions === 'object' &&
        !Array.isArray(effectiveCallOptions)
      ) {
        effectiveCallOptions = {
          ...(runnable.defaultOptions as Record<string, unknown>),
          ...(effectiveCallOptions as Record<string, unknown>),
        };
      } else if (effectiveCallOptions == null) {
        effectiveCallOptions = runnable.defaultOptions;
      }
      if (
        runnable._useResponsesApi?.(effectiveCallOptions) === true ||
        runnable._useResponsesApi?.(undefined) === true
      ) {
        return true;
      }
    } catch {
      // Continue through RunnableSequence/RunnableBinding wrappers.
    }
    if (
      typeof runnable.constructor?.name === 'string' &&
      runnable.constructor.name.includes('Responses')
    ) {
      return true;
    }
    if (runnable.last != null && typeof runnable.last === 'object') {
      candidate = runnable.last;
      continue;
    }
    if (runnable.bound != null && typeof runnable.bound === 'object') {
      candidate = runnable.bound;
      continue;
    }
    return false;
  }
  return false;
}

function resolveProviderMessageProjectionMode(
  model: t.ChatModel,
  provider: Providers,
  callOptions?: unknown
): ProviderMessageProjectionMode {
  return usesNativeOpenAIResponses(model, provider, callOptions)
    ? 'openai-responses'
    : 'chat-messages';
}

interface ProjectMessagesForProviderParams {
  model: t.ChatModel;
  messages: BaseMessage[];
  provider: Providers;
  maxToolResultChars?: number;
  callOptions?: unknown;
}

function projectMessagesForProviderMode({
  messages,
  provider,
  maxToolResultChars,
}: ProjectMessagesForProviderParams,
projectionMode: ProviderMessageProjectionMode
): BaseMessage[] {
  const nativeOpenAIResponses = projectionMode === 'openai-responses';
  const providerInputMessages = projectToolStreamContentForProvider(
    messages,
    nativeOpenAIResponses ? 'native' : 'fallback',
    maxToolResultChars
  );
  if (nativeOpenAIResponses) {
    return projectOpenAIResponsesToolMessageContent(
      stripAnthropicCacheControl(
        stripBedrockCacheControl(providerInputMessages)
      ),
      maxToolResultChars
    );
  }
  if (provider === Providers.OPENROUTER) {
    return projectComputerCallOutputsToText(
      projectOpenRouterToolMessageContent(
        stripBedrockCacheControl(providerInputMessages),
        maxToolResultChars
      )
    );
  }
  if (isOpenAILike(provider)) {
    return projectComputerCallOutputsToText(
      projectOpenAIChatToolMessageContent(
        stripAnthropicCacheControl(
          stripBedrockCacheControl(providerInputMessages)
        ),
        maxToolResultChars
      )
    );
  }
  if (provider === Providers.ANTHROPIC) {
    return projectComputerCallOutputsToText(
      projectSingleTextToolOutputsToText(
        stripBedrockCacheControl(providerInputMessages),
        maxToolResultChars
      )
    );
  }
  if (provider === Providers.BEDROCK) {
    return stripAnthropicCacheControl(
      projectComputerCallOutputsToText(
        projectCacheControlledToolOutputsToText(
          providerInputMessages,
          maxToolResultChars
        )
      )
    );
  }
  return projectComputerCallOutputsToText(
    projectStructuredToolOutputsToText(
      projectSingleTextToolOutputsToText(
        stripAnthropicCacheControl(
          stripBedrockCacheControl(providerInputMessages)
        ),
        maxToolResultChars
      ),
      maxToolResultChars
    )
  );
}

/** Produces the provider-facing representation before adapter serialization. */
export function projectMessagesForProvider(
  params: ProjectMessagesForProviderParams
): BaseMessage[] {
  return projectMessagesForProviderMode(
    params,
    resolveProviderMessageProjectionMode(
      params.model,
      params.provider,
      params.callOptions
    )
  );
}

/** Reads the serving model id through LangChain binding/sequence wrappers. */
export function resolveServingModelId(model: unknown): string | undefined {
  const seen = new Set<unknown>();
  let current: unknown = model;
  while (current != null && typeof current === 'object' && !seen.has(current)) {
    seen.add(current);
    const wrapper = current as {
      model?: unknown;
      bound?: unknown;
      last?: unknown;
      steps?: unknown[];
    };
    if (typeof wrapper.model === 'string' && wrapper.model !== '') {
      return wrapper.model;
    }
    current =
      wrapper.bound ??
      wrapper.last ??
      (Array.isArray(wrapper.steps)
        ? wrapper.steps[wrapper.steps.length - 1]
        : undefined);
  }
  return undefined;
}

/**
 * Finalizes one provider request and measures the exact message array that
 * will be passed to LangChain. Source messages remain untouched.
 */
export function prepareProviderRequest({
  model,
  messages,
  provider,
  context,
  config,
  maxToolResultChars,
  measure,
}: PrepareProviderRequestParams): PreparedProviderRequest {
  const projectionMode = resolveProviderMessageProjectionMode(
    model,
    provider,
    config
  );
  const projected = projectMessagesForProviderMode(
    {
      model,
      messages,
      provider,
      maxToolResultChars,
      callOptions: config,
    },
    projectionMode
  );
  const registry = context?.getOrCreateToolOutputRegistry?.();
  const runId = config?.configurable?.run_id as string | undefined;
  const annotated = annotateMessagesForLLM(projected, registry, runId);
  const isRunProduced = context?.isRunProducedMessage;
  const modelId = resolveServingModelId(model);
  const cued = isAnthropicLike(provider, {
    model: modelId,
  })
    ? appendPredecessorHandoffCue(
      annotated,
      isRunProduced == null
        ? undefined
        : (message): boolean => isRunProduced.call(context, message)
    )
    : removePredecessorHandoffCue(annotated);
  const preparedMessages = strictAlternationProviders.has(provider)
    ? coalesceAdjacentUserTurns(cued)
    : cued;

  const request: PreparedProviderRequestData = {
    model,
    modelId,
    provider,
    projectionMode,
    messages: preparedMessages,
    measurement: measure?.(preparedMessages),
  };
  Object.defineProperty(request, preparedProviderRequestBrand, {
    value: true,
    enumerable: false,
  });
  return Object.freeze(request) as PreparedProviderRequest;
}

export function assertPreparedProviderRequestFor(
  request: PreparedProviderRequest,
  model: t.ChatModel,
  provider: Providers,
  config?: RunnableConfig
): void {
  if (
    !Object.prototype.hasOwnProperty.call(request, preparedProviderRequestBrand)
  ) {
    throw new Error('Invalid prepared provider request');
  }
  if (request.model !== model) {
    throw new Error('Prepared provider request does not match serving model');
  }
  if (request.provider !== provider) {
    throw new Error('Prepared provider request does not match serving provider');
  }
  if (
    request.projectionMode !==
    resolveProviderMessageProjectionMode(model, provider, config)
  ) {
    throw new Error(
      'Prepared provider request does not match invocation options'
    );
  }
}
