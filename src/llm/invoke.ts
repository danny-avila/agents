import { concat } from '@langchain/core/utils/stream';
import { AIMessageChunk } from '@langchain/core/messages';
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import { getCallbackManagerForConfig } from '@langchain/core/runnables';
import type { Serialized } from '@langchain/core/load/serializable';
import type { ChatGeneration } from '@langchain/core/outputs';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { ContextOverflowContext } from '@/utils/errors';
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
  stripAnthropicCacheControl,
  stripBedrockCacheControl,
} from '@/messages/cache';
import { annotateMessagesForLLM } from '@/tools/toolOutputReferences';
import { assertNotTruncatedToolCall } from '@/llm/truncation';
import { Constants, ContentTypes, GraphEvents, Providers } from '@/common';
import { manualToolStreamProviders } from '@/llm/providers';
import { appendCallbacks } from '@/utils/callbacks';
import { safeDispatchCustomEvent } from '@/utils/events';
import { getContextOverflowInfo } from '@/utils/errors';
import { modifyDeltaProperties } from '@/messages';
import { canSealPreempt } from '@/llm/preempt';
import { ChatModelStreamHandler } from '@/stream';
import { initializeModel } from '@/llm/init';
import { isOpenAILike } from '@/utils/llm';

/**
 * Context passed to `attemptInvoke`. Matches the subset of Graph that
 * `ChatModelStreamHandler.handle` needs *plus* the explicit
 * `getOrCreateToolOutputRegistry()` accessor that `attemptInvoke`
 * itself calls to pull the run-scoped tool-output registry off the
 * graph and project each relevant ToolMessage into a transient
 * annotated copy before the provider call.
 *
 * The intersection is intentional: `Parameters<...>[3]` resolves
 * indirectly through the stream handler's signature (which returns
 * `StandardGraph` and already exposes the accessor since #117), but
 * stating it explicitly here surfaces the contract at the call site —
 * a developer reading `attemptInvoke` doesn't have to chase the
 * upstream handler's parameter list to discover that
 * `context?.getOrCreateToolOutputRegistry()` is a real thing. Single
 * optional chain only — the method itself is required on the
 * `StandardGraph` branch of the intersection, so the second `?.` is
 * unnecessary at the call site.
 *
 * `NonNullable<...>` strips `undefined` from the upstream parameter
 * type so the intersection doesn't collapse to `never` on the
 * undefined branch; callers express optionality via `context?:
 * InvokeContext` on the function signature instead.
 *
 * Callers without a registry (e.g. summarization) simply pass no
 * `context` and the transform safely no-ops.
 */
export type InvokeContext = NonNullable<
  Parameters<ChatModelStreamHandler['handle']>[3]
> & {
  getOrCreateToolOutputRegistry?(): ToolOutputReferenceRegistry | undefined;
};

/**
 * Per-chunk callback for custom stream processing.
 * When provided, replaces the default `ChatModelStreamHandler`.
 */
export type OnChunk = (chunk: AIMessageChunk) => void | Promise<void>;

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

/**
 * Produces the exact provider-facing message representation before a model
 * adapter serializes it. This is shared by invocation and Graph's final budget
 * guard so structured tool output cannot grow after the payload was measured.
 */
export function projectMessagesForProvider({
  model,
  messages,
  provider,
  maxToolResultChars,
  callOptions,
}: {
  model: t.ChatModel;
  messages: BaseMessage[];
  provider: Providers;
  maxToolResultChars?: number;
  callOptions?: unknown;
}): BaseMessage[] {
  const providerInputMessages = projectToolStreamContentForProvider(messages);
  if (usesNativeOpenAIResponses(model, provider, callOptions)) {
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

function getRegisteredDefaultChatStreamHandler(
  context?: InvokeContext
): ChatModelStreamHandler | undefined {
  const handler = context?.handlerRegistry?.getHandler(
    GraphEvents.CHAT_MODEL_STREAM
  );
  return handler instanceof ChatModelStreamHandler ? handler : undefined;
}

function hasReasoningDetails(chunk: AIMessageChunk): boolean {
  const reasoningDetails = chunk.additional_kwargs.reasoning_details;
  return Array.isArray(reasoningDetails) && reasoningDetails.length > 0;
}

function removeOpenRouterFinalReasoningReplayContent({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: Providers;
}): AIMessageChunk {
  const content = getOpenRouterFinalReasoningContent({
    current,
    next,
    provider,
  });
  if (content == null || content === next.content) {
    return next;
  }

  return new AIMessageChunk(
    Object.assign({}, next, {
      content,
    })
  );
}

function getOpenRouterFinalReasoningContent({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: Providers;
}): string | undefined {
  if (
    provider !== Providers.OPENROUTER ||
    current == null ||
    !hasReasoningDetails(next) ||
    typeof current.content !== 'string' ||
    current.content === '' ||
    typeof next.content !== 'string' ||
    next.content === ''
  ) {
    return undefined;
  }
  if (!next.content.startsWith(current.content)) {
    return next.content;
  }
  return next.content.slice(current.content.length);
}

function removeReasoningDetails(
  additionalKwargs: AIMessageChunk['additional_kwargs']
): AIMessageChunk['additional_kwargs'] {
  return Object.fromEntries(
    Object.entries(additionalKwargs).filter(
      ([key]) => key !== 'reasoning_details'
    )
  );
}

function getStreamHandlingChunk({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: Providers;
}): AIMessageChunk | undefined {
  const content = getOpenRouterFinalReasoningContent({
    current,
    next,
    provider,
  });
  if (content == null) {
    return next;
  }
  if (content === '') {
    return undefined;
  }
  return new AIMessageChunk(
    Object.assign({}, next, {
      content,
      additional_kwargs: removeReasoningDetails(next.additional_kwargs),
    })
  );
}

/**
 * Best-effort output-token count for a sealed turn, used only when the
 * provider never got to send its usage chunk.
 */
function countSealedTokens(
  context: InvokeContext | undefined,
  metadata: Record<string, unknown> | undefined,
  messages: BaseMessage[]
): number | undefined {
  try {
    const counter = context?.getAgentContext(metadata).tokenCounter;
    if (counter == null) {
      return undefined;
    }
    let total = 0;
    for (const message of messages) {
      total += counter(message);
    }
    return total;
  } catch {
    return undefined;
  }
}

/**
 * Instruction overhead the provider processed but that never appears in the
 * message array: `createCallModel` pipes the model through
 * `agentContext.systemRunnable` and binds tool schemas AFTER `messages` is
 * formed, so the system prompt, dynamic instructions, summary and tool
 * schemas are all billed yet invisible here.
 *
 * Read per-node via `getAgentContext(metadata)` rather than the graph-level
 * accessor, which is hardcoded to `defaultAgentId` and would report the wrong
 * agent's overhead in a `MultiAgentGraph`.
 */
function sealedInstructionOverhead(
  context: InvokeContext | undefined,
  metadata: Record<string, unknown> | undefined
): number {
  try {
    const agentContext = context?.getAgentContext(metadata);
    return (
      agentContext?.resolvedInstructionOverhead ??
      agentContext?.instructionTokens ??
      0
    );
  } catch {
    return 0;
  }
}

/**
 * Best-effort usage for a turn the provider never got to bill us for.
 *
 * The prompt matters as much as the completion: the provider processed the
 * ENTIRE prompt — messages plus instruction overhead — before we sealed, and
 * every resume re-sends it, so under-counting input hides the expensive half
 * of a preempted run.
 *
 * ESTIMATE, NOT MEASUREMENT. Messages are counted with the host's tokenizer
 * rather than the provider's, and `toolSchemaTokens` applies a heuristic
 * multiplier. It is also an over-count on the fallback path, where
 * `tryFallbackProviders` builds a bare model with no `systemRunnable` pipe so
 * the system prompt genuinely is not sent. Accepted rather than threaded
 * through a flag: only the fallback-plus-seal combination is affected, and an
 * over-count is safer than the previous fabricated `input_tokens: 0`.
 *
 * Marked `estimated_usage` so calibration can refuse to learn from it — a
 * ratio derived from the same counter that produced the estimate is
 * self-consistent by construction and would drag a provider's real
 * calibration toward 1.0.
 */
function synthesizeSealedUsage(
  context: InvokeContext | undefined,
  chunk: AIMessageChunk,
  prompt: BaseMessage[],
  metadata: Record<string, unknown> | undefined
): void {
  if (chunk.usage_metadata != null) {
    return;
  }
  const outputTokens = countSealedTokens(context, metadata, [chunk]);
  if (outputTokens == null) {
    return;
  }
  const inputTokens =
    (countSealedTokens(context, metadata, prompt) ?? 0) +
    sealedInstructionOverhead(context, metadata);
  chunk.usage_metadata = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
  };
  chunk.response_metadata = {
    ...chunk.response_metadata,
    estimated_usage: true,
  };
}

function getMessageText(chunk: AIMessageChunk): string {
  if (typeof chunk.content === 'string') {
    return chunk.content;
  }
  let text = '';
  for (const block of chunk.content) {
    if (block.type === ContentTypes.TEXT) {
      const value = block[ContentTypes.TEXT];
      if (typeof value === 'string') {
        text += value;
      }
    }
  }
  return text;
}

/**
 * Ends the real model run for a turn that was sealed mid-stream.
 *
 * Mandatory, not cosmetic. `@langchain/core`'s `_streamIterator` calls
 * `handleLLMEnd` after its try/catch with no `finally`, so breaking out of the
 * consumer's `for await` produces a *return* completion that fires neither
 * `handleLLMError` nor `handleLLMEnd`. The run would stay open in every
 * callback handler: the host records no usage — and since each seal re-sends
 * the whole prompt, N preemptions cost N unrecorded prompts — while LangSmith
 * and Langfuse hold a span that never closes.
 *
 * `runId` cannot be dictated from here (the bound runnable consumes
 * `config.runId` for its own run and hands the chat model a fresh one), but it
 * can be OBSERVED: the capture handler installed at the `model.stream` call
 * records it from `handleChatModelStart`, which fires before the first chunk.
 * Rebuilding the manager against that id closes the real run, and the host's
 * `on_chat_model_end` then arrives through the ordinary `streamEvents` path.
 *
 * Falls back to a custom-event dispatch if the id was never observed, so the
 * host still records usage even when the native close is unavailable.
 */
async function endSealedModelRun(
  context: InvokeContext | undefined,
  chunk: AIMessageChunk,
  prompt: BaseMessage[],
  llmRunId: string | undefined,
  config?: RunnableConfig
): Promise<void> {
  const metadata = config?.metadata as Record<string, unknown> | undefined;
  synthesizeSealedUsage(context, chunk, prompt, metadata);
  if (llmRunId != null) {
    try {
      const callbackManager = await getCallbackManagerForConfig(config);
      if (callbackManager != null) {
        const runManager = new CallbackManagerForLLMRun(
          llmRunId,
          callbackManager.handlers,
          callbackManager.inheritableHandlers,
          callbackManager.tags,
          callbackManager.inheritableTags,
          callbackManager.metadata,
          callbackManager.inheritableMetadata,
          callbackManager.getParentRunId()
        );
        const generation: ChatGeneration = {
          text: getMessageText(chunk),
          message: chunk,
        };
        await runManager.handleLLMEnd({
          generations: [[generation]],
          llmOutput: {},
        });
        return;
      }
    } catch (e) {
      /**
       * A sealed answer that reaches the user is worth more than a tidy
       * trace. Fall through to the custom event rather than failing the run.
       */
      // eslint-disable-next-line no-console
      console.warn(
        '[attemptInvoke] Native close of the sealed model run failed; falling back to a custom event:',
        e instanceof Error ? e.message : e
      );
    }
  }
  await safeDispatchCustomEvent(
    GraphEvents.CHAT_MODEL_END,
    { output: chunk },
    config
  );
}

function appendStreamChunk({
  current,
  next,
  provider,
}: {
  current?: AIMessageChunk;
  next: AIMessageChunk;
  provider: Providers;
}): AIMessageChunk {
  if (current == null) {
    return next;
  }
  return concat(
    current,
    removeOpenRouterFinalReasoningReplayContent({ current, next, provider })
  );
}

/**
 * Invokes a chat model with the given messages, handling both streaming and
 * non-streaming paths.
 *
 * By default, stream chunks are processed through a `ChatModelStreamHandler`
 * that dispatches run steps (MESSAGE_CREATION, TOOL_CALLS) for the graph.
 * Pass an `onChunk` callback to override this with custom chunk processing
 * (e.g. summarization delta events).
 */
export async function attemptInvoke(
  {
    model,
    messages,
    provider,
    context,
    onChunk,
  }: {
    model: t.ChatModel;
    messages: BaseMessage[];
    provider: Providers;
    context?: InvokeContext;
    onChunk?: OnChunk;
  },
  config?: RunnableConfig
): Promise<Partial<t.BaseGraphState>> {
  /**
   * Pull the run-scoped tool output registry off the graph (when one
   * exists) and project ToolMessages carrying ref metadata into a
   * transient annotated copy. The original `messages` array stays
   * untouched so the graph state never sees `[ref: …]` / `_ref`
   * payload.
   */
  const invocationMessages = projectMessagesForProvider({
    model,
    messages,
    provider,
    callOptions: config,
  });
  const registry = context?.getOrCreateToolOutputRegistry();
  const runId = config?.configurable?.run_id as string | undefined;
  const messagesForProvider = annotateMessagesForLLM(
    invocationMessages,
    registry,
    runId
  );

  /**
   * Stamp the provider that is ACTUALLY serving this invocation onto the
   * callback metadata. `attemptInvoke` is the single funnel for primary,
   * fallback, and summarization model calls, so consumers that need
   * provider attribution per call (the subagent usage-capture handler)
   * read this key instead of trusting static agent config — which is
   * wrong for fallback-served calls — or `ls_provider` — which derived
   * providers inherit from their base class.
   */
  config = {
    ...config,
    metadata: {
      ...(config?.metadata ?? {}),
      [Constants.INVOKED_PROVIDER]: provider,
    },
  };

  if (model.stream) {
    /**
     * Observed, not dictated. `handleChatModelStart` fires with the chat
     * model's real run id before the first chunk, which is the only way to
     * name the run a seal has to close — pinning `config.runId` does not
     * survive the bound runnable. Installed only when preemption is
     * configured, so a run that cannot seal carries no extra handler.
     */
    let sealedRunId: string | undefined;
    const streamConfig =
      context?.preemption == null
        ? config
        : {
          ...config,
          callbacks: appendCallbacks(config.callbacks, [
            {
              handleChatModelStart: (
                _llm: Serialized,
                _messages: BaseMessage[][],
                runId: string
              ): void => {
                sealedRunId ??= runId;
              },
            },
          ]),
        };
    const stream = await model.stream(messagesForProvider, streamConfig);
    let finalChunk: AIMessageChunk | undefined;
    let preempted = false;
    const registeredStreamHandler =
      getRegisteredDefaultChatStreamHandler(context);

    if (onChunk) {
      for await (const chunk of stream) {
        await onChunk(chunk);
        finalChunk = appendStreamChunk({
          current: finalChunk,
          next: chunk,
          provider,
        });
      }
    } else if (registeredStreamHandler == null) {
      const metadata = config.metadata as Record<string, unknown> | undefined;
      const streamHandler = new ChatModelStreamHandler();
      for await (const chunk of stream) {
        const handlingChunk = getStreamHandlingChunk({
          current: finalChunk,
          next: chunk,
          provider,
        });
        if (handlingChunk != null) {
          await streamHandler.handle(
            GraphEvents.CHAT_MODEL_STREAM,
            { chunk: handlingChunk },
            metadata,
            context
          );
        }
        finalChunk = appendStreamChunk({
          current: finalChunk,
          next: chunk,
          provider,
        });
        /**
         * Only this loop may seal. The registered-handler branch below
         * dispatches through `run.ts`'s decoupled `streamEvents` consumer,
         * which can lag the accumulated chunk — sealing there would let the
         * host index a content part the user has not been shown yet.
         */
        /**
         * Cheap poll first, shape check second, budget claim last. The claim
         * is what makes this safe under a parallel `MultiAgentGraph`: several
         * agents share one graph and can each see the poll as true, but only
         * one can take the slot, and a chunk that cannot seal never spends it.
         */
        if (
          context?.shouldPreemptStream() === true &&
          canSealPreempt(finalChunk) &&
          context.claimPreemptSeal()
        ) {
          preempted = true;
          break;
        }
      }
    } else {
      const metadata = config.metadata as Record<string, unknown> | undefined;
      for await (const chunk of stream) {
        const handlingChunk = getStreamHandlingChunk({
          current: finalChunk,
          next: chunk,
          provider,
        });
        if (handlingChunk != null && handlingChunk !== chunk) {
          await registeredStreamHandler.handle(
            GraphEvents.CHAT_MODEL_STREAM,
            { chunk: handlingChunk },
            metadata,
            context
          );
        }
        finalChunk = appendStreamChunk({
          current: finalChunk,
          next: chunk,
          provider,
        });
      }
    }

    if (manualToolStreamProviders.has(provider)) {
      finalChunk = modifyDeltaProperties(provider, finalChunk);
    }

    if (preempted && finalChunk != null) {
      finalChunk.response_metadata = {
        ...finalChunk.response_metadata,
        preempted: true,
      };
      await endSealedModelRun(
        context,
        finalChunk,
        messagesForProvider,
        sealedRunId,
        config
      );
    }

    if ((finalChunk?.tool_calls?.length ?? 0) > 0) {
      finalChunk!.tool_calls = finalChunk!.tool_calls?.filter(
        (tool_call: ToolCall) => !!tool_call.name
      );
    }

    assertNotTruncatedToolCall(finalChunk, provider);
    return { messages: [finalChunk as AIMessageChunk] };
  }

  const finalMessage = await model.invoke(messagesForProvider, config);
  if ((finalMessage.tool_calls?.length ?? 0) > 0) {
    finalMessage.tool_calls = finalMessage.tool_calls?.filter(
      (tool_call: ToolCall) => !!tool_call.name
    );
  }
  assertNotTruncatedToolCall(finalMessage, provider);
  return { messages: [finalMessage] };
}

/**
 * Identifies which fallback produced an error, so a caller planning a
 * recovery can reason about the client that actually failed rather than the
 * primary's configuration — their context windows and output allowances
 * differ, which is the whole reason a fallback exists.
 */
export interface FallbackErrorContext {
  provider: Providers;
  clientOptions?: t.ClientOptions;
  maxContextTokens?: number;
}

export interface FallbackOverflowCandidate {
  error: unknown;
  context: FallbackErrorContext;
}

const fallbackErrorContexts = new WeakMap<object, FallbackErrorContext>();
const fallbackOverflowCandidates = new WeakMap<
  object,
  FallbackOverflowCandidate[]
>();

function attachFallbackErrorContext(
  error: unknown,
  fallbackContext: FallbackErrorContext
): void {
  if (typeof error !== 'object' || error === null) {
    return;
  }
  fallbackErrorContexts.set(error, fallbackContext);
}

/** Reads back the fallback attribution attached by `tryFallbackProviders`. */
export function getFallbackErrorContext(
  error: unknown
): FallbackErrorContext | undefined {
  if (typeof error !== 'object' || error === null) {
    return undefined;
  }
  return fallbackErrorContexts.get(error);
}

/** Returns every fallback overflow retained from an exhausted provider chain. */
export function getFallbackOverflowCandidates(
  error: unknown
): FallbackOverflowCandidate[] {
  if (typeof error !== 'object' || error === null) {
    return [];
  }
  return [...(fallbackOverflowCandidates.get(error) ?? [])];
}

/**
 * Best-effort read of the configured model name from client options.
 * Providers disagree on the key (`model` vs `modelName`).
 */
function extractClientOptionsModel(
  clientOptions: t.ClientOptions | undefined
): string | undefined {
  const options = clientOptions as
    | { model?: unknown; modelName?: unknown }
    | undefined;
  if (typeof options?.model === 'string' && options.model !== '') {
    return options.model;
  }
  if (typeof options?.modelName === 'string' && options.modelName !== '') {
    return options.modelName;
  }
  return undefined;
}

/**
 * Attempts each fallback provider in order until one succeeds.
 *
 * When every fallback fails, a context overflow among them is thrown in
 * preference to whichever failure happened to come last. An overflow is the
 * one failure the caller can act on — it compacts and retries — and losing it
 * behind a later unrelated error would surface a dead end instead. Ordinary
 * failures still throw last-error-wins.
 */
export async function tryFallbackProviders({
  fallbacks,
  tools,
  messages,
  config,
  primaryError,
  context,
  onChunk,
  overflowContext,
  prepareProviderMessages,
}: {
  fallbacks: t.FallbackConfig[];
  tools?: t.GraphTools;
  messages: BaseMessage[];
  config?: RunnableConfig;
  primaryError: unknown;
  context?: InvokeContext;
  onChunk?: OnChunk;
  /**
   * Prompt-size corroboration for signatures that are not self-describing.
   * Vertex AI's overflow is a bare `400` with no reason, so without this a
   * fallback that overflows is indistinguishable from any other 400 and would
   * be dropped in favour of whichever failure came last.
   */
  overflowContext?: ContextOverflowContext;
  /**
   * Optional final payload guard used by Graph. It receives the initialized,
   * tool-bound fallback model so Responses-vs-Chat projection is exact before
   * the fallback request is measured and sent.
   */
  prepareProviderMessages?: (input: {
    model: t.ChatModel;
    messages: BaseMessage[];
    provider: Providers;
    clientOptions?: t.ClientOptions;
    maxContextTokens?: number;
    config?: RunnableConfig;
  }) => BaseMessage[] | Promise<BaseMessage[]>;
}): Promise<Partial<t.BaseGraphState> | undefined> {
  const isOverflow = (
    error: unknown,
    contextOverride = overflowContext
  ): boolean => getContextOverflowInfo(error, contextOverride) != null;
  let lastError: unknown = primaryError;
  /**
   * Tracked apart from the primary's overflow. A caller reaching this
   * function with an overflowing primary has already failed to recover from
   * it, so a fallback overflow — which may sit against a different window and
   * output allowance — is the more useful of the two to surface.
   */
  const overflowCandidates: FallbackOverflowCandidate[] = [];
  const primaryOverflowError: unknown = isOverflow(primaryError)
    ? primaryError
    : undefined;
  for (const fb of fallbacks) {
    try {
      const fbModel = initializeModel({
        provider: fb.provider,
        clientOptions: fb.clientOptions,
        tools,
      });
      /**
       * Stamp the fallback's configured model onto callback metadata so
       * per-call attribution (subagent usage capture) doesn't fall back to
       * the PRIMARY config's model when the provider reports no
       * `ls_model_name`. The serving provider is stamped uniformly by
       * `attemptInvoke` (`INVOKED_PROVIDER`).
       */
      const fbModelName = extractClientOptionsModel(fb.clientOptions);
      const fbConfig: RunnableConfig | undefined =
        fbModelName == null
          ? config
          : {
            ...config,
            metadata: {
              ...(config?.metadata ?? {}),
              [Constants.INVOKED_MODEL]: fbModelName,
            },
          };
      const fallbackMessages =
        (await prepareProviderMessages?.({
          model: fbModel as t.ChatModel,
          messages,
          provider: fb.provider,
          clientOptions: fb.clientOptions,
          maxContextTokens: fb.maxContextTokens,
          config: fbConfig,
        })) ?? messages;
      const result = await attemptInvoke(
        {
          model: fbModel as t.ChatModel,
          messages: fallbackMessages,
          provider: fb.provider,
          context,
          onChunk,
        },
        fbConfig
      );
      return result;
    } catch (e) {
      lastError = e;
      const fallbackOverflowContext: ContextOverflowContext = {
        provider: fb.provider,
        maxContextTokens: fb.maxContextTokens,
        ...(overflowContext?.provider === fb.provider
          ? {
            estimatedPromptTokens: overflowContext.estimatedPromptTokens,
          }
          : {}),
      };
      if (isOverflow(e, fallbackOverflowContext)) {
        const errorContext: FallbackErrorContext = {
          provider: fb.provider,
          clientOptions: fb.clientOptions,
          maxContextTokens: fb.maxContextTokens,
        };
        attachFallbackErrorContext(e, errorContext);
        overflowCandidates.push({ error: e, context: errorContext });
      }
      continue;
    }
  }
  /**
   * Preference order: a fallback overflow, then the primary's overflow, then
   * whichever failure came last. An overflow is the only one of the three a
   * caller can act on, and the fallback's carries the client attribution that
   * makes a correct retry budget possible.
   */
  const preferred =
    overflowCandidates[0]?.error ?? primaryOverflowError ?? lastError;
  if (
    overflowCandidates.length > 0 &&
    typeof preferred === 'object' &&
    preferred !== null
  ) {
    fallbackOverflowCandidates.set(preferred, overflowCandidates);
  }
  if (preferred !== undefined) {
    throw preferred;
  }
  return undefined;
}
