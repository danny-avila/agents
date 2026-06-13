import { concat } from '@langchain/core/utils/stream';
import { AIMessageChunk } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessage } from '@langchain/core/messages';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type * as t from '@/types';
import { annotateMessagesForLLM } from '@/tools/toolOutputReferences';
import { Constants, GraphEvents, Providers } from '@/common';
import { manualToolStreamProviders } from '@/llm/providers';
import { modifyDeltaProperties } from '@/messages';
import { ChatModelStreamHandler } from '@/stream';
import { initializeModel } from '@/llm/init';

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
  const registry = context?.getOrCreateToolOutputRegistry();
  const runId = config?.configurable?.run_id as string | undefined;
  const messagesForProvider = annotateMessagesForLLM(messages, registry, runId);

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
    const stream = await model.stream(messagesForProvider, config);
    let finalChunk: AIMessageChunk | undefined;
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

    if ((finalChunk?.tool_calls?.length ?? 0) > 0) {
      finalChunk!.tool_calls = finalChunk!.tool_calls?.filter(
        (tool_call: ToolCall) => !!tool_call.name
      );
    }

    return { messages: [finalChunk as AIMessageChunk] };
  }

  const finalMessage = await model.invoke(messagesForProvider, config);
  if ((finalMessage.tool_calls?.length ?? 0) > 0) {
    finalMessage.tool_calls = finalMessage.tool_calls?.filter(
      (tool_call: ToolCall) => !!tool_call.name
    );
  }
  return { messages: [finalMessage] };
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
 * Throws the last error if all fallbacks fail.
 */
export async function tryFallbackProviders({
  fallbacks,
  tools,
  messages,
  config,
  primaryError,
  context,
  onChunk,
}: {
  fallbacks: Array<{ provider: Providers; clientOptions?: t.ClientOptions }>;
  tools?: t.GraphTools;
  messages: BaseMessage[];
  config?: RunnableConfig;
  primaryError: unknown;
  context?: InvokeContext;
  onChunk?: OnChunk;
}): Promise<Partial<t.BaseGraphState> | undefined> {
  let lastError: unknown = primaryError;
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
      const result = await attemptInvoke(
        {
          model: fbModel as t.ChatModel,
          messages,
          provider: fb.provider,
          context,
          onChunk,
        },
        fbConfig
      );
      return result;
    } catch (e) {
      lastError = e;
      continue;
    }
  }
  if (lastError !== undefined) {
    throw lastError;
  }
  return undefined;
}
