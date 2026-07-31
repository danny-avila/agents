import { CallbackHandler } from '@langfuse/langchain';
import { LangfuseOtelContextKeys } from '@langfuse/core';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { isGraphInterrupt, isParentCommand } from '@langchain/langgraph';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import {
  getLangfuseTracerProvider,
  propagateAttributes,
} from '@langfuse/tracing';
import type {
  AIMessageChunkFields,
  AIMessageFields,
  BaseMessage,
  MessageContent,
  UsageMetadata,
} from '@langchain/core/messages';
import type {
  ChatGeneration,
  Generation,
  LLMResult,
} from '@langchain/core/outputs';
import type { PropagateAttributesParams } from '@langfuse/tracing';
import type { Context } from '@opentelemetry/api';
import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import {
  resolveLangfuseConfigForSpan,
  resolveLangfuseScopeAgentId,
  resolveLangfuseScopeRunId,
  resolveTraceIdSeedForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  resolveToolOutputTracingConfig,
  hasLangfuseEnvConfig,
} from '@/langfuseConfig';
import {
  getLangfuseManagedSpanDestination,
  resolveLangfuseDestinationKey,
} from '@/langfuseSpanRegistry';
import { isPresent, parseBooleanEnv } from '@/utils/misc';

export {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
};

const TRACE_METADATA_MAX_LENGTH = 200;
const LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'LANGFUSE_FORCE_FLUSH_ON_DISPOSE';
const GRAPH_INTERRUPT_CONTROL_FLOW = { controlFlow: 'GraphInterrupt' } as const;
const GRAPH_INTERRUPT_TOOL_OUTPUT = JSON.stringify(
  GRAPH_INTERRUPT_CONTROL_FLOW
);

export type LangfuseTraceMetadata = Record<string, string>;
export type LangfuseTraceAttributes = Record<string, string | number | boolean>;
type LangfuseMetadata = NonNullable<t.LangfuseConfig['metadata']>;
type LangfuseConfigTraceAttributes = NonNullable<
  t.LangfuseConfig['librechatTraceAttributes']
>;

type LangfuseHandlerParams = {
  userId?: string;
  sessionId?: string;
  traceMetadata?: LangfuseTraceMetadata;
  tags?: string[];
  traceIdSeed?: string;
  /** Identity of the run this handler traces; ambient runtime scopes are
   *  only adopted when stamped with the same run (see
   *  `LangfuseRuntimeContext.runId`). */
  runId?: string;
  /** The run's resolved tool-output policy — for multi-agent streams the
   *  conservative aggregate across agents, which `this.langfuse` (the
   *  primary agent's config) cannot reproduce. Applied when a foreign
   *  scope's policy is rejected. */
  toolOutputTracing?: ResolvedLangfuseToolOutputTracingConfig;
  /** The run's propagated trace name, re-propagated when a foreign scope's
   *  attributes are cleared. */
  traceName?: string;
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
};

type HandlerIdentity = {
  userId?: string;
  sessionId?: string;
  tags?: string[];
  metadata?: LangfuseTraceMetadata;
  traceName?: string;
};

type LangfuseAttributeParams = AgentLangfuseHandlerParams & {
  traceName?: string;
};

type FlushableTracerProvider = {
  forceFlush?: () => Promise<void> | void;
};

type BedrockResponseUsage = {
  inputTokens?: number;
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
};

type BedrockResponseMetadata = {
  metadata?: {
    usage?: BedrockResponseUsage;
  };
};

function getLangfuseBedrockUsage(
  message: AIMessage | AIMessageChunk
): UsageMetadata | undefined {
  const usageMetadata = message.usage_metadata;
  const bedrockUsage = (message.response_metadata as BedrockResponseMetadata)
    .metadata?.usage;
  if (
    usageMetadata == null ||
    bedrockUsage == null ||
    usageMetadata.input_tokens !== bedrockUsage.inputTokens
  ) {
    return usageMetadata;
  }

  const cacheRead = bedrockUsage.cacheReadInputTokens ?? 0;
  const cacheCreation = bedrockUsage.cacheWriteInputTokens ?? 0;
  if (cacheRead === 0 && cacheCreation === 0) {
    return usageMetadata;
  }

  return {
    ...usageMetadata,
    input_tokens: usageMetadata.input_tokens + cacheRead + cacheCreation,
  };
}

const OMITTED_IMAGE_URL = '[image URL omitted]';

function summarizeImageUrl(url: unknown): string {
  if (typeof url !== 'string') {
    return OMITTED_IMAGE_URL;
  }
  const match =
    /^data:(image\/[a-zA-Z0-9.+-]+);base64,([a-zA-Z0-9+/]*={0,2})$/i.exec(url);
  if (match == null) {
    return OMITTED_IMAGE_URL;
  }
  const padding = match[2].length - match[2].replace(/=+$/, '').length;
  const bytes = Math.max(0, Math.floor((match[2].length * 3) / 4) - padding);
  return `[image omitted; type=${match[1].toLowerCase()}; bytes=${bytes}]`;
}

function sanitizeImageValue(value: unknown): unknown {
  if (typeof value === 'string') {
    return summarizeImageUrl(value);
  }
  if (value == null || typeof value !== 'object') {
    return OMITTED_IMAGE_URL;
  }
  const record = value as Record<string, unknown>;
  return {
    ...(record.detail != null && { detail: record.detail }),
    url: summarizeImageUrl(record.url),
  };
}

function sanitizeMessageContent(content: MessageContent): MessageContent {
  if (!Array.isArray(content)) {
    return content;
  }
  const sanitized = content.map((block) => {
    const record = block as unknown as Record<string, unknown>;
    if ('image_url' in record) {
      return {
        ...record,
        image_url: sanitizeImageValue(record.image_url),
      } as unknown as typeof block;
    }
    if (
      record.type === 'image' &&
      record.source != null &&
      typeof record.source === 'object'
    ) {
      const source = record.source as Record<string, unknown>;
      return {
        ...record,
        source: {
          ...(source.type != null && { type: source.type }),
          ...(source.media_type != null && { media_type: source.media_type }),
          data: summarizeImageUrl(
            typeof source.data === 'string' && source.media_type != null
              ? `data:${String(source.media_type)};base64,${source.data}`
              : source.data
          ),
        },
      } as unknown as typeof block;
    }
    if (record.inlineData != null && typeof record.inlineData === 'object') {
      const inlineData = record.inlineData as Record<string, unknown>;
      return {
        ...record,
        inlineData: {
          ...(inlineData.mimeType != null && { mimeType: inlineData.mimeType }),
          data: summarizeImageUrl(
            typeof inlineData.data === 'string' && inlineData.mimeType != null
              ? `data:${String(inlineData.mimeType)};base64,${inlineData.data}`
              : inlineData.data
          ),
        },
      } as unknown as typeof block;
    }
    return block;
  });
  return sanitized.some((block, index) => block !== content[index])
    ? sanitized
    : content;
}

function cloneMessageWithContent(message: BaseMessage): BaseMessage {
  const content = sanitizeMessageContent(message.content);
  if (content === message.content) {
    return message;
  }
  const clone = Object.create(
    Object.getPrototypeOf(message),
    Object.getOwnPropertyDescriptors(message)
  ) as BaseMessage;
  clone.content = content;
  return clone;
}

function sanitizeChatMessagesForLangfuse(
  messages: BaseMessage[][]
): BaseMessage[][] {
  return messages.map((batch) => batch.map(cloneMessageWithContent));
}

function cloneMessageForLangfuse(
  message: AIMessage | AIMessageChunk,
  usageMetadata: UsageMetadata | undefined
): AIMessage | AIMessageChunk {
  const fields: AIMessageFields = {
    content: sanitizeMessageContent(message.content),
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
    id: message.id,
    name: message.name,
    tool_calls: message.tool_calls,
    invalid_tool_calls: message.invalid_tool_calls,
    usage_metadata: usageMetadata,
  };

  if (message instanceof AIMessageChunk) {
    const chunkFields: AIMessageChunkFields = {
      ...fields,
      tool_call_chunks: message.tool_call_chunks,
    };
    return new AIMessageChunk(chunkFields);
  }

  return new AIMessage(fields);
}

function normalizeGenerationForLangfuse(generation: Generation): Generation {
  if (!('message' in generation)) {
    return generation;
  }

  const message = (generation as ChatGeneration).message;
  if (!(message instanceof AIMessage || message instanceof AIMessageChunk)) {
    return generation;
  }

  const usageMetadata = getLangfuseBedrockUsage(message);
  const content = sanitizeMessageContent(message.content);
  if (
    (usageMetadata == null || usageMetadata === message.usage_metadata) &&
    content === message.content
  ) {
    return generation;
  }

  const chatGeneration: ChatGeneration = {
    ...(generation as ChatGeneration),
    message: cloneMessageForLangfuse(message, usageMetadata),
  };
  return chatGeneration;
}

function normalizeBedrockUsageForLangfuse(output: LLMResult): LLMResult {
  if (output.generations.length === 0) {
    return output;
  }

  const listIndex = output.generations.length - 1;
  const generationList = output.generations[listIndex];
  if (generationList.length === 0) {
    return output;
  }

  const generationIndex = generationList.length - 1;
  const generation = generationList[generationIndex];
  const normalized = normalizeGenerationForLangfuse(generation);
  if (normalized === generation) {
    return output;
  }

  const generations = [...output.generations];
  generations[listIndex] = [...generationList];
  generations[listIndex][generationIndex] = normalized;
  return { ...output, generations };
}

const LANGGRAPH_NODE_METADATA_KEY = 'langgraph_node';
/** Explicit agent identity in invoke metadata. Every identity-stamping
 *  component (graph model path, ToolNode, summarization node) overwrites the
 *  canonical `agentId` at its own invoke, so spread order guarantees the
 *  closest stamper wins — key priority alone could not (an inherited key of
 *  either casing can name the wrong agent). `agent_id` remains a fallback
 *  for third-party graphs that only stamp the snake-case form. */
const AGENT_ID_METADATA_KEYS = ['agentId', 'agent_id'];
const LANGGRAPH_NODE_AGENT_PREFIXES = ['agent=', 'tools=', 'summarize='];

/** The LangGraph node a callback executes under, from its inherited
 *  `langgraph_node` run metadata. `undefined` when the callback carries no
 *  node identity. */
function getCallbackNode(
  metadata?: Record<string, unknown>
): string | undefined {
  const node = metadata?.[LANGGRAPH_NODE_METADATA_KEY];
  return typeof node === 'string' && node !== '' ? node : undefined;
}

/** Whether a callback's node identifies the given agent. The outer workflow
 *  node carries the agent id VERBATIM — including ids that themselves begin
 *  with an internal prefix (an agent literally named `agent=research`) — so
 *  an exact match is checked before decoding the inner subgraph prefixes
 *  (`agent=` / `tools=` / `summarize=`). */
function callbackNodeMatchesAgent(node: string, agentId: string): boolean {
  if (node === agentId) {
    return true;
  }
  for (const prefix of LANGGRAPH_NODE_AGENT_PREFIXES) {
    if (node.startsWith(prefix) && node.slice(prefix.length) === agentId) {
      return true;
    }
  }
  return false;
}

/**
 * Hosts often execute agent code inside their own OpenTelemetry spans (HTTP
 * server auto-instrumentation on the global provider). Root observations must
 * not inherit that ambient identity: the foreign parent is never exported to
 * Langfuse, which orphans the trace root (root/trace input-output shaping is
 * skipped because the span no longer looks like a root), collapses concurrent
 * runs inside one request context — an agent run and the previous turn's
 * title run — into a single merged trace with racing names and unioned tags,
 * and bypasses the seeded deterministic trace id generator. Only a
 * Langfuse-managed span bound to the same export destination as the starting
 * run is a safe parent — that is the sanctioned way for hosts to group runs
 * under their own Langfuse observations; a managed span from a different
 * destination (another tenant's project) would leave this run's trace
 * dangling in its own destination while inheriting the other trace's id.
 */
function detachForeignAmbientSpan(
  activeContext: Context,
  destinationKey?: string
): Context {
  const activeSpan = otelTrace.getSpan(activeContext);
  if (activeSpan == null) {
    return activeContext;
  }
  const parentDestination = getLangfuseManagedSpanDestination(activeSpan);
  if (parentDestination != null && parentDestination === destinationKey) {
    return activeContext;
  }
  return otelTrace.deleteSpan(activeContext);
}

class ScopedLangfuseCallbackHandler extends CallbackHandler {
  private readonly langfuse?: t.LangfuseConfig;
  private readonly traceIdSeed?: string;
  private readonly runId?: string;
  private readonly identity: HandlerIdentity;
  private readonly toolOutputTracing?: ResolvedLangfuseToolOutputTracingConfig;
  private readonly trackedRunIds = new Set<string>();

  constructor(params?: AgentLangfuseHandlerParams) {
    const {
      langfuse,
      traceIdSeed,
      runId,
      toolOutputTracing,
      traceName,
      ...handlerParams
    } = params ?? {};
    super(handlerParams);
    this.langfuse = langfuse;
    this.traceIdSeed = traceIdSeed;
    this.runId = runId;
    this.toolOutputTracing = toolOutputTracing;
    this.identity = {
      userId: handlerParams.userId,
      sessionId: handlerParams.sessionId,
      tags: handlerParams.tags,
      metadata: handlerParams.traceMetadata,
      traceName,
    };
  }

  private getDeterministicTraceSeed(): string | undefined {
    return this.langfuse?.deterministicTraceId === true
      ? this.traceIdSeed
      : undefined;
  }

  /**
   * Mirrors the base handler's `runMap`: a start callback whose `parentRunId`
   * this handler never observed gets no explicit parent span and falls back
   * to the ambient OTEL context (`startAndRegisterOtelSpan`), so it needs the
   * same foreign-span detachment as a true root. This happens whenever a
   * handler is attached mid-graph — e.g. the per-agent handler created for a
   * detached subagent's model invocations, whose surrounding graph runs were
   * never traced.
   */
  private startsDetachedRun(runId: string, parentRunId?: string): boolean {
    const detached =
      parentRunId == null || !this.trackedRunIds.has(parentRunId);
    this.trackedRunIds.add(runId);
    return detached;
  }

  /**
   * Whether the ambient runtime scope belongs to a different run — or, for
   * per-agent overlay scopes, to a different concurrently executing agent of
   * the same run than the one this callback reports via its inherited
   * `langgraph_node` metadata. Unstamped scopes on unstamped handlers are
   * never foreign (host-managed handler semantics).
   */
  private isForeignScope(
    scopeRunId: string | undefined,
    scopeAgentId: string | undefined,
    callbackMetadata?: Record<string, unknown>
  ): boolean {
    if (this.runId == null || scopeRunId == null) {
      return false;
    }
    if (scopeRunId !== this.runId) {
      return true;
    }
    if (scopeAgentId == null) {
      return false;
    }
    // Explicit agent identity (stamped into invoke metadata by the graph's
    // model path and ToolNode) is unambiguous; node names are a fallback —
    // an agent literally named `agent=research` makes its outer node
    // indistinguishable from agent `research`'s inner model node.
    for (const key of AGENT_ID_METADATA_KEYS) {
      const explicitAgentId = callbackMetadata?.[key];
      if (typeof explicitAgentId === 'string' && explicitAgentId !== '') {
        return explicitAgentId !== scopeAgentId;
      }
    }
    const callbackNode = getCallbackNode(callbackMetadata);
    return (
      callbackNode != null &&
      !callbackNodeMatchesAgent(callbackNode, scopeAgentId)
    );
  }

  /**
   * LangChain executes non-awaited callbacks on a process-wide background
   * queue (`consumeCallback`), so this callback may be running inside a
   * DIFFERENT concurrent run's async context. The ambient runtime scope is
   * therefore only adopted when it belongs to this handler's run (and, for
   * agent sub-scopes, to this callback's agent): scopes are stamped at their
   * call sites, and a foreign stamp means the scope's config would route
   * spans to the wrong destination, its seed would collapse this run's spans
   * into the foreign trace, and its tool-output policy could leak output the
   * foreign run permits but this run redacts — so config, seed, AND redaction
   * policy all fall back to this handler's own run. Unstamped scopes on
   * unstamped handlers keep scope-first semantics (agent overlays and
   * per-path seeds like the title/label scopes, host-managed handlers).
   *
   * Detached runs (roots, or starts whose parent this handler never tracked)
   * take their span parent from the ambient OTEL context, so drop any
   * foreign ambient span first — a run launched from a host's instrumented
   * request context must start its own trace, not join an unexported
   * foreign one.
   */
  private withRuntimeContext<T>(
    action: () => T,
    isDetachedRun = false,
    callbackMetadata?: Record<string, unknown>
  ): T {
    const currentContext = otelContext.active();
    const scopeRunId = resolveLangfuseScopeRunId(currentContext);
    const scopeAgentId = resolveLangfuseScopeAgentId(currentContext);
    if (this.isForeignScope(scopeRunId, scopeAgentId, callbackMetadata)) {
      return this.withForeignScopeRejected(action);
    }
    const langfuse =
      resolveLangfuseConfigForSpan(currentContext) ?? this.langfuse;
    const activeContext = isDetachedRun
      ? detachForeignAmbientSpan(
        currentContext,
        resolveLangfuseDestinationKey(langfuse)
      )
      : currentContext;
    const scoped = (): T =>
      withLangfuseRuntimeScope(
        {
          langfuse,
          traceIdSeed:
            resolveTraceIdSeedForSpan(activeContext) ??
            this.getDeterministicTraceSeed(),
          runId: scopeRunId ?? this.runId,
        },
        action
      );
    return activeContext === currentContext
      ? scoped()
      : otelContext.with(activeContext, scoped);
  }

  /**
   * A foreign concurrent run's context must be replaced wholesale, not
   * merged: this library's scope keys (config, seed, tool-output policy,
   * identity stamps) via the replace-mode runtime scope, `@langfuse/tracing`'s
   * propagated trace attributes (userId, sessionId, tags, metadata, …) by
   * deleting their context keys and re-propagating this handler's own
   * identity, and the foreign active span — removed both so detached runs
   * root their own trace and so `propagateAttributes` cannot stamp this
   * run's identity onto the foreign run's still-recording span.
   */
  private withForeignScopeRejected<T>(action: () => T): T {
    let cleanContext = otelTrace.deleteSpan(otelContext.active());
    for (const key of Object.values(LangfuseOtelContextKeys)) {
      cleanContext = cleanContext.deleteValue(key);
    }
    const scoped = (): T =>
      withLangfuseRuntimeScope(
        {
          langfuse: this.langfuse,
          traceIdSeed: this.getDeterministicTraceSeed(),
          runId: this.runId,
          toolOutputTracing:
            this.toolOutputTracing ??
            resolveToolOutputTracingConfig(this.langfuse),
        },
        action,
        { replace: true }
      );
    const { userId, sessionId, tags, metadata, traceName } = this.identity;
    const hasIdentity =
      userId != null ||
      sessionId != null ||
      metadata != null ||
      traceName != null ||
      (tags?.length ?? 0) > 0;
    return otelContext.with(cleanContext, () =>
      hasIdentity
        ? propagateAttributes(
          { userId, sessionId, tags, metadata, traceName },
          scoped
        )
        : scoped()
    );
  }

  // LangChain may invoke callback handlers outside the caller's OTEL context.
  // Re-enter tenant scope only for callbacks that start Langfuse observations;
  // end/error/token callbacks use spans already bound to a processor at start.
  override handleChainStart(
    ...args: Parameters<CallbackHandler['handleChainStart']>
  ): ReturnType<CallbackHandler['handleChainStart']> {
    return this.withRuntimeContext(
      () => super.handleChainStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }

  override handleChainError(
    ...args: Parameters<CallbackHandler['handleChainError']>
  ): ReturnType<CallbackHandler['handleChainError']> {
    const [error, runId, parentRunId] = args;
    if (error != null && parentRunId != null && isGraphInterrupt(error)) {
      return super.handleChainEnd(
        GRAPH_INTERRUPT_CONTROL_FLOW,
        runId,
        parentRunId
      );
    }
    if (error != null && parentRunId != null && isParentCommand(error)) {
      return super.handleChainEnd(
        { controlFlow: 'ParentCommand' },
        runId,
        parentRunId
      );
    }
    return super.handleChainError(...args);
  }

  override handleAgentAction(
    ...args: Parameters<CallbackHandler['handleAgentAction']>
  ): ReturnType<CallbackHandler['handleAgentAction']> {
    return this.withRuntimeContext(
      () => super.handleAgentAction(...args),
      this.startsDetachedRun(args[1], args[2])
    );
  }

  override handleGenerationStart(
    ...args: Parameters<CallbackHandler['handleGenerationStart']>
  ): ReturnType<CallbackHandler['handleGenerationStart']> {
    return this.withRuntimeContext(
      () => super.handleGenerationStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleChatModelStart(
    ...args: Parameters<CallbackHandler['handleChatModelStart']>
  ): ReturnType<CallbackHandler['handleChatModelStart']> {
    const sanitizedArgs = [...args] as Parameters<
      CallbackHandler['handleChatModelStart']
    >;
    sanitizedArgs[1] = sanitizeChatMessagesForLangfuse(args[1]);
    return this.withRuntimeContext(
      () => super.handleChatModelStart(...sanitizedArgs),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleLLMStart(
    ...args: Parameters<CallbackHandler['handleLLMStart']>
  ): ReturnType<CallbackHandler['handleLLMStart']> {
    return this.withRuntimeContext(
      () => super.handleLLMStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[6]
    );
  }

  override handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string
  ): Promise<void> {
    return super.handleLLMEnd(
      normalizeBedrockUsageForLangfuse(output),
      runId,
      parentRunId
    );
  }

  override handleToolStart(
    ...args: Parameters<CallbackHandler['handleToolStart']>
  ): ReturnType<CallbackHandler['handleToolStart']> {
    return this.withRuntimeContext(
      () => super.handleToolStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }

  override handleToolError(
    ...args: Parameters<CallbackHandler['handleToolError']>
  ): ReturnType<CallbackHandler['handleToolError']> {
    const [error, runId, parentRunId] = args;
    if (error != null && parentRunId != null && isGraphInterrupt(error)) {
      return super.handleToolEnd(
        GRAPH_INTERRUPT_TOOL_OUTPUT,
        runId,
        parentRunId
      );
    }
    return super.handleToolError(...args);
  }

  override handleRetrieverStart(
    ...args: Parameters<CallbackHandler['handleRetrieverStart']>
  ): ReturnType<CallbackHandler['handleRetrieverStart']> {
    return this.withRuntimeContext(
      () => super.handleRetrieverStart(...args),
      this.startsDetachedRun(args[2], args[3]),
      args[5]
    );
  }
}

function hasLangfuseTracingConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.toolNodeTracing != null || langfuse?.toolOutputTracing != null
  );
}

function hasLangfuseTraceAttributes(langfuse?: t.LangfuseConfig): boolean {
  return (
    Object.keys(createTraceMetadata(langfuse?.metadata ?? {})).length > 0 ||
    Object.keys(
      createLibreChatTraceAttributes(langfuse?.librechatTraceAttributes ?? {})
    ).length > 0 ||
    (mergeLangfuseTags(undefined, langfuse?.tags)?.length ?? 0) > 0
  );
}

function hasLangfuseConfigBaseUrl(langfuse?: t.LangfuseConfig): boolean {
  return isPresent(langfuse?.baseUrl);
}

export function isExplicitLangfuseConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.enabled != null ||
    isPresent(langfuse?.publicKey) ||
    isPresent(langfuse?.secretKey) ||
    isPresent(langfuse?.baseUrl) ||
    hasLangfuseTraceAttributes(langfuse) ||
    hasLangfuseTracingConfig(langfuse)
  );
}

function createTraceMetadata(
  metadata: Record<string, unknown>
): LangfuseTraceMetadata {
  const traceMetadata: LangfuseTraceMetadata = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (value == null) {
      continue;
    }
    const stringValue = typeof value === 'string' ? value : String(value);
    if (
      stringValue.trim() === '' ||
      stringValue.length > TRACE_METADATA_MAX_LENGTH
    ) {
      continue;
    }
    traceMetadata[key] = stringValue;
  }
  return traceMetadata;
}

export function createLibreChatTraceAttributes(
  attributes: LangfuseConfigTraceAttributes
): LangfuseTraceAttributes {
  const librechatTraceAttributes: LangfuseTraceAttributes = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (value == null || key.trim() === '') {
      continue;
    }
    if (typeof value === 'string') {
      if (value.trim() === '' || value.length > TRACE_METADATA_MAX_LENGTH) {
        continue;
      }
      librechatTraceAttributes[key] = value;
      continue;
    }
    librechatTraceAttributes[key] = value;
  }
  return librechatTraceAttributes;
}

export function createLangfuseTraceMetadata({
  messageId,
  parentMessageId,
  agentId,
  agentName,
}: {
  messageId?: unknown;
  parentMessageId?: unknown;
  agentId?: unknown;
  agentName?: unknown;
}): LangfuseTraceMetadata {
  return createTraceMetadata({
    messageId,
    parentMessageId,
    agentId,
    agentName,
  });
}

function mergeLangfuseTraceMetadata(
  traceMetadata?: LangfuseTraceMetadata,
  metadata?: LangfuseMetadata
): LangfuseTraceMetadata | undefined {
  const merged = createTraceMetadata({
    ...(metadata ?? {}),
    ...(traceMetadata ?? {}),
  });
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function mergeLangfuseTags(
  tags?: string[],
  configTags?: string[]
): string[] | undefined {
  const merged = [...(tags ?? []), ...(configTags ?? [])].filter(
    (tag) => tag.trim() !== ''
  );
  return merged.length > 0 ? [...new Set(merged)] : undefined;
}

export function getLangfuseTraceName(
  traceMetadata?: LangfuseTraceMetadata,
  fallback: string = 'LibreChat Agent'
): string {
  const agentName = traceMetadata?.agentName;
  return isPresent(agentName) ? `${fallback}: ${agentName}` : fallback;
}

export function shouldCreateLangfuseHandler(
  langfuse?: t.LangfuseConfig
): boolean {
  if (langfuse?.enabled === false) {
    return false;
  }
  return (
    hasLangfuseEnvConfig() ||
    hasLangfuseConfigCredentials(langfuse) ||
    (hasLangfuseConfigBaseUrl(langfuse) && hasLangfuseEnvCredentials())
  );
}

export function createLegacyLangfuseHandler(
  params: LangfuseHandlerParams
): CallbackHandler {
  return new ScopedLangfuseCallbackHandler(params);
}

export function createLangfuseHandler({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  tags,
  traceIdSeed,
  runId,
  toolOutputTracing,
  traceName,
}: AgentLangfuseHandlerParams): CallbackHandler | undefined {
  if (!shouldCreateLangfuseHandler(langfuse)) {
    return undefined;
  }
  return new ScopedLangfuseCallbackHandler({
    userId,
    sessionId,
    traceMetadata: mergeLangfuseTraceMetadata(
      traceMetadata,
      langfuse?.metadata
    ),
    tags: mergeLangfuseTags(tags, langfuse?.tags),
    langfuse,
    traceIdSeed,
    runId,
    toolOutputTracing,
    traceName,
  });
}

function createPropagateAttributeParams({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  traceName,
  tags,
}: LangfuseAttributeParams): PropagateAttributesParams {
  return {
    userId,
    sessionId,
    traceName,
    tags: mergeLangfuseTags(tags, langfuse?.tags),
    metadata: mergeLangfuseTraceMetadata(traceMetadata, langfuse?.metadata),
  };
}

export function withLangfuseAttributes<T>(
  params: LangfuseAttributeParams,
  action: () => T
): T {
  if (!shouldCreateLangfuseHandler(params.langfuse)) {
    return action();
  }
  return propagateAttributes(createPropagateAttributeParams(params), action);
}

export function hasExplicitLangfuseConfig(
  contexts: Iterable<{ langfuse?: t.LangfuseConfig }>
): boolean {
  for (const context of contexts) {
    if (isExplicitLangfuseConfig(context.langfuse)) {
      return true;
    }
  }
  return false;
}

export function isLangfuseCallbackHandler(value: unknown): boolean {
  return value instanceof CallbackHandler;
}

export async function disposeLangfuseHandler(value: unknown): Promise<void> {
  if (
    value == null ||
    parseBooleanEnv(process.env[LANGFUSE_FORCE_FLUSH_ON_DISPOSE]) !== true
  ) {
    return;
  }
  const provider = getLangfuseTracerProvider() as FlushableTracerProvider;
  await provider.forceFlush?.();
}
