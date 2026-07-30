import { CallbackHandler } from '@langfuse/langchain';
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
  UsageMetadata,
} from '@langchain/core/messages';
import type {
  ChatGeneration,
  Generation,
  LLMResult,
} from '@langchain/core/outputs';
import type { PropagateAttributesParams } from '@langfuse/tracing';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  resolveLangfuseConfigForSpan,
  resolveTraceIdSeedForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
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
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
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

function cloneMessageWithUsage(
  message: AIMessage | AIMessageChunk,
  usageMetadata: UsageMetadata
): AIMessage | AIMessageChunk {
  const fields: AIMessageFields = {
    content: message.content,
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
  if (usageMetadata == null || usageMetadata === message.usage_metadata) {
    return generation;
  }

  const chatGeneration: ChatGeneration = {
    ...(generation as ChatGeneration),
    message: cloneMessageWithUsage(message, usageMetadata),
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
  private readonly trackedRunIds = new Set<string>();

  constructor(params?: AgentLangfuseHandlerParams) {
    const { langfuse, traceIdSeed, ...handlerParams } = params ?? {};
    super(handlerParams);
    this.langfuse = langfuse;
    this.traceIdSeed = traceIdSeed;
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
   * Detached runs (roots, or starts whose parent this handler never tracked)
   * take their span parent from the ambient OTEL context, so drop any foreign
   * ambient span first — a run launched from a host's instrumented request
   * context must start its own trace, not join an unexported foreign one.
   * Trace identity otherwise stays scope-first: an active runtime scope
   * (agent overlay, or a per-path seed like the title/label scopes) wins over
   * this handler's own configuration.
   */
  private withRuntimeContext<T>(action: () => T, isDetachedRun = false): T {
    const currentContext = otelContext.active();
    const langfuse =
      resolveLangfuseConfigForSpan(currentContext) ?? this.langfuse;
    const activeContext = isDetachedRun
      ? detachForeignAmbientSpan(
        currentContext,
        resolveLangfuseDestinationKey(langfuse)
      )
      : currentContext;
    const seed = this.getDeterministicTraceSeed();
    const scoped = (): T =>
      withLangfuseRuntimeScope(
        {
          langfuse,
          traceIdSeed: resolveTraceIdSeedForSpan(activeContext) ?? seed,
        },
        action
      );
    return activeContext === currentContext
      ? scoped()
      : otelContext.with(activeContext, scoped);
  }

  // LangChain may invoke callback handlers outside the caller's OTEL context.
  // Re-enter tenant scope only for callbacks that start Langfuse observations;
  // end/error/token callbacks use spans already bound to a processor at start.
  override handleChainStart(
    ...args: Parameters<CallbackHandler['handleChainStart']>
  ): ReturnType<CallbackHandler['handleChainStart']> {
    return this.withRuntimeContext(
      () => super.handleChainStart(...args),
      this.startsDetachedRun(args[2], args[3])
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
      this.startsDetachedRun(args[2], args[3])
    );
  }

  override handleChatModelStart(
    ...args: Parameters<CallbackHandler['handleChatModelStart']>
  ): ReturnType<CallbackHandler['handleChatModelStart']> {
    return this.withRuntimeContext(
      () => super.handleChatModelStart(...args),
      this.startsDetachedRun(args[2], args[3])
    );
  }

  override handleLLMStart(
    ...args: Parameters<CallbackHandler['handleLLMStart']>
  ): ReturnType<CallbackHandler['handleLLMStart']> {
    return this.withRuntimeContext(
      () => super.handleLLMStart(...args),
      this.startsDetachedRun(args[2], args[3])
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
      this.startsDetachedRun(args[2], args[3])
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
      this.startsDetachedRun(args[2], args[3])
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
