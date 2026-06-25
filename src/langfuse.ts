import { CallbackHandler } from '@langfuse/langchain';
import { context as otelContext } from '@opentelemetry/api';
import {
  getLangfuseTracerProvider,
  propagateAttributes,
} from '@langfuse/tracing';
import type { PropagateAttributesParams } from '@langfuse/tracing';
import type * as t from '@/types';
import {
  resolveLangfuseConfigForSpan,
  resolveTraceIdSeedForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import { normalizePromptCacheUsageForLangfuse } from '@/langfuse/promptCacheUsage';
import { isPresent, parseBooleanEnv } from '@/utils/misc';

const TRACE_METADATA_MAX_LENGTH = 200;
const LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'LANGFUSE_FORCE_FLUSH_ON_DISPOSE';

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

class ScopedLangfuseCallbackHandler extends CallbackHandler {
  private readonly langfuse?: t.LangfuseConfig;
  private readonly traceIdSeed?: string;

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

  private withRuntimeContext<T>(action: () => T): T {
    const activeContext = otelContext.active();
    const langfuse =
      resolveLangfuseConfigForSpan(activeContext) ?? this.langfuse;
    const seed = this.getDeterministicTraceSeed();
    return withLangfuseRuntimeScope(
      {
        langfuse,
        traceIdSeed: resolveTraceIdSeedForSpan(activeContext) ?? seed,
      },
      action
    );
  }

  // LangChain may invoke callback handlers outside the caller's OTEL context.
  // Re-enter tenant scope only for callbacks that start Langfuse observations;
  // end/error/token callbacks use spans already bound to a processor at start.
  override handleChainStart(
    ...args: Parameters<CallbackHandler['handleChainStart']>
  ): ReturnType<CallbackHandler['handleChainStart']> {
    return this.withRuntimeContext(() => super.handleChainStart(...args));
  }

  override handleAgentAction(
    ...args: Parameters<CallbackHandler['handleAgentAction']>
  ): ReturnType<CallbackHandler['handleAgentAction']> {
    return this.withRuntimeContext(() => super.handleAgentAction(...args));
  }

  override handleGenerationStart(
    ...args: Parameters<CallbackHandler['handleGenerationStart']>
  ): ReturnType<CallbackHandler['handleGenerationStart']> {
    return this.withRuntimeContext(() => super.handleGenerationStart(...args));
  }

  override handleChatModelStart(
    ...args: Parameters<CallbackHandler['handleChatModelStart']>
  ): ReturnType<CallbackHandler['handleChatModelStart']> {
    return this.withRuntimeContext(() => super.handleChatModelStart(...args));
  }

  override handleLLMStart(
    ...args: Parameters<CallbackHandler['handleLLMStart']>
  ): ReturnType<CallbackHandler['handleLLMStart']> {
    return this.withRuntimeContext(() => super.handleLLMStart(...args));
  }

  override handleLLMEnd(
    output: Parameters<CallbackHandler['handleLLMEnd']>[0],
    runId: Parameters<CallbackHandler['handleLLMEnd']>[1],
    parentRunId?: Parameters<CallbackHandler['handleLLMEnd']>[2]
  ): ReturnType<CallbackHandler['handleLLMEnd']> {
    return super.handleLLMEnd(
      normalizePromptCacheUsageForLangfuse(output),
      runId,
      parentRunId
    );
  }

  override handleToolStart(
    ...args: Parameters<CallbackHandler['handleToolStart']>
  ): ReturnType<CallbackHandler['handleToolStart']> {
    return this.withRuntimeContext(() => super.handleToolStart(...args));
  }

  override handleRetrieverStart(
    ...args: Parameters<CallbackHandler['handleRetrieverStart']>
  ): ReturnType<CallbackHandler['handleRetrieverStart']> {
    return this.withRuntimeContext(() => super.handleRetrieverStart(...args));
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

export function hasLangfuseConfigCredentials(
  langfuse?: t.LangfuseConfig
): langfuse is t.LangfuseConfig & {
  publicKey: string;
  secretKey: string;
} {
  return (
    langfuse != null &&
    isPresent(langfuse.publicKey) &&
    isPresent(langfuse.secretKey)
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

export function hasLangfuseEnvConfig(): boolean {
  return hasLangfuseEnvCredentials();
}

export function hasLangfuseEnvCredentials(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY)
  );
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
    !parseBooleanEnv(process.env[LANGFUSE_FORCE_FLUSH_ON_DISPOSE])
  ) {
    return;
  }
  const provider = getLangfuseTracerProvider() as FlushableTracerProvider;
  await provider.forceFlush?.();
}
