import { getLangfuseTracerProvider } from '@langfuse/tracing';
import { CallbackHandler } from '@langfuse/langchain';
import type * as t from '@/types';
import { isPresent } from '@/utils/misc';

const TRACE_METADATA_MAX_LENGTH = 200;
const LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'LANGFUSE_FORCE_FLUSH_ON_DISPOSE';

export type LangfuseTraceMetadata = Record<string, string>;

type LangfuseHandlerParams = {
  userId?: string;
  sessionId?: string;
  traceMetadata?: LangfuseTraceMetadata;
  tags?: string[];
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
};

type FlushableTracerProvider = {
  forceFlush?: () => Promise<void> | void;
};

function parseBooleanEnv(value?: string): boolean {
  if (value == null) {
    return false;
  }
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

function hasLangfuseTracingConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.toolNodeTracing != null || langfuse?.toolOutputTracing != null
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
  return new CallbackHandler(params);
}

export function createLangfuseHandler({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  tags,
}: AgentLangfuseHandlerParams): CallbackHandler | undefined {
  if (!shouldCreateLangfuseHandler(langfuse)) {
    return undefined;
  }
  return new CallbackHandler({
    userId,
    sessionId,
    traceMetadata,
    tags,
  });
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
