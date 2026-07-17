import { createHash, randomBytes } from 'node:crypto';
import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { context, ROOT_CONTEXT, createContextKey } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import type {
  IdGenerator,
  ReadableSpan,
  Span,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  createLibreChatTraceAttributes,
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
} from '@/langfuse';
import {
  resolveLangfuseConfigForSpan,
  resolveTraceIdSeedForSpan,
} from '@/langfuseRuntimeScope';
import { createLangfuseSpanProcessor } from '@/langfuseToolOutputTracing';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import { isPresent } from '@/utils/misc';

/**
 * Per-run seed for deterministic Langfuse trace ids. When a run opts in
 * (`LangfuseConfig.deterministicTraceId`), it executes its stream inside
 * `runWithTraceIdSeed(runId, ...)` from `./langfuseRuntimeContext`, and the
 * IdGenerator below derives the root trace id from that seed instead of a
 * random one. This lets external systems (e.g. a host app recording user
 * feedback after the fact) attach scores or observations to the trace by
 * regenerating the same id from the run/message id; no trace lookup required.
 * With no active seed it falls back to random ids, so default behavior is
 * unchanged.
 */
class SeededTraceIdGenerator implements IdGenerator {
  generateTraceId(): string {
    const seed = resolveTraceIdSeedForSpan(context.active());
    return isPresent(seed)
      ? traceIdFromSeed(seed)
      : randomBytes(16).toString('hex');
  }

  generateSpanId(): string {
    return randomBytes(8).toString('hex');
  }
}

let langfuseTracerProvider: BasicTracerProvider | undefined;
let langfuseRoutingSpanProcessor: RoutingLangfuseSpanProcessor | undefined;
const contextManagerProbeKey = createContextKey(
  'langfuse-context-manager-probe'
);

function hasActiveContextManager(): boolean {
  return context.with(
    ROOT_CONTEXT.setValue(contextManagerProbeKey, true),
    () => context.active().getValue(contextManagerProbeKey) === true
  );
}

export function ensureOpenTelemetryContextManager(): void {
  if (hasActiveContextManager()) {
    return;
  }

  const contextManager = new AsyncLocalStorageContextManager();
  contextManager.enable();
  if (!context.setGlobalContextManager(contextManager)) {
    contextManager.disable();
  }
}

function resolveLangfuseEnvironment(
  langfuse?: t.LangfuseConfig
): string | undefined {
  return (
    langfuse?.environment ??
    process.env.LANGFUSE_TRACING_ENVIRONMENT ??
    process.env.NODE_ENV
  );
}

function getLangfuseSpanProcessorParams(
  langfuse?: t.LangfuseConfig
): LangfuseSpanProcessorParams | undefined {
  if (langfuse?.enabled === false) {
    return undefined;
  }
  const environment = resolveLangfuseEnvironment(langfuse);
  if (hasLangfuseConfigCredentials(langfuse)) {
    return {
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...(isPresent(langfuse.baseUrl) ? { baseUrl: langfuse.baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  if (hasLangfuseEnvConfig()) {
    const baseUrl =
      langfuse?.baseUrl ??
      process.env.LANGFUSE_BASE_URL ??
      process.env.LANGFUSE_BASEURL;
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      ...(isPresent(baseUrl) ? { baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  if (isPresent(langfuse?.baseUrl) && hasLangfuseEnvCredentials()) {
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      baseUrl: langfuse.baseUrl,
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  return undefined;
}

function hashCacheKeyValue(value: string | undefined): string | undefined {
  return isPresent(value)
    ? createHash('sha256').update(value, 'utf8').digest('hex')
    : undefined;
}

function getLangfuseTracerProviderKey(
  params: LangfuseSpanProcessorParams,
  langfuse?: t.LangfuseConfig
): string {
  return JSON.stringify({
    publicKey: params.publicKey,
    secretKeyHash: hashCacheKeyValue(params.secretKey),
    baseUrl: params.baseUrl,
    environment: params.environment,
    toolOutputTracing: langfuse?.toolOutputTracing,
  });
}

class RoutingLangfuseSpanProcessor implements SpanProcessor {
  // Processors live for the process lifetime. LibreChat tenant Langfuse
  // destinations are expected to be a bounded admin-managed set, and shutdown
  // drains every cached processor when the provider is disposed.
  private readonly processors = new Map<string, SpanProcessor>();
  private readonly spanProcessors = new WeakMap<object, SpanProcessor>();

  ensureProcessor(langfuse?: t.LangfuseConfig): SpanProcessor | undefined {
    const params = getLangfuseSpanProcessorParams(langfuse);
    if (params == null) {
      return undefined;
    }

    const processorKey = getLangfuseTracerProviderKey(params, langfuse);
    const existing = this.processors.get(processorKey);
    if (existing != null) {
      return existing;
    }

    const processor = createLangfuseSpanProcessor(params, langfuse);
    this.processors.set(processorKey, processor);
    return processor;
  }

  onStart(span: Span, parentContext: Context): void {
    const langfuse = resolveLangfuseConfigForSpan(parentContext);
    const processor = this.ensureProcessor(langfuse);
    if (processor == null) {
      return;
    }

    const librechatTraceAttributes = createLibreChatTraceAttributes(
      langfuse?.librechatTraceAttributes ?? {}
    );
    if (Object.keys(librechatTraceAttributes).length > 0) {
      span.setAttributes(librechatTraceAttributes);
    }

    this.spanProcessors.set(span, processor);
    processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    this.spanProcessors.get(span)?.onEnd(span);
  }

  async forceFlush(): Promise<void> {
    await Promise.all(
      Array.from(this.processors.values(), (processor) =>
        processor.forceFlush()
      )
    );
  }

  async shutdown(): Promise<void> {
    await Promise.all(
      Array.from(this.processors.values(), (processor) => processor.shutdown())
    );
  }
}

export function initializeLangfuseTracing(
  langfuse?: t.LangfuseConfig
): BasicTracerProvider | undefined {
  const params = getLangfuseSpanProcessorParams(langfuse);
  if (params == null) {
    return undefined;
  }

  if (langfuseTracerProvider != null) {
    langfuseRoutingSpanProcessor?.ensureProcessor(langfuse);
    return langfuseTracerProvider;
  }

  ensureOpenTelemetryContextManager();
  langfuseRoutingSpanProcessor = new RoutingLangfuseSpanProcessor();
  langfuseRoutingSpanProcessor.ensureProcessor(langfuse);
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseRoutingSpanProcessor],
    idGenerator: new SeededTraceIdGenerator(),
  });

  setLangfuseTracerProvider(langfuseTracerProvider);
  return langfuseTracerProvider;
}

export function initializeLangfuseTracingFromEnv():
  | BasicTracerProvider
  | undefined {
  return initializeLangfuseTracing();
}

initializeLangfuseTracingFromEnv();
