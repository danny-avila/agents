import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { context, ROOT_CONTEXT, createContextKey } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
} from '@/langfuse';
import {
  createLangfuseSpanProcessor,
  getContextLangfuseConfig,
} from '@/langfuseToolOutputTracing';
import { isPresent } from '@/utils/misc';
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';

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

function getLangfuseSpanProcessorParams(
  langfuse?: t.LangfuseConfig
): LangfuseSpanProcessorParams | undefined {
  if (langfuse?.enabled === false) {
    return undefined;
  }
  if (hasLangfuseConfigCredentials(langfuse)) {
    return {
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...(isPresent(langfuse.baseUrl) ? { baseUrl: langfuse.baseUrl } : {}),
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
    };
  }
  if (isPresent(langfuse?.baseUrl) && hasLangfuseEnvCredentials()) {
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      baseUrl: langfuse.baseUrl,
    };
  }
  return undefined;
}

function getLangfuseTracerProviderKey(
  params: LangfuseSpanProcessorParams,
  langfuse?: t.LangfuseConfig
): string {
  return JSON.stringify({
    publicKey: params.publicKey,
    secretKey: params.secretKey,
    baseUrl: params.baseUrl,
    environment: params.environment,
    toolOutputTracing: langfuse?.toolOutputTracing,
  });
}

class RoutingLangfuseSpanProcessor implements SpanProcessor {
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
    const processor = this.ensureProcessor(
      getContextLangfuseConfig(parentContext)
    );
    if (processor == null) {
      return;
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
