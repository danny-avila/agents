import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { context, ROOT_CONTEXT, createContextKey } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
} from '@/langfuse';
import { createLangfuseSpanProcessor } from '@/langfuseToolOutputTracing';
import { isPresent } from '@/utils/misc';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type * as t from '@/types';

let langfuseTracerProvider: BasicTracerProvider | undefined;
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
  const baseUrl = isPresent(langfuse?.baseUrl)
    ? { baseUrl: langfuse.baseUrl }
    : {};
  if (hasLangfuseConfigCredentials(langfuse)) {
    return {
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...baseUrl,
    };
  }
  if (
    hasLangfuseEnvConfig() ||
    (isPresent(langfuse?.baseUrl) && hasLangfuseEnvCredentials())
  ) {
    return baseUrl;
  }
  return undefined;
}

export function initializeLangfuseTracing(
  langfuse?: t.LangfuseConfig
): BasicTracerProvider | undefined {
  if (langfuseTracerProvider != null) {
    return langfuseTracerProvider;
  }

  const params = getLangfuseSpanProcessorParams(langfuse);
  if (params == null) {
    return undefined;
  }

  ensureOpenTelemetryContextManager();
  const langfuseSpanProcessor = createLangfuseSpanProcessor(
    Object.keys(params).length > 0 ? params : undefined
  );
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseSpanProcessor],
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
