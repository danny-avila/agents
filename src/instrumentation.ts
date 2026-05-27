import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { context, ROOT_CONTEXT, createContextKey } from '@opentelemetry/api';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import { hasLangfuseEnvConfig } from '@/langfuse';
import { createLangfuseSpanProcessor } from '@/langfuseToolOutputTracing';

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

export function initializeLangfuseTracingFromEnv():
  | BasicTracerProvider
  | undefined {
  if (langfuseTracerProvider != null) {
    return langfuseTracerProvider;
  }
  if (!hasLangfuseEnvConfig()) {
    return undefined;
  }

  ensureOpenTelemetryContextManager();
  const langfuseSpanProcessor = createLangfuseSpanProcessor();
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseSpanProcessor],
  });

  setLangfuseTracerProvider(langfuseTracerProvider);
  return langfuseTracerProvider;
}

initializeLangfuseTracingFromEnv();
