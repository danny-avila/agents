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
let langfuseTracerProviderKey: string | undefined;
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
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      baseUrl: langfuse?.baseUrl ??
        process.env.LANGFUSE_BASE_URL ??
        process.env.LANGFUSE_BASEURL,
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

export function initializeLangfuseTracing(
  langfuse?: t.LangfuseConfig
): BasicTracerProvider | undefined {
  const params = getLangfuseSpanProcessorParams(langfuse);
  if (params == null) {
    return undefined;
  }

  const providerKey = getLangfuseTracerProviderKey(params, langfuse);
  if (
    langfuseTracerProvider != null &&
    langfuseTracerProviderKey === providerKey
  ) {
    return langfuseTracerProvider;
  }

  ensureOpenTelemetryContextManager();
  const langfuseSpanProcessor = createLangfuseSpanProcessor(
    params,
    langfuse
  );
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseSpanProcessor],
  });
  langfuseTracerProviderKey = providerKey;

  setLangfuseTracerProvider(langfuseTracerProvider);
  return langfuseTracerProvider;
}

export function initializeLangfuseTracingFromEnv():
  | BasicTracerProvider
  | undefined {
  return initializeLangfuseTracing();
}

initializeLangfuseTracingFromEnv();
