import { LangfuseSpanProcessor } from '@langfuse/otel';
import { setLangfuseTracerProvider } from '@langfuse/tracing';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { hasLangfuseEnvConfig } from '@/langfuse';

let langfuseTracerProvider: BasicTracerProvider | undefined;

export function initializeLangfuseTracingFromEnv():
  | BasicTracerProvider
  | undefined {
  if (langfuseTracerProvider != null) {
    return langfuseTracerProvider;
  }
  if (!hasLangfuseEnvConfig()) {
    return undefined;
  }

  const langfuseSpanProcessor = new LangfuseSpanProcessor();
  langfuseTracerProvider = new BasicTracerProvider({
    spanProcessors: [langfuseSpanProcessor],
  });

  setLangfuseTracerProvider(langfuseTracerProvider);
  return langfuseTracerProvider;
}

initializeLangfuseTracingFromEnv();
