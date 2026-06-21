import { context, createContextKey } from '@opentelemetry/api';
import type { Context } from '@opentelemetry/api';
import type {
  LangfuseRuntimeContext,
  ResolvedLangfuseToolOutputTracingConfig,
} from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import {
  getLangfuseRuntimeConfig,
  getLangfuseRuntimeToolOutputTracingConfig,
  getTraceIdSeed,
  hasLangfuseRuntimeContextValue,
  runWithLangfuseRuntimeContext,
} from '@/langfuseRuntimeContext';
import {
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
} from '@/langfuseConfig';

export type LangfuseRuntimeScope = LangfuseRuntimeContext;

export type ResolveLangfuseRuntimeScopeParams = {
  runLangfuse?: t.LangfuseConfig;
  langfuseOverlay?: t.LangfuseConfig;
  traceIdSeed?: string;
};

const langfuseToolOutputTracingConfigKey = createContextKey(
  'librechat.langfuse.tool-output-tracing'
);
const langfuseConfigKey = createContextKey('librechat.langfuse.config');
const langfuseTraceIdSeedKey = createContextKey(
  'librechat.langfuse.trace-id-seed'
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function hasText(value: string | undefined): value is string {
  return value != null && value.trim() !== '';
}

export function getOtelLangfuseConfig(
  activeContext: Context
): t.LangfuseConfig | undefined {
  const value = activeContext.getValue(langfuseConfigKey);
  return isRecord(value) ? (value as t.LangfuseConfig) : undefined;
}

export function getOtelTraceIdSeed(activeContext: Context): string | undefined {
  const value = activeContext.getValue(langfuseTraceIdSeedKey);
  return typeof value === 'string' && value.trim() !== '' ? value : undefined;
}

export function getOtelToolOutputTracingConfig(
  activeContext: Context
): ResolvedLangfuseToolOutputTracingConfig | undefined {
  const value = activeContext.getValue(langfuseToolOutputTracingConfigKey);
  return isRecord(value)
    ? (value as ResolvedLangfuseToolOutputTracingConfig)
    : undefined;
}

export function resolveLangfuseConfigForSpan(
  activeContext: Context
): t.LangfuseConfig | undefined {
  return getLangfuseRuntimeConfig() ?? getOtelLangfuseConfig(activeContext);
}

export function resolveTraceIdSeedForSpan(
  activeContext: Context
): string | undefined {
  return getTraceIdSeed() ?? getOtelTraceIdSeed(activeContext);
}

export function resolveToolOutputTracingConfigForSpan(
  activeContext: Context
): ResolvedLangfuseToolOutputTracingConfig | undefined {
  return (
    getLangfuseRuntimeToolOutputTracingConfig() ??
    getOtelToolOutputTracingConfig(activeContext)
  );
}

export function withLangfuseRuntimeScope<T>(
  scope: LangfuseRuntimeScope,
  action: () => T
): T {
  if (!hasLangfuseRuntimeContextValue(scope)) {
    return action();
  }

  let activeContext = context.active();
  if (scope.langfuse != null) {
    activeContext = activeContext.setValue(langfuseConfigKey, scope.langfuse);
  }
  if (scope.toolOutputTracing != null) {
    activeContext = activeContext.setValue(
      langfuseToolOutputTracingConfigKey,
      scope.toolOutputTracing
    );
  }
  if (hasText(scope.traceIdSeed)) {
    activeContext = activeContext.setValue(
      langfuseTraceIdSeedKey,
      scope.traceIdSeed
    );
  }

  // Span processors receive the OTel parent context in `onStart`, while
  // LangChain callback handlers may run outside that context and need ALS.
  // The trace id generator reads the seed from ALS or OTel context so SDK
  // callbacks that preserve only one of those contexts still keep trace/score
  // cohesion.
  return runWithLangfuseRuntimeContext(scope, () =>
    context.with(activeContext, action)
  );
}

export function resolveLangfuseRuntimeScope({
  runLangfuse,
  langfuseOverlay,
  traceIdSeed,
}: ResolveLangfuseRuntimeScopeParams): LangfuseRuntimeScope {
  const langfuse = resolveLangfuseConfig(runLangfuse, langfuseOverlay);
  const hasNoToolOutputConfig =
    runLangfuse?.toolOutputTracing == null &&
    langfuseOverlay?.toolOutputTracing == null;
  const toolOutputTracing = hasNoToolOutputConfig
    ? undefined
    : resolveToolOutputTracingConfig(runLangfuse, langfuseOverlay);
  return { langfuse, traceIdSeed, toolOutputTracing };
}
