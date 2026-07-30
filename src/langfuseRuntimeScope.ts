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
  getLangfuseScopeAgentId,
  getLangfuseScopeRunId,
  getTraceIdSeed,
  hasLangfuseRuntimeContextValue,
  replaceLangfuseRuntimeContext,
  runWithLangfuseRuntimeContext,
} from '@/langfuseRuntimeContext';
import {
  hasToolOutputTracingConfig,
  resolveLangfuseConfig,
  resolveToolOutputTracingConfig,
} from '@/langfuseConfig';

export type LangfuseRuntimeScope = LangfuseRuntimeContext;

export type ResolveLangfuseRuntimeScopeParams = {
  runLangfuse?: t.LangfuseConfig;
  langfuseOverlay?: t.LangfuseConfig;
  traceIdSeed?: string;
  runId?: string;
  agentId?: string;
};

export type LangfuseRuntimeScopeOptions = {
  /** Replace the surrounding scope entirely instead of merging with it:
   *  fields absent from `scope` are CLEARED rather than inherited. Required
   *  when rejecting a foreign concurrent run's scope, whose explicit values
   *  must not survive merge inheritance. */
  replace?: boolean;
};

const langfuseToolOutputTracingConfigKey = createContextKey(
  'librechat.langfuse.tool-output-tracing'
);
const langfuseConfigKey = createContextKey('librechat.langfuse.config');
const langfuseTraceIdSeedKey = createContextKey(
  'librechat.langfuse.trace-id-seed'
);
const langfuseScopeRunIdKey = createContextKey('librechat.langfuse.run-id');
const langfuseScopeAgentIdKey = createContextKey('librechat.langfuse.agent-id');

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

function getOtelScopeRunId(activeContext: Context): string | undefined {
  const value = activeContext.getValue(langfuseScopeRunIdKey);
  return typeof value === 'string' && value.trim() !== '' ? value : undefined;
}

function getOtelScopeAgentId(activeContext: Context): string | undefined {
  const value = activeContext.getValue(langfuseScopeAgentIdKey);
  return typeof value === 'string' && value.trim() !== '' ? value : undefined;
}

/** The run identity the active scope belongs to (see
 *  `LangfuseRuntimeContext.runId`), or `undefined` for unstamped scopes. */
export function resolveLangfuseScopeRunId(
  activeContext: Context
): string | undefined {
  return getLangfuseScopeRunId() ?? getOtelScopeRunId(activeContext);
}

/** The agent identity of a per-agent overlay scope (see
 *  `LangfuseRuntimeContext.agentId`), or `undefined` for run-level scopes. */
export function resolveLangfuseScopeAgentId(
  activeContext: Context
): string | undefined {
  return getLangfuseScopeAgentId() ?? getOtelScopeAgentId(activeContext);
}

export function resolveToolOutputTracingConfigForSpan(
  activeContext: Context
): ResolvedLangfuseToolOutputTracingConfig | undefined {
  return (
    getLangfuseRuntimeToolOutputTracingConfig() ??
    getOtelToolOutputTracingConfig(activeContext)
  );
}

function setOrClearContextValue(
  activeContext: Context,
  key: Parameters<Context['setValue']>[0],
  value: unknown,
  replace: boolean
): Context {
  if (value != null) {
    return activeContext.setValue(key, value);
  }
  return replace ? activeContext.deleteValue(key) : activeContext;
}

export function withLangfuseRuntimeScope<T>(
  scope: LangfuseRuntimeScope,
  action: () => T,
  options?: LangfuseRuntimeScopeOptions
): T {
  const replace = options?.replace === true;
  if (!replace && !hasLangfuseRuntimeContextValue(scope)) {
    return action();
  }

  let activeContext = context.active();
  activeContext = setOrClearContextValue(
    activeContext,
    langfuseConfigKey,
    scope.langfuse,
    replace
  );
  activeContext = setOrClearContextValue(
    activeContext,
    langfuseToolOutputTracingConfigKey,
    scope.toolOutputTracing,
    replace
  );
  activeContext = setOrClearContextValue(
    activeContext,
    langfuseTraceIdSeedKey,
    hasText(scope.traceIdSeed) ? scope.traceIdSeed : undefined,
    replace
  );
  activeContext = setOrClearContextValue(
    activeContext,
    langfuseScopeRunIdKey,
    hasText(scope.runId) ? scope.runId : undefined,
    replace
  );
  activeContext = setOrClearContextValue(
    activeContext,
    langfuseScopeAgentIdKey,
    hasText(scope.agentId) ? scope.agentId : undefined,
    replace
  );

  // Span processors receive the OTel parent context in `onStart`, while
  // LangChain callback handlers may run outside that context and need ALS.
  // The trace id generator reads the seed from ALS or OTel context so SDK
  // callbacks that preserve only one of those contexts still keep trace/score
  // cohesion.
  const runScoped = replace
    ? replaceLangfuseRuntimeContext
    : runWithLangfuseRuntimeContext;
  return runScoped(scope, () => context.with(activeContext, action));
}

export function resolveLangfuseRuntimeScope({
  runLangfuse,
  langfuseOverlay,
  traceIdSeed,
  runId,
  agentId,
}: ResolveLangfuseRuntimeScopeParams): LangfuseRuntimeScope {
  const langfuse = resolveLangfuseConfig(runLangfuse, langfuseOverlay);
  const toolOutputTracing = !hasToolOutputTracingConfig(
    runLangfuse,
    langfuseOverlay
  )
    ? undefined
    : resolveToolOutputTracingConfig(runLangfuse, langfuseOverlay);
  return { langfuse, traceIdSeed, toolOutputTracing, runId, agentId };
}
