import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';
import { parseBooleanEnv } from '@/utils/misc';

export const LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[tool output redacted]';

function isPresent(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

export function normalizeToolName(name: string): string {
  return name.trim().toLowerCase();
}

function normalizeToolNames(names: string[] | undefined): Set<string> {
  const normalized = new Set<string>();
  for (const name of names ?? []) {
    if (isPresent(name)) {
      normalized.add(normalizeToolName(name));
    }
  }
  return normalized;
}

function parseToolNames(value: string | undefined): string[] | undefined {
  if (!isPresent(value)) {
    return undefined;
  }

  return value
    .split(',')
    .map((name) => name.trim())
    .filter((name) => name !== '');
}

function getEnvToolOutputTracingEnabled(): boolean | undefined {
  const traceToolOutputs = parseBooleanEnv(
    process.env.LANGFUSE_TRACE_TOOL_OUTPUTS
  );
  if (traceToolOutputs != null) {
    return traceToolOutputs;
  }

  const redactToolOutputs = parseBooleanEnv(
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS
  );
  if (redactToolOutputs != null) {
    return !redactToolOutputs;
  }

  return parseBooleanEnv(process.env.LANGFUSE_TOOL_OUTPUT_TRACING_ENABLED);
}

function getEnvRedactedToolNames(): string[] | undefined {
  return (
    parseToolNames(process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAMES) ??
    parseToolNames(process.env.LANGFUSE_REDACT_TOOL_NAMES)
  );
}

function getEnvRedactionText(): string | undefined {
  return isPresent(process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT)
    ? process.env.LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    : undefined;
}

function getEnvToolNameMatchMode(): 'exact' | 'partial' | undefined {
  const mode = (
    process.env.LANGFUSE_REDACT_TOOL_OUTPUT_NAME_MATCH_MODE ??
    process.env.LANGFUSE_REDACT_TOOL_NAME_MATCH_MODE
  )
    ?.trim()
    .toLowerCase();
  if (mode === 'exact' || mode === 'partial') {
    return mode;
  }
  return undefined;
}

function hasEnvToolOutputTracingConfig(): boolean {
  return (
    getEnvToolOutputTracingEnabled() != null ||
    getEnvRedactedToolNames() != null ||
    getEnvRedactionText() != null ||
    getEnvToolNameMatchMode() != null
  );
}

function resolveToolOutputTracingEnabled(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): boolean {
  return (
    agentConfig?.enabled ??
    runConfig?.enabled ??
    getEnvToolOutputTracingEnabled() ??
    true
  );
}

function resolveRedactedToolNames(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): Set<string> {
  return normalizeToolNames([
    ...(getEnvRedactedToolNames() ?? []),
    ...(runConfig?.redactedToolNames ?? []),
    ...(agentConfig?.redactedToolNames ?? []),
  ]);
}

function resolveToolNameMatchMode(
  runConfig?: t.LangfuseToolOutputTracingConfig,
  agentConfig?: t.LangfuseToolOutputTracingConfig
): 'exact' | 'partial' {
  const modes = [
    getEnvToolNameMatchMode(),
    runConfig?.redactedToolNameMatchMode,
    agentConfig?.redactedToolNameMatchMode,
  ];
  return modes.includes('partial') ? 'partial' : 'exact';
}

export function hasToolOutputTracingConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): boolean {
  return (
    runLangfuse?.toolOutputTracing != null ||
    agentLangfuse?.toolOutputTracing != null ||
    hasEnvToolOutputTracingConfig()
  );
}

export function resolveToolOutputTracingConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): ResolvedLangfuseToolOutputTracingConfig {
  const runConfig = runLangfuse?.toolOutputTracing;
  const agentConfig = agentLangfuse?.toolOutputTracing;

  return {
    enabled: resolveToolOutputTracingEnabled(runConfig, agentConfig),
    redactedToolNames: resolveRedactedToolNames(runConfig, agentConfig),
    redactedToolNameMatchMode: resolveToolNameMatchMode(runConfig, agentConfig),
    redactionText:
      agentConfig?.redactionText ??
      runConfig?.redactionText ??
      getEnvRedactionText() ??
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
  };
}

export function resolveLangfuseConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): t.LangfuseConfig | undefined {
  if (runLangfuse == null) {
    return agentLangfuse;
  }
  if (agentLangfuse == null) {
    return runLangfuse;
  }

  const toolNodeTracing =
    runLangfuse.toolNodeTracing != null || agentLangfuse.toolNodeTracing != null
      ? {
        ...runLangfuse.toolNodeTracing,
        ...agentLangfuse.toolNodeTracing,
      }
      : undefined;
  const toolOutputTracing =
    runLangfuse.toolOutputTracing != null ||
    agentLangfuse.toolOutputTracing != null
      ? {
        ...runLangfuse.toolOutputTracing,
        ...agentLangfuse.toolOutputTracing,
      }
      : undefined;
  const metadata =
    runLangfuse.metadata != null || agentLangfuse.metadata != null
      ? {
        ...runLangfuse.metadata,
        ...agentLangfuse.metadata,
      }
      : undefined;
  const librechatTraceAttributes =
    runLangfuse.librechatTraceAttributes != null ||
    agentLangfuse.librechatTraceAttributes != null
      ? {
        ...runLangfuse.librechatTraceAttributes,
        ...agentLangfuse.librechatTraceAttributes,
      }
      : undefined;
  const tags =
    runLangfuse.tags != null || agentLangfuse.tags != null
      ? [
        ...new Set([
          ...(runLangfuse.tags ?? []),
          ...(agentLangfuse.tags ?? []),
        ]),
      ]
      : undefined;

  return {
    ...runLangfuse,
    ...agentLangfuse,
    ...(metadata != null ? { metadata } : {}),
    ...(librechatTraceAttributes != null ? { librechatTraceAttributes } : {}),
    ...(tags != null ? { tags } : {}),
    ...(toolNodeTracing != null ? { toolNodeTracing } : {}),
    ...(toolOutputTracing != null ? { toolOutputTracing } : {}),
  };
}
