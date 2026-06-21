import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type * as t from '@/types';

export const LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[tool output redacted]';

function isPresent(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

function parseBoolean(value: string | undefined): boolean | undefined {
  if (value == null) {
    return undefined;
  }

  const normalized = value.trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) {
    return true;
  }
  if (['0', 'false', 'no', 'off'].includes(normalized)) {
    return false;
  }

  return undefined;
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
  const traceToolOutputs = parseBoolean(
    process.env.LANGFUSE_TRACE_TOOL_OUTPUTS
  );
  if (traceToolOutputs != null) {
    return traceToolOutputs;
  }

  const redactToolOutputs = parseBoolean(
    process.env.LANGFUSE_REDACT_TOOL_OUTPUTS
  );
  if (redactToolOutputs != null) {
    return !redactToolOutputs;
  }

  return parseBoolean(process.env.LANGFUSE_TOOL_OUTPUT_TRACING_ENABLED);
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

export function resolveToolOutputTracingConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): ResolvedLangfuseToolOutputTracingConfig {
  const runConfig = runLangfuse?.toolOutputTracing;
  const agentConfig = agentLangfuse?.toolOutputTracing;

  return {
    enabled:
      agentConfig?.enabled ??
      runConfig?.enabled ??
      getEnvToolOutputTracingEnabled() ??
      true,
    redactedToolNames: normalizeToolNames(
      agentConfig?.redactedToolNames ??
        runConfig?.redactedToolNames ??
        getEnvRedactedToolNames()
    ),
    redactedToolNameMatchMode:
      agentConfig?.redactedToolNameMatchMode ??
      runConfig?.redactedToolNameMatchMode ??
      getEnvToolNameMatchMode() ??
      'exact',
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
    ...(tags != null ? { tags } : {}),
    ...(toolNodeTracing != null ? { toolNodeTracing } : {}),
    ...(toolOutputTracing != null ? { toolOutputTracing } : {}),
  };
}
