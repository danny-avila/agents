import { AsyncLocalStorage } from 'node:async_hooks';
import { createContextKey } from '@opentelemetry/api';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';

export const LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT = '[REDACTED]';

export type SensitivePattern = {
  id: string;
  label: string;
  pattern: RegExp;
};

export type ResolvedLangfuseMessageContentRedactionConfig = {
  enabled: boolean;
  patterns: SensitivePattern[];
  redactionText: string;
};

const messageContentRedactionConfigKey = createContextKey(
  'librechat.langfuse.message-content-redaction'
);
const messageContentRedactionStorage =
  new AsyncLocalStorage<ResolvedLangfuseMessageContentRedactionConfig>();

const SCANNED_SPAN_ATTRIBUTE_KEYS = [
  LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
  LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
  LangfuseOtelSpanAttributes.TRACE_INPUT,
  LangfuseOtelSpanAttributes.TRACE_OUTPUT,
];

/**
 * Built-in credential-shaped patterns. Each pattern's first capture group is
 * the visible prefix that survives redaction (so reviewers can still tell
 * what kind of secret was scrubbed). Order matters: more specific `sk-`
 * variants (Anthropic, Langfuse) precede the generic OpenAI pattern, which
 * uses a negative lookahead to avoid double-matching them.
 */
export const SENSITIVE_VALUE_PATTERNS: SensitivePattern[] = [
  {
    id: 'aws_access_key',
    label: 'AWS access key',
    pattern: /\b(AKIA|ASIA|ABIA|ACCA|A3T[A-Z0-9])[A-Z0-9]{12,16}\b/g,
  },
  {
    id: 'github_token',
    label: 'GitHub token',
    pattern: /\b(gh[poshru]_)[A-Za-z0-9]{36,255}\b/g,
  },
  {
    id: 'slack_token',
    label: 'Slack token',
    pattern: /\b(xox[bpasr]-)[A-Za-z0-9-]{10,200}/g,
  },
  {
    id: 'google_api_key',
    label: 'Google API key',
    pattern: /\b(AIza)[A-Za-z0-9_-]{35}\b/g,
  },
  {
    id: 'stripe_key',
    label: 'Stripe key',
    pattern: /\b((?:sk|pk|rk)_(?:live|test|prod)_)[A-Za-z0-9]{10,99}\b/g,
  },
  {
    id: 'anthropic_api_key',
    label: 'Anthropic API key',
    pattern: /\b(sk-ant-)[A-Za-z0-9_-]{20,}/g,
  },
  {
    id: 'langfuse_key',
    label: 'Langfuse key',
    pattern: /\b((?:pk|sk)-lf-)[A-Za-z0-9-]{20,}/g,
  },
  {
    id: 'openai_api_key',
    label: 'OpenAI API key',
    pattern: /\b(sk-(?!ant-|lf-))[A-Za-z0-9_-]{20,}/g,
  },
  {
    id: 'jwt',
    label: 'JWT',
    pattern: /\b(eyJ)[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,
  },
  {
    id: 'bearer_token',
    label: 'Bearer token',
    pattern: /\b(Bearer )[A-Za-z0-9._\-=]+/g,
  },
  {
    id: 'api_key_header',
    label: 'api-key header',
    pattern: /\b(api-key:\s*)[A-Za-z0-9._\-=]+/gi,
  },
  {
    id: 'api_key_query',
    label: 'api_key query param',
    pattern: /\b(api_key=)[^\s"'&]+/gi,
  },
];

const PATTERNS_BY_ID = new Map<string, SensitivePattern>(
  SENSITIVE_VALUE_PATTERNS.map((entry) => [entry.id, entry])
);

function isPresent(value: unknown): value is string {
  return typeof value === 'string' && value.trim() !== '';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
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

function parsePatternIds(value: string | undefined): string[] | undefined {
  if (!isPresent(value)) {
    return undefined;
  }
  return value
    .split(',')
    .map((id) => id.trim())
    .filter((id) => id !== '');
}

function getEnvEnabled(): boolean | undefined {
  return parseBoolean(process.env.LANGFUSE_REDACT_MESSAGE_CONTENT);
}

function getEnvPatternIds(): string[] | undefined {
  return parsePatternIds(process.env.LANGFUSE_REDACT_MESSAGE_CONTENT_PATTERNS);
}

function getEnvRedactionText(): string | undefined {
  return isPresent(process.env.LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT)
    ? process.env.LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT
    : undefined;
}

function selectPatterns(ids: string[] | undefined): SensitivePattern[] {
  if (ids == null) {
    return SENSITIVE_VALUE_PATTERNS;
  }
  const selected: SensitivePattern[] = [];
  for (const id of ids) {
    const entry = PATTERNS_BY_ID.get(id);
    if (entry != null) {
      selected.push(entry);
    }
  }
  return selected;
}

export function resolveMessageContentRedactionConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): ResolvedLangfuseMessageContentRedactionConfig {
  const runConfig = runLangfuse?.messageContentRedaction;
  const agentConfig = agentLangfuse?.messageContentRedaction;

  const enabled =
    agentConfig?.enabled ?? runConfig?.enabled ?? getEnvEnabled() ?? false;

  const patternIds =
    agentConfig?.patternIds ?? runConfig?.patternIds ?? getEnvPatternIds();

  const redactionText =
    agentConfig?.redactionText ??
    runConfig?.redactionText ??
    getEnvRedactionText() ??
    LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT;

  return {
    enabled,
    patterns: selectPatterns(patternIds),
    redactionText,
  };
}

export function shouldApplyMessageContentRedaction(
  config: ResolvedLangfuseMessageContentRedactionConfig
): boolean {
  return config.enabled && config.patterns.length > 0;
}

function redactString(
  value: string,
  config: ResolvedLangfuseMessageContentRedactionConfig
): { value: string; changed: boolean } {
  let next = value;
  let changed = false;
  for (const { pattern } of config.patterns) {
    pattern.lastIndex = 0;
    const replaced = next.replace(pattern, `$1${config.redactionText}`);
    if (replaced !== next) {
      next = replaced;
      changed = true;
    }
  }
  return { value: next, changed };
}

function redactValue(
  value: unknown,
  config: ResolvedLangfuseMessageContentRedactionConfig
): { value: unknown; changed: boolean } {
  if (typeof value === 'string') {
    return redactString(value, config);
  }

  if (Array.isArray(value)) {
    let changed = false;
    const next: unknown[] = [];
    for (const item of value) {
      const result = redactValue(item, config);
      next.push(result.value);
      if (result.changed) {
        changed = true;
      }
    }
    return changed ? { value: next, changed } : { value, changed: false };
  }

  if (!isRecord(value)) {
    return { value, changed: false };
  }

  let changed = false;
  const next: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const result = redactValue(child, config);
    next[key] = result.value;
    if (result.changed) {
      changed = true;
    }
  }
  return changed ? { value: next, changed } : { value, changed: false };
}

function redactSpanAttribute(
  attributes: Record<string, unknown>,
  key: string,
  config: ResolvedLangfuseMessageContentRedactionConfig
): void {
  if (!(key in attributes)) {
    return;
  }

  const value = attributes[key];

  if (typeof value !== 'string') {
    const result = redactValue(value, config);
    if (result.changed) {
      attributes[key] = result.value;
    }
    return;
  }

  const trimmed = value.trim();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(value) as unknown;
      const result = redactValue(parsed, config);
      if (result.changed) {
        attributes[key] = JSON.stringify(result.value);
        return;
      }
    } catch {
      // fall through to plain-string scrub
    }
  }

  const result = redactString(value, config);
  if (result.changed) {
    attributes[key] = result.value;
  }
}

type SpanWithAttributes = ReadableSpan & {
  attributes: Record<string, unknown>;
};

export function redactLangfuseSpanMessageContent(
  span: ReadableSpan,
  config: ResolvedLangfuseMessageContentRedactionConfig
): void {
  if (!shouldApplyMessageContentRedaction(config)) {
    return;
  }

  const attributes = (span as SpanWithAttributes).attributes;
  for (const key of SCANNED_SPAN_ATTRIBUTE_KEYS) {
    redactSpanAttribute(attributes, key, config);
  }
}

export function getContextMessageContentRedactionConfig(
  activeContext: Context
): ResolvedLangfuseMessageContentRedactionConfig | undefined {
  const asyncConfig = messageContentRedactionStorage.getStore();
  if (asyncConfig != null) {
    return asyncConfig;
  }

  const value = activeContext.getValue(messageContentRedactionConfigKey);
  return isRecord(value)
    ? (value as unknown as ResolvedLangfuseMessageContentRedactionConfig)
    : undefined;
}

export function setMessageContentRedactionConfigOnContext(
  activeContext: Context,
  config: ResolvedLangfuseMessageContentRedactionConfig
): Context {
  return activeContext.setValue(messageContentRedactionConfigKey, config);
}

export function runWithMessageContentRedactionConfig<T>(
  config: ResolvedLangfuseMessageContentRedactionConfig,
  action: () => T
): T {
  return messageContentRedactionStorage.run(config, action);
}

export function hasExplicitMessageContentRedactionConfig(
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): boolean {
  return (
    runLangfuse?.messageContentRedaction != null ||
    agentLangfuse?.messageContentRedaction != null
  );
}
