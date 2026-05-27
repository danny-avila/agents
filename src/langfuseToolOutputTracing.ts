import { context, createContextKey } from '@opentelemetry/api';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import { AsyncLocalStorage } from 'node:async_hooks';
import type {
  ReadableSpan,
  Span,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';

export const LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT = '[tool output redacted]';

const langfuseToolOutputTracingConfigKey = createContextKey(
  'librechat.langfuse.tool-output-tracing'
);
const toolOutputTracingStorage =
  new AsyncLocalStorage<ResolvedLangfuseToolOutputTracingConfig>();

const CHAT_ROLES = new Set([
  'assistant',
  'developer',
  'human',
  'system',
  'user',
]);

export type ResolvedLangfuseToolOutputTracingConfig = {
  enabled: boolean;
  redactedToolNames: Set<string>;
  redactedToolNameMatchMode: 'exact' | 'partial';
  redactionText: string;
};

type SpanWithAttributes = ReadableSpan & {
  attributes: Record<string, unknown>;
};

type RedactionResult = {
  value: unknown;
  changed: boolean;
};

type RedactionContext = {
  toolNamesByCallId: Map<string, string>;
};

const TOOL_OUTPUT_FIELD_KEYS = ['content', 'artifact'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

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

function normalizeToolName(name: string): string {
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

function resolveToolOutputTracingConfig(
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

function shouldApplyToolOutputRedaction(
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  return config.enabled === false || config.redactedToolNames.size > 0;
}

function toolNameMatches(
  toolName: string | undefined,
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  if (!isPresent(toolName)) {
    return false;
  }

  const normalizedToolName = normalizeToolName(toolName);
  if (config.redactedToolNameMatchMode === 'partial') {
    for (const redactedToolName of config.redactedToolNames) {
      if (normalizedToolName.includes(redactedToolName)) {
        return true;
      }
    }
    return false;
  }

  return config.redactedToolNames.has(normalizedToolName);
}

function shouldRedactTool(
  toolName: string | undefined,
  config: ResolvedLangfuseToolOutputTracingConfig
): boolean {
  return config.enabled === false || toolNameMatches(toolName, config);
}

function getStringField(
  value: Record<string, unknown>,
  key: string
): string | undefined {
  const field = value[key];
  return typeof field === 'string' ? field : undefined;
}

function getNestedStringField(
  value: Record<string, unknown>,
  objectKey: string,
  fieldKey: string
): string | undefined {
  const nested = value[objectKey];
  if (!isRecord(nested)) {
    return undefined;
  }
  return getStringField(nested, fieldKey);
}

function getSerializedToolCallId(
  value: Record<string, unknown>
): string | undefined {
  return (
    getStringField(value, 'tool_call_id') ??
    getNestedStringField(value, 'kwargs', 'tool_call_id') ??
    getNestedStringField(value, 'additional_kwargs', 'tool_call_id') ??
    getNestedStringField(value, 'data', 'tool_call_id') ??
    (typeof value.id === 'string' ? value.id : undefined)
  );
}

function getSerializedToolName(
  value: Record<string, unknown>,
  redactionContext?: RedactionContext
): string | undefined {
  const role = getStringField(value, 'role');
  const explicitName =
    getStringField(value, 'name') ??
    getStringField(value, 'tool_name') ??
    getNestedStringField(value, 'function', 'name') ??
    getNestedStringField(value, 'kwargs', 'name') ??
    getNestedStringField(value, 'additional_kwargs', 'name') ??
    getNestedStringField(value, 'data', 'name') ??
    (role != null && role.toLowerCase() !== 'tool' ? role : undefined);

  if (explicitName != null) {
    return explicitName;
  }

  const toolCallId = getSerializedToolCallId(value);
  return toolCallId != null
    ? redactionContext?.toolNamesByCallId.get(toolCallId)
    : undefined;
}

function hasToolMessageIdentity(value: Record<string, unknown>): boolean {
  const type = getStringField(value, 'type') ?? getStringField(value, '_type');
  if (type === 'tool' || type === 'tool_message') {
    return true;
  }

  const id = value.id;
  if (
    Array.isArray(id) &&
    id.some((part) => typeof part === 'string' && part.includes('ToolMessage'))
  ) {
    return true;
  }

  if (
    'tool_call_id' in value ||
    getNestedStringField(value, 'kwargs', 'tool_call_id') != null ||
    getNestedStringField(value, 'additional_kwargs', 'tool_call_id') != null
  ) {
    return true;
  }

  const role = getStringField(value, 'role');
  return (
    role != null &&
    !CHAT_ROLES.has(role.toLowerCase()) &&
    ('content' in value || isRecord(value.kwargs) || isRecord(value.data))
  );
}

function redactToolContentFields(
  value: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig
): Record<string, unknown> {
  const next = { ...value };

  for (const outputKey of TOOL_OUTPUT_FIELD_KEYS) {
    if (outputKey in next) {
      next[outputKey] = config.redactionText;
    }
  }

  for (const nestedKey of ['kwargs', 'data', 'additional_kwargs']) {
    const nested = next[nestedKey];
    if (!isRecord(nested)) {
      continue;
    }
    const nextNested = { ...nested };
    let changed = false;
    for (const outputKey of TOOL_OUTPUT_FIELD_KEYS) {
      if (outputKey in nextNested) {
        nextNested[outputKey] = config.redactionText;
        changed = true;
      }
    }
    if (changed) {
      next[nestedKey] = nextNested;
    }
  }

  return next;
}

function collectToolCallNames(
  value: unknown,
  redactionContext: RedactionContext
): void {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectToolCallNames(item, redactionContext);
    }
    return;
  }

  if (!isRecord(value)) {
    return;
  }

  const toolCallId = getSerializedToolCallId(value);
  const toolName = getSerializedToolName(value);
  if (toolCallId != null && toolName != null) {
    redactionContext.toolNamesByCallId.set(toolCallId, toolName);
  }

  for (const child of Object.values(value)) {
    collectToolCallNames(child, redactionContext);
  }
}

function redactValue(
  value: unknown,
  config: ResolvedLangfuseToolOutputTracingConfig,
  redactionContext: RedactionContext
): RedactionResult {
  if (Array.isArray(value)) {
    let changed = false;
    const next: unknown[] = [];
    for (const item of value) {
      const result = redactValue(item, config, redactionContext);
      if (result.changed) {
        changed = true;
      }
      next.push(result.value);
    }
    return changed ? { value: next, changed } : { value, changed };
  }

  if (!isRecord(value)) {
    return { value, changed: false };
  }

  const toolName = getSerializedToolName(value, redactionContext);
  if (hasToolMessageIdentity(value) && shouldRedactTool(toolName, config)) {
    return {
      value: redactToolContentFields(value, config),
      changed: true,
    };
  }

  let changed = false;
  const next: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const result = redactValue(child, config, redactionContext);
    if (result.changed) {
      changed = true;
    }
    next[key] = result.value;
  }

  return changed ? { value: next, changed } : { value, changed };
}

function redactSerializedValue(
  value: unknown,
  config: ResolvedLangfuseToolOutputTracingConfig
): RedactionResult {
  const redactionContext: RedactionContext = {
    toolNamesByCallId: new Map(),
  };
  if (typeof value !== 'string') {
    collectToolCallNames(value, redactionContext);
    return redactValue(value, config, redactionContext);
  }

  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return { value, changed: false };
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    collectToolCallNames(parsed, redactionContext);
    const result = redactValue(parsed, config, redactionContext);
    return result.changed
      ? { value: JSON.stringify(result.value), changed: true }
      : { value, changed: false };
  } catch {
    return { value, changed: false };
  }
}

function redactAttribute(
  attributes: Record<string, unknown>,
  key: string,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (!(key in attributes)) {
    return;
  }

  const result = redactSerializedValue(attributes[key], config);
  if (result.changed) {
    attributes[key] = result.value;
  }
}

function isToolObservation(attributes: Record<string, unknown>): boolean {
  const type = attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  return typeof type === 'string' && type.toLowerCase() === 'tool';
}

function redactToolObservationOutput(
  span: ReadableSpan,
  attributes: Record<string, unknown>,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (
    !(
      isToolObservation(attributes) &&
      shouldRedactTool(span.name, config) &&
      LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT in attributes
    )
  ) {
    return;
  }

  attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] =
    config.redactionText;
}

export function redactLangfuseSpanToolOutputs(
  span: ReadableSpan,
  config: ResolvedLangfuseToolOutputTracingConfig
): void {
  if (!shouldApplyToolOutputRedaction(config)) {
    return;
  }

  const attributes = (span as SpanWithAttributes).attributes;
  redactToolObservationOutput(span, attributes, config);

  for (const key of [
    LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
    LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT,
    LangfuseOtelSpanAttributes.TRACE_INPUT,
    LangfuseOtelSpanAttributes.TRACE_OUTPUT,
  ]) {
    redactAttribute(attributes, key, config);
  }
}

function getContextToolOutputTracingConfig(
  activeContext: Context
): ResolvedLangfuseToolOutputTracingConfig | undefined {
  const asyncConfig = toolOutputTracingStorage.getStore();
  if (asyncConfig != null) {
    return asyncConfig;
  }

  const value = activeContext.getValue(langfuseToolOutputTracingConfigKey);
  return isRecord(value)
    ? (value as ResolvedLangfuseToolOutputTracingConfig)
    : undefined;
}

class ToolOutputRedactingLangfuseSpanProcessor implements SpanProcessor {
  private readonly processor: LangfuseSpanProcessor;
  private readonly fallbackConfig?: ResolvedLangfuseToolOutputTracingConfig;
  private readonly spanConfigs = new WeakMap<
    object,
    ResolvedLangfuseToolOutputTracingConfig
  >();

  constructor(
    params?: LangfuseSpanProcessorParams,
    fallbackConfig?: ResolvedLangfuseToolOutputTracingConfig
  ) {
    this.processor = new LangfuseSpanProcessor(params);
    this.fallbackConfig = fallbackConfig;
  }

  onStart(span: Span, parentContext: Context): void {
    const config =
      this.fallbackConfig ?? getContextToolOutputTracingConfig(parentContext);
    if (config != null) {
      this.spanConfigs.set(span, config);
    }
    this.processor.onStart(span, parentContext);
  }

  onEnd(span: ReadableSpan): void {
    const config =
      this.spanConfigs.get(span) ??
      toolOutputTracingStorage.getStore() ??
      this.fallbackConfig ??
      resolveToolOutputTracingConfig();
    redactLangfuseSpanToolOutputs(span, config);
    this.processor.onEnd(span);
  }

  forceFlush(): Promise<void> {
    return this.processor.forceFlush();
  }

  shutdown(): Promise<void> {
    return this.processor.shutdown();
  }
}

export function createLangfuseSpanProcessor(
  params?: LangfuseSpanProcessorParams,
  runLangfuse?: t.LangfuseConfig,
  agentLangfuse?: t.LangfuseConfig
): SpanProcessor {
  const fallbackConfig =
    runLangfuse != null || agentLangfuse != null
      ? resolveToolOutputTracingConfig(runLangfuse, agentLangfuse)
      : undefined;
  return new ToolOutputRedactingLangfuseSpanProcessor(params, fallbackConfig);
}

export function withLangfuseToolOutputTracingConfig<T>(
  runLangfuse: t.LangfuseConfig | undefined,
  action: () => T,
  agentLangfuse?: t.LangfuseConfig
): T {
  if (
    runLangfuse?.toolOutputTracing == null &&
    agentLangfuse?.toolOutputTracing == null
  ) {
    return action();
  }

  const config = resolveToolOutputTracingConfig(runLangfuse, agentLangfuse);
  const activeContext = context
    .active()
    .setValue(langfuseToolOutputTracingConfigKey, config);
  return toolOutputTracingStorage.run(config, () =>
    context.with(activeContext, action)
  );
}

function hasLangfuseEnvKeys(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
    isPresent(process.env.LANGFUSE_BASE_URL ?? process.env.LANGFUSE_BASEURL)
  );
}

export function shouldTraceToolNodeForLangfuse({
  runLangfuse,
  agentLangfuse,
}: {
  runLangfuse?: t.LangfuseConfig;
  agentLangfuse?: t.LangfuseConfig;
}): boolean {
  const explicit =
    agentLangfuse?.toolNodeTracing?.enabled ??
    runLangfuse?.toolNodeTracing?.enabled;
  if (explicit != null) {
    return explicit && hasLangfuseEnvKeys();
  }

  if (agentLangfuse?.enabled === false || runLangfuse?.enabled === false) {
    return false;
  }

  return hasLangfuseEnvKeys();
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

  return {
    ...runLangfuse,
    ...agentLangfuse,
    toolNodeTracing: {
      ...runLangfuse.toolNodeTracing,
      ...agentLangfuse.toolNodeTracing,
    },
    toolOutputTracing: {
      ...runLangfuse.toolOutputTracing,
      ...agentLangfuse.toolOutputTracing,
    },
  };
}
