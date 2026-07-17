import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';

const LANGGRAPH_START_NODE = '__start__';
const ANONYMOUS_LAMBDA_NAME = 'RunnableLambda';
const LANGGRAPH_AGENT_NODE_PREFIX = 'agent=';
const LANGGRAPH_TOOL_NODE_PREFIX = 'tools=';
const TOOL_BATCH_RUN_NAME = 'tool_batch';
const AGENT_NODE_SPAN_NAME = 'agent';
const GENERATION_SPAN_NAME = 'llm';
const ROOT_OBSERVATION_TYPE = 'agent';

type MutableSpan = ReadableSpan & {
  name: string;
  attributes: Record<string, unknown>;
};

type SerializedToolCall = {
  name: string;
  args: unknown;
  id?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function parseAttributeValue(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return value;
  }
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

function getMessageArray(
  value: unknown
): Record<string, unknown>[] | undefined {
  if (Array.isArray(value)) {
    const records = value.filter(isRecord);
    return records.length > 0 ? records : undefined;
  }
  if (!isRecord(value)) {
    return undefined;
  }
  return (
    getMessageArray(value.messages) ??
    getMessageArray(value.input) ??
    getMessageArray(value.output)
  );
}

function getMessageRole(message: Record<string, unknown>): string | undefined {
  const id = message.id;
  if (Array.isArray(id)) {
    const className = id[id.length - 1];
    if (typeof className === 'string') {
      if (className.includes('Human')) {
        return 'user';
      }
      if (className.includes('AI')) {
        return 'assistant';
      }
      if (className.includes('System')) {
        return 'system';
      }
      if (className.includes('Tool')) {
        return 'tool';
      }
    }
  }
  const rawRole =
    message.type ?? message._type ?? message.role ?? message.sender;
  if (typeof rawRole !== 'string') {
    const kwargs = message.kwargs;
    return isRecord(kwargs) ? getMessageRole(kwargs) : undefined;
  }
  const normalized = rawRole.toLowerCase();
  if (normalized === 'human') {
    return 'user';
  }
  if (normalized === 'ai') {
    return 'assistant';
  }
  return normalized;
}

function getMessageText(message: Record<string, unknown>): string | undefined {
  const content =
    message.content ??
    (isRecord(message.kwargs) ? message.kwargs.content : undefined) ??
    (isRecord(message.data) ? message.data.content : undefined);
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return undefined;
  }
  const text = content
    .filter(isRecord)
    .map((part) => (typeof part.text === 'string' ? part.text : ''))
    .join('');
  return text === '' ? undefined : text;
}

function findLastMessageText(value: unknown, role: string): string | undefined {
  const messages = getMessageArray(value);
  if (messages == null) {
    return undefined;
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    if (getMessageRole(messages[i]) !== role) {
      continue;
    }
    const text = getMessageText(messages[i]);
    if (text != null && text.trim() !== '') {
      return text;
    }
  }
  return undefined;
}

function normalizeToolCall(value: unknown): SerializedToolCall | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const fn = value.function;
  if (isRecord(fn) && typeof fn.name === 'string') {
    return {
      name: fn.name,
      args: parseAttributeValue(fn.arguments),
      ...(typeof value.id === 'string' ? { id: value.id } : {}),
    };
  }
  if (typeof value.name !== 'string') {
    return undefined;
  }
  return {
    name: value.name,
    args: value.args ?? {},
    ...(typeof value.id === 'string' ? { id: value.id } : {}),
  };
}

function getMessageToolCalls(
  message: Record<string, unknown>
): SerializedToolCall[] {
  const rawCalls =
    message.tool_calls ??
    (isRecord(message.kwargs) ? message.kwargs.tool_calls : undefined) ??
    (isRecord(message.additional_kwargs)
      ? message.additional_kwargs.tool_calls
      : undefined) ??
    (isRecord(message.data) ? message.data.tool_calls : undefined);
  if (!Array.isArray(rawCalls)) {
    return [];
  }
  const calls: SerializedToolCall[] = [];
  for (const rawCall of rawCalls) {
    const call = normalizeToolCall(rawCall);
    if (call != null) {
      calls.push(call);
    }
  }
  return calls;
}

/** Latest assistant turn's tool calls — the calls this tool node is executing. */
function findPendingToolCalls(value: unknown): SerializedToolCall[] {
  const messages = getMessageArray(value);
  if (messages == null) {
    return [];
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    if (getMessageRole(messages[i]) !== 'assistant') {
      continue;
    }
    const calls = getMessageToolCalls(messages[i]);
    if (calls.length > 0) {
      return calls;
    }
  }
  return [];
}

function getRootSpanParentId(span: ReadableSpan): string | undefined {
  const legacyParent = (span as { parentSpanId?: string }).parentSpanId;
  if (typeof legacyParent === 'string' && legacyParent !== '') {
    return legacyParent;
  }
  const parentContext = (span as { parentSpanContext?: { spanId?: string } })
    .parentSpanContext;
  const spanId = parentContext?.spanId;
  return typeof spanId === 'string' && spanId !== '' ? spanId : undefined;
}

function isRootSpan(span: ReadableSpan): boolean {
  return getRootSpanParentId(span) == null;
}

/**
 * LangGraph plumbing observations that add noise without information:
 * the duplicated `__start__` channel-seed nodes, anonymous `RunnableLambda`
 * pass-throughs, and the `tool_batch` run that duplicates its parent
 * `tools=<id>` node span verbatim (Langfuse team feedback items 4 & 5).
 */
export function shouldDropLangfuseSpan(spanName: string): boolean {
  return (
    spanName === LANGGRAPH_START_NODE ||
    spanName === ANONYMOUS_LAMBDA_NAME ||
    spanName === TOOL_BATCH_RUN_NAME
  );
}

function shapeToolNodeSpan(span: MutableSpan): void {
  const inputKey = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
  const calls = findPendingToolCalls(
    parseAttributeValue(span.attributes[inputKey])
  );
  if (calls.length === 0) {
    return;
  }
  span.name = [...new Set(calls.map((call) => call.name))].join(', ');
  span.attributes[inputKey] = JSON.stringify(
    calls.map(({ name, args }) => ({ name, args }))
  );
}

function shapeRootSpan(span: MutableSpan): void {
  const inputKey = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
  const outputKey = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT;
  const question = findLastMessageText(
    parseAttributeValue(span.attributes[inputKey]),
    'user'
  );
  const answer = findLastMessageText(
    parseAttributeValue(span.attributes[outputKey]),
    'assistant'
  );
  if (question != null) {
    span.attributes[inputKey] = question;
    span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT] = question;
  }
  if (answer != null) {
    span.attributes[outputKey] = answer;
    span.attributes[LangfuseOtelSpanAttributes.TRACE_OUTPUT] = answer;
  }
}

function isGenerationSpan(span: MutableSpan): boolean {
  const type = span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE];
  return typeof type === 'string' && type.toLowerCase() === 'generation';
}

/**
 * Reshapes spans per Langfuse-team feedback before export:
 * - `agent=<id>` / `tools=<id>` node names carry the ephemeral agent id
 *   (`provider__model`) — strip it so switching models doesn't break
 *   name-based logic (item 1).
 * - LLM generation spans keep the provider client class name (`ChatOpenAI`,
 *   `AzureChatOpenAI`, …); rename them to a provider-agnostic `llm` so the
 *   name reflects the operation, not the model (the model stays on the
 *   generation's model attribute).
 * - Tool node spans are renamed to the actual tool name(s) and their
 *   input scoped to the pending tool-call args instead of the whole
 *   chat history (items 3 & 4).
 * - The root span becomes a top-level `agent` observation whose input/output
 *   (and the trace input/output derived from it) are the user question and
 *   assistant response, so the session view reads as a conversation (item 2).
 */
export function shapeLangfuseSpan(span: ReadableSpan): void {
  const mutable = span as MutableSpan;
  if (mutable.name.startsWith(LANGGRAPH_AGENT_NODE_PREFIX)) {
    mutable.name = AGENT_NODE_SPAN_NAME;
    return;
  }
  if (mutable.name.startsWith(LANGGRAPH_TOOL_NODE_PREFIX)) {
    shapeToolNodeSpan(mutable);
    return;
  }
  if (isGenerationSpan(mutable)) {
    mutable.name = GENERATION_SPAN_NAME;
    return;
  }
  if (isRootSpan(span)) {
    mutable.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE] =
      ROOT_OBSERVATION_TYPE;
    shapeRootSpan(mutable);
  }
}
