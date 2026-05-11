import {
  AIMessage,
  HumanMessage,
  RemoveMessage,
  SystemMessage,
  ToolMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { UsageMetadata } from '@langchain/core/messages';
import type { JsonObject, JsonValue, SerializedSessionMessage } from './types';

type MessageExtras = {
  id?: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
  usage_metadata?: UsageMetadata;
  additional_kwargs?: unknown;
  response_metadata?: unknown;
};

const CIRCULAR_REFERENCE = '[Circular]';

function isJsonPrimitive(
  value: unknown
): value is string | number | boolean | null {
  return (
    value == null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  );
}

function toJsonValueInternal(value: unknown, seen: WeakSet<object>): JsonValue {
  if (isJsonPrimitive(value)) {
    return Number.isNaN(value) ? null : value;
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return CIRCULAR_REFERENCE;
    }
    seen.add(value);
    const result = value.map((item) => toJsonValueInternal(item, seen));
    seen.delete(value);
    return result;
  }
  if (value instanceof Error) {
    if (seen.has(value)) {
      return CIRCULAR_REFERENCE;
    }
    seen.add(value);
    const result: JsonObject = {
      name: value.name,
      message: value.message,
    };
    if (value.stack != null && value.stack !== '') {
      result.stack = value.stack;
    }
    if ('cause' in value && typeof value.cause !== 'undefined') {
      result.cause = toJsonValueInternal(value.cause, seen);
    }
    for (const [key, nested] of Object.entries(value)) {
      if (typeof nested !== 'undefined') {
        result[key] = toJsonValueInternal(nested, seen);
      }
    }
    seen.delete(value);
    return result;
  }
  if (value !== null && typeof value === 'object') {
    if (seen.has(value)) {
      return CIRCULAR_REFERENCE;
    }
    seen.add(value);
    const result: JsonObject = {};
    for (const [key, nested] of Object.entries(value)) {
      if (typeof nested !== 'undefined') {
        result[key] = toJsonValueInternal(nested, seen);
      }
    }
    seen.delete(value);
    return result;
  }
  return String(value);
}

export function toJsonValue(value: unknown): JsonValue {
  return toJsonValueInternal(value, new WeakSet<object>());
}

function toJsonObject(value: unknown): JsonObject | undefined {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }
  return toJsonValue(value) as JsonObject;
}

function fromJsonValue(value: JsonValue): unknown {
  return value;
}

export function serializeMessage(
  message: BaseMessage
): SerializedSessionMessage {
  const extras = message as BaseMessage & MessageExtras;
  const serialized: SerializedSessionMessage = {
    messageType: message._getType(),
    content: toJsonValue(message.content),
  };
  const additionalKwargs = toJsonObject(extras.additional_kwargs);
  const responseMetadata = toJsonObject(extras.response_metadata);
  const usageMetadata = toJsonObject(extras.usage_metadata);
  if (additionalKwargs) {
    serialized.additionalKwargs = additionalKwargs;
  }
  if (responseMetadata) {
    serialized.responseMetadata = responseMetadata;
  }
  if (usageMetadata) {
    serialized.usageMetadata = usageMetadata;
  }
  if (extras.id != null && extras.id !== '') {
    serialized.id = extras.id;
  }
  if (extras.name != null && extras.name !== '') {
    serialized.name = extras.name;
  }
  if (extras.tool_call_id != null && extras.tool_call_id !== '') {
    serialized.toolCallId = extras.tool_call_id;
  }
  if (extras.tool_calls) {
    serialized.toolCalls = toJsonValue(extras.tool_calls);
  }
  return serialized;
}

export function deserializeMessage(
  serialized: SerializedSessionMessage
): BaseMessage {
  const common = {
    content: fromJsonValue(serialized.content) as BaseMessage['content'],
    additional_kwargs: serialized.additionalKwargs,
    response_metadata: serialized.responseMetadata,
    id: serialized.id,
    name: serialized.name,
  };
  if (serialized.messageType === 'human') {
    return new HumanMessage(common);
  }
  if (serialized.messageType === 'ai') {
    return new AIMessage({
      ...common,
      tool_calls: serialized.toolCalls as ToolCall[] | undefined,
      usage_metadata: serialized.usageMetadata as UsageMetadata | undefined,
    });
  }
  if (serialized.messageType === 'system') {
    return new SystemMessage(common);
  }
  if (serialized.messageType === 'tool') {
    return new ToolMessage({
      ...common,
      tool_call_id: serialized.toolCallId ?? '',
    });
  }
  if (serialized.messageType === 'remove') {
    return new RemoveMessage({
      additional_kwargs: serialized.additionalKwargs,
      response_metadata: serialized.responseMetadata,
      id: serialized.id ?? '',
      name: serialized.name,
    });
  }
  return new HumanMessage(common);
}

export function getMessageRole(message: BaseMessage): string {
  const type = message._getType();
  if (type === 'human') {
    return 'user';
  }
  if (type === 'ai') {
    return 'assistant';
  }
  return type;
}

export function extractTextFromContent(content: JsonValue): string {
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return '';
  }
  const chunks: string[] = [];
  for (const part of content) {
    if (part != null && typeof part === 'object' && !Array.isArray(part)) {
      const text = part.text;
      if (typeof text === 'string') {
        chunks.push(text);
      }
    }
  }
  return chunks.join('');
}
