import { isProxy } from 'node:util/types';
import { ToolMessage, type BaseMessage } from '@langchain/core/messages';
import { truncateToolResultContent } from './truncation';

type ToolContent = BaseMessage['content'];
type ToolContentBlock = Exclude<ToolContent, string>[number];
type TextToolContentBlock = ToolContentBlock & { type: 'text'; text: string };

const ATOMIC_CONTENT_TYPES = new Set([
  'audio',
  'computer_screenshot',
  'document',
  'file',
  'image',
  'image_file',
  'image_url',
  'input_audio',
  'media',
  'resource',
  'resource_link',
  'video',
  'video_url',
]);

const PROVIDER_NATIVE_TOOL_RESULT_TYPES = new Set([
  'search_result',
  'server_tool_call_result',
  'tool_result',
  'toolResponse',
  'web_search_result',
  'web_search_tool_result',
]);

/**
 * Serializes values that providers render as JSON without throwing on values
 * such as bigint or circular diagnostic metadata.
 */
export function serializeStructuredValue(value: unknown): string {
  const ancestors: object[] = [];
  try {
    const serialized: unknown = JSON.stringify(
      value,
      function (_key, nestedValue: unknown) {
        if (typeof nestedValue === 'bigint') {
          return nestedValue.toString();
        }
        if (
          nestedValue instanceof ArrayBuffer ||
          ArrayBuffer.isView(nestedValue)
        ) {
          const name =
            nestedValue instanceof ArrayBuffer
              ? 'ArrayBuffer'
              : nestedValue.constructor.name;
          return `[${name}: ${nestedValue.byteLength} bytes]`;
        }
        if (nestedValue != null && typeof nestedValue === 'object') {
          while (
            ancestors.length > 0 &&
            ancestors[ancestors.length - 1] !== this
          ) {
            ancestors.pop();
          }
          if (ancestors.includes(nestedValue)) {
            return '[Circular]';
          }
          ancestors.push(nestedValue);
        }
        return nestedValue;
      }
    );
    return typeof serialized === 'string' ? serialized : String(value);
  } catch {
    try {
      return String(value);
    } catch {
      return '[Unserializable structured value]';
    }
  }
}

const SERIALIZATION_CHUNK_SIZE = 4_096;
const MAX_STRUCTURED_SERIALIZATION_DEPTH = 200;
const MAX_STRUCTURED_SERIALIZATION_WORK = 1_000_000;
const PROXY_VALUE_PLACEHOLDER = '[Proxy value omitted]';

type StructuredWorkState = {
  remaining: number;
  exceeded: boolean;
};

function createStructuredWorkState(): StructuredWorkState {
  return {
    remaining: MAX_STRUCTURED_SERIALIZATION_WORK,
    exceeded: false,
  };
}

function consumeStructuredWork(state: StructuredWorkState): boolean {
  if (state.remaining <= 0) {
    state.exceeded = true;
    return false;
  }
  state.remaining--;
  return true;
}

function conservativeStructuredLength(limit: number): number {
  return limit >= Number.MAX_SAFE_INTEGER ? Number.MAX_SAFE_INTEGER : limit + 1;
}

function jsonStringLength(value: string): number {
  let length = 2;
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (
      code === 0x08 ||
      code === 0x09 ||
      code === 0x0a ||
      code === 0x0c ||
      code === 0x0d ||
      code === 0x22 ||
      code === 0x5c
    ) {
      length += 2;
    } else if (code < 0x20) {
      length += 6;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        length += 2;
        i++;
      } else {
        length += 6;
      }
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      length += 6;
    } else {
      length++;
    }
  }
  return length;
}

function estimateStructuredChars(
  value: unknown,
  limit = Number.MAX_SAFE_INTEGER,
  ancestors: object[] = [],
  work = createStructuredWorkState(),
  depth = 0
): number {
  if (typeof value === 'string') {
    return jsonStringLength(value);
  }
  if (typeof value === 'bigint') {
    return jsonStringLength(value.toString());
  }
  if (
    typeof value === 'number' ||
    typeof value === 'boolean' ||
    value == null
  ) {
    return String(value).length;
  }
  if (isProxy(value)) {
    work.exceeded = true;
    return conservativeStructuredLength(limit);
  }
  if (typeof value === 'undefined' || typeof value === 'function') {
    return 4;
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return Math.min(limit + 1, Math.ceil((value.byteLength * 4) / 3) + 32);
  }
  if (value instanceof Date) {
    const time = Date.prototype.getTime.call(value);
    return Number.isFinite(time)
      ? jsonStringLength(Date.prototype.toISOString.call(value))
      : 4;
  }
  if (value instanceof String) {
    return jsonStringLength(String.prototype.valueOf.call(value));
  }
  if (value instanceof Number) {
    const number = Number.prototype.valueOf.call(value);
    return Number.isFinite(number)
      ? String(number === 0 ? 0 : number).length
      : 4;
  }
  if (value instanceof Boolean) {
    return String(Boolean.prototype.valueOf.call(value)).length;
  }
  if (typeof value !== 'object') {
    return jsonStringLength(String(value));
  }
  if (depth >= MAX_STRUCTURED_SERIALIZATION_DEPTH) {
    return jsonStringLength('[Max serialization depth]');
  }
  if (ancestors.includes(value)) {
    return 12;
  }

  ancestors.push(value);
  let length = 2;
  try {
    if (Array.isArray(value)) {
      const arrayLength = readStructuredArrayLength(value, work);
      if (arrayLength == null) {
        return conservativeStructuredLength(limit);
      }
      for (let i = 0; i < arrayLength && length <= limit; i++) {
        if (!consumeStructuredWork(work)) {
          return conservativeStructuredLength(limit);
        }
        if (i > 0) {
          length++;
        }
        length += estimateStructuredChars(
          readStructuredProperty(value, i, work),
          Math.max(0, limit - length),
          ancestors,
          work,
          depth + 1
        );
      }
    } else {
      let emitted = 0;
      for (const key in value) {
        if (!Object.prototype.propertyIsEnumerable.call(value, key)) {
          continue;
        }
        if (!consumeStructuredWork(work)) {
          return conservativeStructuredLength(limit);
        }
        if (length > limit) {
          break;
        }
        const nested = readStructuredProperty(
          value as Record<string, unknown>,
          key,
          work
        );
        if (
          typeof nested === 'undefined' ||
          typeof nested === 'function' ||
          typeof nested === 'symbol'
        ) {
          continue;
        }
        if (emitted > 0) {
          length++;
        }
        length += jsonStringLength(key) + 1;
        length += estimateStructuredChars(
          nested,
          Math.max(0, limit - length),
          ancestors,
          work,
          depth + 1
        );
        emitted++;
      }
    }
  } catch {
    work.exceeded = true;
    return conservativeStructuredLength(limit);
  } finally {
    ancestors.pop();
  }
  if (work.exceeded) {
    return conservativeStructuredLength(limit);
  }
  return Math.min(limit + 1, length);
}

type StructuredSerializationCollector = {
  totalChars: number;
  prefix: SegmentedStringBuffer;
  suffix: SegmentedStringBuffer;
  prefixLimit: number;
  suffixLimit: number;
  work: StructuredWorkState;
};

type SegmentedStringBuffer = {
  segments: string[];
  head: number;
  pending: string[];
  pendingChars: number;
  length: number;
};

export type BoundedStructuredSerialization = {
  /** Provider-facing head/tail preview, bounded by `maxChars`. */
  content: string;
  /** Exact serialized prefix up to the defensive traversal-work ceiling. */
  prefix: string;
  /** Exact length, or `Number.MAX_SAFE_INTEGER` when traversal was capped. */
  originalChars: number;
  truncated: boolean;
};

function normalizeCharLimit(limit: number): number {
  return Number.isFinite(limit)
    ? Math.max(0, Math.floor(limit))
    : Number.MAX_SAFE_INTEGER;
}

function createSegmentedStringBuffer(): SegmentedStringBuffer {
  return {
    segments: [],
    head: 0,
    pending: [],
    pendingChars: 0,
    length: 0,
  };
}

function flushSegmentedStringBuffer(buffer: SegmentedStringBuffer): void {
  if (buffer.pendingChars === 0) {
    return;
  }
  buffer.segments.push(buffer.pending.join(''));
  buffer.pending = [];
  buffer.pendingChars = 0;
}

function appendToSegmentedStringBuffer(
  buffer: SegmentedStringBuffer,
  chunk: string
): void {
  if (chunk.length === 0) {
    return;
  }
  buffer.pending.push(chunk);
  buffer.pendingChars += chunk.length;
  buffer.length += chunk.length;
  if (buffer.pendingChars >= SERIALIZATION_CHUNK_SIZE) {
    flushSegmentedStringBuffer(buffer);
  }
}

function trimSegmentedStringBufferStart(
  buffer: SegmentedStringBuffer,
  chars: number
): void {
  let remaining = Math.min(chars, buffer.length);
  if (remaining <= 0) {
    return;
  }
  flushSegmentedStringBuffer(buffer);
  while (remaining > 0 && buffer.head < buffer.segments.length) {
    const segment = buffer.segments[buffer.head];
    if (segment.length <= remaining) {
      remaining -= segment.length;
      buffer.length -= segment.length;
      buffer.head++;
    } else {
      buffer.segments[buffer.head] = segment.slice(remaining);
      buffer.length -= remaining;
      remaining = 0;
    }
  }
  if (buffer.head >= 1_024 && buffer.head * 2 >= buffer.segments.length) {
    buffer.segments = buffer.segments.slice(buffer.head);
    buffer.head = 0;
  }
}

function materializeSegmentedStringBuffer(
  buffer: SegmentedStringBuffer
): string {
  const segments = buffer.segments.slice(buffer.head);
  if (buffer.pendingChars > 0) {
    segments.push(buffer.pending.join(''));
  }
  return segments.join('');
}

function appendSerializedChunk(
  collector: StructuredSerializationCollector,
  chunk: string
): void {
  if (chunk.length === 0) {
    return;
  }

  collector.totalChars += chunk.length;

  const prefixRemaining = collector.prefixLimit - collector.prefix.length;
  if (prefixRemaining > 0) {
    appendToSegmentedStringBuffer(
      collector.prefix,
      chunk.slice(0, prefixRemaining)
    );
  }

  if (collector.suffixLimit <= 0) {
    return;
  }
  appendToSegmentedStringBuffer(collector.suffix, chunk);
  if (
    collector.suffix.length >=
    collector.suffixLimit + SERIALIZATION_CHUNK_SIZE
  ) {
    trimSegmentedStringBufferStart(
      collector.suffix,
      collector.suffix.length - collector.suffixLimit
    );
  }
}

function appendJsonSafeRange(
  collector: StructuredSerializationCollector,
  value: string,
  start: number,
  end: number
): void {
  let offset = start;
  while (offset < end) {
    let chunkEnd = Math.min(end, offset + SERIALIZATION_CHUNK_SIZE);
    if (
      chunkEnd < end &&
      chunkEnd > offset &&
      value.charCodeAt(chunkEnd - 1) >= 0xd800 &&
      value.charCodeAt(chunkEnd - 1) <= 0xdbff &&
      value.charCodeAt(chunkEnd) >= 0xdc00 &&
      value.charCodeAt(chunkEnd) <= 0xdfff
    ) {
      chunkEnd--;
    }
    appendSerializedChunk(collector, value.slice(offset, chunkEnd));
    offset = chunkEnd;
  }
}

/**
 * Emits one JSON string incrementally. Calling JSON.stringify on the complete
 * string would allocate an escaped copy before the output cap could apply.
 */
function appendJsonString(
  collector: StructuredSerializationCollector,
  value: string
): void {
  appendSerializedChunk(collector, '"');
  let safeStart = 0;

  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    let escaped: string | undefined;
    switch (code) {
    case 0x08:
      escaped = '\\b';
      break;
    case 0x09:
      escaped = '\\t';
      break;
    case 0x0a:
      escaped = '\\n';
      break;
    case 0x0c:
      escaped = '\\f';
      break;
    case 0x0d:
      escaped = '\\r';
      break;
    case 0x22:
      escaped = '\\"';
      break;
    case 0x5c:
      escaped = '\\\\';
      break;
    default:
      if (code < 0x20) {
        escaped = `\\u${code.toString(16).padStart(4, '0')}`;
      } else if (code >= 0xd800 && code <= 0xdbff) {
        const next = value.charCodeAt(i + 1);
        if (next >= 0xdc00 && next <= 0xdfff) {
          i++;
        } else {
          escaped = `\\u${code.toString(16).padStart(4, '0')}`;
        }
      } else if (code >= 0xdc00 && code <= 0xdfff) {
        escaped = `\\u${code.toString(16).padStart(4, '0')}`;
      }
    }

    if (escaped != null) {
      appendJsonSafeRange(collector, value, safeStart, i);
      appendSerializedChunk(collector, escaped);
      safeStart = i + 1;
    }
  }
  appendJsonSafeRange(collector, value, safeStart, value.length);
  appendSerializedChunk(collector, '"');
}

function getArrayBufferViewName(value: ArrayBufferView): string {
  try {
    const prototype = Object.getPrototypeOf(value) as {
      constructor?: { name?: unknown };
    } | null;
    const name = prototype?.constructor?.name;
    return typeof name === 'string' && name !== '' ? name : 'ArrayBufferView';
  } catch {
    return 'ArrayBufferView';
  }
}

function readStructuredProperty(
  value: Record<string, unknown> | unknown[],
  key: string | number,
  work: StructuredWorkState
): unknown {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(key));
    if (descriptor == null) {
      return undefined;
    }
    if ('value' in descriptor) {
      return descriptor.value;
    }
    return '[Property accessor omitted]';
  } catch {
    work.exceeded = true;
    work.remaining = 0;
    return '[Unreadable property omitted]';
  }
}

function readStructuredArrayLength(
  value: unknown[],
  work: StructuredWorkState
): number | undefined {
  const length = readStructuredProperty(value, 'length', work);
  if (
    typeof length === 'number' &&
    Number.isSafeInteger(length) &&
    length >= 0
  ) {
    return length;
  }
  work.exceeded = true;
  work.remaining = 0;
  return undefined;
}

/**
 * Provider-neutral JSON writer which deliberately does not invoke `toJSON` or
 * property accessors. Native `JSON.stringify` can invoke both before a replacer
 * can bound their output.
 */
function appendStructuredValue(
  collector: StructuredSerializationCollector,
  value: unknown,
  ancestors: Set<object>,
  depth: number,
  position: 'top' | 'array' | 'object'
): boolean {
  if (typeof value === 'string') {
    appendJsonString(collector, value);
    return true;
  }
  if (typeof value === 'bigint') {
    appendJsonString(collector, value.toString());
    return true;
  }
  if (typeof value === 'number') {
    appendSerializedChunk(
      collector,
      Number.isFinite(value) ? String(value === 0 ? 0 : value) : 'null'
    );
    return true;
  }
  if (typeof value === 'boolean' || value === null) {
    appendSerializedChunk(collector, String(value));
    return true;
  }
  if (isProxy(value)) {
    collector.work.exceeded = true;
    appendJsonString(collector, PROXY_VALUE_PLACEHOLDER);
    return true;
  }
  if (
    typeof value === 'undefined' ||
    typeof value === 'function' ||
    typeof value === 'symbol'
  ) {
    if (position === 'object') {
      return false;
    }
    if (position === 'array') {
      appendSerializedChunk(collector, 'null');
    } else if (typeof value === 'undefined') {
      appendSerializedChunk(collector, 'undefined');
    } else {
      appendJsonString(
        collector,
        typeof value === 'function'
          ? `[Function${value.name ? `: ${value.name}` : ''}]`
          : String(value)
      );
    }
    return true;
  }
  if (value instanceof ArrayBuffer) {
    appendJsonString(collector, `[ArrayBuffer: ${value.byteLength} bytes]`);
    return true;
  }
  if (ArrayBuffer.isView(value)) {
    appendJsonString(
      collector,
      `[${getArrayBufferViewName(value)}: ${value.byteLength} bytes]`
    );
    return true;
  }
  if (value instanceof Date) {
    const time = Date.prototype.getTime.call(value);
    if (!Number.isFinite(time)) {
      appendSerializedChunk(collector, 'null');
    } else {
      appendJsonString(collector, Date.prototype.toISOString.call(value));
    }
    return true;
  }
  if (value instanceof String) {
    appendJsonString(collector, String.prototype.valueOf.call(value));
    return true;
  }
  if (value instanceof Number) {
    const number = Number.prototype.valueOf.call(value);
    appendSerializedChunk(
      collector,
      Number.isFinite(number) ? String(number === 0 ? 0 : number) : 'null'
    );
    return true;
  }
  if (value instanceof Boolean) {
    appendSerializedChunk(
      collector,
      String(Boolean.prototype.valueOf.call(value))
    );
    return true;
  }
  if (depth >= MAX_STRUCTURED_SERIALIZATION_DEPTH) {
    appendJsonString(collector, '[Max serialization depth]');
    return true;
  }
  if (ancestors.has(value)) {
    appendJsonString(collector, '[Circular]');
    return true;
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      const arrayLength = readStructuredArrayLength(value, collector.work);
      if (arrayLength == null) {
        appendJsonString(collector, '[Unreadable array omitted]');
        return true;
      }
      appendSerializedChunk(collector, '[');
      for (let i = 0; i < arrayLength; i++) {
        if (!consumeStructuredWork(collector.work)) {
          if (i > 0) {
            appendSerializedChunk(collector, ',');
          }
          appendJsonString(
            collector,
            `[Traversal limit exceeded: ${arrayLength - i} array entries omitted]`
          );
          break;
        }
        if (i > 0) {
          appendSerializedChunk(collector, ',');
        }
        appendStructuredValue(
          collector,
          readStructuredProperty(value, i, collector.work),
          ancestors,
          depth + 1,
          'array'
        );
      }
      appendSerializedChunk(collector, ']');
      return true;
    }

    appendSerializedChunk(collector, '{');
    let emitted = 0;
    // `Object.keys()` allocates an array proportional to the complete object
    // before either output limit applies. Filter a streaming `for…in` walk to
    // retain JSON's own-enumerable-key semantics without that extra array.
    for (const key in value) {
      if (!Object.prototype.propertyIsEnumerable.call(value, key)) {
        continue;
      }
      if (!consumeStructuredWork(collector.work)) {
        if (emitted > 0) {
          appendSerializedChunk(collector, ',');
        }
        appendJsonString(collector, '_truncated');
        appendSerializedChunk(collector, ':');
        appendJsonString(collector, '[Traversal limit exceeded]');
        break;
      }
      const nested = readStructuredProperty(
        value as Record<string, unknown>,
        key,
        collector.work
      );
      if (
        typeof nested === 'undefined' ||
        typeof nested === 'function' ||
        typeof nested === 'symbol'
      ) {
        continue;
      }
      if (emitted > 0) {
        appendSerializedChunk(collector, ',');
      }
      appendJsonString(collector, key);
      appendSerializedChunk(collector, ':');
      appendStructuredValue(collector, nested, ancestors, depth + 1, 'object');
      emitted++;
    }
    appendSerializedChunk(collector, '}');
    return true;
  } finally {
    ancestors.delete(value);
  }
}

function formatBoundedStructuredContent(
  collector: StructuredSerializationCollector,
  maxChars: number,
  prefix: string,
  suffix: string
): string {
  if (collector.totalChars <= maxChars) {
    return prefix.slice(0, collector.totalChars);
  }

  const indicator =
    `\n\n… [truncated: ${collector.totalChars} chars exceeded ` +
    `${maxChars} limit] …\n\n`;
  const available = maxChars - indicator.length;
  if (available <= 0) {
    return prefix.slice(0, maxChars);
  }
  if (available < 200) {
    return prefix.slice(0, available) + indicator.trimEnd();
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;
  let headEnd = headSize;
  const headNewline = prefix.lastIndexOf('\n', headSize);
  if (headNewline > headSize - 200 && headNewline > 0) {
    headEnd = headNewline;
  }

  let tailStart = Math.max(0, suffix.length - tailSize);
  const tailNewline = suffix.indexOf('\n', tailStart);
  if (tailNewline > 0 && tailNewline < tailStart + 200) {
    tailStart = tailNewline + 1;
  }

  return prefix.slice(0, headEnd) + indicator + suffix.slice(tailStart);
}

/**
 * Serializes structured output while retaining at most the requested provider
 * preview and registry prefix. The traversal never calls `toJSON` and never
 * materializes the complete serialized string when it exceeds those limits.
 *
 * `prefixChars` is useful for the tool-output registry: it receives the exact
 * serialized prefix up to `registry.perOutputLimit`, while `content` retains
 * the provider-facing head/tail truncation behavior at `maxChars`. Property
 * accessors are represented by a fixed placeholder instead of being invoked.
 */
export function serializeStructuredValueBounded(
  value: unknown,
  maxChars: number,
  prefixChars = 0
): BoundedStructuredSerialization {
  const normalizedMaxChars = normalizeCharLimit(maxChars);
  const normalizedPrefixChars = normalizeCharLimit(prefixChars);
  const prefixLimit = Math.max(normalizedMaxChars, normalizedPrefixChars);
  const collector: StructuredSerializationCollector = {
    totalChars: 0,
    prefix: createSegmentedStringBuffer(),
    suffix: createSegmentedStringBuffer(),
    prefixLimit,
    suffixLimit:
      normalizedMaxChars === Number.MAX_SAFE_INTEGER ? 0 : normalizedMaxChars,
    work: createStructuredWorkState(),
  };

  try {
    appendStructuredValue(collector, value, new Set<object>(), 0, 'top');
  } catch {
    collector.totalChars = 0;
    collector.prefix = createSegmentedStringBuffer();
    collector.suffix = createSegmentedStringBuffer();
    collector.work = createStructuredWorkState();
    collector.work.exceeded = true;
    appendJsonString(collector, '[Unserializable structured value]');
  }

  const prefix = materializeSegmentedStringBuffer(collector.prefix);
  const suffix = materializeSegmentedStringBuffer(collector.suffix);
  const originalChars = collector.work.exceeded
    ? Number.MAX_SAFE_INTEGER
    : collector.totalChars;
  return {
    content: formatBoundedStructuredContent(
      collector,
      normalizedMaxChars,
      prefix,
      suffix
    ),
    prefix: prefix.slice(0, normalizedPrefixChars),
    originalChars,
    truncated:
      collector.work.exceeded || collector.totalChars > normalizedMaxChars,
  };
}

function serializeStructuredValueWithinLimit(
  value: unknown,
  maxChars: number
): string {
  return serializeStructuredValueBounded(value, maxChars).content;
}

function isTextBlock(value: unknown): value is TextToolContentBlock {
  if (!isRecord(value) || hasUnsafeCallableToJSON(value)) {
    return false;
  }
  const type = readOwnEnumerableDataProperty(value, 'type');
  const text = readOwnEnumerableDataProperty(value, 'text');
  return (
    type.found &&
    type.value === 'text' &&
    text.found &&
    typeof text.value === 'string'
  );
}

function isArray(value: unknown): value is unknown[] {
  try {
    return Array.isArray(value);
  } catch {
    return false;
  }
}

function getDenseTextBlockArrayLength(content: unknown[]): number | undefined {
  if (isProxy(content)) {
    return undefined;
  }
  let contentLength: number;
  try {
    contentLength = content.length;
  } catch {
    return undefined;
  }
  if (
    contentLength === 0 ||
    contentLength > MAX_STRUCTURED_SERIALIZATION_WORK
  ) {
    return undefined;
  }

  let length = contentLength - 1;
  try {
    for (let i = 0; i < contentLength; i++) {
      if (!Object.prototype.hasOwnProperty.call(content, i)) {
        return undefined;
      }
      const block = content[i];
      if (!isTextBlock(block)) {
        return undefined;
      }
      length += block.text.length;
      if (length >= Number.MAX_SAFE_INTEGER) {
        return Number.MAX_SAFE_INTEGER;
      }
    }
  } catch {
    return undefined;
  }
  return length;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null;
}

type OwnDataProperty =
  | { found: true; value: unknown }
  | { found: false; value?: never };

function readOwnEnumerableDataProperty(
  record: Record<string, unknown>,
  key: string
): OwnDataProperty {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(record, key);
    if (
      descriptor == null ||
      descriptor.enumerable !== true ||
      !('value' in descriptor)
    ) {
      return { found: false };
    }
    return { found: true, value: descriptor.value };
  } catch {
    return { found: false };
  }
}

function hasString(record: Record<string, unknown>, key: string): boolean {
  const property = readOwnEnumerableDataProperty(record, key);
  return (
    property.found &&
    typeof property.value === 'string' &&
    property.value !== ''
  );
}

function hasValue(record: Record<string, unknown>, key: string): boolean {
  const property = readOwnEnumerableDataProperty(record, key);
  return property.found && property.value !== undefined;
}

function isStringPayload(value: unknown): boolean {
  return typeof value === 'string' && value.length > 0;
}

function isBinaryPayload(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.length > 0;
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return value.byteLength > 0;
  }
  return false;
}

function isContentPayload(value: unknown): boolean {
  return isArray(value) && value.length > 0;
}

function hasPayloadPath(
  record: Record<string, unknown>,
  path: string[],
  validator: (value: unknown) => boolean = isBinaryPayload
): boolean {
  let value: unknown = record;
  for (const key of path) {
    if (!isRecord(value)) {
      return false;
    }
    const property = readOwnEnumerableDataProperty(value, key);
    if (!property.found) {
      return false;
    }
    value = property.value;
  }
  return validator(value);
}

function hasAnyPayloadPath(
  record: Record<string, unknown>,
  paths: string[][],
  validator: (value: unknown) => boolean = isBinaryPayload
): boolean {
  return paths.some((path) => hasPayloadPath(record, path, validator));
}

function isProviderNativeToolResultBlock(
  record: Record<string, unknown>,
  type: string
): boolean {
  if (type === 'search_result') {
    return (
      hasString(record, 'title') &&
      hasString(record, 'source') &&
      hasPayloadPath(record, ['content'], isArray)
    );
  }
  if (type === 'web_search_result') {
    return hasString(record, 'url');
  }
  if (type === 'web_search_tool_result') {
    const content = readOwnEnumerableDataProperty(record, 'content');
    return (
      hasString(record, 'tool_use_id') &&
      content.found &&
      (isArray(content.value) || isRecord(content.value))
    );
  }
  if (type === 'tool_result') {
    return hasString(record, 'tool_use_id');
  }
  if (type === 'server_tool_call_result') {
    const status = readOwnEnumerableDataProperty(record, 'status');
    return (
      hasString(record, 'toolCallId') &&
      status.found &&
      (status.value === 'success' || status.value === 'error') &&
      hasValue(record, 'output')
    );
  }
  if (type === 'toolResponse') {
    const response = readOwnEnumerableDataProperty(record, 'toolResponse');
    return response.found && isRecord(response.value);
  }
  return false;
}

const MAX_ATOMIC_VALIDATION_WORK = 10_000;
const MAX_ATOMIC_VALIDATION_DEPTH = 50;
const NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER = Object.getOwnPropertyDescriptor(
  ArrayBuffer.prototype,
  'byteLength'
)?.get;
const NATIVE_DATE_GET_TIME = Date.prototype.getTime;
const NATIVE_DATE_TO_JSON = Date.prototype.toJSON;
const NATIVE_DATE_TO_ISO_STRING = Date.prototype.toISOString;
const NATIVE_DATE_VALUE_OF = Date.prototype.valueOf;
const NATIVE_DATE_TO_STRING = Date.prototype.toString;
const NATIVE_DATE_TO_PRIMITIVE = Date.prototype[Symbol.toPrimitive];
const NATIVE_DATE_PROTOTYPE_METHODS: ReadonlyArray<
  readonly [PropertyKey, unknown]
> = [
  ['getTime', NATIVE_DATE_GET_TIME],
  ['toJSON', NATIVE_DATE_TO_JSON],
  ['toISOString', NATIVE_DATE_TO_ISO_STRING],
  ['valueOf', NATIVE_DATE_VALUE_OF],
  ['toString', NATIVE_DATE_TO_STRING],
  [Symbol.toPrimitive, NATIVE_DATE_TO_PRIMITIVE],
];

function hasPotentiallyCallableToJSON(value: object): boolean {
  let current: object | null = value;
  const seen = new Set<object>();
  for (let depth = 0; current != null && depth < 100; depth++) {
    if (seen.has(current)) {
      return true;
    }
    seen.add(current);
    const descriptor = Object.getOwnPropertyDescriptor(current, 'toJSON');
    if (descriptor != null) {
      return (
        typeof descriptor.value === 'function' ||
        typeof descriptor.get === 'function'
      );
    }
    current = Object.getPrototypeOf(current) as object | null;
  }
  return current != null;
}

function isSafeNativeArrayBuffer(value: ArrayBuffer): boolean {
  try {
    if (
      Object.getPrototypeOf(value) !== ArrayBuffer.prototype ||
      NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER == null ||
      Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength')
        ?.get !== NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER ||
      hasPotentiallyCallableToJSON(value)
    ) {
      return false;
    }
    NATIVE_ARRAY_BUFFER_BYTE_LENGTH_GETTER.call(value);
    for (const key in value) {
      if (Object.prototype.propertyIsEnumerable.call(value, key)) {
        return false;
      }
    }
    return Object.getOwnPropertyDescriptor(value, 'byteLength') == null;
  } catch {
    return false;
  }
}

function isSafeNativeDate(value: Date): boolean {
  try {
    if (Object.getPrototypeOf(value) !== Date.prototype) {
      return false;
    }
    NATIVE_DATE_GET_TIME.call(value);
    for (const [key, nativeMethod] of NATIVE_DATE_PROTOTYPE_METHODS) {
      if (
        Object.getOwnPropertyDescriptor(value, key) != null ||
        Object.getOwnPropertyDescriptor(Date.prototype, key)?.value !==
          nativeMethod
      ) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

function hasUnsafeCallableToJSON(
  value: unknown,
  seen = new Set<object>(),
  remaining = { value: MAX_ATOMIC_VALIDATION_WORK },
  depth = 0
): boolean {
  if (isProxy(value)) {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  try {
    if (value instanceof ArrayBuffer) {
      return !isSafeNativeArrayBuffer(value);
    }
    if (value instanceof Date) {
      return !isSafeNativeDate(value);
    }
    if (ArrayBuffer.isView(value)) {
      return hasPotentiallyCallableToJSON(value);
    }
  } catch {
    return true;
  }
  if (seen.has(value)) {
    return true;
  }
  if (depth >= MAX_ATOMIC_VALIDATION_DEPTH || remaining.value <= 0) {
    return true;
  }

  remaining.value--;
  seen.add(value);
  try {
    if (hasPotentiallyCallableToJSON(value)) {
      return true;
    }
    if (isArray(value)) {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Array.prototype && prototype !== null) {
        return true;
      }
      const lengthDescriptor = Object.getOwnPropertyDescriptor(value, 'length');
      const length = lengthDescriptor?.value;
      if (
        typeof length !== 'number' ||
        !Number.isSafeInteger(length) ||
        length < 0 ||
        length > remaining.value
      ) {
        return true;
      }
      remaining.value -= length;
      for (let i = 0; i < length; i++) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(i));
        if (
          descriptor == null ||
          typeof descriptor.get === 'function' ||
          typeof descriptor.set === 'function'
        ) {
          return true;
        }
        if (
          hasUnsafeCallableToJSON(descriptor.value, seen, remaining, depth + 1)
        ) {
          return true;
        }
      }
      return false;
    }

    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      return true;
    }
    for (const key in value) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (descriptor == null) {
        return true;
      }
      if (descriptor.enumerable !== true) {
        continue;
      }
      if (
        remaining.value <= 0 ||
        typeof descriptor.get === 'function' ||
        typeof descriptor.set === 'function'
      ) {
        return true;
      }
      remaining.value--;
      if (
        hasUnsafeCallableToJSON(descriptor.value, seen, remaining, depth + 1)
      ) {
        return true;
      }
    }
    return false;
  } catch {
    return true;
  } finally {
    seen.delete(value);
  }
}

export function isAtomicToolContentBlock(value: unknown): boolean {
  try {
    return (
      !hasUnsafeCallableToJSON(value) &&
      isAtomicToolContentBlockUnchecked(value)
    );
  } catch {
    return false;
  }
}

function isAtomicToolContentBlockUnchecked(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  const record = value;
  const typeProperty = readOwnEnumerableDataProperty(record, 'type');
  if (!typeProperty.found) {
    const cachePointProperty = readOwnEnumerableDataProperty(
      record,
      'cachePoint'
    );
    if (!cachePointProperty.found || !isRecord(cachePointProperty.value)) {
      return false;
    }
    const cachePointType = readOwnEnumerableDataProperty(
      cachePointProperty.value,
      'type'
    );
    return cachePointType.found && cachePointType.value === 'default';
  }
  const type = typeof typeProperty.value === 'string' ? typeProperty.value : '';
  if (
    !ATOMIC_CONTENT_TYPES.has(type) &&
    !PROVIDER_NATIVE_TOOL_RESULT_TYPES.has(type) &&
    !(type.includes('/') && type.split('/').length === 2)
  ) {
    return false;
  }

  if (PROVIDER_NATIVE_TOOL_RESULT_TYPES.has(type)) {
    return isProviderNativeToolResultBlock(record, type);
  }

  if (type === 'computer_screenshot') {
    return hasPayloadPath(record, ['image_url'], isStringPayload);
  }
  if (type === 'image_url') {
    return hasAnyPayloadPath(
      record,
      [['image_url'], ['image_url', 'url']],
      isStringPayload
    );
  }
  if (type === 'image_file') {
    return hasAnyPayloadPath(
      record,
      [
        ['image_file', 'file_id'],
        ['image_file', 'fileId'],
      ],
      isStringPayload
    );
  }
  if (type === 'input_audio') {
    return hasAnyPayloadPath(record, [
      ['input_audio', 'data'],
      ['input_audio', 'bytes'],
    ]);
  }
  if (type === 'resource_link') {
    return hasString(record, 'uri');
  }
  if (type === 'video_url') {
    return hasAnyPayloadPath(
      record,
      [['video_url'], ['video_url', 'url']],
      isStringPayload
    );
  }
  if (type === 'resource') {
    return (
      hasAnyPayloadPath(record, [
        ['resource', 'blob'],
        ['resource', 'bytes'],
        ['resource', 'data'],
        ['resource', 'text'],
      ]) || hasPayloadPath(record, ['resource', 'content'], isContentPayload)
    );
  }

  if (type.includes('/') && type.split('/').length === 2) {
    return hasPayloadPath(record, ['data'], isStringPayload);
  }

  if (type === 'image' || type === 'audio' || type === 'video') {
    return hasAnyPayloadPath(record, [
      ['data'],
      ['bytes'],
      ['url'],
      ['source', 'data'],
      ['source', 'bytes'],
      ['source', 'url'],
      [type, 'data'],
      [type, 'bytes'],
      [type, 'url'],
      [type, 'source', 'data'],
      [type, 'source', 'bytes'],
      [type, 'source', 'url'],
    ]);
  }

  if (type === 'media') {
    const hasMimeType =
      hasString(record, 'mimeType') || hasString(record, 'mime_type');
    return (
      hasMimeType &&
      hasAnyPayloadPath(record, [
        ['data'],
        ['bytes'],
        ['fileUri'],
        ['media', 'data'],
        ['media', 'bytes'],
        ['media', 'fileUri'],
        ['media', 'source', 'bytes'],
      ])
    );
  }

  if (type === 'document' || type === 'file') {
    const nestedType = type;
    return (
      hasAnyPayloadPath(record, [
        ['data'],
        ['bytes'],
        ['url'],
        ['file_data'],
        ['file_id'],
        ['fileId'],
        ['fileUri'],
        ['source', 'data'],
        ['source', 'bytes'],
        ['source', 'url'],
        ['source', 'file_data'],
        ['source', 'file_id'],
        ['source', 'fileId'],
        ['source', 'fileUri'],
        [nestedType, 'data'],
        [nestedType, 'bytes'],
        [nestedType, 'url'],
        [nestedType, 'file_data'],
        [nestedType, 'file_id'],
        [nestedType, 'fileId'],
        [nestedType, 'fileUri'],
        [nestedType, 'source', 'data'],
        [nestedType, 'source', 'bytes'],
        [nestedType, 'source', 'url'],
      ]) ||
      hasPayloadPath(record, ['source', 'content'], isContentPayload) ||
      hasPayloadPath(
        record,
        [nestedType, 'source', 'content'],
        isContentPayload
      ) ||
      (readOwnEnumerableDataProperty(record, 'source_type').value === 'text' &&
        hasString(record, 'text')) ||
      (readOwnEnumerableDataProperty(record, 'source_type').value === 'id' &&
        hasString(record, 'id'))
    );
  }

  return false;
}

/**
 * Produces the text representation providers use for structured tool results.
 * Text-only block arrays stay readable; opaque blocks use canonical JSON.
 */
export function serializeToolContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  if (!isArray(content)) {
    return serializeStructuredValue(content);
  }
  if (getDenseTextBlockArrayLength(content) != null) {
    return (content as TextToolContentBlock[])
      .map((block) => block.text)
      .join('\n');
  }
  return serializeStructuredValue(content);
}

export function getToolContentCharLength(content: unknown): number {
  if (typeof content === 'string') {
    return content.length;
  }
  if (isArray(content)) {
    const denseTextLength = getDenseTextBlockArrayLength(content);
    if (denseTextLength != null) {
      return denseTextLength;
    }
  }
  try {
    return estimateStructuredChars(content);
  } catch {
    return Number.MAX_SAFE_INTEGER;
  }
}

/**
 * Produces provider-facing tool-result text without ever exceeding `maxChars`.
 */
export function serializeToolContentBounded(
  content: unknown,
  maxChars: number
): string {
  const normalizedMaxChars = normalizeCharLimit(maxChars);
  if (typeof content === 'string') {
    return truncateToolResultContent(content, normalizedMaxChars);
  }
  if (!isArray(content)) {
    return serializeStructuredValueWithinLimit(content, normalizedMaxChars);
  }

  const originalChars = getDenseTextBlockArrayLength(content);
  if (originalChars == null) {
    return serializeStructuredValueWithinLimit(content, normalizedMaxChars);
  }
  if (originalChars <= normalizedMaxChars) {
    return (content as TextToolContentBlock[])
      .map((block) => block.text)
      .join('\n');
  }
  const indicator =
    `\n\n… [truncated: ${originalChars} chars exceeded ` +
    `${normalizedMaxChars} limit] …`;
  const available = Math.max(0, normalizedMaxChars - indicator.length);
  let preview = '';
  const textBlocks = content as TextToolContentBlock[];
  for (let i = 0; i < content.length && preview.length < available; i++) {
    if (i > 0) {
      preview += '\n';
    }
    preview += textBlocks[i].text.slice(0, available - preview.length);
  }
  return (preview + indicator).slice(0, normalizedMaxChars);
}

export type CompactToolContentResult = {
  content: ToolContent;
  changed: boolean;
  originalChars: number;
};

/**
 * Bounds structured tool output without slicing binary/media payloads.
 * Purely textual/structured arrays become a universally valid string tool
 * result. For mixed-media arrays, compactable blocks become one text preview
 * while atomic blocks remain intact.
 */
export function compactToolContent(
  content: unknown,
  maxChars: number
): CompactToolContentResult {
  const normalizedMaxChars = Number.isFinite(maxChars)
    ? Math.max(0, Math.floor(maxChars))
    : Number.MAX_SAFE_INTEGER;
  if (typeof content === 'string') {
    return {
      content: truncateToolResultContent(content, normalizedMaxChars),
      changed: content.length > normalizedMaxChars,
      originalChars: content.length,
    };
  }
  if (!isArray(content)) {
    const serialized = serializeStructuredValueBounded(
      content,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
  if (isProxy(content)) {
    const serialized = serializeStructuredValueBounded(
      content,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
  const contentBlocks = content as ToolContentBlock[];

  let contentLength: number;
  try {
    contentLength = contentBlocks.length;
  } catch {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: Number.MAX_SAFE_INTEGER,
    };
  }
  if (contentLength > MAX_STRUCTURED_SERIALIZATION_WORK) {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }

  try {
    const originalChars = getToolContentCharLength(contentBlocks);
    const denseTextLength = getDenseTextBlockArrayLength(contentBlocks);
    const atomicBlocks = contentBlocks.filter(isAtomicToolContentBlock);
    if (atomicBlocks.length === 0) {
      if (denseTextLength != null && originalChars <= normalizedMaxChars) {
        return {
          content: contentBlocks,
          changed: false,
          originalChars,
        };
      }
      const serialized = serializeToolContentBounded(
        contentBlocks,
        normalizedMaxChars
      );
      return {
        content: truncateToolResultContent(serialized, normalizedMaxChars),
        // Opaque arrays are normalized even when they are small so every
        // provider receives a universally valid tool-result string.
        changed: true,
        originalChars,
      };
    }

    const hasOpaqueBlocks = contentBlocks.some(
      (block) => !isAtomicToolContentBlock(block) && !isTextBlock(block)
    );
    const normalizedBlocks: ToolContentBlock[] = [];
    let opaqueRun: unknown[] = [];
    const flushOpaqueRun = (): void => {
      if (opaqueRun.length === 0) {
        return;
      }
      normalizedBlocks.push({
        type: 'text',
        text: serializeStructuredValueWithinLimit(
          opaqueRun,
          normalizedMaxChars
        ),
      });
      opaqueRun = [];
    };
    for (const block of contentBlocks) {
      if (isAtomicToolContentBlock(block) || isTextBlock(block)) {
        flushOpaqueRun();
        normalizedBlocks.push(block);
      } else {
        opaqueRun.push(block);
      }
    }
    flushOpaqueRun();

    const normalizedLength = getToolContentCharLength(normalizedBlocks);
    if (normalizedLength <= normalizedMaxChars) {
      return {
        content: hasOpaqueBlocks ? normalizedBlocks : contentBlocks,
        changed: hasOpaqueBlocks,
        originalChars,
      };
    }

    // Keep small media/resource blocks intact, but never let a single inline
    // base64/blob block defeat the cap. Half of the budget is reserved for
    // compactable text so the model still receives an explanation.
    const atomicBudget = Math.floor(normalizedMaxChars / 2);
    let atomicChars = 0;
    const preservedAtomicBlocks = new Set<ToolContentBlock>();
    const omittedAtomicTypes: string[] = [];
    for (const block of normalizedBlocks) {
      if (!isAtomicToolContentBlock(block)) {
        continue;
      }
      const blockChars = estimateStructuredChars(
        block,
        atomicBudget - atomicChars
      );
      if (atomicChars + blockChars <= atomicBudget) {
        preservedAtomicBlocks.add(block);
        atomicChars += blockChars;
      } else {
        const typeProperty = isRecord(block)
          ? readOwnEnumerableDataProperty(block, 'type')
          : { found: false as const };
        omittedAtomicTypes.push(
          typeProperty.found && typeof typeProperty.value === 'string'
            ? typeProperty.value
            : 'media'
        );
      }
    }

    const compactableBlocks = normalizedBlocks.filter(
      (block) => !isAtomicToolContentBlock(block)
    );
    let previewSource = serializeToolContentBounded(
      compactableBlocks,
      normalizedMaxChars
    );
    if (omittedAtomicTypes.length > 0) {
      const omittedNotice =
        `[omitted ${omittedAtomicTypes.length} oversized atomic tool-content ` +
        `block${omittedAtomicTypes.length === 1 ? '' : 's'}: ` +
        `${omittedAtomicTypes.join(', ')}]`;
      previewSource =
        previewSource.length > 0
          ? `${previewSource}\n${omittedNotice}`
          : omittedNotice;
    }

    const structuralAllowance =
      preservedAtomicBlocks.size * 4 + (previewSource.length > 0 ? 32 : 2);
    const previewBudget = Math.max(
      0,
      normalizedMaxChars - atomicChars - structuralAllowance
    );
    const preview = truncateToolResultContent(previewSource, previewBudget);
    const compacted: ToolContentBlock[] = [];
    let previewInserted = false;
    for (const block of normalizedBlocks) {
      if (isAtomicToolContentBlock(block)) {
        if (preservedAtomicBlocks.has(block)) {
          compacted.push(block);
        }
      } else if (!previewInserted) {
        if (preview.length > 0) {
          compacted.push({ type: 'text', text: preview });
        }
        previewInserted = true;
      }
    }
    if (!previewInserted && preview.length > 0) {
      compacted.push({ type: 'text', text: preview });
    }

    // JSON framing can make the block array a few bytes larger than the
    // character budget. Fall back to bounded provider-neutral text rather than
    // leaking an oversized media payload.
    if (getToolContentCharLength(compacted) > normalizedMaxChars) {
      return {
        content: serializeToolContentBounded(
          normalizedBlocks,
          normalizedMaxChars
        ),
        changed: true,
        originalChars,
      };
    }

    return {
      content: compacted,
      changed: true,
      originalChars,
    };
  } catch {
    const serialized = serializeStructuredValueBounded(
      contentBlocks,
      normalizedMaxChars
    );
    return {
      content: serialized.content,
      changed: true,
      originalChars: serialized.originalChars,
    };
  }
}

/** Clones a ToolMessage while retaining fields omitted by ad-hoc clones. */
export function cloneToolMessageWithContent(
  message: ToolMessage,
  content: ToolContent,
  artifact: unknown = message.artifact
): ToolMessage {
  return new ToolMessage({
    content,
    tool_call_id: message.tool_call_id,
    name: message.name,
    id: message.id,
    status: message.status,
    artifact,
    metadata: message.metadata,
    additional_kwargs: message.additional_kwargs,
    response_metadata: message.response_metadata,
  });
}
