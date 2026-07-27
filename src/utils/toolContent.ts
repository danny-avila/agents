import { ToolMessage, type BaseMessage } from '@langchain/core/messages';
import { truncateToolResultContent } from './truncation';

type ToolContent = BaseMessage['content'];
type ToolContentBlock = Exclude<ToolContent, string>[number];

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
    return String(value);
  }
}

function jsonStringLength(value: string): number {
  let length = 2;
  for (let i = 0; i < value.length; i++) {
    const code = value.charCodeAt(i);
    if (code === 0x22 || code === 0x5c || code === 0x08 || code === 0x0c) {
      length += 2;
    } else if (code < 0x20) {
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
  ancestors: object[] = []
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
  if (typeof value === 'undefined' || typeof value === 'function') {
    return 4;
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return Math.min(limit + 1, Math.ceil((value.byteLength * 4) / 3) + 32);
  }
  if (typeof value !== 'object') {
    return jsonStringLength(String(value));
  }
  if (ancestors.includes(value)) {
    return 12;
  }

  ancestors.push(value);
  let length = 2;
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length && length <= limit; i++) {
      if (i > 0) {
        length++;
      }
      length += estimateStructuredChars(
        value[i],
        Math.max(0, limit - length),
        ancestors
      );
    }
  } else {
    let emitted = 0;
    for (const key of Object.keys(value)) {
      if (length > limit) {
        break;
      }
      const nested = (value as Record<string, unknown>)[key];
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
        ancestors
      );
      emitted++;
    }
  }
  ancestors.pop();
  return Math.min(limit + 1, length);
}

type PreviewState = {
  remaining: number;
  ancestors: object[];
};

function createStructuredPreview(value: unknown, state: PreviewState): unknown {
  if (typeof value === 'string') {
    const available = Math.max(0, state.remaining - 24);
    const preview =
      value.length > available
        ? `${value.slice(0, available)}…[truncated]`
        : value;
    state.remaining = Math.max(0, state.remaining - preview.length);
    return preview;
  }
  if (typeof value === 'bigint') {
    return value.toString();
  }
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    const name =
      value instanceof ArrayBuffer ? 'ArrayBuffer' : value.constructor.name;
    return `[${name}: ${value.byteLength} bytes]`;
  }
  if (value == null || typeof value !== 'object') {
    return value;
  }
  if (state.ancestors.includes(value)) {
    return '[Circular]';
  }

  state.ancestors.push(value);
  if (Array.isArray(value)) {
    const preview: unknown[] = [];
    let i = 0;
    for (; i < value.length && state.remaining > 48; i++) {
      state.remaining -= 2;
      preview.push(createStructuredPreview(value[i], state));
    }
    if (i < value.length) {
      preview.push(`[truncated ${value.length - i} item(s)]`);
    }
    state.ancestors.pop();
    return preview;
  }

  const preview: Record<string, unknown> = {};
  const entries = Object.entries(value);
  let i = 0;
  for (; i < entries.length && state.remaining > 48; i++) {
    const [key, nested] = entries[i];
    state.remaining -= key.length + 4;
    preview[key] = createStructuredPreview(nested, state);
  }
  if (i < entries.length) {
    preview._truncated = `${entries.length - i} field(s)`;
  }
  state.ancestors.pop();
  return preview;
}

function serializeStructuredValueWithinLimit(
  value: unknown,
  maxChars: number
): string {
  const normalizedMaxChars = Math.max(0, Math.floor(maxChars));
  if (
    estimateStructuredChars(value, normalizedMaxChars) <= normalizedMaxChars
  ) {
    return serializeStructuredValue(value);
  }
  const preview = createStructuredPreview(value, {
    remaining: normalizedMaxChars,
    ancestors: [],
  });
  return truncateToolResultContent(
    serializeStructuredValue(preview),
    normalizedMaxChars
  );
}

function isTextBlock(
  value: unknown
): value is ToolContentBlock & { type: 'text'; text: string } {
  if (typeof value !== 'object' || value == null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return record.type === 'text' && typeof record.text === 'string';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null;
}

function hasString(record: Record<string, unknown>, key: string): boolean {
  return typeof record[key] === 'string' && record[key] !== '';
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
  return Array.isArray(value) && value.length > 0;
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
    value = value[key];
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

export function isAtomicToolContentBlock(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  const record = value;
  if (!('type' in record)) {
    const cachePoint = record.cachePoint;
    return isRecord(cachePoint) && cachePoint.type === 'default';
  }
  const type = typeof record.type === 'string' ? record.type : '';
  if (
    !ATOMIC_CONTENT_TYPES.has(type) &&
    !(type.includes('/') && type.split('/').length === 2)
  ) {
    return false;
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
      (record.source_type === 'text' && hasString(record, 'text')) ||
      (record.source_type === 'id' && hasString(record, 'id'))
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
  if (!Array.isArray(content)) {
    return serializeStructuredValue(content);
  }
  if (content.length > 0 && content.every(isTextBlock)) {
    return content.map((block) => block.text).join('\n');
  }
  return serializeStructuredValue(content);
}

export function getToolContentCharLength(content: unknown): number {
  if (typeof content === 'string') {
    return content.length;
  }
  if (
    Array.isArray(content) &&
    content.length > 0 &&
    content.every(isTextBlock)
  ) {
    let length = content.length - 1;
    for (const block of content) {
      length += block.text.length;
    }
    return length;
  }
  return estimateStructuredChars(content);
}

function serializeToolContentWithinLimit(
  content: unknown,
  maxChars: number
): string {
  const normalizedMaxChars = Math.max(0, Math.floor(maxChars));
  if (typeof content === 'string') {
    return truncateToolResultContent(content, normalizedMaxChars);
  }
  if (
    !Array.isArray(content) ||
    content.length === 0 ||
    !content.every(isTextBlock)
  ) {
    return serializeStructuredValueWithinLimit(content, normalizedMaxChars);
  }

  const originalChars = getToolContentCharLength(content);
  if (originalChars <= normalizedMaxChars) {
    return content.map((block) => block.text).join('\n');
  }
  const indicator =
    `\n\n… [truncated: ${originalChars} chars exceeded ` +
    `${normalizedMaxChars} limit] …`;
  const available = Math.max(0, normalizedMaxChars - indicator.length);
  let preview = '';
  for (let i = 0; i < content.length && preview.length < available; i++) {
    if (i > 0) {
      preview += '\n';
    }
    preview += content[i].text.slice(0, available - preview.length);
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
  if (!Array.isArray(content)) {
    const originalChars = estimateStructuredChars(content);
    const serialized = serializeStructuredValueWithinLimit(
      content,
      normalizedMaxChars
    );
    return {
      content: serialized,
      changed: true,
      originalChars,
    };
  }

  const originalChars = getToolContentCharLength(content);
  const atomicBlocks = content.filter(isAtomicToolContentBlock);
  if (atomicBlocks.length === 0) {
    if (
      content.length > 0 &&
      content.every(isTextBlock) &&
      originalChars <= normalizedMaxChars
    ) {
      return {
        content,
        changed: false,
        originalChars,
      };
    }
    const serialized = serializeToolContentWithinLimit(
      content,
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

  const hasOpaqueBlocks = content.some(
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
      text: serializeStructuredValueWithinLimit(opaqueRun, normalizedMaxChars),
    });
    opaqueRun = [];
  };
  for (const block of content) {
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
      content: hasOpaqueBlocks ? normalizedBlocks : content,
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
      const record = block as Record<string, unknown>;
      omittedAtomicTypes.push(
        typeof record.type === 'string' ? record.type : 'media'
      );
    }
  }

  const compactableBlocks = normalizedBlocks.filter(
    (block) => !isAtomicToolContentBlock(block)
  );
  let previewSource = serializeToolContentWithinLimit(
    compactableBlocks,
    normalizedMaxChars
  );
  if (omittedAtomicTypes.length > 0) {
    const omittedNotice =
      `[omitted ${omittedAtomicTypes.length} oversized media/resource ` +
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
      content: serializeToolContentWithinLimit(
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
