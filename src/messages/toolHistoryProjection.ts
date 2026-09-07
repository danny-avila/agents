import { isProxy } from 'node:util/types';
import type { BaseMessage, ContentBlock } from '@langchain/core/messages';
import { getProviderToolCallPartDescriptor } from './toolResultTypes';
import { serializeStructuredValueBounded } from '@/utils/toolContent';

interface ToolHistoryCallMirror {
  readonly name?: string;
  readonly arguments: string;
}

export type ToolHistoryCallMirrors = Map<string, ToolHistoryCallMirror | null>;

/** Only complete argument representations can establish that a call is a mirror. */
export function recordToolHistoryCallMirror(
  mirrors: ToolHistoryCallMirrors,
  block: unknown
): void {
  const descriptor = getProviderToolCallPartDescriptor(block);
  if (descriptor == null) {
    return;
  }
  if (mirrors.has(descriptor.callId)) {
    mirrors.set(descriptor.callId, null);
    return;
  }
  const call =
    readData(block, 'tool_call') ??
    readData(block, 'toolUse') ??
    readData(block, 'toolCall') ??
    block;
  const args = readData(call, 'args') ?? readData(call, 'input');
  const serialized = serializeStructuredValueBounded(args, 8000);
  mirrors.set(
    descriptor.callId,
    serialized.truncated
      ? null
      : {
        name: descriptor.name,
        arguments: serialized.content,
      }
  );
}

export function isToolHistoryCallMirror(
  mirrors: ToolHistoryCallMirrors | undefined,
  id: unknown,
  name: unknown,
  args: unknown
): boolean {
  if (typeof id !== 'string') {
    return false;
  }
  const mirror = mirrors?.get(id);
  if (mirror == null || mirror.name !== name) {
    return false;
  }
  const serialized = serializeStructuredValueBounded(args, 8000);
  return !serialized.truncated && serialized.content === mirror.arguments;
}

export const OPENAI_RESPONSES_REPLAY_POSITIONS_KEY =
  '__openai_responses_replay_positions__';

export type ResponsesReplayPosition = {
  contentIndex?: number;
  itemId: string;
  kind: 'message' | 'output' | 'reasoning' | 'text';
  outputIndex: number;
};

export function isResponsesReplayPosition(
  value: unknown
): value is ResponsesReplayPosition {
  const kind = readData(value, 'kind');
  const itemId = readData(value, 'itemId');
  const outputIndex = readData(value, 'outputIndex');
  const contentIndex = readData(value, 'contentIndex');
  return (
    (kind === 'message' ||
      kind === 'output' ||
      kind === 'reasoning' ||
      kind === 'text') &&
    typeof itemId === 'string' &&
    itemId.length > 0 &&
    typeof outputIndex === 'number' &&
    Number.isSafeInteger(outputIndex) &&
    outputIndex >= 0 &&
    (contentIndex == null ||
      (typeof contentIndex === 'number' &&
        Number.isSafeInteger(contentIndex) &&
        contentIndex >= 0))
  );
}

/** Provider evidence is selected once; a sidecar never replaces message content. */
export interface ResponsesHistorySource {
  readonly coverage: 'complete-output' | 'tool-sidecar';
  readonly items: unknown[];
  readonly message?: BaseMessage;
}

export interface OrderedToolHistoryProjection {
  readonly source: ResponsesHistorySource;
  readonly contributions: readonly ResponsesHistoryContribution[];
  readonly hasToolContent: boolean;
  readonly truncated: boolean;
}

export interface ToolHistoryPreparation {
  get(message: BaseMessage): OrderedToolHistoryProjection | undefined;
}

/** A preparation owns this cache; changed messages must be copy-on-write. */
export function createToolHistoryPreparation(): ToolHistoryPreparation {
  const projections = new WeakMap<BaseMessage, OrderedToolHistoryProjection>();
  let remainingWork = 100_000;
  return {
    get(message): OrderedToolHistoryProjection | undefined {
      const cached = projections.get(message);
      if (cached != null) {
        return cached;
      }
      const source = getResponsesHistorySource(message);
      if (source == null || source.items.length === 0) {
        return undefined;
      }
      const contributions: ResponsesHistoryContribution[] = [];
      let hasToolContent = false;
      let truncated = false;
      for (const contribution of projectResponsesHistory(source, () => {
        if (remainingWork <= 0) {
          truncated = true;
          return false;
        }
        remainingWork--;
        return true;
      })) {
        contributions.push(contribution);
        hasToolContent ||=
          contribution.kind !== 'text' &&
          !(
            contribution.kind === 'provider-item' &&
            contribution.providerType === 'reasoning'
          );
      }
      const projection = {
        source,
        contributions,
        hasToolContent: hasToolContent || truncated,
        truncated,
      };
      projections.set(message, projection);
      return projection;
    },
  };
}

export interface ToolHistoryPosition {
  readonly outputIndex: number;
  readonly contentIndex?: number;
  readonly itemId?: string;
}

export type ResponsesHistoryContribution = ToolHistoryPosition &
  (
    | { readonly kind: 'text'; readonly actor: 'model'; readonly text: string }
    | {
        readonly kind: 'call';
        readonly actor: 'model';
        readonly name: string;
        readonly callId?: string;
        readonly arguments: unknown;
      }
    | {
        readonly kind: 'image';
        readonly actor: 'tool';
        readonly image: ContentBlock.Multimodal.Image;
      }
    | {
        readonly kind: 'provider-item';
        readonly actor: 'model' | 'tool';
        readonly providerType: string;
        readonly value: unknown;
      }
  );

function readData(value: unknown, key: string): unknown {
  if (value == null || typeof value !== 'object' || isProxy(value)) {
    return undefined;
  }
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  return descriptor != null && 'value' in descriptor
    ? descriptor.value
    : undefined;
}

function readArray(value: unknown): unknown[] | undefined {
  return Array.isArray(value) && !isProxy(value) ? value : undefined;
}

export function getResponsesHistorySource(
  message: BaseMessage
): ResponsesHistorySource | undefined {
  const output = readArray(readData(message.response_metadata, 'output'));
  if (output != null && output.length > 0) {
    return { coverage: 'complete-output', items: output, message };
  }
  const sidecar = readArray(
    readData(message.additional_kwargs, 'tool_outputs')
  );
  return sidecar != null
    ? { coverage: 'tool-sidecar', items: sidecar, message }
    : undefined;
}

/** Complete generated images retain their actual format across replay and folding. */
export function getGeneratedImageMimeType(data: string): string {
  const bytes = Buffer.from(data.slice(0, 16), 'base64');
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return 'image/webp';
  }
  return 'image/png';
}

/**
 * Lazy traversal bounds outer items and nested content together. The caller's
 * shared work allowance also spans subsequent messages. False stops before the
 * next provider value is inspected; payload serialization remains consumer policy.
 */
function* projectResponsesItems(
  source: ResponsesHistorySource,
  consumeWork: () => boolean
): Generator<ResponsesHistoryContribution> {
  for (let outputIndex = 0; outputIndex < source.items.length; outputIndex++) {
    if (!consumeWork()) {
      return;
    }
    const item = readData(source.items, String(outputIndex));
    const type = readData(item, 'type');
    if (typeof type !== 'string') {
      continue;
    }
    const id = readData(item, 'id');
    const position: ToolHistoryPosition = {
      outputIndex,
      ...(typeof id === 'string' && { itemId: id }),
    };
    if (type === 'message') {
      const content = readArray(readData(item, 'content'));
      if (content == null) {
        continue;
      }
      for (
        let contentIndex = 0;
        contentIndex < content.length;
        contentIndex++
      ) {
        if (!consumeWork()) {
          return;
        }
        const part = readData(content, String(contentIndex));
        const partType = readData(part, 'type');
        const text = readData(
          part,
          partType === 'refusal' ? 'refusal' : 'text'
        );
        if (typeof text === 'string' && text.length > 0) {
          yield {
            ...position,
            contentIndex,
            kind: 'text',
            actor: 'model',
            text,
          };
        }
      }
      continue;
    }
    if (type === 'computer_call') {
      const callId = readData(item, 'call_id');
      yield {
        ...position,
        kind: 'call',
        actor: 'model',
        name: 'computer',
        arguments: readData(item, 'action'),
        ...(typeof callId === 'string' && { callId }),
      };
      continue;
    }
    if (type === 'function_call' || type === 'custom_tool_call') {
      const name = readData(item, 'name');
      const args = readData(
        item,
        type === 'custom_tool_call' ? 'input' : 'arguments'
      );
      const callId = readData(item, 'call_id');
      yield {
        ...position,
        kind: 'call',
        actor: 'model',
        name: typeof name === 'string' ? name : '',
        arguments: typeof args === 'string' ? args : '',
        ...(typeof callId === 'string' && { callId }),
      };
      continue;
    }
    if (type === 'image_generation_call') {
      const data = readData(item, 'result');
      if (
        readData(item, 'status') === 'completed' &&
        typeof data === 'string' &&
        data
      ) {
        yield {
          ...position,
          kind: 'image',
          actor: 'tool',
          image: {
            type: 'image',
            mimeType: getGeneratedImageMimeType(data),
            data,
            ...(typeof id === 'string' && { id }),
            metadata: { status: 'completed' },
          },
        };
      }
      if (
        readData(item, 'status') !== 'completed' ||
        typeof data !== 'string' ||
        !data
      ) {
        yield {
          ...position,
          kind: 'provider-item',
          actor: 'tool',
          providerType: type,
          value: {
            type,
            status: readData(item, 'status'),
            result: '[image unavailable]',
          },
        };
      }
      continue;
    }
    yield {
      ...position,
      kind: 'provider-item',
      actor:
        type === 'reasoning' || type === 'mcp_approval_request'
          ? 'model'
          : 'tool',
      providerType: type,
      value: item,
    };
  }
}

/** Streaming positions order sidecar evidence only when text positions map exactly. */
export function* projectResponsesHistory(
  source: ResponsesHistorySource,
  consumeWork: () => boolean
): Generator<ResponsesHistoryContribution> {
  if (source.coverage === 'complete-output' || source.message == null) {
    yield* projectResponsesItems(source, consumeWork);
    return;
  }
  const positions = readArray(
    readData(
      source.message.additional_kwargs,
      OPENAI_RESPONSES_REPLAY_POSITIONS_KEY
    )
  );
  const outputPositions = new Map<string, number>();
  const textPositions: ResponsesReplayPosition[] = [];
  const seenPositions = new Set<string>();
  for (let index = 0; positions != null && index < positions.length; index++) {
    if (!consumeWork()) {
      return;
    }
    const position = readData(positions, String(index));
    if (!isResponsesReplayPosition(position)) {
      continue;
    }
    if (position.kind === 'output') {
      outputPositions.set(position.itemId, position.outputIndex);
    } else if (position.kind === 'text') {
      const key = `${position.itemId}:${position.outputIndex}:${position.contentIndex ?? 0}`;
      if (!seenPositions.has(key)) {
        seenPositions.add(key);
        textPositions.push(position);
      }
    }
  }
  textPositions.sort(
    (left, right) =>
      left.outputIndex - right.outputIndex ||
      (left.contentIndex ?? 0) - (right.contentIndex ?? 0)
  );
  const text: string[] = [];
  const content = source.message.content;
  if (typeof content === 'string' && content) {
    text.push(content);
  } else if (Array.isArray(content)) {
    for (let index = 0; index < content.length; index++) {
      if (!consumeWork()) {
        return;
      }
      const block = readData(content, String(index));
      const value = readData(block, 'text');
      if (typeof value === 'string' && value) {
        text.push(value);
      }
    }
  }
  const contributions: ResponsesHistoryContribution[] = [];
  for (let index = 0; index < text.length; index++) {
    const position =
      text.length === textPositions.length ? textPositions[index] : undefined;
    contributions.push({
      kind: 'text',
      actor: 'model',
      text: text[index],
      outputIndex: position?.outputIndex ?? -1,
      contentIndex: position?.contentIndex ?? index,
      ...(position != null && { itemId: position.itemId }),
    });
  }
  for (const contribution of projectResponsesItems(source, consumeWork)) {
    contributions.push({
      ...contribution,
      outputIndex:
        contribution.itemId != null
          ? (outputPositions.get(contribution.itemId) ??
            Number.MAX_SAFE_INTEGER)
          : Number.MAX_SAFE_INTEGER,
    });
  }
  contributions.sort(
    (left, right) =>
      left.outputIndex - right.outputIndex ||
      (left.contentIndex ?? 0) - (right.contentIndex ?? 0)
  );
  for (const contribution of contributions) {
    yield contribution;
  }
}
