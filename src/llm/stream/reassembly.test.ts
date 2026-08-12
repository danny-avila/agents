import { concat } from '@langchain/core/utils/stream';
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import type { SmoothItem } from './smoother';
import { toSmoothItem } from '@/llm/openai';
import {
  toGenerationSmoothItem,
  getReasoningKwargsText,
} from './chunkAdapters';
import { findStreamChunkBoundary } from './smoother';

/**
 * Lossless-reassembly property: for ANY chunk shape an adapter classifies,
 * aggregating the emitted pieces must reproduce the original chunk's
 * observable payload — text, content, kwargs, response metadata, usage,
 * tool calls. This is the invariant behind every split-corruption bug class
 * (duplicated reasoning/kwargs, summed usage, cloned tool args): if a field
 * cannot survive slicing, the adapter must classify the chunk atomic or
 * scope the field to a single piece, and this property proves it did.
 */

type AdapterName = 'openai-family' | 'generic';

const ADAPTERS: Record<
  AdapterName,
  (chunk: ChatGenerationChunk) => SmoothItem<ChatGenerationChunk>
> = {
  'openai-family': (chunk) => toSmoothItem(chunk),
  generic: (chunk) => toGenerationSmoothItem(chunk, getReasoningKwargsText),
};

/** Emits the item the way the engine would: split smooth+splittable items
 * into ≥2 boundary-aligned pieces; atomic and passthrough emit whole. */
function emitPieces(item: SmoothItem<ChatGenerationChunk>): ChatGenerationChunk[] {
  if (!item.smooth || item.atomic === true || item.text.length < 8) {
    return [item.emit({ text: item.text, isFirst: true, isLast: true })];
  }
  const pieces: ChatGenerationChunk[] = [];
  let offset = 0;
  while (offset < item.text.length) {
    const end = offset + findStreamChunkBoundary(item.text.slice(offset), 4);
    pieces.push(
      item.emit({
        text: item.text.slice(offset, end),
        isFirst: offset === 0,
        isLast: end === item.text.length,
      })
    );
    offset = end;
  }
  return pieces;
}

function aggregate(pieces: ChatGenerationChunk[]): AIMessageChunk {
  let merged = pieces[0].message as AIMessageChunk;
  for (const piece of pieces.slice(1)) {
    merged = concat(merged, piece.message as AIMessageChunk);
  }
  return merged;
}

function assertLossless(
  adapter: AdapterName,
  chunk: ChatGenerationChunk
): void {
  const item = ADAPTERS[adapter](chunk);
  const pieces = emitPieces(item);
  const original = chunk.message as AIMessageChunk;

  const joinedText = pieces.map((p) => p.text).join('');
  if (item.smooth) {
    expect(joinedText === chunk.text || joinedText === item.text).toBe(true);
  }

  const merged = aggregate(pieces);
  expect(merged.content).toEqual(original.content);
  expect(merged.additional_kwargs).toEqual(original.additional_kwargs);
  expect(merged.response_metadata).toEqual(original.response_metadata);
  /** concat() adds empty *_token_details objects during merge; token COUNTS
   * are the corruption signal (replicated usage would sum). */
  expect(merged.usage_metadata?.input_tokens).toBe(
    original.usage_metadata?.input_tokens
  );
  expect(merged.usage_metadata?.output_tokens).toBe(
    original.usage_metadata?.output_tokens
  );
  expect(merged.usage_metadata?.total_tokens).toBe(
    original.usage_metadata?.total_tokens
  );
  expect(merged.tool_call_chunks ?? []).toEqual(
    original.tool_call_chunks ?? []
  );
}

type MatrixCase = {
  name: string;
  chunk: () => ChatGenerationChunk;
  adapters: AdapterName[];
};

const TEXT = 'the quick brown fox jumps over the lazy dog again and again ';

const MATRIX: MatrixCase[] = [
  {
    name: 'plain string text',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({ content: TEXT }),
      }),
  },
  {
    name: 'text with usage metadata',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          usage_metadata: { input_tokens: 7, output_tokens: 11, total_tokens: 18 },
        }),
      }),
  },
  {
    name: 'text with an unknown string kwargs field',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          additional_kwargs: { annotation: 'gateway-added-value' },
        }),
      }),
  },
  {
    name: 'text with scalar response metadata',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          response_metadata: { model_name: 'test-model', system_fingerprint: 'fp_1' },
        }),
      }),
  },
  {
    name: 'text with reasoning_content kwargs',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          additional_kwargs: { reasoning_content: 'a hidden thought' },
        }),
      }),
  },
  {
    name: 'text with reasoning_details kwargs',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          additional_kwargs: {
            reasoning_details: [{ type: 'reasoning.text', text: 'thought' }],
          },
        }),
      }),
  },
  {
    name: 'text alongside tool_call_chunks',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: TEXT,
          tool_call_chunks: [
            { name: 'lookup', args: '{"q":1}', id: 'call_1', index: 0, type: 'tool_call_chunk' },
          ],
        }),
      }),
  },
  {
    name: 'indexed text-part array content (google shape)',
    adapters: ['generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: [{ type: 'text', text: TEXT, index: 0 }],
        }),
      }),
  },
  {
    name: 'index-less text-part array content (no merge key)',
    adapters: ['generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: TEXT,
        message: new AIMessageChunk({
          content: [{ type: 'text', text: TEXT }],
        }),
      }),
  },
  {
    name: 'reasoning-only delta',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          additional_kwargs: { reasoning_content: 'thought only delta here' },
        }),
      }),
  },
  {
    name: 'usage-only delta',
    adapters: ['openai-family', 'generic'],
    chunk: () =>
      new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
        }),
      }),
  },
];

describe.each(MATRIX)('lossless reassembly: $name', ({ chunk, adapters }) => {
  it.each(adapters)('%s adapter', (adapter) => {
    assertLossless(adapter, chunk());
  });
});
