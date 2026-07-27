import type { BaseMessage } from '@langchain/core/messages';
import {
  compactToolContent,
  isAtomicToolContentBlock,
  serializeStructuredValue,
  serializeToolContent,
} from './toolContent';

type ToolContent = BaseMessage['content'];

describe('toolContent', () => {
  it('normalizes a small opaque array to provider-neutral text', () => {
    const content = [{ type: 'json', rows: [{ id: 1 }] }] as ToolContent;

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(true);
    expect(result.content).toBe(JSON.stringify(content));
  });

  it('does not trust a media-looking type without a valid media payload', () => {
    const invalidBlocks = [
      {
        type: 'image_url',
        rows: [{ id: 1, value: 'not an image' }],
      },
      { type: 'image', text: 'ordinary row' },
      { type: 'file', content: [{ id: 1 }] },
      { type: 'media', uri: 'not-a-supported-media-payload' },
      { type: 'image_url', image_url: [{ id: 1 }] },
      {
        type: 'media',
        mimeType: 'application/octet-stream',
        data: [{ id: 1 }],
      },
    ];

    for (const block of invalidBlocks) {
      const content = [block] as ToolContent;
      const result = compactToolContent(content, 10_000);

      expect(result.changed).toBe(true);
      expect(result.content).toBe(JSON.stringify(content));
    }
  });

  it('recognizes native byte and resource payload shapes as atomic', () => {
    const bytes = new Uint8Array([1, 2, 3]);

    expect(
      isAtomicToolContentBlock({
        type: 'video',
        video: { source: { bytes } },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'document',
        document: {
          source: { content: [{ type: 'text', text: 'page' }] },
        },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'resource',
        resource: { uri: 'file:///result.bin', blob: bytes },
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'file',
        source_type: 'id',
        id: 'file_123',
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'computer_screenshot',
        image_url: 'data:image/png;base64,AAAA',
      })
    ).toBe(true);
    expect(
      isAtomicToolContentBlock({
        type: 'application/pdf',
        data: 'JVBERi0xLjQ=',
      })
    ).toBe(true);
  });

  it('normalizes a direct object result to provider-neutral text', () => {
    const content = { rows: [{ id: 1 }], count: 1 };

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(true);
    expect(result.content).toBe(JSON.stringify(content));
  });

  it('keeps valid text and small media blocks intact', () => {
    const image = {
      type: 'image_url',
      image_url: { url: 'https://example.com/chart.png' },
    };
    const content = [
      { type: 'text', text: 'Query result chart' },
      image,
    ] as ToolContent;

    const result = compactToolContent(content, 10_000);

    expect(result.changed).toBe(false);
    expect(result.content).toBe(content);
  });

  it('omits an oversized inline media block instead of leaking past the cap', () => {
    const base64 = 'A'.repeat(10_000);
    const content = [
      {
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${base64}` },
      },
    ] as ToolContent;

    const result = compactToolContent(content, 200);

    expect(result.changed).toBe(true);
    expect(serializeToolContent(result.content).length).toBeLessThanOrEqual(
      200
    );
    expect(serializeToolContent(result.content)).toContain('omitted');
    expect(serializeToolContent(result.content)).not.toContain(base64);
  });

  it('sizes large native byte payloads without JSON-expanding them', () => {
    const bytes = new Uint8Array(1_000_000);
    const content = [
      {
        type: 'video',
        video: { source: { bytes } },
      },
    ] as ToolContent;

    const uncapped = compactToolContent(content, Number.MAX_SAFE_INTEGER);
    const compacted = compactToolContent(content, 200);

    expect(uncapped.content).toBe(content);
    expect(compacted.changed).toBe(true);
    expect(compacted.originalChars).toBeGreaterThan(1_000_000);
    expect(serializeToolContent(compacted.content).length).toBeLessThanOrEqual(
      200
    );
    expect(serializeToolContent(compacted.content)).toContain('omitted');
  });

  it('preserves small document blocks while compacting adjacent text', () => {
    const document = {
      type: 'document',
      source: { type: 'url', url: 'https://example.com/report.pdf' },
    };
    const content = [
      { type: 'text', text: 'x'.repeat(2_000) },
      document,
    ] as ToolContent;

    const result = compactToolContent(content, 400);

    expect(result.changed).toBe(true);
    expect(Array.isArray(result.content)).toBe(true);
    expect(result.content).toContain(document);
    expect(serializeToolContent(result.content).length).toBeLessThanOrEqual(
      400
    );
  });

  it('repeats shared non-cyclic values and marks only true cycles', () => {
    const shared = { value: 'repeated' };
    expect(serializeStructuredValue([shared, shared])).toBe(
      '[{"value":"repeated"},{"value":"repeated"}]'
    );

    const cyclic: { self?: unknown } = {};
    cyclic.self = cyclic;
    expect(serializeStructuredValue(cyclic)).toBe('{"self":"[Circular]"}');
  });
});
