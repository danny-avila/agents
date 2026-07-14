/**
 * Inherited Anthropic content/util tests, ported from @langchain/anthropic@1.5.1.
 *
 * Upstream sources (vitest), consolidated here and adapted to this fork (jest):
 *  - src/v1/tests/standard_content.test.ts        (v1 standard content-block conversion)
 *  - src/utils/tests/message_outputs.test.ts      (message_outputs — copied at utils/message_outputs.ts)
 *  - src/utils/tests/tools.test.ts                (tools util — copied at utils/tools.ts)
 *
 * Adaptation notes:
 *  - vitest -> jest; `zod/v3` -> `zod`; no `vi.*` usage was needed.
 *  - The fork copies `message_outputs.ts` and `tools.ts` near-verbatim, so those
 *    cases are imported and asserted directly against our copies.
 *  - Upstream's `_formatStandardContent` operates on the v1 `ContentBlock.Multimodal`
 *    model (`fileId`/`text-plain`/`data: Uint8Array`/`metadata`). This fork does not
 *    implement that function or the v1 block model; instead it converts the deprecated
 *    v0.3 `source_type`-based standard blocks via the private `standardContentBlockConverter`,
 *    reached through the public `_convertMessagesToAnthropicPayload`. The standard-content
 *    cases below are therefore routed through that path and assert the fork's equivalent
 *    output. Cases with no fork equivalent are dropped inline with a reason.
 */
import { HumanMessage } from '@langchain/core/messages';
import { _makeMessageChunkFromAnthropicEvent } from './utils/message_outputs';
import { _convertMessagesToAnthropicPayload } from './utils/message_inputs';
import { handleToolChoice } from './utils/tools';

/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Helper mirroring upstream's `createAnthropicMessage`, but using this fork's
 * conversion entry point. Upstream wrapped raw v1 `contentBlocks` and called
 * `_formatStandardContent`; here we send deprecated v0.3 standard blocks through
 * `_convertMessagesToAnthropicPayload` and read back the converted human-turn content.
 */
function convertStandardBlock(block: Record<string, unknown>): unknown {
  const payload = _convertMessagesToAnthropicPayload([
    new HumanMessage({ content: [block as any] }),
  ]);
  const human = payload.messages.find((m: any) => m.role === 'user')!;
  return (human.content as unknown[])[0];
}

describe('standard content-block conversion (ported from standard_content.test.ts)', () => {
  // Adapted: upstream converted a v1 `file` block backed by a `fileId` into a
  // `{ source: { type: 'file', file_id } }` document. This fork has no fileId
  // source type; the metadata passthrough (cache_control/citations/context/title)
  // is the load-bearing behavior, so we assert it on a URL-backed file document.
  it('converts file blocks into Anthropic documents and forwards metadata', () => {
    const content = convertStandardBlock({
      type: 'file',
      source_type: 'url',
      url: 'https://example.com/doc.pdf',
      mime_type: 'application/pdf',
      metadata: {
        cache_control: { type: 'ephemeral', ttl: '5m' },
        citations: { enabled: true },
        context: 'source context',
        title: 'My Document',
      },
    });

    expect(content).toMatchObject({
      type: 'document',
      source: { type: 'url', url: 'https://example.com/doc.pdf' },
      cache_control: { type: 'ephemeral', ttl: '5m' },
      citations: { enabled: true },
      context: 'source context',
      title: 'My Document',
    });
  });

  it('converts inlined text files into text document sources', () => {
    const content = convertStandardBlock({
      type: 'file',
      source_type: 'text',
      text: 'Plain text body',
      mime_type: 'text/plain',
    });

    expect(content).toMatchObject({
      type: 'document',
      source: {
        type: 'text',
        data: 'Plain text body',
        media_type: 'text/plain',
      },
    });
  });

  it('wraps base64 image file payloads in document content blocks', () => {
    // Upstream passed a Uint8Array and asserted Buffer.from(..).toString('base64');
    // this fork takes a pre-encoded base64 string, so we pass the encoded value.
    const data = Buffer.from(Uint8Array.from([1, 2, 3])).toString('base64');
    const content = convertStandardBlock({
      type: 'file',
      source_type: 'base64',
      data,
      mime_type: 'image/png',
    });

    expect(content).toMatchObject({
      type: 'document',
      source: {
        type: 'content',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              data,
              media_type: 'image/png',
            },
          },
        ],
      },
    });
  });

  it('converts standard image blocks with metadata', () => {
    const content = convertStandardBlock({
      type: 'image',
      source_type: 'url',
      url: 'https://example.com/image.png',
      metadata: {
        cache_control: { type: 'ephemeral', ttl: '1h' },
      },
    });

    expect(content).toMatchObject({
      type: 'image',
      source: { type: 'url', url: 'https://example.com/image.png' },
      cache_control: { type: 'ephemeral', ttl: '1h' },
    });
  });

  // Dropped (inherited): upstream's "promotes plain text blocks to Anthropic text
  // documents" exercises the v1 `text-plain` content-block type, which this fork does
  // not model. The equivalent text->document-source path is already covered by the
  // "inlined text files" case above (v0.3 `file`/`source_type: 'text'`).

  it('throws for unsupported audio blocks', () => {
    // Upstream threw /does not support audio/i. This fork throws from the standard
    // converter because it intentionally implements no `fromStandardAudioBlock`,
    // so we assert the fork's actual message.
    expect(() =>
      convertStandardBlock({
        type: 'audio',
        source_type: 'base64',
        data: 'AQID',
        mime_type: 'audio/mpeg',
      })
    ).toThrow(/fromStandardAudioBlock/);
  });
});

describe('_makeMessageChunkFromAnthropicEvent (ported from message_outputs.test.ts)', () => {
  const fields = { streamUsage: true, coerceContentToString: false };

  it('message_start chunk contains correct cache token counts', () => {
    const event = {
      type: 'message_start' as const,
      message: {
        id: 'msg_01',
        type: 'message' as const,
        role: 'assistant' as const,
        content: [],
        model: 'claude-3-5-haiku-latest',
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 100,
          output_tokens: 0,
          cache_creation_input_tokens: 500,
          cache_read_input_tokens: 1000,
        },
      },
    };

    const result = _makeMessageChunkFromAnthropicEvent(event as any, fields);
    expect(result).not.toBeNull();

    const usage = result!.chunk.usage_metadata!;
    // input_tokens in LangChain = input_tokens + cache_creation + cache_read
    expect(usage.input_tokens).toBe(1600);
    expect(usage.input_token_details?.cache_creation).toBe(500);
    expect(usage.input_token_details?.cache_read).toBe(1000);
  });

  it('message_delta chunk preserves cumulative input and cache usage', () => {
    const event = {
      type: 'message_delta' as const,
      delta: { stop_reason: 'end_turn' as const, stop_sequence: null },
      usage: {
        input_tokens: 100,
        output_tokens: 42,
        // Anthropic API returns cumulative cache values here — same as message_start
        cache_creation_input_tokens: 500,
        cache_read_input_tokens: 1000,
      },
    };

    const result = _makeMessageChunkFromAnthropicEvent(event as any, fields);
    expect(result).not.toBeNull();

    const usage = result!.chunk.usage_metadata!;
    expect(usage).toEqual({
      input_tokens: 1600,
      output_tokens: 42,
      total_tokens: 1642,
      input_token_details: {
        cache_creation: 500,
        cache_read: 1000,
      },
    });
  });
});

describe('handleToolChoice (ported from tools.test.ts)', () => {
  it('should return undefined for undefined input', () => {
    expect(handleToolChoice(undefined)).toBeUndefined();
  });

  it("should handle 'any' tool choice", () => {
    expect(handleToolChoice('any')).toEqual({ type: 'any' });
  });

  it("maps OpenAI-style 'required' to Anthropic 'any'", () => {
    expect(handleToolChoice('required')).toEqual({ type: 'any' });
  });

  it("should handle 'auto' tool choice", () => {
    expect(handleToolChoice('auto')).toEqual({ type: 'auto' });
  });

  it("maps 'none' to Anthropic 'none' (disables tools)", () => {
    expect(handleToolChoice('none')).toEqual({ type: 'none' });
  });

  it('should handle specific tool name as string', () => {
    expect(handleToolChoice('my_custom_tool')).toEqual({
      type: 'tool',
      name: 'my_custom_tool',
    });
  });

  it('should pass through object tool choice', () => {
    const toolChoice = { type: 'tool' as const, name: 'specific_tool' };
    expect(handleToolChoice(toolChoice)).toEqual(toolChoice);
  });
});
