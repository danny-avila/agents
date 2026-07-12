import { Tokenizer } from 'ai-tokenizer';
import type { BaseMessage } from '@langchain/core/messages';
import { ContentTypes } from '@/common/enum';

export type EncodingName = 'o200k_base' | 'claude';

/** Anthropic minimum image token cost. */
const ANTHROPIC_IMAGE_MIN_TOKENS = 1024;
/** Anthropic divisor: tokens = width × height / 750. */
const ANTHROPIC_IMAGE_DIVISOR = 750;
/** OpenAI low-detail fixed cost. */
const OPENAI_IMAGE_LOW_TOKENS = 85;
/** OpenAI high-detail tile size. */
const OPENAI_IMAGE_TILE_SIZE = 512;
/** OpenAI high-detail tokens per tile. */
const OPENAI_IMAGE_TOKENS_PER_TILE = 170;
/** Google Gemini fixed per-image cost. */
const _GEMINI_IMAGE_TOKENS = 258;
/** Safety margin for image and document token estimates (5% overestimate). */
export const IMAGE_TOKEN_SAFETY_MARGIN = 1.05;

/**
 * Anthropic PDF: each page costs image tokens + text tokens.
 * Typical range is 1500-3000 tokens/page. Using 2000 as midpoint.
 */
const ANTHROPIC_PDF_TOKENS_PER_PAGE = 2000;
/** OpenAI PDF: each page rendered as high-detail image. ~1500 tokens typical. */
const OPENAI_PDF_TOKENS_PER_PAGE = 1500;
/** Gemini PDF: fixed 258 tokens per page. */
const _GEMINI_PDF_TOKENS_PER_PAGE = 258;
/** Approximate base64 bytes per PDF page for page count estimation. */
const BASE64_BYTES_PER_PDF_PAGE = 75_000;
/** Fallback token cost for URL-referenced documents without local data. */
const URL_DOCUMENT_FALLBACK_TOKENS = 2000;

/**
 * Timed media (video/audio) is priced by DURATION, not size, and the content
 * carries no duration — so duration is estimated from encoded size at
 * representative bitrates and priced at Gemini's rates (the dominant provider
 * for native video/audio). Deliberately rough; superseded by real provider
 * usage after the first turn.
 */
/** Gemini video ≈ 258 tokens/frame (1 fps) + 32 tokens/s audio ≈ 300 tokens/s. */
const VIDEO_TOKENS_PER_SECOND = 300;
/** Gemini audio: 32 tokens/s (single channel). */
const AUDIO_TOKENS_PER_SECOND = 32;
/** Representative encoded byte rates for duration-from-size estimation. */
const VIDEO_BYTES_PER_SECOND = 250_000; // ~2 Mbps
const AUDIO_BYTES_PER_SECOND = 16_000; // ~128 kbps
/** Flat fallback when only a URL is present (no size to estimate from) — ~30s. */
const VIDEO_URL_FALLBACK_TOKENS = 9000;
const AUDIO_URL_FALLBACK_TOKENS = 960;
/** Content block types that can carry timed media (video/audio). `media` is
 *  generic (Google) so it is classified by MIME; the rest are unambiguous. */
const TIMED_MEDIA_TYPES = new Set([
  'media',
  'video',
  'audio',
  'video_url',
  'input_audio',
]);

/**
 * Extracts image dimensions from the first bytes of a base64-encoded
 * PNG, JPEG, GIF, or WebP without decoding the full image.
 * Returns null if the format is unrecognized or data is too short.
 */
export function extractImageDimensions(
  base64Data: string
): { width: number; height: number } | null {
  const raw = base64Data.startsWith('data:')
    ? base64Data.slice(base64Data.indexOf(',') + 1)
    : base64Data;

  if (raw.length < 32) {
    return null;
  }

  const bytes = new Uint8Array(Buffer.from(raw.slice(0, 80), 'base64'));

  if (bytes[0] === 0x89 && bytes[1] === 0x50) {
    // PNG: width at bytes 16-19, height at 20-23 (big-endian)
    const width =
      (bytes[16] << 24) | (bytes[17] << 16) | (bytes[18] << 8) | bytes[19];
    const height =
      (bytes[20] << 24) | (bytes[21] << 16) | (bytes[22] << 8) | bytes[23];
    return { width, height };
  }

  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    // JPEG: scan for SOF0 (0xFFC0) or SOF2 (0xFFC2) marker
    for (let i = 2; i < bytes.length - 9; i++) {
      if (
        bytes[i] === 0xff &&
        (bytes[i + 1] === 0xc0 || bytes[i + 1] === 0xc2)
      ) {
        const height = (bytes[i + 5] << 8) | bytes[i + 6];
        const width = (bytes[i + 7] << 8) | bytes[i + 8];
        return { width, height };
      }
    }
    return null;
  }

  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) {
    // GIF: width at bytes 6-7, height at 8-9 (little-endian)
    const width = bytes[6] | (bytes[7] << 8);
    const height = bytes[8] | (bytes[9] << 8);
    return { width, height };
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
    // WebP VP8: width at bytes 26-27, height at 28-29
    if (bytes.length > 29) {
      const width = (bytes[26] | (bytes[27] << 8)) & 0x3fff;
      const height = (bytes[28] | (bytes[29] << 8)) & 0x3fff;
      return { width, height };
    }
    return null;
  }

  return null;
}

/** Estimates image token cost for Anthropic/Bedrock (Claude). */
export function estimateAnthropicImageTokens(
  width: number,
  height: number
): number {
  return Math.max(
    ANTHROPIC_IMAGE_MIN_TOKENS,
    Math.ceil((width * height) / ANTHROPIC_IMAGE_DIVISOR)
  );
}

/** Estimates image token cost for OpenAI (high detail). */
export function estimateOpenAIImageTokens(
  width: number,
  height: number,
  detail: string = 'high'
): number {
  if (detail === 'low') {
    return OPENAI_IMAGE_LOW_TOKENS;
  }
  const tiles =
    Math.ceil(width / OPENAI_IMAGE_TILE_SIZE) *
    Math.ceil(height / OPENAI_IMAGE_TILE_SIZE);
  return OPENAI_IMAGE_LOW_TOKENS + tiles * OPENAI_IMAGE_TOKENS_PER_TILE;
}

/**
 * Estimates token cost for an image content block.
 * Extracts dimensions from base64 header when available.
 * Falls back to Anthropic minimum (1024) when dimensions can't be determined.
 */
export function estimateImageBlockTokens(
  block: Record<string, unknown>,
  encoding: EncodingName
): number {
  let base64Data: string | undefined;

  if (block.type === ContentTypes.IMAGE_URL || block.type === 'image_url') {
    const imageUrl = block.image_url as string | { url?: string } | undefined;
    const url = typeof imageUrl === 'string' ? imageUrl : imageUrl?.url;
    if (typeof url === 'string' && url.startsWith('data:')) {
      base64Data = url;
    } else {
      return ANTHROPIC_IMAGE_MIN_TOKENS;
    }
  } else if (block.type === 'image') {
    const source = block.source as { type?: string; data?: string } | undefined;
    if (source?.type === 'base64' && typeof source.data === 'string') {
      base64Data = source.data;
    } else {
      return ANTHROPIC_IMAGE_MIN_TOKENS;
    }
  } else {
    return ANTHROPIC_IMAGE_MIN_TOKENS;
  }

  const dims = extractImageDimensions(base64Data);
  if (dims == null) {
    return ANTHROPIC_IMAGE_MIN_TOKENS;
  }

  if (encoding === 'claude') {
    return estimateAnthropicImageTokens(dims.width, dims.height);
  }
  return estimateOpenAIImageTokens(dims.width, dims.height);
}

/**
 * Estimates token cost for a document/file content block.
 * Handles both LangChain standard format (`type: 'file'` with `source_type`)
 * and Anthropic format (`type: 'document'` with `source`).
 *
 * - Plain text: tokenized directly via `getTokenCount`.
 * - Base64 PDF: page count estimated from base64 length × per-page cost.
 * - URL reference: conservative flat estimate.
 */
export function estimateDocumentBlockTokens(
  block: Record<string, unknown>,
  encoding: EncodingName,
  getTokenCount: (text: string) => number
): number {
  const pdfTokensPerPage =
    encoding === 'claude'
      ? ANTHROPIC_PDF_TOKENS_PER_PAGE
      : OPENAI_PDF_TOKENS_PER_PAGE;

  // LangChain standard format: type='file', source_type, data/text/url, mime_type
  const sourceType = block.source_type as string | undefined;
  if (typeof sourceType === 'string') {
    const mimeType = ((block.mime_type as string | undefined) ?? '').split(
      ';'
    )[0];

    if (sourceType === 'text' && typeof block.text === 'string') {
      return getTokenCount(block.text as string);
    }

    if (sourceType === 'base64' && typeof block.data === 'string') {
      if (mimeType === 'application/pdf' || mimeType === '') {
        const pageEstimate = Math.max(
          1,
          Math.ceil((block.data as string).length / BASE64_BYTES_PER_PDF_PAGE)
        );
        return pageEstimate * pdfTokensPerPage;
      }
      // Image inside a file block — delegate to image estimation
      if (mimeType.startsWith('image/')) {
        return estimateImageBlockTokens(
          {
            ...block,
            type: 'image',
            source: { type: 'base64', data: block.data },
          },
          encoding
        );
      }
      return getTokenCount(block.data as string);
    }

    if (sourceType === 'url') {
      return URL_DOCUMENT_FALLBACK_TOKENS;
    }

    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  // Anthropic format: type='document', source: { type, data, media_type }
  const source = block.source as
    | {
        type?: string;
        data?: string;
        media_type?: string;
        content?: unknown[];
      }
    | undefined;

  if (source == null) {
    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  if (source.type === 'text' && typeof source.data === 'string') {
    return getTokenCount(source.data);
  }

  if (source.type === 'base64' && typeof source.data === 'string') {
    const mediaType = (source.media_type ?? '').split(';')[0];
    if (mediaType === 'application/pdf' || mediaType === '') {
      const pageEstimate = Math.max(
        1,
        Math.ceil(source.data.length / BASE64_BYTES_PER_PDF_PAGE)
      );
      return pageEstimate * pdfTokensPerPage;
    }
    if (mediaType.startsWith('image/')) {
      return estimateImageBlockTokens(
        { type: 'image', source: { type: 'base64', data: source.data } },
        encoding
      );
    }
    return getTokenCount(source.data);
  }

  if (source.type === 'url') {
    return URL_DOCUMENT_FALLBACK_TOKENS;
  }

  // content-type source (wraps other blocks like images)
  if (source.type === 'content' && Array.isArray(source.content)) {
    let total = 0;
    for (const inner of source.content) {
      if (inner != null && typeof inner === 'object' && 'type' in inner) {
        const innerBlock = inner as Record<string, unknown>;
        if (innerBlock.type === 'image') {
          total += estimateImageBlockTokens(innerBlock, encoding);
        }
      }
    }
    return total > 0 ? total : URL_DOCUMENT_FALLBACK_TOKENS;
  }

  return URL_DOCUMENT_FALLBACK_TOKENS;
}

/** Decoded byte length of base64 or a `data:` URL; 0 for any remote URI (http,
 *  gs, s3, file, …) or empty. Base64 never contains `:`, so any scheme prefix
 *  marks a remote reference with no local size. */
function base64ByteLength(value: string | undefined): number {
  if (typeof value !== 'string' || value.length === 0) {
    return 0;
  }
  if (value.startsWith('data:')) {
    const comma = value.indexOf(',');
    return comma < 0 ? 0 : Math.floor(((value.length - comma - 1) * 3) / 4);
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(value)) {
    return 0;
  }
  return Math.floor((value.length * 3) / 4);
}

function timedMediaTokens(
  bytes: number,
  bytesPerSecond: number,
  tokensPerSecond: number,
  urlFallback: number
): number {
  if (bytes <= 0) {
    return urlFallback;
  }
  return Math.max(
    tokensPerSecond,
    Math.ceil((bytes / bytesPerSecond) * tokensPerSecond)
  );
}

/** Decoded byte length of a media block payload — top-level `data` (base64
 *  string or `Uint8Array`) or base64 `url`, or the nested `video|audio.{data,
 *  url, source.bytes}` shapes (`formatMessage` media arrays / native Bedrock);
 *  0 for a bare remote URL, `fileId`/`fileUri`, S3 location, or empty. */
function mediaBlockByteLength(block: Record<string, unknown>): number {
  const data = block.data;
  if (typeof data === 'string') {
    return base64ByteLength(data);
  }
  if (data instanceof Uint8Array) {
    return data.length;
  }
  if (typeof block.url === 'string') {
    return base64ByteLength(block.url);
  }
  const nested = (block.video ?? block.audio) as
    | { data?: unknown; url?: unknown; source?: { bytes?: unknown } }
    | undefined;
  if (nested != null) {
    if (typeof nested.data === 'string') {
      return base64ByteLength(nested.data);
    }
    if (typeof nested.url === 'string') {
      return base64ByteLength(nested.url);
    }
    if (nested.source?.bytes instanceof Uint8Array) {
      return nested.source.bytes.length;
    }
  }
  return 0;
}

/** Classifies a block as timed media, or null when it is not video/audio — e.g.
 *  a generic Google `media` block carrying an image/document MIME, which must
 *  NOT be priced as video. Also handles the Google shape where `type` IS the
 *  MIME string (e.g. `{ type: 'audio/wav', data }`). */
function timedMediaKind(
  block: Record<string, unknown>
): 'video' | 'audio' | null {
  const type = typeof block.type === 'string' ? block.type : '';
  if (type === 'input_audio' || type === 'audio') {
    return 'audio';
  }
  if (type === 'video_url' || type === 'video') {
    return 'video';
  }
  const mime =
    type === 'media' && typeof block.mimeType === 'string' ? block.mimeType : type;
  if (mime.startsWith('audio/')) {
    return 'audio';
  }
  if (mime.startsWith('video/')) {
    return 'video';
  }
  return null;
}

/** Whether a content-block `type` can carry timed media — the fixed set plus the
 *  Google MIME-as-type shape (`audio/*` / `video/*`). Gates callers before they
 *  hand the block to {@link estimateTimedMediaBlockTokens}. */
function isTimedMediaType(type: string): boolean {
  return (
    TIMED_MEDIA_TYPES.has(type) ||
    type.startsWith('audio/') ||
    type.startsWith('video/')
  );
}

/**
 * Estimates token cost for a timed-media block (video/audio). Handles Google
 * `{ type: 'media', mimeType, data|url|fileUri }`, OpenRouter
 * `{ type: 'video_url' }` / `{ type: 'input_audio' }`, and standard
 * `{ type: 'video' }` / `{ type: 'audio' }` blocks (payload as `data` base64 or
 * `Uint8Array`, base64 `url`, or `fileId`). Duration is inferred from encoded
 * size (providers price by duration, which the block does not carry) at Gemini's
 * rates; a payload with no size (bare URL / file id) falls back to a ~30s
 * estimate. Returns 0 for non-timed media (e.g. an image/document `media` block)
 * so it is never mispriced as video.
 */
export function estimateTimedMediaBlockTokens(
  block: Record<string, unknown>
): number {
  const kind = timedMediaKind(block);
  if (kind == null) {
    return 0;
  }
  let bytes: number;
  if (block.type === 'input_audio') {
    const audio = block.input_audio as { data?: string } | undefined;
    bytes = base64ByteLength(audio?.data);
  } else if (block.type === 'video_url') {
    const video = block.video_url as string | { url?: string } | undefined;
    bytes = base64ByteLength(typeof video === 'string' ? video : video?.url);
  } else {
    bytes = mediaBlockByteLength(block);
  }
  if (kind === 'audio') {
    return timedMediaTokens(
      bytes,
      AUDIO_BYTES_PER_SECOND,
      AUDIO_TOKENS_PER_SECOND,
      AUDIO_URL_FALLBACK_TOKENS
    );
  }
  return timedMediaTokens(
    bytes,
    VIDEO_BYTES_PER_SECOND,
    VIDEO_TOKENS_PER_SECOND,
    VIDEO_URL_FALLBACK_TOKENS
  );
}

const tokenizers: Partial<Record<EncodingName, Tokenizer>> = {};

async function getTokenizer(
  encoding: EncodingName = 'o200k_base'
): Promise<Tokenizer> {
  const cached = tokenizers[encoding];
  if (cached) {
    return cached;
  }
  const data =
    encoding === 'claude'
      ? await import('ai-tokenizer/encoding/claude')
      : await import('ai-tokenizer/encoding/o200k_base');
  const instance = new Tokenizer(data);
  tokenizers[encoding] = instance;
  return instance;
}

export function encodingForModel(model: string): EncodingName {
  if (model.toLowerCase().includes('claude')) {
    return 'claude';
  }
  return 'o200k_base';
}

export function getTokenCountForMessage(
  message: BaseMessage,
  getTokenCount: (text: string) => number,
  encoding: EncodingName = 'o200k_base'
): number {
  const tokensPerMessage = 3;

  type ContentBlock = Record<string, unknown> & {
    type?: string;
    tool_call?: { name?: string; args?: string; output?: string };
  };

  const processValue = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const raw of value) {
        const item = raw as ContentBlock | null | undefined;
        if (item == null || typeof item.type !== 'string') {
          continue;
        }
        if (item.type === ContentTypes.ERROR) {
          continue;
        }

        if (
          item.type === ContentTypes.IMAGE_URL ||
          item.type === 'image_url' ||
          item.type === 'image'
        ) {
          numTokens += Math.ceil(
            estimateImageBlockTokens(item, encoding) * IMAGE_TOKEN_SAFETY_MARGIN
          );
          continue;
        }

        if (
          item.type === 'document' ||
          item.type === 'file' ||
          item.type === ContentTypes.IMAGE_FILE
        ) {
          numTokens += Math.ceil(
            estimateDocumentBlockTokens(item, encoding, getTokenCount) *
              IMAGE_TOKEN_SAFETY_MARGIN
          );
          continue;
        }

        if (isTimedMediaType(item.type)) {
          numTokens += Math.ceil(
            estimateTimedMediaBlockTokens(item) * IMAGE_TOKEN_SAFETY_MARGIN
          );
          continue;
        }

        if (item.type === ContentTypes.TOOL_CALL && item.tool_call != null) {
          const toolName = item.tool_call.name;
          if (typeof toolName === 'string' && toolName.length > 0) {
            numTokens += getTokenCount(toolName);
          }
          const args = item.tool_call.args;
          if (typeof args === 'string' && args.length > 0) {
            numTokens += getTokenCount(args);
          }
          const output = item.tool_call.output;
          if (typeof output === 'string' && output.length > 0) {
            numTokens += getTokenCount(output);
          }
          continue;
        }

        const nestedValue = item[item.type];
        if (nestedValue == null) {
          continue;
        }

        processValue(nestedValue);
      }
    } else if (typeof value === 'string') {
      numTokens += getTokenCount(value);
    } else if (typeof value === 'number') {
      numTokens += getTokenCount(value.toString());
    } else if (typeof value === 'boolean') {
      numTokens += getTokenCount(value.toString());
    }
  };

  let numTokens = tokensPerMessage;
  processValue(message.content);
  return numTokens;
}

/**
 * Largest-remainder apportionment: scales each count by `multiplier` and
 * distributes the rounding remainder so the results sum exactly to
 * `targetTotal`. Keeps per-item breakdowns reconciled with an aggregate
 * computed as a single rounded product of the summed raw counts.
 */
export function apportionTokenCounts(
  rawCounts: Record<string, number>,
  multiplier: number,
  targetTotal: number
): Record<string, number> {
  const result: Record<string, number> = Object.create(null);
  const remainders: Array<{ name: string; remainder: number }> = [];
  let floorSum = 0;
  for (const [name, rawCount] of Object.entries(rawCounts)) {
    const scaled = rawCount * multiplier;
    const floored = Math.floor(scaled);
    result[name] = floored;
    floorSum += floored;
    remainders.push({ name, remainder: scaled - floored });
  }
  let leftover = targetTotal - floorSum;
  if (leftover <= 0 || remainders.length === 0) {
    return result;
  }
  remainders.sort((a, b) => b.remainder - a.remainder);
  for (let i = 0; leftover > 0; i = (i + 1) % remainders.length) {
    result[remainders[i].name] += 1;
    leftover--;
  }
  return result;
}

/**
 * Anthropic's API consistently reports ~10% more tokens than the local
 * claude tokenizer due to internal message framing and content encoding.
 * Verified empirically across content types via the count_tokens endpoint.
 */
const CLAUDE_TOKEN_CORRECTION = 1.1;

/**
 * Creates a token counter function using the specified encoding.
 * Lazily loads the encoding data on first use via dynamic import.
 */
export const createTokenCounter = async (
  encoding: EncodingName = 'o200k_base'
): Promise<(message: BaseMessage) => number> => {
  const tok = await getTokenizer(encoding);
  const countTokens = (text: string): number => tok.count(text);
  const isClaude = encoding === 'claude';
  return (message: BaseMessage): number => {
    const count = getTokenCountForMessage(message, countTokens, encoding);
    return isClaude ? Math.ceil(count * CLAUDE_TOKEN_CORRECTION) : count;
  };
};

/** Utility to manage the token encoder lifecycle explicitly. */
export const TokenEncoderManager = {
  async initialize(): Promise<void> {
    // No-op: ai-tokenizer is synchronously initialized from bundled data.
  },

  reset(): void {
    for (const key of Object.keys(tokenizers)) {
      delete tokenizers[key as EncodingName];
    }
  },

  isInitialized(): boolean {
    return Object.keys(tokenizers).length > 0;
  },
};
