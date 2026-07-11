import { HumanMessage } from '@langchain/core/messages';
import {
  encodingForModel,
  createTokenCounter,
  TokenEncoderManager,
  estimateImageBlockTokens,
  estimateDocumentBlockTokens,
  estimateTimedMediaBlockTokens,
} from '@/utils/tokens';

/** Builds a minimal PNG data URI whose IHDR encodes the given dimensions. */
function pngDataUri(width: number, height: number): string {
  const buf = Buffer.alloc(48);
  buf[0] = 0x89;
  buf[1] = 0x50;
  buf[2] = 0x4e;
  buf[3] = 0x47;
  buf[4] = 0x0d;
  buf[5] = 0x0a;
  buf[6] = 0x1a;
  buf[7] = 0x0a;
  buf.writeUInt32BE(width, 16);
  buf.writeUInt32BE(height, 20);
  return `data:image/png;base64,${buf.toString('base64')}`;
}

describe('encodingForModel', () => {
  test('returns claude for Claude model strings', () => {
    expect(encodingForModel('claude-3-5-sonnet-20241022')).toBe('claude');
    expect(encodingForModel('claude-3-haiku-20240307')).toBe('claude');
  });

  test('handles Bedrock Claude ARNs', () => {
    expect(encodingForModel('anthropic.claude-3-5-sonnet-20241022-v2:0')).toBe(
      'claude'
    );
  });

  test('is case-insensitive', () => {
    expect(encodingForModel('CLAUDE-3-HAIKU')).toBe('claude');
    expect(encodingForModel('Claude-3-Opus')).toBe('claude');
  });

  test('returns o200k_base for non-Claude models', () => {
    expect(encodingForModel('gpt-4o')).toBe('o200k_base');
    expect(encodingForModel('gemini-2.0-flash')).toBe('o200k_base');
    expect(encodingForModel('mistral-large')).toBe('o200k_base');
  });

  test('returns o200k_base for empty string', () => {
    expect(encodingForModel('')).toBe('o200k_base');
  });
});

describe('createTokenCounter with different encodings', () => {
  beforeEach(() => {
    TokenEncoderManager.reset();
  });

  test('claude encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('claude');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('o200k_base encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('o200k_base');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('both encodings can be initialized and used independently', async () => {
    const claudeCounter = await createTokenCounter('claude');
    const o200kCounter = await createTokenCounter('o200k_base');
    expect(TokenEncoderManager.isInitialized()).toBe(true);

    const msg = new HumanMessage('Test message for both encodings');
    expect(claudeCounter(msg)).toBeGreaterThan(0);
    expect(o200kCounter(msg)).toBeGreaterThan(0);
  });
});

describe('estimateImageBlockTokens', () => {
  test('claude: tokens = ceil(w*h/750), floored at 1024', () => {
    const block = { type: 'image_url', image_url: { url: pngDataUri(1024, 768) } };
    // 1024*768/750 = 1048.58 -> ceil 1049 (> 1024 floor)
    expect(estimateImageBlockTokens(block, 'claude')).toBe(1049);
  });

  test('openai: tokens = 85 + tiles*170 (512px tiles)', () => {
    const block = { type: 'image_url', image_url: { url: pngDataUri(1024, 768) } };
    // ceil(1024/512)*ceil(768/512) = 2*2 = 4 tiles -> 85 + 680 = 765
    expect(estimateImageBlockTokens(block, 'o200k_base')).toBe(765);
  });

  test('falls back to the Anthropic minimum (1024) without base64 data', () => {
    expect(
      estimateImageBlockTokens(
        { type: 'image_url', image_url: { url: 'https://example.com/a.png' } },
        'claude'
      )
    ).toBe(1024);
  });
});

describe('estimateDocumentBlockTokens', () => {
  const countChars = (text: string): number => text.length;

  test('text document is tokenized directly via getTokenCount', () => {
    const block = { type: 'file', source_type: 'text', text: 'hello world' };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(11);
  });

  test('base64 PDF is priced per estimated page', () => {
    const block = {
      type: 'file',
      source_type: 'base64',
      mime_type: 'application/pdf',
      data: 'x'.repeat(150_000),
    };
    // ceil(150000/75000) = 2 pages
    expect(estimateDocumentBlockTokens(block, 'claude', countChars)).toBe(4000); // 2 * 2000
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(3000); // 2 * 1500
  });

  test('url-referenced document uses the conservative fallback', () => {
    const block = { type: 'file', source_type: 'url', url: 'https://x/y.pdf' };
    expect(estimateDocumentBlockTokens(block, 'o200k_base', countChars)).toBe(2000);
  });
});

describe('estimateTimedMediaBlockTokens', () => {
  const B64 = (chars: number): string => 'A'.repeat(chars);

  test('Google video (type=media, video/*): duration from size at ~300 tok/s', () => {
    // 1,000,000 b64 -> 750,000 bytes / 250,000 Bps = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'video/mp4', data: B64(1_000_000) }),
    ).toBe(900);
  });

  test('Google audio (type=media, audio/*): 32 tok/s at ~16KB/s', () => {
    // 320,000 b64 -> 240,000 bytes / 16,000 Bps = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'audio/mp3', data: B64(320_000) }),
    ).toBe(480);
  });

  test('OpenRouter input_audio: estimates from base64 data', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'input_audio', input_audio: { data: B64(320_000) } }),
    ).toBe(480);
  });

  test('OpenRouter video_url with a data: URL estimates from size', () => {
    const url = `data:video/mp4;base64,${B64(1_000_000)}`;
    expect(estimateTimedMediaBlockTokens({ type: 'video_url', video_url: { url } })).toBe(900);
  });

  test('bare remote URL falls back to the flat ~30s estimate', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'video_url', video_url: { url: 'https://x/v.mp4' } }),
    ).toBe(9000);
    expect(estimateTimedMediaBlockTokens({ type: 'input_audio', input_audio: {} })).toBe(960);
  });

  test('clamps to at least one second of tokens for tiny payloads', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'audio/wav', data: 'AAAA' }),
    ).toBe(32);
  });

  test('standard type=video / type=audio blocks (Bedrock converter shape)', () => {
    // 750,000 bytes video / 250,000 = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({ type: 'video', mimeType: 'video/mp4', data: B64(1_000_000) }),
    ).toBe(900);
    // 240,000 bytes audio / 16,000 = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio', mimeType: 'audio/mpeg', data: B64(320_000) }),
    ).toBe(480);
  });

  test('reads Uint8Array data and base64 url payloads', () => {
    // 240,000-byte Uint8Array audio -> 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({
        type: 'audio',
        mimeType: 'audio/wav',
        data: new Uint8Array(240_000),
      }),
    ).toBe(480);
    // base64 data url on a bare video block
    expect(
      estimateTimedMediaBlockTokens({
        type: 'video',
        mimeType: 'video/mp4',
        url: `data:video/mp4;base64,${B64(1_000_000)}`,
      }),
    ).toBe(900);
  });

  test('non-video/audio media (image/document MIME) is NOT priced as video', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'image/png', fileUri: 's3://x' }),
    ).toBe(0);
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'image/png', data: B64(400_000) }),
    ).toBe(0);
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'application/pdf', data: B64(400) }),
    ).toBe(0);
  });

  test('fileId / bare URL with no size falls back to the ~30s estimate', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'video', mimeType: 'video/mp4', fileId: 's3://v' }),
    ).toBe(9000);
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio', mimeType: 'audio/mp3', fileId: 's3://a' }),
    ).toBe(960);
  });

  test('reads native Bedrock nested source.bytes (video/audio)', () => {
    expect(
      estimateTimedMediaBlockTokens({ type: 'video', video: { source: { bytes: new Uint8Array(750_000) } } }),
    ).toBe(900); // 750,000 / 250,000 = 3s * 300
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio', audio: { source: { bytes: new Uint8Array(240_000) } } }),
    ).toBe(480); // 240,000 / 16,000 = 15s * 32
  });

  test('non-data URI schemes (gs://, s3://) are treated as remote, not base64', () => {
    // gs:// audio must hit the ~30s remote fallback, not clamp to 32
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio', mimeType: 'audio/mp3', url: 'gs://bucket/a.mp3' }),
    ).toBe(960);
    expect(
      estimateTimedMediaBlockTokens({ type: 'media', mimeType: 'video/mp4', data: 's3://bucket/v.mp4' }),
    ).toBe(9000);
  });

  test('classifies Google MIME-as-type blocks (type is the mime string)', () => {
    // { type: 'audio/wav', data } -> 240,000 bytes / 16,000 = 15s * 32 = 480
    expect(
      estimateTimedMediaBlockTokens({ type: 'audio/wav', data: B64(320_000) }),
    ).toBe(480);
    // { type: 'video/mp4', data } -> 750,000 / 250,000 = 3s * 300 = 900
    expect(
      estimateTimedMediaBlockTokens({ type: 'video/mp4', data: B64(1_000_000) }),
    ).toBe(900);
  });

  test('returns 0 for non-timed-media blocks', () => {
    expect(estimateTimedMediaBlockTokens({ type: 'text' })).toBe(0);
    expect(estimateTimedMediaBlockTokens({ type: 'image/png', data: B64(400) })).toBe(0);
  });
});
