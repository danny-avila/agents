/**
 * Detects whether a file on disk is an LLM-renderable attachment
 * (image / PDF) and produces the LangChain `MessageContentComplex[]`
 * payload a `ToolMessage` needs to actually surface those bytes to
 * the vision-capable model.
 *
 * Same approach as LibreChat's `api/server/utils/files.js`: sniff the
 * magic bytes (NOT the extension) so a mislabelled `.png` that's
 * really a binary blob doesn't get embedded as an image. Inlined for
 * the five formats we actually care about (PNG / JPEG / GIF / WebP /
 * PDF) instead of pulling the ESM-only `file-type` package — keeps
 * the test setup CJS-clean.
 *
 * Provider compatibility:
 *   - Anthropic: tool_result content arrays accept `image` / `image_url`
 *     blocks; LangChain's anthropic adapter at
 *     `node_modules/@langchain/anthropic/dist/utils/message_inputs.js`
 *     converts them to native `image` source blocks.
 *   - OpenAI Chat Completions: image_url blocks in tool messages are
 *     accepted on vision-capable models.
 *   - OpenAI Responses API: tool messages are flattened to plain text;
 *     image_url blocks degrade to a JSON description (still useful as
 *     a textual hint to the model).
 *   - Google: image blocks in tool responses are accepted on Gemini
 *     vision models.
 *
 * Configuration:
 *   - `local.attachReadAttachments` (default `'images-only'`) controls
 *     which file kinds are returned as inline attachments. Other kinds
 *     fall through to the existing binary-stub path.
 *   - `local.maxAttachmentBytes` (default 5 MB) caps the pre-encoding
 *     size; oversize attachments degrade to a stub describing the
 *     refusal so the model isn't surprised.
 */

import { open as fsOpen, readFile as fsReadFile } from 'fs/promises';
import type { WorkspaceFS } from './workspaceFS';

/**
 * Magic-byte sniff for the small set of image/PDF formats we care
 * about. We avoided pulling in `file-type` (ESM-only, awkward under
 * ts-jest) since the universe of attachments we want to embed is
 * tiny: PNG, JPEG, GIF, WebP, PDF. All have well-known signatures in
 * the first 12 bytes.
 *
 * Returns `undefined` on no match — caller treats as text/unknown.
 */
function sniffMime(buffer: Buffer): string | undefined {
  if (buffer.length < 4) return undefined;
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (
    buffer.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47 &&
    buffer[4] === 0x0d &&
    buffer[5] === 0x0a &&
    buffer[6] === 0x1a &&
    buffer[7] === 0x0a
  ) {
    return 'image/png';
  }
  // JPEG: FF D8 FF
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return 'image/jpeg';
  }
  // GIF: "GIF87a" or "GIF89a"
  if (
    buffer.length >= 6 &&
    buffer[0] === 0x47 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x38 &&
    (buffer[4] === 0x37 || buffer[4] === 0x39) &&
    buffer[5] === 0x61
  ) {
    return 'image/gif';
  }
  // WebP: "RIFF" .... "WEBP"
  if (
    buffer.length >= 12 &&
    buffer[0] === 0x52 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x46 &&
    buffer[8] === 0x57 &&
    buffer[9] === 0x45 &&
    buffer[10] === 0x42 &&
    buffer[11] === 0x50
  ) {
    return 'image/webp';
  }
  // PDF: "%PDF-"
  if (
    buffer.length >= 5 &&
    buffer[0] === 0x25 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x44 &&
    buffer[3] === 0x46 &&
    buffer[4] === 0x2d
  ) {
    return 'application/pdf';
  }
  return undefined;
}

const SUPPORTED_IMAGE_MIMES = new Set<string>([
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
]);

/** Mime types that get returned to the model as inline attachments. */
const SUPPORTED_ATTACHMENT_MIMES = new Set<string>([
  ...SUPPORTED_IMAGE_MIMES,
  'application/pdf',
]);

export type AttachmentMode = 'images-only' | 'images-and-pdf' | 'off';

export type Attachment =
  | {
      kind: 'image';
      mime: string;
      bytes: number;
      dataUrl: string;
    }
  | {
      kind: 'pdf';
      mime: 'application/pdf';
      bytes: number;
      dataUrl: string;
    }
  | {
      kind: 'binary';
      mime: string;
      bytes: number;
    }
  | {
      kind: 'oversize';
      mime: string;
      bytes: number;
      maxBytes: number;
    }
  | {
      kind: 'text-or-unknown';
      bytes: number;
    };

export async function classifyAttachment(args: {
  path: string;
  bytes: number;
  mode: AttachmentMode;
  maxBytes: number;
  /**
   * WorkspaceFS to route I/O through — defaults to host fs/promises
   * for backward compat. Manual review (finding F): without this
   * routing, custom/remote FS implementations could either fail to
   * embed valid attachments or accidentally read a host path with
   * the same absolute name (since `read_file` itself does go through
   * the configured WorkspaceFS).
   */
  fs?: WorkspaceFS;
}): Promise<Attachment> {
  if (args.bytes === 0) {
    return { kind: 'text-or-unknown', bytes: 0 };
  }

  // MIME sniffing only needs the first 12 bytes — read just the
  // header so a 9 MB PNG (under the 10 MB read cap, over the 5 MB
  // attachment cap) doesn't pull the whole buffer into memory before
  // we discover it's oversize. Full read happens only when we're
  // about to base64-embed.
  const open = args.fs?.open ?? fsOpen;
  const handle = await open(args.path, 'r');
  const header = Buffer.alloc(12);
  let mime: string | undefined;
  try {
    await handle.read(header, 0, 12, 0);
    mime = sniffMime(header);
  } finally {
    await handle.close();
  }

  if (mime == null) {
    return { kind: 'text-or-unknown', bytes: args.bytes };
  }

  const wantsImage =
    args.mode === 'images-only' || args.mode === 'images-and-pdf';
  const wantsPdf = args.mode === 'images-and-pdf';

  const isImage = wantsImage && SUPPORTED_IMAGE_MIMES.has(mime);
  const isPdf = wantsPdf && mime === 'application/pdf';

  if (!isImage && !isPdf) {
    if (SUPPORTED_ATTACHMENT_MIMES.has(mime)) {
      return { kind: 'binary', mime, bytes: args.bytes };
    }
    return { kind: 'binary', mime, bytes: args.bytes };
  }

  if (args.bytes > args.maxBytes) {
    return {
      kind: 'oversize',
      mime,
      bytes: args.bytes,
      maxBytes: args.maxBytes,
    };
  }

  const readFile = args.fs?.readFile ?? fsReadFile;
  const buffer = (await readFile(args.path)) as Buffer;
  const base64 = buffer.toString('base64');
  const dataUrl = `data:${mime};base64,${base64}`;

  if (isImage) {
    return { kind: 'image', mime, bytes: args.bytes, dataUrl };
  }
  return {
    kind: 'pdf',
    mime: 'application/pdf' as const,
    bytes: args.bytes,
    dataUrl,
  };
}

/** Build the LangChain content array for an image attachment. */
export function imageAttachmentContent(
  path: string,
  attachment: Extract<Attachment, { kind: 'image' }>
): Array<{
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string };
}> {
  return [
    {
      type: 'text',
      text:
        `Read ${path} (${attachment.mime}, ${attachment.bytes} bytes). ` +
        'The image is attached below for vision-capable models.',
    },
    {
      type: 'image_url',
      image_url: { url: attachment.dataUrl },
    },
  ];
}
