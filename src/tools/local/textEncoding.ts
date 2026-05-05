/**
 * BOM and line-ending preservation helpers for the local engine's
 * file-mutating tools. We never *introduce* a BOM or change line
 * endings — only preserve what was already on disk so a Windows-
 * checked-in source file stays CRLF and a UTF-8-with-BOM JSON file
 * keeps its BOM after an edit.
 *
 * Inspired by opencode's `Bom` helper. Trimmed to the cases that
 * actually matter for editing source code (UTF-8 BOM only;
 * UTF-16/UTF-32 are out of scope).
 */

const UTF8_BOM = '﻿';

export interface EncodedFile {
  /** File contents with BOM stripped. */
  text: string;
  /** Whether the on-disk file started with a UTF-8 BOM. */
  hasBom: boolean;
  /** Detected newline style. CRLF wins if any CRLF is present. */
  newline: '\n' | '\r\n';
}

export function decodeFile(raw: string): EncodedFile {
  const hasBom = raw.startsWith(UTF8_BOM);
  const stripped = hasBom ? raw.slice(1) : raw;
  const newline = stripped.includes('\r\n') ? '\r\n' : '\n';
  // Internally we always work in LF; encode() restores CRLF on write.
  const lf = newline === '\r\n' ? stripped.replace(/\r\n/g, '\n') : stripped;
  return { text: lf, hasBom, newline };
}

export function encodeFile(text: string, encoding: EncodedFile): string {
  const out =
    encoding.newline === '\r\n' ? text.replace(/\n/g, '\r\n') : text;
  return encoding.hasBom ? `${UTF8_BOM}${out}` : out;
}
