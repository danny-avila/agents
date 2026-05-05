/**
 * Single-occurrence string-replacement strategies for `edit_file`.
 *
 * The LLM frequently emits an `oldString` whose whitespace, indentation,
 * or escape sequences are slightly off from the on-disk content. Rather
 * than failing the call (which forces a re-read + retry round-trip),
 * we walk a chain of progressively looser matchers, stopping at the
 * first one that locates exactly one match. The matched on-disk slice
 * is then literally replaced with `newString` — we never modify
 * `newString`, only the search.
 *
 * Strategies are ordered from strict to lenient so we don't accidentally
 * over-match a more specific pattern with a looser one. Inspired by
 * opencode's nine-strategy chain (sst/opencode), trimmed to the four
 * highest-yield strategies for a first cut. Add more (block-anchor +
 * Levenshtein, escape-normalized, etc.) as needed.
 */

export interface EditMatch {
  /** Strategy name that produced the match, for telemetry/diagnostics. */
  strategy: string;
  /** Starting offset in the source. */
  start: number;
  /** Ending offset (exclusive). */
  end: number;
}

export type EditStrategy = (
  source: string,
  oldString: string
) => EditMatch | null;

const exactStrategy: EditStrategy = (source, oldString) => {
  if (oldString === '') return null;
  const first = source.indexOf(oldString);
  if (first === -1) return null;
  const second = source.indexOf(oldString, first + oldString.length);
  if (second !== -1) return null;
  return { strategy: 'exact', start: first, end: first + oldString.length };
};

/**
 * Match per-line, ignoring trailing whitespace differences. Useful for
 * the very common case where the LLM stripped trailing spaces or added
 * an extra blank.
 */
const lineTrimmedStrategy: EditStrategy = (source, oldString) => {
  if (oldString === '') return null;
  const sourceLines = source.split('\n');
  const oldLines = oldString.split('\n');
  if (oldLines.length === 0 || oldLines.length > sourceLines.length) {
    return null;
  }

  let foundAt = -1;
  for (let i = 0; i <= sourceLines.length - oldLines.length; i++) {
    let ok = true;
    for (let j = 0; j < oldLines.length; j++) {
      if (sourceLines[i + j].trimEnd() !== oldLines[j].trimEnd()) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    if (foundAt !== -1) return null; // multiple matches
    foundAt = i;
  }
  if (foundAt === -1) return null;

  let start = 0;
  for (let i = 0; i < foundAt; i++) start += sourceLines[i].length + 1;
  let end = start;
  for (let i = 0; i < oldLines.length; i++) {
    end += sourceLines[foundAt + i].length;
    if (i < oldLines.length - 1) end += 1;
  }
  return { strategy: 'line-trimmed', start, end };
};

/**
 * Collapse all runs of whitespace to a single space and match. Catches
 * cases where the LLM normalised tabs to spaces or vice-versa.
 */
const whitespaceNormalizedStrategy: EditStrategy = (source, oldString) => {
  if (oldString === '') return null;
  const norm = (s: string): string => s.replace(/\s+/g, ' ').trim();
  const normalizedNeedle = norm(oldString);
  if (normalizedNeedle === '') return null;

  const sourceLines = source.split('\n');
  const needleLines = oldString.split('\n');
  if (needleLines.length > sourceLines.length) return null;

  let foundAt = -1;
  for (let i = 0; i <= sourceLines.length - needleLines.length; i++) {
    const candidate = sourceLines
      .slice(i, i + needleLines.length)
      .join('\n');
    if (norm(candidate) !== normalizedNeedle) continue;
    if (foundAt !== -1) return null;
    foundAt = i;
  }
  if (foundAt === -1) return null;

  let start = 0;
  for (let i = 0; i < foundAt; i++) start += sourceLines[i].length + 1;
  let end = start;
  for (let i = 0; i < needleLines.length; i++) {
    end += sourceLines[foundAt + i].length;
    if (i < needleLines.length - 1) end += 1;
  }
  return { strategy: 'whitespace-normalized', start, end };
};

/**
 * Strip the common leading-indent from each line of the needle and
 * each candidate window of the source. Catches the very common case
 * where the LLM omitted the indentation it should have copied.
 */
const indentationFlexibleStrategy: EditStrategy = (source, oldString) => {
  if (oldString === '') return null;

  const stripCommonIndent = (block: string): string => {
    const lines = block.split('\n');
    let common = Number.POSITIVE_INFINITY;
    for (const line of lines) {
      if (line.trim() === '') continue;
      const m = /^(\s*)/.exec(line);
      const indent = m ? m[1].length : 0;
      if (indent < common) common = indent;
      if (common === 0) break;
    }
    if (!Number.isFinite(common) || common === 0) return block;
    return lines
      .map((l) => (l.length >= common ? l.slice(common) : l))
      .join('\n');
  };

  const normalizedNeedle = stripCommonIndent(oldString);
  if (normalizedNeedle === '') return null;

  const sourceLines = source.split('\n');
  const needleLines = oldString.split('\n');
  if (needleLines.length > sourceLines.length) return null;

  let foundAt = -1;
  for (let i = 0; i <= sourceLines.length - needleLines.length; i++) {
    const window = sourceLines.slice(i, i + needleLines.length).join('\n');
    if (stripCommonIndent(window) !== normalizedNeedle) continue;
    if (foundAt !== -1) return null;
    foundAt = i;
  }
  if (foundAt === -1) return null;

  let start = 0;
  for (let i = 0; i < foundAt; i++) start += sourceLines[i].length + 1;
  let end = start;
  for (let i = 0; i < needleLines.length; i++) {
    end += sourceLines[foundAt + i].length;
    if (i < needleLines.length - 1) end += 1;
  }
  return { strategy: 'indentation-flexible', start, end };
};

const STRATEGY_CHAIN: EditStrategy[] = [
  exactStrategy,
  lineTrimmedStrategy,
  whitespaceNormalizedStrategy,
  indentationFlexibleStrategy,
];

export function locateEdit(source: string, oldString: string): EditMatch | null {
  for (const strategy of STRATEGY_CHAIN) {
    const match = strategy(source, oldString);
    if (match != null) {
      return match;
    }
  }
  return null;
}

export function applyEdit(
  source: string,
  match: EditMatch,
  newString: string
): string {
  return source.slice(0, match.start) + newString + source.slice(match.end);
}
