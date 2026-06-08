export const DEFAULT_REDACTION_TEXT = '[REDACTED]';

/**
 * Regex shape contract:
 *   - `pattern` must be `/g`-flagged (validated at call time; throws
 *     otherwise — see resolveConfig).
 *   - The first capture group is treated as the visible prefix that
 *     survives redaction. Replacement is always `$1<redactionText>`,
 *     so the regex must place the prefix at the start of the match
 *     (e.g. `/\b(sk-)[A-Za-z0-9_-]+/g`). A pattern whose first group
 *     is somewhere in the middle (e.g. `/secret=([a-z]+)/g`) will
 *     produce broken output because the bytes between the match start
 *     and the capture group are also dropped.
 *   - Use a zero-width empty group `()` at the start if you want no
 *     visible prefix preserved.
 *   - `id` must be unique across the supplied pattern list (match
 *     aggregation keys by id; duplicates throw at config-resolve time).
 */
export type SensitivePattern = {
  id: string;
  label: string;
  pattern: RegExp;
};

export type PatternMatch = {
  patternId: string;
  patternLabel: string;
  count: number;
};

export type MessageContentRedactionConfig = {
  patterns: SensitivePattern[];
  redactionText?: string;
};

type ResolvedConfig = {
  patterns: SensitivePattern[];
  redactionText: string;
};

function resolveConfig(config: MessageContentRedactionConfig): ResolvedConfig {
  const seenIds = new Set<string>();
  for (const { id, pattern } of config.patterns) {
    if (seenIds.has(id)) {
      throw new TypeError(
        `[messageContentRedaction] duplicate pattern id "${id}"; ` +
          'each pattern must have a unique id (match aggregation keys by id).'
      );
    }
    seenIds.add(id);
    if (!pattern.global) {
      throw new TypeError(
        `[messageContentRedaction] pattern "${id}" must use the global (g) flag; ` +
          'without it, only the first match in a string is scrubbed and ' +
          'subsequent matches silently leak.'
      );
    }
  }
  return {
    patterns: config.patterns,
    redactionText: config.redactionText ?? DEFAULT_REDACTION_TEXT,
  };
}

function recordMatch(
  aggregate: Map<string, PatternMatch>,
  pattern: SensitivePattern,
  count: number
): void {
  const existing = aggregate.get(pattern.id);
  if (existing != null) {
    existing.count += count;
    return;
  }
  aggregate.set(pattern.id, {
    patternId: pattern.id,
    patternLabel: pattern.label,
    count,
  });
}

function redactStringInto(
  text: string,
  resolved: ResolvedConfig,
  aggregate: Map<string, PatternMatch>
): string {
  let next = text;
  for (const pattern of resolved.patterns) {
    pattern.pattern.lastIndex = 0;
    let count = 0;
    const replaced = next.replace(pattern.pattern, (...args: unknown[]) => {
      count += 1;
      const groups = args.slice(1, -2);
      const prefix = typeof groups[0] === 'string' ? groups[0] : '';
      return `${prefix}${resolved.redactionText}`;
    });
    if (count > 0) {
      next = replaced;
      recordMatch(aggregate, pattern, count);
    }
  }
  return next;
}

/**
 * Scrubs credential-shaped substrings from a single string using the
 * caller-supplied patterns. Each match is replaced with the configured
 * redaction text (default `[REDACTED]`) while the leading prefix
 * capture group is preserved so reviewers can still identify which
 * family of secret was matched.
 *
 * The library ships no built-in patterns by design — consumers like
 * LibreChat define and maintain the patterns they care about.
 */
export function redactSensitiveText(
  text: string,
  config: MessageContentRedactionConfig
): { text: string; matches: PatternMatch[] } {
  const resolved = resolveConfig(config);
  const aggregate = new Map<string, PatternMatch>();
  const next = redactStringInto(text, resolved, aggregate);
  return { text: next, matches: Array.from(aggregate.values()) };
}
