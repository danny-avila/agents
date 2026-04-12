// src/hooks/matchers.ts

/**
 * Upper bound on hook-matcher pattern length. Patterns longer than this
 * are rejected outright — the goal is a cheap cap on pathological inputs
 * (repeated quantifiers, huge alternation groups) without pulling in a
 * safe-regex dependency.
 *
 * Legitimate matchers are almost always under 50 characters (tool names,
 * short alternations, simple prefix anchors); 512 leaves generous
 * headroom while preventing 10KB regexes.
 */
export const MAX_PATTERN_LENGTH = 512;

/**
 * Upper bound on the compilation cache. Chosen to comfortably hold every
 * distinct pattern a single multi-tenant run is likely to see (tools,
 * agent types, basename filters) without growing without bound.
 *
 * Under hosts that register unique patterns per tenant, LRU eviction
 * keeps the working set bounded — cold patterns are re-compiled on next
 * use, which is the correct cost trade-off for long-running processes
 * that must not leak memory.
 */
export const MAX_CACHE_SIZE = 256;

interface CacheEntry {
  regex: RegExp | null;
}

/**
 * Module-level LRU cache keyed by pattern string. Map iteration order is
 * insertion order in ECMAScript, so refreshing an entry's position means
 * "delete then re-set". On overflow we evict the first key (least
 * recently used).
 *
 * Failed compiles are cached as `{ regex: null }` so a malformed pattern
 * does not re-enter the compiler — and so a tenant spamming bad patterns
 * doesn't burn CPU on every call.
 */
const patternCache: Map<string, CacheEntry> = new Map();

function touchCacheEntry(pattern: string, entry: CacheEntry): void {
  patternCache.delete(pattern);
  patternCache.set(pattern, entry);
}

function setCacheEntry(pattern: string, entry: CacheEntry): void {
  if (patternCache.size >= MAX_CACHE_SIZE) {
    const oldestKey = patternCache.keys().next().value;
    if (oldestKey !== undefined) {
      patternCache.delete(oldestKey);
    }
  }
  patternCache.set(pattern, entry);
}

/**
 * Cheap syntactic detector for the most common catastrophic-backtracking
 * shape: a quantified group that contains another quantifier (e.g.
 * `(a+)+`, `(.*)*`, `(\w+)+$`). This is the "nested quantifier" class of
 * ReDoS — runs in polynomial-or-worse time on adversarial inputs.
 *
 * The scan walks the pattern linearly, tracking parenthesis depth and
 * whether the innermost group has seen a quantifier. When a group closes,
 * if it contained a quantifier AND the closing paren is followed by a
 * quantifier, the pattern is flagged.
 *
 * This is a heuristic, not a proof — it rejects many unsafe patterns
 * and allows some that safe-regex libraries would flag. It is a floor,
 * not a ceiling: hosts that accept user-supplied patterns must still
 * validate upstream. The goal is to stop trivially bad patterns from
 * reaching the compiler on a multi-tenant hot path.
 */
export function hasNestedQuantifier(pattern: string): boolean {
  const quantifiedAtDepth: boolean[] = [];
  let depth = 0;
  let i = 0;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === '\\') {
      i += 2;
      continue;
    }
    if (ch === '[') {
      const end = findCharClassEnd(pattern, i);
      i = end + 1;
      continue;
    }
    if (ch === '(') {
      depth++;
      quantifiedAtDepth[depth] = false;
      i++;
      continue;
    }
    if (ch === ')') {
      const innerHadQuantifier = quantifiedAtDepth[depth] === true;
      depth--;
      const next = pattern[i + 1];
      if (
        innerHadQuantifier &&
        (next === '*' || next === '+' || next === '?' || next === '{')
      ) {
        return true;
      }
      i++;
      continue;
    }
    if (ch === '*' || ch === '+' || ch === '?' || ch === '{') {
      if (depth > 0) {
        quantifiedAtDepth[depth] = true;
      }
    }
    i++;
  }
  return false;
}

function findCharClassEnd(pattern: string, start: number): number {
  let i = start + 1;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === '\\') {
      i += 2;
      continue;
    }
    if (ch === ']') {
      return i;
    }
    i++;
  }
  return pattern.length - 1;
}

function compile(pattern: string): RegExp | null {
  const cached = patternCache.get(pattern);
  if (cached !== undefined) {
    touchCacheEntry(pattern, cached);
    return cached.regex;
  }
  if (pattern.length > MAX_PATTERN_LENGTH) {
    setCacheEntry(pattern, { regex: null });
    return null;
  }
  if (hasNestedQuantifier(pattern)) {
    setCacheEntry(pattern, { regex: null });
    return null;
  }
  try {
    const regex = new RegExp(pattern);
    setCacheEntry(pattern, { regex });
    return regex;
  } catch {
    setCacheEntry(pattern, { regex: null });
    return null;
  }
}

/**
 * Tests whether a hook matcher pattern matches the given query string.
 *
 * ## Semantics
 *
 * - `undefined` or empty `pattern` matches any query (wildcard). This is
 *   the intended shape for events that do not supply a query string at
 *   all (`RunStart`, `Stop`, etc.) — register such matchers without a
 *   pattern.
 * - `undefined` or empty `query` with a non-empty `pattern` never matches.
 *   Setting a pattern on a query-less event is therefore inert: the
 *   matcher will simply never fire. This is intentional — it keeps
 *   query-based filtering out of event types where "query" has no meaning,
 *   and is documented on `HookMatcher.pattern`.
 * - Otherwise, the pattern is compiled once (via a bounded LRU cache) and
 *   tested against the query.
 * - Invalid regex patterns never throw — a failed compile is cached as
 *   "never matches" so a single malformed pattern cannot take out a whole
 *   `executeHooks` batch.
 *
 * ## ReDoS mitigations
 *
 * Patterns compile through three cheap gates before reaching `new RegExp`:
 *
 * 1. {@link MAX_PATTERN_LENGTH} length cap rejects oversized inputs.
 * 2. {@link hasNestedQuantifier} rejects the most common catastrophic-
 *    backtracking shape (quantified group containing a quantifier).
 * 3. Successful compiles are cached in a bounded LRU so repeated calls
 *    never re-enter the regex compiler.
 *
 * These are a floor, not a ceiling. Hosts that accept user-supplied
 * patterns should still validate upstream. The design report §3.8 routes
 * persistable hooks through a host-side compiler before they reach this
 * module.
 */
export function matchesQuery(
  pattern: string | undefined,
  query: string | undefined
): boolean {
  if (pattern === undefined || pattern === '') {
    return true;
  }
  if (query === undefined || query === '') {
    return false;
  }
  const regex = compile(pattern);
  if (regex === null) {
    return false;
  }
  return regex.test(query);
}

/** Clears the regex compilation cache. Intended for test isolation. */
export function clearMatcherCache(): void {
  patternCache.clear();
}

/** Returns the current size of the compilation cache. Intended for tests. */
export function getMatcherCacheSize(): number {
  return patternCache.size;
}
