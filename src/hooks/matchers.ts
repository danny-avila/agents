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

/**
 * Threshold above which `touchCacheEntry` actually performs the LRU
 * refresh. Below this watermark the cache has zero eviction pressure, so
 * the delete+set on every hit would be pure overhead. Above it we refresh
 * properly so hot patterns survive evictions. 75% of capacity is the
 * standard sweet spot.
 */
const LRU_REFRESH_THRESHOLD = Math.floor((MAX_CACHE_SIZE * 3) / 4);

function touchCacheEntry(pattern: string, entry: CacheEntry): void {
  if (patternCache.size < LRU_REFRESH_THRESHOLD) {
    return;
  }
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

interface QuantifierFrame {
  hasBacktrackRisk: boolean;
}

function skipGroupSyntaxPrefix(pattern: string, start: number): number {
  if (start >= pattern.length || pattern[start] !== '?') {
    return start;
  }
  let i = start + 1;
  if (i >= pattern.length) {
    return i;
  }
  const modifier = pattern[i];
  if (modifier === ':' || modifier === '=' || modifier === '!') {
    return i + 1;
  }
  if (modifier !== '<') {
    return i;
  }
  i++;
  if (i < pattern.length && (pattern[i] === '=' || pattern[i] === '!')) {
    return i + 1;
  }
  while (i < pattern.length && pattern[i] !== '>') {
    i++;
  }
  if (i < pattern.length) {
    i++;
  }
  return i;
}

/**
 * Cheap syntactic detector for the most common catastrophic-backtracking
 * shape: a quantified group that contains another quantifier (e.g.
 * `(a+)+`, `(.*)*`, `(\w+)+$`, `(?:(a+))+`). This is the "nested
 * quantifier" class of ReDoS — runs in polynomial-or-worse time on
 * adversarial inputs.
 *
 * The scan walks the pattern linearly using an explicit stack of group
 * frames. For each group it tracks whether the group's contents include
 * "backtrack risk" — meaning a direct quantifier OR a nested group that
 * carries risk up. When a group closes with a trailing quantifier AND its
 * frame carries backtrack risk, the pattern is flagged. Risk propagates
 * to the enclosing frame when a child group closes (whether the child
 * itself was quantified or not), so `(?:(a+))+` — equivalent to `(a+)+`
 * — is flagged correctly even though the outer non-capturing wrapper is
 * one level removed from the inner quantifier.
 *
 * ## Group-syntax prefixes
 *
 * Non-capturing groups (`(?:`), lookaheads (`(?=`, `(?!`), lookbehinds
 * (`(?<=`, `(?<!`), and named groups (`(?<name>`) are skipped over at
 * the `(` so their `?` is not misread as a quantifier. Without this,
 * `(?:pre_)?tool_name` would be incorrectly rejected because the scanner
 * would see the group-syntax `?` as a quantifier at depth 1.
 *
 * ## Heuristic, not a proof
 *
 * This catches the common forms but not all. Ambiguous-alternation ReDoS
 * like `(a|a)+` is not detected. Pathologically long patterns are also
 * caught by {@link MAX_PATTERN_LENGTH}. Hosts that accept user-supplied
 * patterns must still validate upstream.
 */
export function hasNestedQuantifier(pattern: string): boolean {
  const stack: QuantifierFrame[] = [];
  let i = 0;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === '\\') {
      i += 2;
      continue;
    }
    if (ch === '[') {
      i = findCharClassEnd(pattern, i) + 1;
      continue;
    }
    if (ch === '(') {
      stack.push({ hasBacktrackRisk: false });
      i = skipGroupSyntaxPrefix(pattern, i + 1);
      continue;
    }
    if (ch === ')') {
      const frame = stack.pop();
      if (frame === undefined) {
        i++;
        continue;
      }
      const next = pattern[i + 1];
      const isQuantifier =
        next === '*' || next === '+' || next === '?' || next === '{';
      if (isQuantifier && frame.hasBacktrackRisk) {
        return true;
      }
      if (stack.length > 0 && (frame.hasBacktrackRisk || isQuantifier)) {
        stack[stack.length - 1].hasBacktrackRisk = true;
      }
      i++;
      continue;
    }
    if (ch === '*' || ch === '+' || ch === '?' || ch === '{') {
      if (stack.length > 0) {
        stack[stack.length - 1].hasBacktrackRisk = true;
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
