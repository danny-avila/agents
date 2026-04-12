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

interface CacheEntry {
  regex: RegExp | null;
}

/**
 * Module-level compilation cache. Patterns are compiled on first use and
 * reused on subsequent calls — keeping hot paths (every tool dispatch
 * filtering matchers) out of the regex compiler.
 *
 * Failed compiles are cached as `{ regex: null }` so a malformed pattern
 * does not re-enter the compiler on each call.
 *
 * A `Map` (not `Record`) is used so entries can be evicted in the future
 * without object-churn. The cache is unbounded for now; bounding it
 * requires a use-case that generates unbounded unique patterns, which is
 * not expected from current hook consumers — every pattern is baked into
 * a matcher registration and is reused across calls.
 */
const patternCache: Map<string, CacheEntry> = new Map();

function compile(pattern: string): RegExp | null {
  const cached = patternCache.get(pattern);
  if (cached !== undefined) {
    return cached.regex;
  }
  if (pattern.length > MAX_PATTERN_LENGTH) {
    patternCache.set(pattern, { regex: null });
    return null;
  }
  try {
    const regex = new RegExp(pattern);
    patternCache.set(pattern, { regex });
    return regex;
  } catch {
    patternCache.set(pattern, { regex: null });
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
 * - Otherwise, the pattern is compiled once (via a module-level cache) and
 *   tested against the query. Patterns longer than {@link MAX_PATTERN_LENGTH}
 *   never compile and never match.
 * - Invalid regex patterns never throw — a failed compile is cached as
 *   "never matches" so a single malformed pattern cannot take out a whole
 *   `executeHooks` batch.
 *
 * ## Trust model for pattern authors
 *
 * Patterns are compiled with `new RegExp(pattern)` without any runtime
 * sandbox, so catastrophic backtracking on a pathological pattern can
 * block the event loop. Hosts are expected to treat pattern registration
 * as a trusted-code operation: the design report (§3.8) routes
 * persistable hooks through a host-side compiler before they reach this
 * module. Patterns that originate from end-user input — for example, a
 * host that exposes hook configuration in an agent-editing UI — must be
 * length-bounded, validated, or run through a safe-regex check upstream.
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
