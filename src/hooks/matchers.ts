// src/hooks/matchers.ts

/**
 * Tests whether a hook matcher pattern matches the given query string.
 *
 * Rules:
 * - `undefined` or empty `pattern` matches any query (wildcard).
 * - `undefined` or empty `query` only matches wildcard patterns.
 * - Invalid regex patterns never match — they silently return `false`
 *   rather than throwing, so a bad matcher registered by one hook cannot
 *   take out the whole executeHooks batch.
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
  try {
    return new RegExp(pattern).test(query);
  } catch {
    return false;
  }
}
