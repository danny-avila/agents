// src/hooks/__tests__/matchers.test.ts
import { matchesQuery } from '../matchers';

describe('matchesQuery', () => {
  it('treats undefined pattern as a wildcard match', () => {
    expect(matchesQuery(undefined, 'Bash')).toBe(true);
    expect(matchesQuery(undefined, '')).toBe(true);
    expect(matchesQuery(undefined, undefined)).toBe(true);
  });

  it('treats empty-string pattern as a wildcard match', () => {
    expect(matchesQuery('', 'Bash')).toBe(true);
    expect(matchesQuery('', undefined)).toBe(true);
  });

  it('returns false when the pattern is set but the query is absent', () => {
    expect(matchesQuery('Bash', undefined)).toBe(false);
    expect(matchesQuery('Bash', '')).toBe(false);
  });

  it('runs the pattern as a regex against the query', () => {
    expect(matchesQuery('Bash', 'Bash')).toBe(true);
    expect(matchesQuery('^Bash$', 'Bash')).toBe(true);
    expect(matchesQuery('^Bash$', 'BashExtra')).toBe(false);
    expect(matchesQuery('Bash|Shell', 'Shell')).toBe(true);
    expect(matchesQuery('mcp_.*_search', 'mcp_github_search')).toBe(true);
  });

  it('does not throw on invalid regex and returns false instead', () => {
    expect(() => matchesQuery('[unclosed', 'anything')).not.toThrow();
    expect(matchesQuery('[unclosed', 'anything')).toBe(false);
  });
});
