// src/hooks/__tests__/matchers.test.ts
import {
  matchesQuery,
  clearMatcherCache,
  MAX_PATTERN_LENGTH,
} from '../matchers';

describe('matchesQuery', () => {
  beforeEach(() => {
    clearMatcherCache();
  });

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

  describe('pattern length bound', () => {
    it('rejects patterns longer than MAX_PATTERN_LENGTH', () => {
      const tooLong = 'a'.repeat(MAX_PATTERN_LENGTH + 1);
      expect(matchesQuery(tooLong, 'aaa')).toBe(false);
    });

    it('accepts patterns exactly at MAX_PATTERN_LENGTH', () => {
      const atLimit = 'a'.repeat(MAX_PATTERN_LENGTH);
      expect(matchesQuery(atLimit, 'a'.repeat(MAX_PATTERN_LENGTH))).toBe(true);
    });
  });

  describe('compilation cache', () => {
    it('caches successful compiles so the same RegExp object is reused', () => {
      const spy = jest.spyOn(global, 'RegExp');
      try {
        matchesQuery('^Bash$', 'Bash');
        matchesQuery('^Bash$', 'Edit');
        matchesQuery('^Bash$', 'Bash');
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        spy.mockRestore();
      }
    });

    it('caches failed compiles so invalid patterns do not re-enter the compiler', () => {
      const spy = jest.spyOn(global, 'RegExp');
      try {
        matchesQuery('[unclosed', 'any');
        matchesQuery('[unclosed', 'any');
        matchesQuery('[unclosed', 'other');
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        spy.mockRestore();
      }
    });

    it('clearMatcherCache drops cached compiles', () => {
      matchesQuery('^Bash$', 'Bash');
      clearMatcherCache();
      const spy = jest.spyOn(global, 'RegExp');
      try {
        matchesQuery('^Bash$', 'Bash');
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        spy.mockRestore();
      }
    });
  });
});
