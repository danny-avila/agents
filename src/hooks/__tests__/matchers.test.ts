// src/hooks/__tests__/matchers.test.ts
import {
  matchesQuery,
  clearMatcherCache,
  getMatcherCacheSize,
  hasNestedQuantifier,
  MAX_PATTERN_LENGTH,
  MAX_CACHE_SIZE,
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

    it('evicts the oldest entry once the cache is full (LRU)', () => {
      for (let i = 0; i < MAX_CACHE_SIZE; i++) {
        matchesQuery(`^pattern${i}$`, `pattern${i}`);
      }
      expect(getMatcherCacheSize()).toBe(MAX_CACHE_SIZE);

      matchesQuery('^overflow$', 'overflow');
      expect(getMatcherCacheSize()).toBe(MAX_CACHE_SIZE);

      const spy = jest.spyOn(global, 'RegExp');
      try {
        matchesQuery('^pattern0$', 'pattern0');
        expect(spy).toHaveBeenCalledTimes(1);
      } finally {
        spy.mockRestore();
      }
    });

    it('refreshes LRU position on hit so hot patterns are not evicted', () => {
      const hotPattern = '^hot$';
      matchesQuery(hotPattern, 'hot');
      for (let i = 0; i < MAX_CACHE_SIZE - 1; i++) {
        matchesQuery(`^cold${i}$`, `cold${i}`);
      }
      matchesQuery(hotPattern, 'hot');

      matchesQuery('^overflow$', 'overflow');

      const spy = jest.spyOn(global, 'RegExp');
      try {
        matchesQuery(hotPattern, 'hot');
        expect(spy).not.toHaveBeenCalled();
      } finally {
        spy.mockRestore();
      }
    });
  });

  describe('hasNestedQuantifier', () => {
    it('detects the classic (a+)+ shape', () => {
      expect(hasNestedQuantifier('(a+)+')).toBe(true);
      expect(hasNestedQuantifier('(a+)+$')).toBe(true);
    });

    it('detects (.*)* and (.+)+', () => {
      expect(hasNestedQuantifier('(.*)*')).toBe(true);
      expect(hasNestedQuantifier('(.+)+')).toBe(true);
    });

    it('detects nested quantifier with ? outside', () => {
      expect(hasNestedQuantifier('(a+)?')).toBe(true);
    });

    it('detects nested quantifier with {n,} outside', () => {
      expect(hasNestedQuantifier('(a+){2,}')).toBe(true);
    });

    it('detects nested quantifier inside deeper groups', () => {
      expect(hasNestedQuantifier('((a+)+)')).toBe(true);
      expect(hasNestedQuantifier('prefix(\\w+)+suffix')).toBe(true);
    });

    it('allows quantifiers that are not nested', () => {
      expect(hasNestedQuantifier('a+')).toBe(false);
      expect(hasNestedQuantifier('^Bash$')).toBe(false);
      expect(hasNestedQuantifier('(a)(b)')).toBe(false);
      expect(hasNestedQuantifier('(a)+(b)')).toBe(false);
      expect(hasNestedQuantifier('(ab)+')).toBe(false);
      expect(hasNestedQuantifier('mcp_\\w+_search')).toBe(false);
    });

    it('ignores quantifier-looking chars inside character classes', () => {
      expect(hasNestedQuantifier('([a+b])+')).toBe(false);
      expect(hasNestedQuantifier('[*+?]+')).toBe(false);
    });

    it('ignores escaped quantifier characters', () => {
      expect(hasNestedQuantifier('(\\+)+')).toBe(false);
      expect(hasNestedQuantifier('(a\\*)+')).toBe(false);
    });
  });

  describe('ReDoS mitigation via matchesQuery', () => {
    it('rejects nested-quantifier patterns as never-matching', () => {
      expect(matchesQuery('(a+)+', 'aaaaaaaaaa')).toBe(false);
      expect(matchesQuery('(.*)*', 'hello')).toBe(false);
    });

    it('does not stall on an adversarial input that would backtrack', () => {
      const adversarial = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!';
      const start = Date.now();
      const result = matchesQuery('(a+)+$', adversarial);
      const elapsed = Date.now() - start;
      expect(result).toBe(false);
      expect(elapsed).toBeLessThan(50);
    });
  });
});
