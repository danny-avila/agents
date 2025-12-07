// src/tools/__tests__/ToolSearchRegex.test.ts
/**
 * Unit tests for Tool Search Regex.
 * Tests helper functions and sanitization logic without hitting the API.
 */
import { describe, it, expect } from '@jest/globals';
import {
  sanitizeRegex,
  escapeRegexSpecialChars,
  isDangerousPattern,
  countNestedGroups,
  hasNestedQuantifiers,
} from '../ToolSearchRegex';

describe('ToolSearchRegex', () => {
  describe('escapeRegexSpecialChars', () => {
    it('escapes special regex characters', () => {
      expect(escapeRegexSpecialChars('hello.world')).toBe('hello\\.world');
      expect(escapeRegexSpecialChars('test*pattern')).toBe('test\\*pattern');
      expect(escapeRegexSpecialChars('query+result')).toBe('query\\+result');
      expect(escapeRegexSpecialChars('a?b')).toBe('a\\?b');
      expect(escapeRegexSpecialChars('(group)')).toBe('\\(group\\)');
      expect(escapeRegexSpecialChars('[abc]')).toBe('\\[abc\\]');
      expect(escapeRegexSpecialChars('a|b')).toBe('a\\|b');
      expect(escapeRegexSpecialChars('a^b$c')).toBe('a\\^b\\$c');
      expect(escapeRegexSpecialChars('a{2,3}')).toBe('a\\{2,3\\}');
    });

    it('handles empty string', () => {
      expect(escapeRegexSpecialChars('')).toBe('');
    });

    it('handles string with no special chars', () => {
      expect(escapeRegexSpecialChars('hello_world')).toBe('hello_world');
      expect(escapeRegexSpecialChars('test123')).toBe('test123');
    });

    it('handles multiple consecutive special chars', () => {
      expect(escapeRegexSpecialChars('...')).toBe('\\.\\.\\.');
      expect(escapeRegexSpecialChars('***')).toBe('\\*\\*\\*');
    });
  });

  describe('countNestedGroups', () => {
    it('counts simple nesting', () => {
      expect(countNestedGroups('(a)')).toBe(1);
      expect(countNestedGroups('((a))')).toBe(2);
      expect(countNestedGroups('(((a)))')).toBe(3);
    });

    it('counts maximum depth with multiple groups', () => {
      expect(countNestedGroups('(a)(b)(c)')).toBe(1);
      expect(countNestedGroups('(a(b)c)')).toBe(2);
      expect(countNestedGroups('(a(b(c)))')).toBe(3);
    });

    it('handles mixed nesting levels', () => {
      expect(countNestedGroups('(a)((b)(c))')).toBe(2);
      expect(countNestedGroups('((a)(b))((c))')).toBe(2);
    });

    it('ignores escaped parentheses', () => {
      expect(countNestedGroups('\\(not a group\\)')).toBe(0);
      expect(countNestedGroups('(a\\(b\\)c)')).toBe(1);
    });

    it('handles no groups', () => {
      expect(countNestedGroups('abc')).toBe(0);
      expect(countNestedGroups('test.*pattern')).toBe(0);
    });

    it('handles unbalanced groups', () => {
      expect(countNestedGroups('((a)')).toBe(2);
      expect(countNestedGroups('(a))')).toBe(1);
    });
  });

  describe('hasNestedQuantifiers', () => {
    it('detects nested quantifiers', () => {
      expect(hasNestedQuantifiers('(a+)+')).toBe(true);
      expect(hasNestedQuantifiers('(a*)*')).toBe(true);
      expect(hasNestedQuantifiers('(a+)*')).toBe(true);
      expect(hasNestedQuantifiers('(a*)?')).toBe(true);
    });

    it('allows safe quantifiers', () => {
      expect(hasNestedQuantifiers('a+')).toBe(false);
      expect(hasNestedQuantifiers('(abc)+')).toBe(false);
      expect(hasNestedQuantifiers('a+b*c?')).toBe(false);
    });

    it('handles complex patterns', () => {
      expect(hasNestedQuantifiers('(a|b)+')).toBe(false);
      expect(hasNestedQuantifiers('((a|b)+)+')).toBe(true);
    });
  });

  describe('isDangerousPattern', () => {
    it('detects nested quantifiers', () => {
      expect(isDangerousPattern('(a+)+')).toBe(true);
      expect(isDangerousPattern('(a*)*')).toBe(true);
      expect(isDangerousPattern('(.+)+')).toBe(true);
      expect(isDangerousPattern('(.*)*')).toBe(true);
    });

    it('detects excessive nesting', () => {
      expect(isDangerousPattern('((((((a))))))')).toBe(true); // Depth > 5
    });

    it('detects excessive wildcards', () => {
      const pattern = '.{1000,}';
      expect(isDangerousPattern(pattern)).toBe(true);
    });

    it('allows safe patterns', () => {
      expect(isDangerousPattern('weather')).toBe(false);
      expect(isDangerousPattern('get_.*_data')).toBe(false);
      expect(isDangerousPattern('(a|b|c)')).toBe(false);
      expect(isDangerousPattern('test\\d+')).toBe(false);
    });

    it('detects various dangerous patterns', () => {
      expect(isDangerousPattern('(.*)+')).toBe(true);
      expect(isDangerousPattern('(.+)*')).toBe(true);
    });
  });

  describe('sanitizeRegex', () => {
    it('returns safe pattern unchanged', () => {
      const result = sanitizeRegex('weather');
      expect(result.safe).toBe('weather');
      expect(result.wasEscaped).toBe(false);
    });

    it('escapes dangerous patterns', () => {
      const result = sanitizeRegex('(a+)+');
      expect(result.safe).toBe('\\(a\\+\\)\\+');
      expect(result.wasEscaped).toBe(true);
    });

    it('escapes invalid regex', () => {
      const result = sanitizeRegex('(unclosed');
      expect(result.wasEscaped).toBe(true);
      expect(result.safe).toContain('\\(');
    });

    it('allows complex but safe patterns', () => {
      const result = sanitizeRegex('get_[a-z]+_data');
      expect(result.safe).toBe('get_[a-z]+_data');
      expect(result.wasEscaped).toBe(false);
    });

    it('handles alternation patterns', () => {
      const result = sanitizeRegex('weather|forecast');
      expect(result.safe).toBe('weather|forecast');
      expect(result.wasEscaped).toBe(false);
    });
  });

  describe('Pattern Validation Edge Cases', () => {
    it('handles empty pattern', () => {
      expect(countNestedGroups('')).toBe(0);
      expect(hasNestedQuantifiers('')).toBe(false);
      expect(isDangerousPattern('')).toBe(false);
    });

    it('handles pattern with only quantifiers', () => {
      expect(hasNestedQuantifiers('+++')).toBe(false);
      expect(hasNestedQuantifiers('***')).toBe(false);
    });

    it('handles escaped special sequences', () => {
      const result = sanitizeRegex('\\d+\\w*\\s?');
      expect(result.wasEscaped).toBe(false);
    });

    it('sanitizes exponential backtracking patterns', () => {
      // These can cause catastrophic backtracking
      expect(isDangerousPattern('(a+)+')).toBe(true);
      expect(isDangerousPattern('(a*)*')).toBe(true);
      expect(isDangerousPattern('(.*)*')).toBe(true);
    });
  });

  describe('Real-World Pattern Examples', () => {
    it('handles common search patterns safely', () => {
      const patterns = [
        'expense',
        'get_.*',
        'weather|forecast',
        'data.*query',
        '(?i)email',
        '^create_',
        '_tool$',
        'get_[a-z]+_info',
      ];

      for (const pattern of patterns) {
        const result = sanitizeRegex(pattern);
        expect(result.wasEscaped).toBe(false);
      }
    });

    it('escapes malicious patterns', () => {
      const maliciousPatterns = [
        '(a+)+',
        '(.*)+',
        '(.+)*',
        '(a|a)*',
        '((((((a))))))',
      ];

      for (const pattern of maliciousPatterns) {
        const result = sanitizeRegex(pattern);
        expect(result.wasEscaped).toBe(true);
      }
    });
  });
});
