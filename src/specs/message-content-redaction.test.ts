import {
  DEFAULT_REDACTION_TEXT,
  redactSensitiveText,
  type MessageContentRedactionConfig,
  type PatternMatch,
  type SensitivePattern,
} from '@/messageContentRedaction';

const TEST_PATTERNS: SensitivePattern[] = [
  {
    id: 'anthropic_key',
    label: 'Anthropic key',
    pattern: /\b(sk-ant-)[A-Za-z0-9_-]{10,}/g,
  },
  {
    id: 'github_token',
    label: 'GitHub token',
    pattern: /\b(ghp_)[A-Za-z0-9]{20,}/g,
  },
  {
    id: 'aws_key',
    label: 'AWS key',
    pattern: /\b(AKIA)[A-Z0-9]{12,16}\b/g,
  },
  {
    id: 'jwt',
    label: 'JWT',
    pattern: /\b(eyJ)[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,
  },
];

const ALL: MessageContentRedactionConfig = { patterns: TEST_PATTERNS };

function findMatch(
  matches: PatternMatch[],
  patternId: string
): PatternMatch | undefined {
  return matches.find((match) => match.patternId === patternId);
}

describe('redactSensitiveText', () => {
  it('redacts a single match and reports it', () => {
    const { text, matches } = redactSensitiveText(
      'token sk-ant-abcdefghijklmn please',
      ALL
    );
    expect(text).toBe('token sk-ant-[REDACTED] please');
    expect(matches).toEqual([
      { patternId: 'anthropic_key', patternLabel: 'Anthropic key', count: 1 },
    ]);
  });

  it('counts multiple matches of the same pattern', () => {
    const { text, matches } = redactSensitiveText(
      'first sk-ant-aaaaaaaaaa second sk-ant-bbbbbbbbbb',
      ALL
    );
    expect(text).toBe('first sk-ant-[REDACTED] second sk-ant-[REDACTED]');
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(2);
  });

  it('aggregates matches across different patterns', () => {
    const { text, matches } = redactSensitiveText(
      'anthropic sk-ant-abcdefghijklmn and github ghp_abcdefghijklmnopqrstuvwxyz0123',
      ALL
    );
    expect(text).toBe('anthropic sk-ant-[REDACTED] and github ghp_[REDACTED]');
    expect(matches).toHaveLength(2);
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(1);
    expect(findMatch(matches, 'github_token')?.count).toBe(1);
  });

  it('returns empty matches when nothing fires', () => {
    const { text, matches } = redactSensitiveText('plain words only', ALL);
    expect(text).toBe('plain words only');
    expect(matches).toEqual([]);
  });

  it('honors a custom redaction text', () => {
    const { text } = redactSensitiveText('token sk-ant-abcdefghijklmn', {
      patterns: TEST_PATTERNS,
      redactionText: '[scrubbed]',
    });
    expect(text).toBe('token sk-ant-[scrubbed]');
  });

  it('limits scanning to the caller-supplied pattern subset', () => {
    const onlyJwt: SensitivePattern[] = [TEST_PATTERNS[3]];
    const { text, matches } = redactSensitiveText(
      'jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.signature and key sk-ant-abcdefghijklmn',
      { patterns: onlyJwt }
    );
    expect(text).toBe('jwt eyJ[REDACTED] and key sk-ant-abcdefghijklmn');
    expect(matches.map((m) => m.patternId)).toEqual(['jwt']);
  });

  it('returns no matches when patterns array is empty', () => {
    const { text, matches } = redactSensitiveText(
      'token sk-ant-abcdefghijklmn',
      { patterns: [] }
    );
    expect(text).toBe('token sk-ant-abcdefghijklmn');
    expect(matches).toEqual([]);
  });

  it('exposes the default redaction text constant', () => {
    expect(DEFAULT_REDACTION_TEXT).toBe('[REDACTED]');
  });

  it.each(['task-runner failed', 'mask-value computed', 'monkey=10 bananas'])(
    'leaves false-positive %s alone',
    (input) => {
      const { text, matches } = redactSensitiveText(input, ALL);
      expect(text).toBe(input);
      expect(matches).toEqual([]);
    }
  );
});

describe('redactSensitiveText config validation', () => {
  it('throws when a pattern lacks the global flag', () => {
    expect(() =>
      redactSensitiveText('any', {
        patterns: [
          {
            id: 'bad',
            label: 'missing /g',
            pattern: /sk-[A-Za-z0-9]+/,
          },
        ],
      })
    ).toThrow(/global \(g\) flag/);
  });

  it('throws when two patterns share an id', () => {
    expect(() =>
      redactSensitiveText('any', {
        patterns: [
          { id: 'dup', label: 'first', pattern: /sk-1/g },
          { id: 'dup', label: 'second', pattern: /sk-2/g },
        ],
      })
    ).toThrow(/duplicate pattern id/);
  });
});
