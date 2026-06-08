import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';
import {
  DEFAULT_REDACTION_TEXT,
  filterMessageContent,
  redactSensitiveText,
  redactSensitiveValue,
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
    expect(matches).toHaveLength(1);
    expect(findMatch(matches, 'anthropic_key')).toEqual({
      patternId: 'anthropic_key',
      patternLabel: 'Anthropic key',
      count: 1,
    });
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

  it('exposes the default redaction text constant', () => {
    expect(DEFAULT_REDACTION_TEXT).toBe('[REDACTED]');
  });

  it('returns no matches when patterns array is empty', () => {
    const { text, matches } = redactSensitiveText(
      'token sk-ant-abcdefghijklmn',
      { patterns: [] }
    );
    expect(text).toBe('token sk-ant-abcdefghijklmn');
    expect(matches).toEqual([]);
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

describe('redactSensitiveValue', () => {
  it('walks plain strings', () => {
    const { value, matches } = redactSensitiveValue(
      'key sk-ant-abcdefghijklmn',
      ALL
    );
    expect(value).toBe('key sk-ant-[REDACTED]');
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(1);
  });

  it('walks nested objects and arrays', () => {
    const input = {
      user: {
        text: 'hi sk-ant-abcdefghijklmn',
        items: [
          'plain',
          'ghp_abcdefghijklmnopqrstuvwxyz0123',
          { nested: 'AKIAIOSFODNN7EXAMPLE' },
        ],
      },
      meta: 42,
    };

    const { value, matches } = redactSensitiveValue(input, ALL);
    expect(value).toEqual({
      user: {
        text: 'hi sk-ant-[REDACTED]',
        items: ['plain', 'ghp_[REDACTED]', { nested: 'AKIA[REDACTED]' }],
      },
      meta: 42,
    });
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(1);
    expect(findMatch(matches, 'github_token')?.count).toBe(1);
    expect(findMatch(matches, 'aws_key')?.count).toBe(1);
  });

  it('preserves reference equality when nothing matches', () => {
    const input = { user: { text: 'plain words', items: ['a', 'b'] } };
    const { value, matches } = redactSensitiveValue(input, ALL);
    expect(value).toBe(input);
    expect(matches).toEqual([]);
  });

  it('returns primitives untouched', () => {
    expect(redactSensitiveValue(42, ALL).value).toBe(42);
    expect(redactSensitiveValue(true, ALL).value).toBe(true);
    expect(redactSensitiveValue(null, ALL).value).toBe(null);
    expect(redactSensitiveValue(undefined, ALL).value).toBe(undefined);
  });
});

describe('filterMessageContent', () => {
  it('scrubs string content on a HumanMessage and reports matches', () => {
    const input: HumanMessage[] = [
      new HumanMessage('key sk-ant-abcdefghijklmn please'),
    ];
    const { messages, matches } = filterMessageContent(input, ALL);

    expect(messages[0].content).toBe('key sk-ant-[REDACTED] please');
    expect(messages[0]).toBeInstanceOf(HumanMessage);
    expect(messages[0]).not.toBe(input[0]);
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(1);
  });

  it('preserves reference equality for messages without matches', () => {
    const clean = new SystemMessage('be helpful');
    const dirty = new HumanMessage('hi sk-ant-abcdefghijklmn');
    const { messages } = filterMessageContent([clean, dirty], ALL);

    expect(messages[0]).toBe(clean);
    expect(messages[1]).not.toBe(dirty);
  });

  it('scrubs text parts inside multimodal content arrays and leaves other parts alone', () => {
    const input = new HumanMessage({
      content: [
        { type: 'text', text: 'key ghp_abcdefghijklmnopqrstuvwxyz0123 ok' },
        {
          type: 'image_url',
          image_url: { url: 'https://example.test/img.png' },
        },
      ],
    });
    const { messages, matches } = filterMessageContent([input], ALL);

    const parts = messages[0].content as Array<
      | { type: 'text'; text: string }
      | { type: 'image_url'; image_url: { url: string } }
    >;
    expect(parts[0]).toEqual({ type: 'text', text: 'key ghp_[REDACTED] ok' });
    expect(parts[1]).toEqual({
      type: 'image_url',
      image_url: { url: 'https://example.test/img.png' },
    });
    expect(findMatch(matches, 'github_token')?.count).toBe(1);
  });

  it('aggregates match counts across multiple messages', () => {
    const input = [
      new HumanMessage('first sk-ant-aaaaaaaaaa'),
      new AIMessage('seen sk-ant-bbbbbbbbbb earlier'),
      new SystemMessage('clean'),
    ];
    const { messages, matches } = filterMessageContent(input, ALL);

    expect(messages[0].content).toBe('first sk-ant-[REDACTED]');
    expect(messages[1].content).toBe('seen sk-ant-[REDACTED] earlier');
    expect(messages[2]).toBe(input[2]);
    expect(findMatch(matches, 'anthropic_key')?.count).toBe(2);
  });

  it('returns an empty result for an empty input', () => {
    const { messages, matches } = filterMessageContent([], ALL);
    expect(messages).toEqual([]);
    expect(matches).toEqual([]);
  });
});
