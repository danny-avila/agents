import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import {
  LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT,
  SENSITIVE_VALUE_PATTERNS,
  redactLangfuseSpanMessageContent,
  resolveMessageContentRedactionConfig,
  shouldApplyMessageContentRedaction,
  type ResolvedLangfuseMessageContentRedactionConfig,
} from '@/langfuseContentRedaction';

function createSpan(
  name: string,
  attributes: Record<string, unknown>
): ReadableSpan {
  return { name, attributes } as unknown as ReadableSpan;
}

function enabledConfig(
  overrides: Partial<ResolvedLangfuseMessageContentRedactionConfig> = {}
): ResolvedLangfuseMessageContentRedactionConfig {
  return {
    enabled: true,
    patterns: SENSITIVE_VALUE_PATTERNS,
    redactionText: LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT,
    ...overrides,
  };
}

describe('Langfuse message content redaction', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe('built-in patterns', () => {
    it.each([
      [
        'OpenAI key',
        'here is sk-proj-AbCdEfGhIjKlMnOpQrStUv09',
        'here is sk-[REDACTED]',
      ],
      [
        'Anthropic key',
        'use sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv09_-',
        'use sk-ant-[REDACTED]',
      ],
      [
        'Langfuse public key',
        'pk-lf-AbCdEfGhIjKlMnOpQrStUv',
        'pk-lf-[REDACTED]',
      ],
      [
        'Langfuse secret key',
        'sk-lf-AbCdEfGhIjKlMnOpQrStUv',
        'sk-lf-[REDACTED]',
      ],
      ['AWS access key', 'AKIAIOSFODNN7EXAMPLE oops', 'AKIA[REDACTED] oops'],
      ['AWS session key', 'ASIAIOSFODNN7EXAMPLE', 'ASIA[REDACTED]'],
      [
        'GitHub PAT',
        'token ghp_abcdefghijklmnopqrstuvwxyz0123456789',
        'token ghp_[REDACTED]',
      ],
      [
        'GitHub OAuth',
        'token gho_abcdefghijklmnopqrstuvwxyz0123456789',
        'token gho_[REDACTED]',
      ],
      ['Slack bot token', 'xoxb-1234567890-abcdefghij', 'xoxb-[REDACTED]'],
      ['Slack user token', 'xoxp-1234567890-abcdefghij', 'xoxp-[REDACTED]'],
      [
        'Google API key',
        'AIzaSyA-1234567890abcdefghijklmnopqrstu',
        'AIza[REDACTED]',
      ],
      ['Stripe live secret', 'sk_live_1234567890abcdef', 'sk_live_[REDACTED]'],
      [
        'Stripe test publishable',
        'pk_test_1234567890abcdef',
        'pk_test_[REDACTED]',
      ],
      ['Bearer token', 'auth Bearer abc.def-ghi=jkl', 'auth Bearer [REDACTED]'],
      ['api-key header', 'api-key: abc-def-ghi', 'api-key: [REDACTED]'],
      [
        'api_key query',
        'https://example.test/?api_key=abc123&next=true',
        'https://example.test/?api_key=[REDACTED]&next=true',
      ],
      [
        'JWT triplet',
        'jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.abcdefghijk rest',
        'jwt eyJ[REDACTED] rest',
      ],
    ])('redacts %s', (_label, input, expected) => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]: input,
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        expected
      );
    });
  });

  describe('false-positive resistance', () => {
    it.each([
      'task-runner failed',
      'mask-value computed',
      'monkey=10 bananas',
      'the secret of comedy is timing',
      'plain words without prefixes',
    ])('leaves %s alone', (input) => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]: input,
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        input
      );
    });
  });

  describe('multi-prefix disambiguation', () => {
    it('does not let the OpenAI pattern eat Anthropic keys', () => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]:
          'sk-ant-api03-1234567890abcdefghij and sk-proj-1234567890abcdefghij',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        'sk-ant-[REDACTED] and sk-[REDACTED]'
      );
    });

    it('does not let the OpenAI pattern eat Langfuse keys', () => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]: 'sk-lf-1234567890abcdefghij',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        'sk-lf-[REDACTED]'
      );
    });
  });

  describe('serialized message arrays', () => {
    it('redacts content inside a JSON-serialized chat history', () => {
      const messages = [
        {
          role: 'user',
          content: 'my key is sk-proj-1234567890abcdefghij please help',
        },
        { role: 'assistant', content: 'never share keys' },
      ];
      const span = createSpan('gpt-4o', {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
          JSON.stringify(messages),
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      const parsed = JSON.parse(
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] as string
      ) as Array<{ role: string; content: string }>;
      expect(parsed[0].content).toBe('my key is sk-[REDACTED] please help');
      expect(parsed[1].content).toBe('never share keys');
    });

    it('redacts text parts inside multimodal content arrays', () => {
      const messages = [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'token ghp_abcdefghijklmnopqrstuvwxyz0123456789 ok',
            },
            {
              type: 'image_url',
              image_url: { url: 'https://example.test/img.png' },
            },
          ],
        },
      ];
      const span = createSpan('gpt-4o', {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
          JSON.stringify(messages),
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      const parsed = JSON.parse(
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] as string
      ) as Array<{
        content: Array<
          | { type: 'text'; text: string }
          | { type: 'image_url'; image_url: { url: string } }
        >;
      }>;
      const parts = parsed[0].content;
      expect(parts[0]).toEqual({
        type: 'text',
        text: 'token ghp_[REDACTED] ok',
      });
      expect(parts[1]).toEqual({
        type: 'image_url',
        image_url: { url: 'https://example.test/img.png' },
      });
    });

    it('preserves tool inputs while redacting credentials in user text', () => {
      const messages = [
        {
          role: 'user',
          content: 'run select 1; my key is sk-ant-api03-1234567890abcdefghij',
        },
        {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_sql',
              name: 'execute_sql',
              args: { query: 'SELECT 1' },
            },
          ],
        },
      ];
      const span = createSpan('gpt-4o', {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
          JSON.stringify(messages),
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      const parsed = JSON.parse(
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] as string
      ) as Array<{
        role: string;
        content: string;
        tool_calls?: Array<{ args: { query: string } }>;
      }>;
      expect(parsed[0].content).toBe(
        'run select 1; my key is sk-ant-[REDACTED]'
      );
      expect(parsed[1].tool_calls?.[0].args.query).toBe('SELECT 1');
    });

    it('leaves non-JSON string attributes alone when no patterns match', () => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_OUTPUT]: 'plain assistant response',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_OUTPUT]).toBe(
        'plain assistant response'
      );
    });
  });

  describe('config resolution', () => {
    it('is disabled by default', () => {
      delete process.env.LANGFUSE_REDACT_MESSAGE_CONTENT;
      const config = resolveMessageContentRedactionConfig();
      expect(config.enabled).toBe(false);
      expect(shouldApplyMessageContentRedaction(config)).toBe(false);
    });

    it('reads LANGFUSE_REDACT_MESSAGE_CONTENT from env', () => {
      process.env.LANGFUSE_REDACT_MESSAGE_CONTENT = 'true';
      const config = resolveMessageContentRedactionConfig();
      expect(config.enabled).toBe(true);
      expect(shouldApplyMessageContentRedaction(config)).toBe(true);
    });

    it('limits patterns to the configured allowlist', () => {
      process.env.LANGFUSE_REDACT_MESSAGE_CONTENT = 'true';
      process.env.LANGFUSE_REDACT_MESSAGE_CONTENT_PATTERNS =
        'openai_api_key,unknown_id';
      const config = resolveMessageContentRedactionConfig();
      expect(config.patterns.map((p) => p.id)).toEqual(['openai_api_key']);
    });

    it('honors a custom redaction text', () => {
      process.env.LANGFUSE_REDACT_MESSAGE_CONTENT = 'true';
      process.env.LANGFUSE_MESSAGE_CONTENT_REDACTION_TEXT = '[scrubbed]';
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]:
          'key sk-proj-1234567890abcdefghij ok',
      });

      redactLangfuseSpanMessageContent(
        span,
        resolveMessageContentRedactionConfig()
      );

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        'key sk-[scrubbed] ok'
      );
    });

    it('lets agent config override run config', () => {
      const config = resolveMessageContentRedactionConfig(
        { messageContentRedaction: { enabled: false } },
        { messageContentRedaction: { enabled: true } }
      );
      expect(config.enabled).toBe(true);
    });

    it('is a no-op when disabled even if patterns match', () => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.TRACE_INPUT]:
          'sk-proj-1234567890abcdefghij',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig({ enabled: false }));

      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        'sk-proj-1234567890abcdefghij'
      );
    });
  });

  describe('span attribute coverage', () => {
    it('walks all four input/output attribute keys', () => {
      const span = createSpan('chat', {
        [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
          'in sk-proj-1234567890abcdefghij',
        [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]:
          'out sk-proj-1234567890abcdefghij',
        [LangfuseOtelSpanAttributes.TRACE_INPUT]:
          'trace in sk-proj-1234567890abcdefghij',
        [LangfuseOtelSpanAttributes.TRACE_OUTPUT]:
          'trace out sk-proj-1234567890abcdefghij',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]
      ).toBe('in sk-[REDACTED]');
      expect(
        span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]
      ).toBe('out sk-[REDACTED]');
      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_INPUT]).toBe(
        'trace in sk-[REDACTED]'
      );
      expect(span.attributes[LangfuseOtelSpanAttributes.TRACE_OUTPUT]).toBe(
        'trace out sk-[REDACTED]'
      );
    });

    it('ignores attribute keys that are not configured for scanning', () => {
      const span = createSpan('chat', {
        'librechat.unrelated': 'sk-proj-1234567890abcdefghij',
      });

      redactLangfuseSpanMessageContent(span, enabledConfig());

      expect(span.attributes['librechat.unrelated']).toBe(
        'sk-proj-1234567890abcdefghij'
      );
    });
  });
});
