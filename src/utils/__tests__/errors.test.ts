import { describe, expect, it } from '@jest/globals';
import {
  getContextOverflowInfo,
  isLikelyContextOverflowError,
  isContextOverflowError,
  extractErrorMessage,
} from '@/utils/errors';
import {
  OVERFLOW_SIGNATURES,
  NON_OVERFLOW_SIGNATURES,
} from './fixtures/contextOverflowSignatures';
import { Providers } from '@/common';

/** Enough pressure to satisfy the corroboration gate for ambiguous errors. */
const UNDER_PRESSURE = {
  estimatedPromptTokens: 190_000,
  maxContextTokens: 200_000,
};

describe('getContextOverflowInfo — captured provider signatures', () => {
  for (const signature of OVERFLOW_SIGNATURES) {
    const label = `${signature.provider}/${signature.model} (${signature.thrownAs})`;

    it(`classifies ${label}`, () => {
      const info = getContextOverflowInfo(signature.error, {
        provider: signature.provider,
        ...(signature.requiresContextPressure === true ? UNDER_PRESSURE : {}),
      });

      expect(info).not.toBeNull();
      expect(info?.kind).toBe(signature.expected.kind);
      expect(info?.provider).toBe(signature.provider);
      expect(info?.limitTokens).toBe(signature.expected.limitTokens);
      expect(info?.requestedTokens).toBe(signature.expected.requestedTokens);
    });
  }

  it('covers every provider the SDK ships a client for, or records why not', () => {
    const covered = new Set(OVERFLOW_SIGNATURES.map((s) => s.provider));
    /**
     * Azure and Moonshot reuse the OpenAI client verbatim, and MistralAI is
     * the same client as Mistral, so the captured OpenAI/Mistral signatures
     * apply unchanged. Every other provider has its own captured entry.
     */
    const byOpenAIClient = new Set([
      Providers.AZURE,
      Providers.MOONSHOT,
      Providers.MISTRALAI,
    ]);
    for (const provider of Object.values(Providers)) {
      expect(covered.has(provider) || byOpenAIClient.has(provider)).toBe(true);
    }
  });
});

describe('getContextOverflowInfo — errors compaction cannot fix', () => {
  for (const signature of NON_OVERFLOW_SIGNATURES) {
    it(`ignores ${signature.label}`, () => {
      expect(getContextOverflowInfo(signature.error)).toBeNull();
      expect(isContextOverflowError(signature.error)).toBe(false);
    });
  }

  it('ignores an unrelated error entirely', () => {
    expect(getContextOverflowInfo(new Error('socket hang up'))).toBeNull();
    expect(getContextOverflowInfo(undefined)).toBeNull();
    expect(getContextOverflowInfo(null)).toBeNull();
  });
});

describe('token-bucket rejections', () => {
  const requestTooLarge = OVERFLOW_SIGNATURES.find(
    (s) => s.expected.kind === 'request_too_large'
  );

  it('treats a request that alone overruns the bucket as recoverable', () => {
    const info = getContextOverflowInfo(requestTooLarge?.error);
    expect(info?.kind).toBe('request_too_large');
    expect(info?.requestedTokens).toBeGreaterThan(info?.limitTokens ?? 0);
  });

  it('treats an exactly-filling request as throttling, not overflow', () => {
    /** It fits an empty bucket, so waiting can succeed — compaction would lose context needlessly. */
    const exactFit = {
      name: 'Error',
      status: 429,
      code: 'rate_limit_exceeded',
      type: 'tokens',
      message:
        '429 Request too large for gpt-4o on tokens per min (TPM): Limit 30000, Requested 30000.',
    };
    expect(getContextOverflowInfo(exactFit)).toBeNull();
  });

  it('leaves ordinary throttling alone even when phrased with the same fields', () => {
    const throttled = {
      name: 'Error',
      status: 429,
      code: 'rate_limit_exceeded',
      type: 'tokens',
      message:
        '429 Request too large for gpt-4o on tokens per min (TPM): Limit 30000, Requested 900.',
    };
    expect(getContextOverflowInfo(throttled)).toBeNull();
  });
});

describe('ambiguous signatures require corroboration', () => {
  const vertex = OVERFLOW_SIGNATURES.find(
    (s) => s.requiresContextPressure === true
  );

  it('does not fire without a caller-side pressure signal', () => {
    expect(
      getContextOverflowInfo(vertex?.error, { provider: Providers.VERTEXAI })
    ).toBeNull();
  });

  it('does not fire when the prompt was comfortably inside the budget', () => {
    expect(
      getContextOverflowInfo(vertex?.error, {
        provider: Providers.VERTEXAI,
        estimatedPromptTokens: 20_000,
        maxContextTokens: 200_000,
      })
    ).toBeNull();
  });

  it('fires once the prompt is near the budget', () => {
    expect(
      getContextOverflowInfo(vertex?.error, {
        provider: Providers.VERTEXAI,
        ...UNDER_PRESSURE,
      })
    ).not.toBeNull();
  });

  it('still reports a Vertex error that did carry a reason', () => {
    const withBody = {
      name: 'Error',
      message:
        'Google request failed with status code 400: {"error":{"code":400,"message":"The input token count exceeds the maximum number of tokens allowed (1048576).","status":"INVALID_ARGUMENT"}}',
    };
    const info = getContextOverflowInfo(withBody, {
      provider: Providers.VERTEXAI,
    });
    expect(info?.limitTokens).toBe(1_048_576);
  });
});

describe('prompt size is separated from completion-inclusive totals', () => {
  it('reads the input-only figure out of OpenRouter’s breakdown', () => {
    const openrouter = OVERFLOW_SIGNATURES.find(
      (s) => s.provider === Providers.OPENROUTER
    );
    const info = getContextOverflowInfo(openrouter?.error);
    expect(info?.requestedTokens).toBe(56_827);
    /** 56,811 of text input; the remaining 16 were the output allowance. */
    expect(info?.promptTokens).toBe(56_811);
  });

  it('reads the input-only figure out of DeepSeek’s breakdown', () => {
    const deepseek = OVERFLOW_SIGNATURES.find(
      (s) => s.provider === Providers.DEEPSEEK
    );
    const info = getContextOverflowInfo(deepseek?.error);
    expect(info?.requestedTokens).toBe(1_179_668);
    expect(info?.promptTokens).toBe(1_179_652);
  });

  it('treats an unqualified input-only count as the prompt', () => {
    const anthropic = OVERFLOW_SIGNATURES.find(
      (s) => s.provider === Providers.ANTHROPIC
    );
    const info = getContextOverflowInfo(anthropic?.error);
    expect(info?.promptTokens).toBe(info?.requestedTokens);
  });

  it('refuses to read a token-bucket total as a prompt measurement', () => {
    const openai = OVERFLOW_SIGNATURES.find(
      (s) => s.expected.kind === 'request_too_large'
    );
    const info = getContextOverflowInfo(openai?.error);
    /** "Requested 480002" folds in the completion allowance. */
    expect(info?.requestedTokens).toBe(480_002);
    expect(info?.promptTokens).toBeUndefined();
  });

  it('refuses to read xAI’s request total as a prompt measurement', () => {
    const xai = OVERFLOW_SIGNATURES.find((s) => s.provider === Providers.XAI);
    const info = getContextOverflowInfo(xai?.error);
    expect(info?.requestedTokens).toBe(332_986);
    expect(info?.promptTokens).toBeUndefined();
  });
});

describe('reported numbers are usable for recalibration', () => {
  it('reads the true ceiling even when it differs from what we configured', () => {
    const deepseek = OVERFLOW_SIGNATURES.find(
      (s) => s.provider === Providers.DEEPSEEK
    );
    const info = getContextOverflowInfo(deepseek?.error);
    expect(info?.limitTokens).toBe(1_048_565);
  });

  it('reads the provider-side count of the prompt we sent', () => {
    const bedrock = OVERFLOW_SIGNATURES.find(
      (s) => s.model === 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
    );
    const info = getContextOverflowInfo(bedrock?.error);
    expect(info?.requestedTokens).toBe(207_848);
    expect(info?.limitTokens).toBe(200_000);
  });
});

describe('back-compatible helpers', () => {
  it('accepts a bare message string', () => {
    expect(
      isContextOverflowError(
        'prompt is too long: 274468 tokens > 200000 maximum'
      )
    ).toBe(true);
    expect(isContextOverflowError('connection reset')).toBe(false);
  });

  it('retains the legacy space-separated context overflow signature', () => {
    expect(isContextOverflowError('context length exceeded')).toBe(true);
  });

  it('treats body-size rejections as likely overflow only in the loose check', () => {
    const payload = { status: 413, message: 'request entity too large' };
    expect(isContextOverflowError(payload)).toBe(false);
    expect(isLikelyContextOverflowError(payload)).toBe(true);
  });

  it('keeps the loose check away from throttling and auth', () => {
    for (const signature of NON_OVERFLOW_SIGNATURES) {
      expect(isLikelyContextOverflowError(signature.error)).toBe(false);
    }
  });

  it('extracts messages from nested provider shapes', () => {
    expect(extractErrorMessage(new Error('boom'))).toBe('boom');
    expect(extractErrorMessage({ error: { message: 'nested' } })).toBe(
      'nested'
    );
    expect(extractErrorMessage('raw')).toBe('raw');
    expect(extractErrorMessage(null)).toBe('');
  });
});
