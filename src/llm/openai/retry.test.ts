import { describe, expect, it, jest } from '@jest/globals';
import { createOpenAIRetryCaller, handleOpenAIFailedAttempt } from './retry';

function providerError(
  status?: number,
  code?: string
): Error & {
  code?: string;
  status?: number;
  retryAfterMs?: number;
  headers?: Record<string, string>;
} {
  return Object.assign(new Error('provider failure'), { status, code });
}

describe('OpenAI retry policy', () => {
  it.each([400, 401, 403, 409, 413, 422, 499])(
    'does not retry client status %i',
    (status) => {
      const error = providerError(status);
      expect(() => handleOpenAIFailedAttempt(error)).toThrow(error);
    }
  );

  it.each([408, 429, 500, 503])('retries transient status %i', (status) => {
    expect(() =>
      handleOpenAIFailedAttempt(providerError(status))
    ).not.toThrow();
  });

  it('retries recognized network failures but not programming errors', () => {
    expect(() =>
      handleOpenAIFailedAttempt(providerError(undefined, 'ECONNRESET'))
    ).not.toThrow();
    expect(() => handleOpenAIFailedAttempt(new Error('bad state'))).toThrow(
      'bad state'
    );
  });

  it('honors bounded Retry-After and rejects an excessive delay', () => {
    const bounded = providerError(429);
    bounded.headers = { 'Retry-After': '2' };
    handleOpenAIFailedAttempt(bounded);
    expect(bounded.retryAfterMs).toBe(2000);

    const excessive = providerError(429);
    excessive.headers = { 'Retry-After': '120' };
    expect(() => handleOpenAIFailedAttempt(excessive)).toThrow(excessive);
  });

  it('uses one bounded retry layer', async () => {
    const caller = createOpenAIRetryCaller({ maxRetries: 1 });
    const operation = jest
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(providerError(503))
      .mockResolvedValueOnce('ok');

    await expect(caller.call(operation)).resolves.toBe('ok');
    expect(operation).toHaveBeenCalledTimes(2);
  });
});
