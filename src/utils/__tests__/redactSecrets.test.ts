import { redactSecrets } from '@/utils/redactSecrets';

describe('redactSecrets', () => {
  it('redacts secret-bearing keys at every nesting depth', () => {
    const value = {
      message: 'Bad Request',
      config: {
        headers: {
          Authorization: 'Bearer live-secret',
          'x-api-key': 'live-key',
          Accept: 'application/json',
        },
      },
      response: {
        data: [{ password: 'hunter2', detail: 'overflow' }],
      },
    };

    const serialized = JSON.stringify(redactSecrets(value));

    expect(serialized).toContain('Bad Request');
    expect(serialized).toContain('application/json');
    expect(serialized).toContain('overflow');
    expect(serialized).not.toContain('live-secret');
    expect(serialized).not.toContain('live-key');
    expect(serialized).not.toContain('hunter2');
  });

  it('handles circular diagnostic payloads without exposing nested secrets', () => {
    const value: { request?: object; token?: string } = {
      token: 'live-token',
    };
    value.request = value;

    expect(redactSecrets(value)).toEqual({
      token: '[REDACTED]',
      request: '[CIRCULAR]',
    });
  });

  it('redacts credentials embedded in diagnostic URLs', () => {
    const value = {
      request: {
        url: 'https://user:pass@example.com/path?api_key=live-key&X-Amz-Signature=live-signature&version=1',
      },
    };

    const serialized = JSON.stringify(redactSecrets(value));

    expect(serialized).toContain('version=1');
    expect(serialized).not.toContain('live-key');
    expect(serialized).not.toContain('live-signature');
    expect(serialized).not.toContain('user');
    expect(serialized).not.toContain('pass');
  });

  it('redacts credential shapes embedded in diagnostic free text', () => {
    const value = {
      message: 'rejected Bearer sk-live-abcdef123456 at the gateway',
      stack:
        'eyJhbGciOiJSUzI1NiIsImtpZCI6ImsxIn0.eyJzdWIiOiJ1c2VyIn0.c2lnbmF0dXJl expired',
      body: '{"error":"unauthorized","token":"abc123def456"}',
      detail: 'connect https://user:pass@codeapi.internal/exec failed',
    };

    const serialized = JSON.stringify(redactSecrets(value));

    expect(serialized).not.toContain('sk-live-abcdef123456');
    expect(serialized).not.toContain('eyJhbGciOiJSUzI1NiIsImtpZCI6ImsxIn0');
    expect(serialized).not.toContain('abc123def456');
    expect(serialized).not.toContain('user:pass');
    expect(serialized).toContain('at the gateway');
    expect(serialized).toContain('expired');
    expect(serialized).toContain('unauthorized');
    expect(serialized).toContain('codeapi.internal');
  });

  it('keeps prose that names a secret without carrying one', () => {
    const value = {
      message:
        'credential helper failed for secret codeapi-signing-key in namespace internal-auth',
      body: '{"error":"rate_limited","retry_after_seconds":8.2}',
      stack: 'at Object.signJwt (/app/dist/index.cjs:1234:5)',
    };

    expect(redactSecrets(value)).toEqual(value);
  });

  it('redacts a whole credential-bearing header line, parameter lists included', () => {
    const value = {
      body: [
        'Authorization: Digest username="Mufasa", realm="x", nonce="dcd98", response="6629fae49393a05397450978507c4ef1"',
        'Cookie: session=abcdef123456',
        'X-Request-Id: req-42',
      ].join('\n'),
    };

    const serialized = JSON.stringify(redactSecrets(value));

    expect(serialized).not.toContain('Mufasa');
    expect(serialized).not.toContain('6629fae49393a05397450978507c4ef1');
    expect(serialized).not.toContain('abcdef123456');
    expect(serialized).toContain('X-Request-Id: req-42');
  });

  it('redacts a username-only credential in an embedded URL', () => {
    const value = {
      message: 'request to https://live-token@example.com/private failed',
    };

    const serialized = JSON.stringify(redactSecrets(value));

    expect(serialized).not.toContain('live-token');
    expect(serialized).toContain('example.com/private');
  });

  it('leaves an already-redacted payload unchanged', () => {
    const value = {
      body: '{"error":"unauthorized","token":"abc123def456","status":401}',
      message: 'Authorization: Bearer sk-live-abcdef123456 rejected',
    };

    const once = redactSecrets(value) as typeof value;

    expect(redactSecrets(once)).toEqual(once);
    expect(once.body).toContain('"status":401');
    expect(once.body).not.toContain('abc123def456');
  });

  it('scans a large body in linear time', () => {
    const body = 'akey'.repeat(50_000);

    const started = process.hrtime.bigint();
    redactSecrets({ body });
    const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;

    expect(elapsedMs).toBeLessThan(250);
  });
});
