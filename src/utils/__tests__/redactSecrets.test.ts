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
});
