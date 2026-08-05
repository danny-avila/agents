import { HttpsProxyAgent } from 'https-proxy-agent';
import { SocksProxyAgent } from 'socks-proxy-agent';
import { resolveFetchProxyAgent, shouldBypassProxy } from './proxy';

const CODE_ENDPOINT = 'http://codeapi:3112/v1/exec';

describe('shouldBypassProxy', () => {
  it('is false without a NO_PROXY list', () => {
    expect(shouldBypassProxy(CODE_ENDPOINT, undefined)).toBe(false);
    expect(shouldBypassProxy(CODE_ENDPOINT, '  ')).toBe(false);
  });

  it('matches a bare hostname', () => {
    expect(shouldBypassProxy(CODE_ENDPOINT, 'codeapi')).toBe(true);
    expect(shouldBypassProxy(CODE_ENDPOINT, 'other')).toBe(false);
  });

  it('accepts comma and whitespace separated lists', () => {
    expect(
      shouldBypassProxy(CODE_ENDPOINT, 'localhost,codeapi,127.0.0.1')
    ).toBe(true);
    expect(shouldBypassProxy(CODE_ENDPOINT, 'localhost codeapi')).toBe(true);
  });

  it('treats a leading dot as host-and-subdomains', () => {
    expect(
      shouldBypassProxy('http://codeapi.internal:3112/v1', '.internal')
    ).toBe(true);
    expect(shouldBypassProxy('http://internal:3112/v1', '.internal')).toBe(
      true
    );
    expect(shouldBypassProxy('http://notinternal:3112/v1', '.internal')).toBe(
      false
    );
  });

  it('honours a port suffix only when it matches', () => {
    expect(shouldBypassProxy(CODE_ENDPOINT, 'codeapi:3112')).toBe(true);
    expect(shouldBypassProxy(CODE_ENDPOINT, 'codeapi:9999')).toBe(false);
  });

  it('defaults the port from the scheme when the URL omits it', () => {
    expect(
      shouldBypassProxy('https://api.example.com/v1', 'api.example.com:443')
    ).toBe(true);
    expect(
      shouldBypassProxy('http://api.example.com/v1', 'api.example.com:80')
    ).toBe(true);
    expect(
      shouldBypassProxy('https://api.example.com/v1', 'api.example.com:80')
    ).toBe(false);
  });

  it('supports the wildcard entry', () => {
    expect(shouldBypassProxy('https://anything.example.com', '*')).toBe(true);
  });

  it('does not claim a bypass for an unparsable target', () => {
    expect(shouldBypassProxy('not a url', '*')).toBe(false);
  });
});

describe('resolveFetchProxyAgent', () => {
  const saved = { ...process.env };

  afterEach(() => {
    for (const key of ['PROXY', 'NO_PROXY', 'no_proxy']) {
      delete process.env[key];
    }
    Object.assign(process.env, saved);
  });

  function clearProxyEnv(): void {
    for (const key of ['PROXY', 'NO_PROXY', 'no_proxy']) {
      delete process.env[key];
    }
  }

  it('returns undefined when no proxy is configured', () => {
    clearProxyEnv();
    expect(resolveFetchProxyAgent(CODE_ENDPOINT)).toBeUndefined();
  });

  /**
   * The defect this module exists for: HttpsProxyAgent speaks HTTP CONNECT, so
   * driving a socks5 proxy with it produces "Proxy connection ended before
   * receiving CONNECT response" rather than a working tunnel.
   */
  it('uses a SOCKS agent for socks schemes', () => {
    clearProxyEnv();
    process.env.PROXY = 'socks5://127.0.0.1:1080';

    const agent = resolveFetchProxyAgent('https://api.example.com/v1');

    expect(agent).toBeInstanceOf(SocksProxyAgent);
    expect(agent).not.toBeInstanceOf(HttpsProxyAgent);
  });

  it.each(['socks://host:1080', 'socks4://host:1080', 'socks5h://host:1080'])(
    'recognises %s as SOCKS',
    (proxy) => {
      clearProxyEnv();
      process.env.PROXY = proxy;
      expect(
        resolveFetchProxyAgent('https://api.example.com/v1')
      ).toBeInstanceOf(SocksProxyAgent);
    }
  );

  it('uses an HTTP agent for http schemes', () => {
    clearProxyEnv();
    process.env.PROXY = 'http://proxy.internal:3128';

    expect(resolveFetchProxyAgent('https://api.example.com/v1')).toBeInstanceOf(
      HttpsProxyAgent
    );
  });

  /**
   * Code execution targets an internal endpoint. Without NO_PROXY support the
   * feature is unusable for anyone with PROXY set, because the internal host is
   * not reachable through an external proxy and there is no way to opt out.
   */
  it('bypasses the proxy for a NO_PROXY host', () => {
    clearProxyEnv();
    process.env.PROXY = 'socks5://127.0.0.1:1080';
    process.env.NO_PROXY = 'codeapi,localhost';

    expect(resolveFetchProxyAgent(CODE_ENDPOINT)).toBeUndefined();
  });

  it('still proxies hosts outside NO_PROXY', () => {
    clearProxyEnv();
    process.env.PROXY = 'socks5://127.0.0.1:1080';
    process.env.NO_PROXY = 'codeapi';

    expect(resolveFetchProxyAgent('https://api.example.com/v1')).toBeInstanceOf(
      SocksProxyAgent
    );
  });

  it('honours the lowercase no_proxy spelling', () => {
    clearProxyEnv();
    process.env.PROXY = 'http://proxy.internal:3128';
    process.env.no_proxy = 'codeapi';

    expect(resolveFetchProxyAgent(CODE_ENDPOINT)).toBeUndefined();
  });

  it('prefers an explicit proxy argument over the environment', () => {
    clearProxyEnv();
    process.env.PROXY = 'http://from-env:3128';

    expect(
      resolveFetchProxyAgent(
        'https://api.example.com/v1',
        'socks5://explicit:1080'
      )
    ).toBeInstanceOf(SocksProxyAgent);
  });

  /**
   * An empty explicit proxy means "no proxy for this call", not "fall back to
   * the environment" — matching the original `proxy != null && proxy !== ''`
   * guard, where callers resolve `initParams.proxy ?? process.env.PROXY`
   * themselves and an empty result disables proxying.
   */
  it('treats an empty explicit proxy as no proxy rather than falling back to env', () => {
    clearProxyEnv();
    process.env.PROXY = 'http://from-env:3128';

    expect(
      resolveFetchProxyAgent('https://api.example.com/v1', '')
    ).toBeUndefined();
  });
});
