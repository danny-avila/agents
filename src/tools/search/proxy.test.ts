import { HttpsProxyAgent } from 'https-proxy-agent';

import { getAxiosProxyOptions } from './proxy';

const PROXY_ENV_VARS = [
  'PROXY',
  'HTTP_PROXY',
  'HTTPS_PROXY',
  'NO_PROXY',
  'http_proxy',
  'https_proxy',
  'no_proxy',
] as const;

describe('getAxiosProxyOptions', () => {
  const savedEnv: Partial<Record<string, string>> = {};

  beforeEach(() => {
    for (const name of PROXY_ENV_VARS) {
      savedEnv[name] = process.env[name];
      delete process.env[name];
    }
  });

  afterEach(() => {
    for (const name of PROXY_ENV_VARS) {
      const value = savedEnv[name];
      if (value == null) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  it('returns no options when no proxy is configured', () => {
    expect(getAxiosProxyOptions('https://api.tavily.com/search')).toEqual({});
  });

  it('returns no options for non-HTTPS URLs', () => {
    process.env.PROXY = 'http://proxy.internal:8080';
    expect(getAxiosProxyOptions('http://searxng.local/search')).toEqual({});
  });

  it('tunnels through the PROXY environment variable', () => {
    process.env.PROXY = 'http://proxy.internal:8080';
    const options = getAxiosProxyOptions('https://api.tavily.com/search');
    expect(options.httpsAgent).toBeInstanceOf(HttpsProxyAgent);
    expect(options.proxy).toBe(false);
  });

  it('tunnels through HTTPS_PROXY when PROXY is not set', () => {
    process.env.HTTPS_PROXY = 'http://proxy.internal:8080';
    const options = getAxiosProxyOptions('https://api.cohere.com/v2/rerank');
    expect(options.httpsAgent).toBeInstanceOf(HttpsProxyAgent);
    expect(options.proxy).toBe(false);
  });

  it('prefers PROXY over HTTPS_PROXY', () => {
    process.env.PROXY = 'http://primary.internal:8080';
    process.env.HTTPS_PROXY = 'http://secondary.internal:3128';
    const options = getAxiosProxyOptions('https://api.jina.ai/v1/rerank');
    expect(options.httpsAgent?.proxy.href).toBe(
      'http://primary.internal:8080/'
    );
  });

  it('ignores an empty PROXY value', () => {
    process.env.PROXY = '';
    expect(getAxiosProxyOptions('https://api.tavily.com/search')).toEqual({});
  });

  it('honors NO_PROXY exclusions for standard variables', () => {
    process.env.HTTPS_PROXY = 'http://proxy.internal:8080';
    process.env.NO_PROXY = 'firecrawl.internal';
    expect(
      getAxiosProxyOptions('https://firecrawl.internal/v1/scrape')
    ).toEqual({});
    expect(
      getAxiosProxyOptions('https://api.tavily.com/search').httpsAgent
    ).toBeInstanceOf(HttpsProxyAgent);
  });
});
