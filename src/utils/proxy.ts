import { HttpsProxyAgent } from 'https-proxy-agent';
import { SocksProxyAgent } from 'socks-proxy-agent';
import type { RequestInit } from 'node-fetch';

type FetchAgent = NonNullable<RequestInit['agent']>;

/**
 * Whether `targetUrl` is excluded from proxying by a NO_PROXY list.
 *
 * Follows the de facto convention: comma or whitespace separated entries, an
 * optional leading dot meaning "this host and its subdomains", an optional
 * `:port` suffix that must match when present, and `*` meaning "bypass
 * everything".
 */
export function shouldBypassProxy(
  targetUrl: string,
  noProxy?: string
): boolean {
  const list = (noProxy ?? '').trim();
  if (list === '') {
    return false;
  }

  let hostname: string;
  let port: string;
  try {
    const url = new URL(targetUrl);
    hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, '');
    port = url.port || (url.protocol === 'https:' ? '443' : '80');
  } catch {
    /* An unparsable target tells us nothing, so do not claim a bypass. */
    return false;
  }

  return list
    .split(/[\s,]+/)
    .filter(Boolean)
    .some((raw) => {
      if (raw === '*') {
        return true;
      }
      const match = /^(.*?)(?::(\d+))?$/.exec(raw);
      if (!match) {
        return false;
      }
      const entryHost = (match[1] ?? '').replace(/^\./, '').toLowerCase();
      const entryPort = match[2];
      if (entryHost === '') {
        return false;
      }
      if (entryPort != null && entryPort !== port) {
        return false;
      }
      return hostname === entryHost || hostname.endsWith(`.${entryHost}`);
    });
}

/**
 * Build the proxy agent for a `node-fetch` request, or `undefined` when the
 * request should go direct.
 *
 * Two rules that a bare `new HttpsProxyAgent(process.env.PROXY)` gets wrong:
 *
 *  - The agent must match the proxy scheme. `HttpsProxyAgent` speaks HTTP
 *    CONNECT; handing it a `socks5://` URL produces a request the proxy never
 *    answers in that protocol, surfacing as "Proxy connection ended before
 *    receiving CONNECT response".
 *  - NO_PROXY has to be honoured. Code execution targets an internal endpoint
 *    (`http://codeapi:3112`), and forcing that through an external proxy makes
 *    the whole feature unusable for anyone who has PROXY set, with no way to
 *    opt out.
 *
 * @param targetUrl Absolute URL the request is going to.
 * @param proxyUrl Explicit proxy; falls back to the PROXY environment variable.
 */
export function resolveFetchProxyAgent(
  targetUrl: string,
  proxyUrl?: string
): FetchAgent | undefined {
  const proxy = (proxyUrl ?? process.env.PROXY ?? '').trim();
  if (proxy === '') {
    return undefined;
  }

  const noProxy = process.env.NO_PROXY ?? process.env.no_proxy;
  if (shouldBypassProxy(targetUrl, noProxy)) {
    return undefined;
  }

  return /^socks/i.test(proxy)
    ? (new SocksProxyAgent(proxy) as unknown as FetchAgent)
    : (new HttpsProxyAgent(proxy) as unknown as FetchAgent);
}
