import { getProxyForUrl } from 'proxy-from-env';
import { HttpsProxyAgent } from 'https-proxy-agent';

export interface AxiosProxyOptions {
  httpsAgent?: HttpsProxyAgent<string>;
  proxy?: false;
}

/**
 * Resolves proxy options for axios requests to HTTPS search/reranker endpoints.
 *
 * Axios' built-in proxy handling does not establish a CONNECT tunnel for HTTPS
 * targets, so requests through corporate forward proxies fail (typically with
 * a 502). Tunneling through `HttpsProxyAgent` with axios' own proxy handling
 * disabled matches how the rest of the codebase performs proxied requests.
 *
 * Resolution order: the `PROXY` environment variable (codebase convention,
 * applied unconditionally), then the standard `HTTPS_PROXY`/`HTTP_PROXY`/
 * `NO_PROXY` variables via `proxy-from-env`. Non-HTTPS URLs are left to
 * axios' default behavior, which handles plain HTTP proxying correctly.
 */
export const getAxiosProxyOptions = (url: string): AxiosProxyOptions => {
  if (!url.startsWith('https:')) {
    return {};
  }
  const proxyUrl =
    process.env.PROXY != null && process.env.PROXY !== ''
      ? process.env.PROXY
      : getProxyForUrl(url);
  if (proxyUrl === '') {
    return {};
  }
  return { httpsAgent: new HttpsProxyAgent(proxyUrl), proxy: false };
};
