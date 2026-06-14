import axios from 'axios';
import type * as t from './types';

const DEFAULT_CRW_TIMEOUT = 15000;

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return link;
  }
};

export const createCrwAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.CrwSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  // NOTE: fastCRW /v1/search is cloud-only, but we still allow a base-URL
  // override for parity with the scraper. Self-host may have no auth, so —
  // unlike Tavily (createTavilyAPI throws on missing key) — we do NOT throw
  // here; we only attach the Authorization header when a key is present.
  const base = (apiUrl ?? process.env.CRW_API_URL ?? 'https://fastcrw.com/api')
    .replace(/\/+$/, '');
  const config = {
    apiKey: apiKey ?? process.env.CRW_API_KEY,
    apiUrl: `${base}/v1/search`,
    timeout: options?.timeout ?? DEFAULT_CRW_TIMEOUT,
  };

  const getSources = async ({
    query,
    numResults = 8,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const limit = Math.min(Math.max(1, options?.maxResults ?? numResults), 20);
      const sources: Array<'web' | 'images'> =
        type === 'images' || options?.includeImages === true
          ? ['web', 'images']
          : ['web'];

      const payload: t.CrwSearchPayload = { query, limit, sources };

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (config.apiKey != null && config.apiKey !== '') {
        headers.Authorization = `Bearer ${config.apiKey}`;
      }

      const response = await axios.post<t.CrwSearchResponse>(
        config.apiUrl,
        payload,
        { headers, timeout: config.timeout }
      );

      const body = response.data;
      if (body.success === false) {
        return {
          success: false,
          error: `fastCRW search failed: ${
            body.error_code != null ? `[${body.error_code}] ` : ''
          }${body.error ?? 'Unknown error'}`,
        };
      }

      // OrganicResult.link is a REQUIRED string (types.ts). fastCRW marks
      // result.url as present, but defend against null/empty so we never feed a
      // broken '' link into the scraper.
      const organicResults: t.OrganicResult[] = (body.data ?? [])
        .filter((r) => r.url != null && r.url !== '')
        .map((r) => ({
          title: r.title ?? '',
          link: r.url as string,
          snippet: r.description ?? '',
        }));

      // fastCRW /v1/search exposes no native news/video verticals.
      // Mirror Tavily by deriving `news` from organic when news is requested.
      const newsResults: t.NewsResult[] =
        type === 'news'
          ? organicResults.map((r) => ({
            title: r.title,
            link: r.link,
            snippet: r.snippet,
            source: getHostname(r.link),
          }))
          : [];

      const results: t.SearchResultData = {
        organic: organicResults,
        images: [],
        topStories: [],
        videos: [],
        news: newsResults,
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `fastCRW search request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
