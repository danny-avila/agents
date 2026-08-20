import axios from 'axios';
import type * as t from './types';
import { DATE_RANGE } from './schema';

const DEFAULT_YOU_TIMEOUT = 15000;

/** Authenticated and keyless endpoints. You.com works without an API key by
 * falling back to the public agents endpoint; a key lifts rate limits and
 * unlocks the full Search API surface. The keyless endpoint rejects an
 * `X-API-Key` header, so the endpoint and the headers are always chosen
 * together — never send a stale key to the public path. */
const YOU_DEFAULT_API_URL = 'https://api.you.com/v1/search';
const YOU_PUBLIC_API_URL = 'https://api.you.com/v1/agents/search';
/** You.com's `freshness` filter has no sub-day granularity, so PAST_HOUR
 * widens to the narrowest bucket it does support. */
const YOU_DATE_RANGES: Record<DATE_RANGE, string> = {
  [DATE_RANGE.PAST_HOUR]: 'day',
  [DATE_RANGE.PAST_24_HOURS]: 'day',
  [DATE_RANGE.PAST_WEEK]: 'week',
  [DATE_RANGE.PAST_MONTH]: 'month',
  [DATE_RANGE.PAST_YEAR]: 'year',
};
const YOU_SAFE_SEARCH = ['off', 'moderate', 'strict'] as const;
/** `count` is applied per response section, and the API caps it at 100. */
const YOU_MAX_COUNT = 100;

/** Web hits carry several extracted passages from the page body; news hits
 * carry only a meta description. Prefer the passages when present. */
const resolveSnippet = (result: t.YouSearchResult): string => {
  const snippets = Array.isArray(result.snippets)
    ? result.snippets.filter((snippet) => snippet.trim() !== '')
    : [];
  if (snippets.length > 0) {
    return snippets.join('\n');
  }
  return result.description ?? '';
};

export const createYouAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.YouSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const resolvedKey = apiKey ?? process.env.YDC_API_KEY;
  const hasKey = resolvedKey != null && resolvedKey !== '';
  const timeout = options?.timeout ?? DEFAULT_YOU_TIMEOUT;
  const resolvedUrl =
    apiUrl ??
    process.env.YDC_API_URL ??
    (hasKey ? YOU_DEFAULT_API_URL : YOU_PUBLIC_API_URL);

  /** Constant for the provider's lifetime. The User-Agent identifies the host
   * application to You.com and is the only attribution signal available on the
   * keyless endpoint, where there is no key to tie usage to. The API key is
   * sent only when present, and only to the authenticated endpoint. */
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'User-Agent': `${options?.attributionTitle ?? 'LibreChat'} youdotcom-integration/danny-avila-agents`,
  };
  if (hasKey) {
    headers['X-API-Key'] = resolvedKey;
  }

  const getSources = async ({
    query,
    date,
    country,
    safeSearch,
    numResults = 8,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    const maxResults = Math.min(
      Math.max(1, options?.maxResults ?? numResults),
      YOU_MAX_COUNT
    );

    try {
      const params: t.YouSearchParams = { query, count: maxResults };
      if (date != null) {
        params.freshness = YOU_DATE_RANGES[date];
      }
      if (country != null && country !== '') {
        params.country = country.toUpperCase();
      }
      if (safeSearch != null) {
        params.safesearch = YOU_SAFE_SEARCH[safeSearch] ?? 'moderate';
      }

      const response = await axios.get<t.YouSearchResponse>(resolvedUrl, {
        headers,
        params,
        timeout,
        httpAgent: options?.httpAgent,
        httpsAgent: options?.httpsAgent,
      });

      const sections = response.data.results ?? {};
      const webResults = Array.isArray(sections.web) ? sections.web : [];
      const newsResults = Array.isArray(sections.news) ? sections.news : [];

      const organic: t.OrganicResult[] = webResults
        .slice(0, maxResults)
        .map((result, index) => ({
          position: index + 1,
          title: result.title ?? '',
          link: result.url ?? '',
          snippet: resolveSnippet(result),
          date: result.page_age,
        }));

      const topStories: t.TopStoryResult[] = newsResults
        .slice(0, maxResults)
        .map((result) => ({
          title: result.title ?? '',
          link: result.url ?? '',
          date: result.page_age,
          imageUrl: result.thumbnail_url,
        }));

      return { success: true, data: { organic, topStories } };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `You.com API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
