import axios from 'axios';
import type * as t from './types';

const DEFAULT_KEENABLE_TIMEOUT = 15000;

/** Authenticated and keyless endpoints. Keenable works without an API key by
 * falling back to the public endpoint; a key only lifts rate limits. */
const KEENABLE_DEFAULT_API_URL = 'https://api.keenable.ai/v1/search';
const KEENABLE_PUBLIC_API_URL = 'https://api.keenable.ai/v1/search/public';

export const createKeenableAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.KeenableSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const resolvedKey = apiKey ?? process.env.KEENABLE_API_KEY;
  const hasKey = resolvedKey != null && resolvedKey !== '';
  const timeout = options?.timeout ?? DEFAULT_KEENABLE_TIMEOUT;
  const resolvedUrl =
    apiUrl ??
    process.env.KEENABLE_API_URL ??
    (hasKey ? KEENABLE_DEFAULT_API_URL : KEENABLE_PUBLIC_API_URL);

  /** Constant for the provider's lifetime. X-Keenable-Title is required for
   * keyless requests and used for traffic attribution; the API key only lifts
   * rate limits, so it is sent only when present. */
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Keenable-Title': options?.attributionTitle ?? 'LibreChat',
  };
  if (hasKey) {
    headers['X-API-Key'] = resolvedKey;
  }

  const getSources = async ({
    query,
    numResults = 8,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      /** Keenable's endpoint has no result-count parameter; the count is
       * applied client-side after the response (see slice below). */
      const payload: t.KeenableSearchPayload = { query };
      if (options?.site != null && options.site !== '') {
        payload.site = options.site;
      }

      const response = await axios.post<t.KeenableSearchResponse>(
        resolvedUrl,
        payload,
        { headers, timeout }
      );

      const maxResults = Math.min(
        Math.max(1, options?.maxResults ?? numResults),
        20
      );
      const rawResults = Array.isArray(response.data.results)
        ? response.data.results.slice(0, maxResults)
        : [];

      const organic: t.OrganicResult[] = rawResults.map((result) => ({
        title: result.title ?? '',
        link: result.url ?? '',
        snippet: result.description ?? result.snippet ?? '',
        date: result.published_at,
      }));

      return { success: true, data: { organic } };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Keenable API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
