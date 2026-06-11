import axios from 'axios';
import type * as t from './types';

const DEFAULT_PARALLEL_TIMEOUT = 30_000;
const DEFAULT_PARALLEL_SEARCH_URL = 'https://api.parallel.ai/v1/search';
const PARALLEL_MAX_SEARCH_QUERIES = 5;
const PARALLEL_MIN_EXCERPT_CHARS = 1_000;

const ISO_COUNTRY_RE = /^[a-z]{2}$/;

const normalizeParallelLocation = (country?: string): string | undefined => {
  const normalized = country?.trim().toLowerCase();
  if (normalized == null || normalized === '') {
    return undefined;
  }
  return ISO_COUNTRY_RE.test(normalized) ? normalized : undefined;
};

const trimQuery = (value: string): string => value.trim();

const buildSearchQueries = (query: string, extras?: string[]): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  const candidates = [query, ...(extras ?? [])].map(trimQuery);
  for (const candidate of candidates) {
    if (candidate.length === 0) continue;
    const key = candidate.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(candidate);
    if (result.length >= PARALLEL_MAX_SEARCH_QUERIES) break;
  }
  return result;
};

const buildObjective = (
  query: string,
  fixedObjective?: string
): string | undefined => {
  const trimmedQuery = trimQuery(query);
  const trimmedFixed = fixedObjective?.trim() ?? '';
  if (trimmedFixed.length > 0) {
    return trimmedQuery.length > 0
      ? `${trimmedFixed}\n\n${trimmedQuery}`
      : trimmedFixed;
  }
  return trimmedQuery.length > 0 ? trimmedQuery : undefined;
};

const toOrganicResults = (
  results: t.ParallelSearchResult[] | undefined
): t.OrganicResult[] => {
  if (!Array.isArray(results)) return [];
  return results.map((result) => ({
    title: result.title ?? '',
    link: result.url,
    snippet: Array.isArray(result.excerpts) ? result.excerpts.join('\n\n') : '',
    date: result.publish_date ?? undefined,
  }));
};

/**
 * Parallel Web Systems Search API client.
 *
 * Docs: https://docs.parallel.ai/api-reference/search/search
 * Best practices: https://docs.parallel.ai/search/best-practices
 *
 * Notes:
 * - Auth header is `x-api-key`, not `Authorization: Bearer`.
 * - The LLM-supplied `query` is used as both the `objective` and a
 *   `search_queries[]` entry; admins can layer a fixed objective and
 *   additional keyword queries via `parallelSearchOptions`.
 * - Per Parallel best practices, restrictive `source_policy`, `location`,
 *   and `max_results` are only emitted when explicitly configured.
 */
export const createParallelAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.ParallelSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.PARALLEL_API_KEY,
    apiUrl:
      apiUrl ?? process.env.PARALLEL_SEARCH_URL ?? DEFAULT_PARALLEL_SEARCH_URL,
    timeout: options?.timeout ?? DEFAULT_PARALLEL_TIMEOUT,
  };

  if (config.apiKey === undefined || config.apiKey === '') {
    throw new Error('PARALLEL_API_KEY is required for Parallel Search API');
  }

  const getSources = async ({
    query,
    country,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const searchQueries = buildSearchQueries(query, options?.search_queries);
      if (searchQueries.length === 0) {
        return { success: false, error: 'Query cannot be empty' };
      }

      const payload: t.ParallelSearchPayload = {
        search_queries: searchQueries,
      };

      const objective = buildObjective(query, options?.objective);
      if (objective != null) {
        payload.objective = objective;
      }
      if (options?.mode != null) {
        payload.mode = options.mode;
      }
      if (options?.max_chars_total != null) {
        payload.max_chars_total = options.max_chars_total;
      }
      if (options?.client_model != null) {
        payload.client_model = options.client_model;
      }

      const advanced: NonNullable<
        t.ParallelSearchPayload['advanced_settings']
      > = {};
      const includeDomains =
        options?.include_domains?.filter((d) => d.trim().length > 0) ?? [];
      const excludeDomains =
        options?.exclude_domains?.filter((d) => d.trim().length > 0) ?? [];
      if (includeDomains.length > 0 || excludeDomains.length > 0) {
        advanced.source_policy = {
          ...(includeDomains.length > 0 && { include_domains: includeDomains }),
          ...(excludeDomains.length > 0 && { exclude_domains: excludeDomains }),
        };
      }
      const location =
        normalizeParallelLocation(options?.location) ??
        normalizeParallelLocation(country);
      if (location != null) {
        advanced.location = location;
      }
      if (options?.max_results != null) {
        advanced.max_results = options.max_results;
      }
      if (options?.max_chars_per_result != null) {
        advanced.excerpt_settings = {
          max_chars_per_result: Math.max(
            PARALLEL_MIN_EXCERPT_CHARS,
            options.max_chars_per_result
          ),
        };
      }
      if (Object.keys(advanced).length > 0) {
        payload.advanced_settings = advanced;
      }

      const response = await axios.post<t.ParallelSearchResponse>(
        config.apiUrl,
        payload,
        {
          headers: {
            'x-api-key': config.apiKey,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
        }
      );

      const results: t.SearchResultData = {
        organic: toOrganicResults(response.data.results),
        images: [],
        topStories: [],
        videos: [],
        news: [],
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Parallel Search API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
