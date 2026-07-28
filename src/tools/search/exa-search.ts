import axios from 'axios';
import type * as t from './types';
import { DATE_RANGE } from './schema';

const DEFAULT_EXA_TIMEOUT = 15000;
const EXA_DEFAULT_BASE_URL = 'https://api.exa.ai';

const EXA_DATE_RANGE_HOURS: Record<DATE_RANGE, number> = {
  [DATE_RANGE.PAST_HOUR]: 1,
  [DATE_RANGE.PAST_24_HOURS]: 24,
  [DATE_RANGE.PAST_WEEK]: 24 * 7,
  [DATE_RANGE.PAST_MONTH]: 24 * 30,
  [DATE_RANGE.PAST_YEAR]: 24 * 365,
};

export const resolveExaSearchUrl = (apiUrl?: string): string => {
  const baseUrl = apiUrl ?? process.env.EXA_API_URL ?? EXA_DEFAULT_BASE_URL;
  return `${baseUrl.replace(/\/+$/, '')}/search`;
};

const toStartPublishedDate = (date: DATE_RANGE): string =>
  new Date(Date.now() - EXA_DATE_RANGE_HOURS[date] * 3600000).toISOString();

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return link;
  }
};

/** Highlights are the token-efficient default; full `text` is only present
 * when the provider is configured with `text: true`. */
const extractSnippet = (result: t.ExaSearchResult): string => {
  if (result.highlights != null && result.highlights.length > 0) {
    return result.highlights.join('\n...\n');
  }
  return result.text ?? '';
};

export const createExaAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.ExaSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.EXA_API_KEY,
    apiUrl: resolveExaSearchUrl(apiUrl),
    timeout: options?.timeout ?? DEFAULT_EXA_TIMEOUT,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('EXA_API_KEY is required for Exa API');
  }

  const getSources = async ({
    query,
    date,
    country,
    numResults = 8,
    type,
    news,
    safeSearch,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const isNews = news === true || type === 'news';
      const maxResults = Math.min(
        Math.max(1, options?.maxResults ?? numResults),
        20
      );
      const contents: t.ExaContentsRequest =
        options?.text === true ? { text: true } : { highlights: true };
      if (options?.maxAgeHours != null) {
        contents.maxAgeHours = options.maxAgeHours;
      }

      const payload: t.ExaSearchPayload = {
        query,
        type: options?.searchType ?? 'auto',
        numResults: maxResults,
        contents,
        moderation: (safeSearch ?? 1) !== 0,
      };

      const category = isNews ? 'news' : options?.category;
      if (category != null) {
        payload.category = category;
      }
      if (date != null) {
        payload.startPublishedDate = toStartPublishedDate(date);
      }
      const userLocation = country?.trim().toLowerCase();
      if (userLocation != null && /^[a-z]{2}$/.test(userLocation)) {
        payload.userLocation = userLocation;
      }
      if (
        options?.includeDomains != null &&
        options.includeDomains.length > 0
      ) {
        payload.includeDomains = options.includeDomains;
      }
      if (
        options?.excludeDomains != null &&
        options.excludeDomains.length > 0
      ) {
        payload.excludeDomains = options.excludeDomains;
      }

      const response = await axios.post<t.ExaSearchResponse>(
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

      const organic: t.OrganicResult[] = (response.data.results ?? []).map(
        (result: t.ExaSearchResult) => ({
          title: result.title ?? '',
          link: result.url ?? '',
          snippet: extractSnippet(result),
          date: result.publishedDate,
        })
      );

      const newsResults: t.NewsResult[] = isNews
        ? organic.map((r) => ({
          title: r.title,
          link: r.link,
          snippet: r.snippet,
          date: r.date,
          source: getHostname(r.link),
        }))
        : [];

      const results: t.SearchResultData = {
        organic,
        topStories: [],
        images: [],
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
        error: `Exa API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
