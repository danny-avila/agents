import axios from 'axios';
import type * as t from './types';
import { getHostname } from './utils';

const DEFAULT_BRAVE_API_URL = 'https://api.search.brave.com/res';
const DEFAULT_BRAVE_TIMEOUT = 15000;

const BRAVE_WEB_MAX_COUNT = 20;
const BRAVE_NEWS_MAX_COUNT = 50;
const BRAVE_IMAGES_MAX_COUNT = 200;
const BRAVE_VIDEOS_MAX_COUNT = 50;

const BRAVE_SUPPORTED_COUNTRIES = new Set([
  'AR',
  'AU',
  'AT',
  'BE',
  'BR',
  'CA',
  'CL',
  'DK',
  'FI',
  'FR',
  'DE',
  'GR',
  'HK',
  'IN',
  'ID',
  'IT',
  'JP',
  'KR',
  'MY',
  'MX',
  'NL',
  'NZ',
  'NO',
  'CN',
  'PL',
  'PT',
  'PH',
  'RU',
  'SA',
  'ZA',
  'ES',
  'SE',
  'CH',
  'TW',
  'TR',
  'GB',
  'US',
  'ALL',
]);
const BRAVE_SAFE_SEARCH_OPTIONS: t.BraveSafeSearch[] = [
  'off',
  'moderate',
  'strict',
];

const mapTimeRangeToBraveFreshnessParam = (
  date?: t.GetSourcesParams['date']
): t.BraveTimeRange | undefined => {
  switch (date) {
  case 'h':
  case 'd':
    return 'pd';
  case 'w':
    return 'pw';
  case 'm':
    return 'pm';
  case 'y':
    return 'py';
  default:
    return undefined;
  }
};

const normalizeBraveCountry = (country?: string): string | undefined => {
  if (country == null) {
    return undefined;
  }
  const trimmed = country.trim();
  if (trimmed === '' || !/^[a-zA-Z]{2}$/.test(trimmed)) {
    return undefined;
  }
  const upper = trimmed.toUpperCase();
  return BRAVE_SUPPORTED_COUNTRIES.has(upper) ? upper : undefined;
};

const convertResponseToOrganicResult = (
  results: t.BraveWebResult[] | undefined
): t.OrganicResult[] =>
  (results ?? []).reduce<t.OrganicResult[]>((acc, result) => {
    if (typeof result.url !== 'string' || result.url === '') {
      return acc;
    }
    acc.push({
      position: acc.length + 1,
      title: result.title ?? '',
      link: result.url,
      snippet: result.description ?? result.extra_snippets?.[0] ?? '',
      date: result.age ?? result.page_age,
    });
    return acc;
  }, []);

const convertResponseToNewsResult = (
  results: t.BraveNewsResult[] | undefined
): t.NewsResult[] =>
  (results ?? []).reduce<t.NewsResult[]>((acc, result) => {
    if (typeof result.url !== 'string' || result.url === '') {
      return acc;
    }
    acc.push({
      position: acc.length + 1,
      title: result.title,
      link: result.url,
      snippet: result.description ?? result.extra_snippets?.[0],
      date: result.age ?? result.page_age,
      source:
        result.source ??
        result.profile?.name ??
        result.meta_url?.hostname ??
        getHostname(result.url),
      imageUrl: result.thumbnail?.src,
    });
    return acc;
  }, []);

const convertResponseToImageResult = (
  results: t.BraveImageResult[] | undefined
): t.ImageResult[] =>
  (results ?? []).reduce<t.ImageResult[]>((acc, result) => {
    const imageUrl = result.properties?.url ?? result.thumbnail?.src;
    if (imageUrl == null || imageUrl === '') {
      return acc;
    }
    acc.push({
      position: acc.length + 1,
      title: result.title,
      imageUrl,
      thumbnailUrl: result.thumbnail?.src,
      thumbnailWidth: result.thumbnail?.width,
      thumbnailHeight: result.thumbnail?.height,
      source: result.source,
      link: result.url,
    });
    return acc;
  }, []);

const convertResponseToVideoResult = (
  results: t.BraveVideoResult[] | undefined
): t.VideoResult[] =>
  (results ?? []).reduce<t.VideoResult[]>((acc, result) => {
    if (typeof result.url !== 'string' || result.url === '') {
      return acc;
    }
    acc.push({
      position: acc.length + 1,
      title: result.title,
      link: result.url,
      snippet: result.description,
      duration: result.video?.duration,
      channel: result.video?.creator ?? result.video?.author?.name,
      source:
        result.video?.publisher ??
        result.meta_url?.hostname ??
        getHostname(result.url),
      date: result.age ?? result.page_age,
      imageUrl: result.thumbnail?.src,
    });
    return acc;
  }, []);

export const createBraveAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.BraveSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.BRAVE_API_KEY,
    apiUrl: apiUrl ?? process.env.BRAVE_API_URL ?? DEFAULT_BRAVE_API_URL,
    timeout: options?.timeout ?? DEFAULT_BRAVE_TIMEOUT,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('BRAVE_API_KEY is required for Brave API');
  }

  const headers = {
    Accept: 'application/json',
    'X-Subscription-Token': config.apiKey,
  };

  const fetchWeb = async (
    query: string,
    freshness: t.BraveTimeRange | undefined,
    country: string | undefined,
    safeSearch: t.BraveSafeSearch,
    count: number,
    goggles: string | undefined
  ): Promise<{
    organic: t.OrganicResult[];
    news: t.NewsResult[];
    videos: t.VideoResult[];
  }> => {
    const params: Record<string, string | number> = {
      q: query,
      ...(freshness != null ? { freshness } : {}),
      ...(country != null ? { country } : {}),
      safesearch: safeSearch,
      count: Math.min(Math.max(1, count), BRAVE_WEB_MAX_COUNT),
      ...(goggles != null && goggles !== '' ? { goggles } : {}),
    };
    const response = await axios.get<t.BraveWebSearchResponse>(
      `${config.apiUrl}/v1/web/search`,
      { headers, params, timeout: config.timeout }
    );
    return {
      organic: convertResponseToOrganicResult(response.data.web?.results),
      news: convertResponseToNewsResult(response.data.news?.results),
      videos: convertResponseToVideoResult(response.data.videos?.results),
    };
  };

  const fetchNews = async (
    query: string,
    freshness: t.BraveTimeRange | undefined,
    country: string | undefined,
    safeSearch: t.BraveSafeSearch,
    count: number
  ): Promise<t.NewsResult[]> => {
    const params: Record<string, string | number> = {
      q: query,
      ...(freshness != null ? { freshness } : {}),
      ...(country != null ? { country } : {}),
      safesearch: safeSearch,
      count: Math.min(Math.max(1, count), BRAVE_NEWS_MAX_COUNT),
    };
    const response = await axios.get<t.BraveNewsSearchResponse>(
      `${config.apiUrl}/v1/news/search`,
      { headers, params, timeout: config.timeout }
    );
    return convertResponseToNewsResult(response.data.results);
  };

  /** Brave's image-search API only supports 'strict' or 'off'. Default to 'strict'. */
  const fetchImages = async (
    query: string,
    country: string | undefined,
    safeSearch: t.BraveSafeSearch,
    count: number
  ): Promise<t.ImageResult[]> => {
    const params: Record<string, string | number> = {
      q: query,
      ...(country != null ? { country } : {}),
      safesearch: safeSearch === 'off' ? 'off' : 'strict',
      count: Math.min(Math.max(1, count), BRAVE_IMAGES_MAX_COUNT),
    };
    const response = await axios.get<t.BraveImageSearchResponse>(
      `${config.apiUrl}/v1/images/search`,
      { headers, params, timeout: config.timeout }
    );
    return convertResponseToImageResult(response.data.results);
  };

  const fetchVideos = async (
    query: string,
    freshness: t.BraveTimeRange | undefined,
    country: string | undefined,
    safeSearch: t.BraveSafeSearch,
    count: number
  ): Promise<t.VideoResult[]> => {
    const params: Record<string, string | number> = {
      q: query,
      ...(freshness != null ? { freshness } : {}),
      ...(country != null ? { country } : {}),
      safesearch: safeSearch,
      count: Math.min(Math.max(1, count), BRAVE_VIDEOS_MAX_COUNT),
    };
    const response = await axios.get<t.BraveVideoSearchResponse>(
      `${config.apiUrl}/v1/videos/search`,
      { headers, params, timeout: config.timeout }
    );
    return convertResponseToVideoResult(response.data.results);
  };

  const getSources = async ({
    query,
    date,
    country,
    numResults = 20,
    safeSearch,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const effectiveCount = options?.count ?? numResults;
      const effectiveFreshness =
        options?.freshness ?? mapTimeRangeToBraveFreshnessParam(date);
      const effectiveSafeSearch =
        options?.safesearch ?? BRAVE_SAFE_SEARCH_OPTIONS[safeSearch ?? 1];
      const normalizedCountry = normalizeBraveCountry(country);

      if (type === 'images') {
        const images = await fetchImages(
          query,
          normalizedCountry,
          effectiveSafeSearch,
          effectiveCount
        );
        return {
          success: true,
          data: {
            organic: [],
            images,
            videos: [],
            news: [],
            topStories: [],
            relatedSearches: [],
          },
        };
      }

      if (type === 'videos') {
        const videos = await fetchVideos(
          query,
          effectiveFreshness,
          normalizedCountry,
          effectiveSafeSearch,
          effectiveCount
        );
        return {
          success: true,
          data: {
            organic: [],
            images: [],
            videos,
            news: [],
            topStories: [],
            relatedSearches: [],
          },
        };
      }

      if (type === 'news') {
        const news = await fetchNews(
          query,
          effectiveFreshness,
          normalizedCountry,
          effectiveSafeSearch,
          effectiveCount
        );
        return {
          success: true,
          data: {
            organic: [],
            images: [],
            videos: [],
            news,
            topStories: [],
            relatedSearches: [],
          },
        };
      }

      const webResponse = await fetchWeb(
        query,
        effectiveFreshness,
        normalizedCountry,
        effectiveSafeSearch,
        effectiveCount,
        options?.goggles
      );

      const results: t.SearchResultData = {
        organic: webResponse.organic,
        images: [],
        topStories: [],
        videos: webResponse.videos,
        news: webResponse.news,
        answerBox: undefined,
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Brave API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
