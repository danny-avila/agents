import axios from 'axios';
import type * as t from './types';

const DEFAULT_BASE_URL = 'https://api.microsoft.ai/v3';
const DEFAULT_MAX_RESULTS = 10;
const DEFAULT_SNIPPET_LENGTH = 2000;
const REQUEST_TIMEOUT = 15000;

interface MicrosoftWebResult {
  title?: string;
  url?: string;
  content?: string;
  crawledAt?: string;
  lastUpdatedAt?: string;
  language?: string;
  isAdult?: boolean;
}

interface MicrosoftNewsResult {
  title?: string;
  url?: string;
  content?: string;
  snippet?: string;
  thumbnail?: { url?: string; width?: number; height?: number };
  lastUpdatedAt?: string;
  source?: string;
  isAdult?: boolean;
}

interface MicrosoftVideoResult {
  title?: string;
  url?: string;
  description?: string;
  summary?: string;
  publishedBy?: string;
  length?: string;
  lastUpdatedAt?: string;
  thumbnailUrl?: string;
  isAdult?: boolean;
}

interface MicrosoftImageResult {
  title?: string;
  url?: string;
  hostPageUrl?: string;
  caption?: string;
  width?: number;
  height?: number;
  thumbnailUrl?: string;
  lastUpdatedAt?: string;
  isAdult?: boolean;
}

interface MicrosoftWebResponse {
  webResults?: MicrosoftWebResult[];
  traceId?: string;
}

interface MicrosoftNewsResponse {
  newsResults?: MicrosoftNewsResult[];
  traceId?: string;
}

interface MicrosoftVideosResponse {
  videoResults?: MicrosoftVideoResult[];
  traceId?: string;
}

interface MicrosoftImagesResponse {
  imageResults?: MicrosoftImageResult[];
  traceId?: string;
}

const truncate = (text: string, maxLength: number): string =>
  text.length > maxLength ? text.slice(0, maxLength) : text;

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return '';
  }
};

/**
 * Microsoft Web IQ search API.
 *
 * Web search hits `POST /v3/search/web` with `contentFormat: 'passage'` so the
 * returned `content` is a query-relevant extract used as the organic snippet;
 * full-text grounding is left to the Browse scraper. News search hits
 * `POST /v3/search/news` and is mapped to topStories, mirroring how Serper
 * folds news results into topStories. Video search hits
 * `POST /v3/search/videos` and is mapped to the `videos` collection; image
 * search hits `POST /v3/search/images` and is mapped to the `images`
 * collection.
 */
export const createMicrosoftSearchAPI = (
  config: t.SearchConfig
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const apiKey =
    config.microsoftWebIQApiKey ?? process.env.MICROSOFT_WEBIQ_API_KEY;
  const baseUrl =
    config.microsoftWebIQBaseUrl ??
    process.env.MICROSOFT_WEBIQ_BASE_URL ??
    DEFAULT_BASE_URL;
  const options = config.microsoftWebIQSearchOptions ?? {};

  if (apiKey == null || apiKey === '') {
    throw new Error(
      'MICROSOFT_WEBIQ_API_KEY is required for Microsoft Web IQ search'
    );
  }

  const headers = {
    'x-apikey': apiKey,
    'content-type': 'application/json',
  };

  const buildBasePayload = (
    query: string,
    numResults: number
  ): Record<string, string | number> => {
    const payload: Record<string, string | number> = {
      query,
      maxResults: options.maxResults ?? numResults,
      language: options.language ?? 'en',
      region: options.region ?? 'US',
      maxLength: options.maxLength ?? 10000,
    };
    if (options.location != null && options.location !== '') {
      payload.location = options.location;
    }
    return payload;
  };

  const searchNews = async (
    query: string,
    numResults: number
  ): Promise<t.SearchResult> => {
    const payload = buildBasePayload(query, numResults);
    const response = await axios.post<MicrosoftNewsResponse>(
      `${baseUrl}/search/news`,
      payload,
      { headers, timeout: REQUEST_TIMEOUT }
    );

    const newsResults = response.data.newsResults ?? [];
    const news: t.NewsResult[] = newsResults
      .filter((item) => item.url != null && item.url !== '')
      .map((item, index) => ({
        title: item.title ?? '',
        link: item.url ?? '',
        snippet: item.snippet ?? '',
        date: item.lastUpdatedAt ?? '',
        source: item.source ?? '',
        imageUrl: item.thumbnail?.url ?? '',
        position: index + 1,
      }));

    const topStories: t.TopStoryResult[] = news.map((item) => ({
      title: item.title ?? '',
      link: item.link ?? '',
      source: item.source ?? '',
      date: item.date ?? '',
      imageUrl: item.imageUrl ?? '',
    }));

    const data: t.SearchResultData = {
      organic: [],
      topStories,
      news,
      images: [],
      videos: [],
      relatedSearches: [],
    };

    return { success: true, data };
  };

  const searchWeb = async (
    query: string,
    numResults: number
  ): Promise<t.SearchResult> => {
    const payload = {
      ...buildBasePayload(query, numResults),
      contentFormat: options.contentFormat ?? 'passage',
    };
    const response = await axios.post<MicrosoftWebResponse>(
      `${baseUrl}/search/web`,
      payload,
      { headers, timeout: REQUEST_TIMEOUT }
    );

    const webResults = response.data.webResults ?? [];
    const organic: t.OrganicResult[] = webResults
      .filter((item) => item.url != null && item.url !== '')
      .map((item, index) => ({
        position: index + 1,
        title: item.title ?? '',
        link: item.url ?? '',
        snippet: truncate(item.content ?? '', DEFAULT_SNIPPET_LENGTH),
        date: item.lastUpdatedAt ?? '',
      }));

    const data: t.SearchResultData = {
      organic,
      topStories: [],
      images: [],
      videos: [],
      news: [],
      relatedSearches: [],
    };

    return { success: true, data };
  };

  const searchVideos = async (
    query: string,
    numResults: number,
    safeSearch?: t.SafeSearchLevel
  ): Promise<t.SearchResult> => {
    const payload: Record<string, string | number> = {
      query,
      maxResults: options.maxResults ?? numResults,
      language: options.language ?? 'en',
      region: options.region ?? 'US',
      safeSearch: safeSearch === 0 ? 'off' : 'strict',
    };
    if (options.freshness != null && options.freshness !== '') {
      payload.freshness = options.freshness;
    }
    const response = await axios.post<MicrosoftVideosResponse>(
      `${baseUrl}/search/videos`,
      payload,
      { headers, timeout: REQUEST_TIMEOUT }
    );

    const videoResults = response.data.videoResults ?? [];
    const videos: t.VideoResult[] = videoResults
      .filter((item) => item.url != null && item.url !== '')
      .map((item, index) => ({
        position: index + 1,
        title: item.title ?? '',
        link: item.url ?? '',
        snippet: item.description ?? item.summary ?? '',
        imageUrl: item.thumbnailUrl ?? '',
        duration: item.length ?? '',
        channel: item.publishedBy ?? '',
        source: getHostname(item.url ?? ''),
        date: item.lastUpdatedAt ?? '',
      }));

    const data: t.SearchResultData = {
      organic: [],
      topStories: [],
      images: [],
      videos,
      news: [],
      relatedSearches: [],
    };

    return { success: true, data };
  };

  const searchImages = async (
    query: string,
    numResults: number,
    safeSearch?: t.SafeSearchLevel
  ): Promise<t.SearchResult> => {
    const payload: Record<string, string | number> = {
      query,
      maxResults: options.maxResults ?? numResults,
      language: options.language ?? 'en',
      region: options.region ?? 'US',
      safeSearch: safeSearch === 0 ? 'off' : 'strict',
    };
    const response = await axios.post<MicrosoftImagesResponse>(
      `${baseUrl}/search/images`,
      payload,
      { headers, timeout: REQUEST_TIMEOUT }
    );

    const imageResults = response.data.imageResults ?? [];
    const images: t.ImageResult[] = imageResults
      .filter((item) => item.url != null && item.url !== '')
      .map((item, index) => ({
        position: index + 1,
        title: item.title ?? '',
        imageUrl: item.url ?? '',
        imageWidth: item.width,
        imageHeight: item.height,
        thumbnailUrl: item.thumbnailUrl ?? '',
        link: item.hostPageUrl ?? '',
        source: getHostname(item.hostPageUrl ?? ''),
        domain: getHostname(item.hostPageUrl ?? ''),
      }));

    const data: t.SearchResultData = {
      organic: [],
      topStories: [],
      images,
      videos: [],
      news: [],
      relatedSearches: [],
    };

    return { success: true, data };
  };

  const getSources = async ({
    query,
    numResults = DEFAULT_MAX_RESULTS,
    type,
    news,
    safeSearch,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      if (type === 'images') {
        return await searchImages(query, numResults, safeSearch);
      }
      if (type === 'videos') {
        return await searchVideos(query, numResults, safeSearch);
      }
      if (news === true || type === 'news') {
        return await searchNews(query, numResults);
      }
      return await searchWeb(query, numResults);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Microsoft Web IQ search request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
