import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_BASIC_TIMEOUT = 15000;
const DEFAULT_ADVANCED_TIMEOUT = 30000;
const MAX_BATCH_SIZE = 20;

const getDefaultTimeout = (extractDepth: 'basic' | 'advanced'): number =>
  extractDepth === 'advanced'
    ? DEFAULT_ADVANCED_TIMEOUT
    : DEFAULT_BASIC_TIMEOUT;

const normalizeUrlKey = (url: string): string => {
  try {
    const parsedUrl = new URL(url);
    parsedUrl.hash = '';
    if (parsedUrl.pathname.length > 1) {
      parsedUrl.pathname = parsedUrl.pathname.replace(/\/+$/, '');
    }
    return parsedUrl.toString();
  } catch {
    return url;
  }
};

const setUrlResult = (
  map: Map<string, t.TavilyExtractResult>,
  result: t.TavilyExtractResult
): void => {
  map.set(result.url, result);
  const normalizedUrl = normalizeUrlKey(result.url);
  if (!map.has(normalizedUrl)) {
    map.set(normalizedUrl, result);
  }
};

export class TavilyScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private payloadTimeout: number | undefined;
  private logger: t.Logger;
  private extractDepth: 'basic' | 'advanced';
  private includeImages: boolean;
  private includeFavicon: boolean;
  private format: 'markdown' | 'text' | undefined;

  constructor(config: t.TavilyScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.TAVILY_API_KEY ?? '';
    this.apiUrl =
      config.apiUrl ??
      process.env.TAVILY_EXTRACT_URL ??
      'https://api.tavily.com/extract';
    this.payloadTimeout = config.timeout;
    this.extractDepth = config.extractDepth ?? 'basic';
    this.timeout = config.timeout ?? getDefaultTimeout(this.extractDepth);
    this.includeImages = config.includeImages ?? false;
    this.includeFavicon = config.includeFavicon ?? false;
    this.format = config.format;
    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('TAVILY_API_KEY is not set. Scraping will not work.');
    }
  }

  async scrapeUrl(
    url: string,
    options: t.TavilyScrapeOptions = {}
  ): Promise<[string, t.TavilyScrapeResponse]> {
    const results = await this.scrapeUrls([url], options);
    return results[0];
  }

  async scrapeUrls(
    urls: string[],
    options: t.TavilyScrapeOptions = {}
  ): Promise<Array<[string, t.TavilyScrapeResponse]>> {
    if (!this.apiKey) {
      return urls.map((url) => [
        url,
        { success: false, error: 'TAVILY_API_KEY is not set' },
      ]);
    }

    const batches: string[][] = [];
    for (let i = 0; i < urls.length; i += MAX_BATCH_SIZE) {
      batches.push(urls.slice(i, i + MAX_BATCH_SIZE));
    }

    const allResults: Array<[string, t.TavilyScrapeResponse]> = [];

    for (const batch of batches) {
      const batchResults = await this.extractBatch(batch, options);
      allResults.push(...batchResults);
    }

    return allResults;
  }

  private async extractBatch(
    urls: string[],
    options: t.TavilyScrapeOptions = {}
  ): Promise<Array<[string, t.TavilyScrapeResponse]>> {
    try {
      const includeFavicon = options.includeFavicon ?? this.includeFavicon;
      const format = options.format ?? this.format;
      const extractDepth = options.extractDepth ?? this.extractDepth;
      const payload: t.TavilyExtractPayload = {
        urls,
        extract_depth: extractDepth,
        include_images: options.includeImages ?? this.includeImages,
      };

      if (includeFavicon) {
        payload.include_favicon = true;
      }
      if (format != null) {
        payload.format = format;
      }

      const effectiveTimeout =
        options.timeout ??
        this.payloadTimeout ??
        (options.extractDepth != null
          ? getDefaultTimeout(extractDepth)
          : this.timeout);
      const payloadTimeout = options.timeout ?? this.payloadTimeout;
      if (payloadTimeout != null) {
        payload.timeout = Math.min(Math.max(payloadTimeout / 1000, 1), 60);
      }

      const response = await axios.post<{
        results?: t.TavilyExtractResult[];
        failed_results?: t.TavilyExtractResult[];
      }>(this.apiUrl, payload, {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: effectiveTimeout,
      });

      const data = response.data;
      const successMap = new Map<string, t.TavilyExtractResult>();
      const failedMap = new Map<string, t.TavilyExtractResult>();

      for (const result of data.results ?? []) {
        setUrlResult(successMap, result);
      }
      for (const result of data.failed_results ?? []) {
        setUrlResult(failedMap, result);
      }

      return urls.map((url): [string, t.TavilyScrapeResponse] => {
        const success =
          successMap.get(url) ?? successMap.get(normalizeUrlKey(url));
        if (success && success.error == null) {
          return [
            url,
            {
              success: true,
              data: {
                rawContent: success.raw_content ?? '',
                images: success.images ?? [],
                favicon: success.favicon,
              },
            },
          ];
        }

        const failed =
          failedMap.get(url) ?? failedMap.get(normalizeUrlKey(url));
        const error =
          success?.error ??
          failed?.error ??
          'URL not found in Tavily Extract response';
        return [url, { success: false, error }];
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return urls.map((url) => [
        url,
        {
          success: false,
          error: `Tavily Extract API request failed: ${errorMessage}`,
        },
      ]);
    }
  }

  extractContent(
    response: t.TavilyScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    const content = response.data.rawContent ?? '';
    const images = response.data.images ?? [];

    const references: t.References | undefined =
      images.length > 0
        ? {
          links: [],
          images: images.map((imageUrl) => ({ originalUrl: imageUrl })),
          videos: [],
        }
        : undefined;

    return [content, references];
  }

  extractMetadata(response: t.TavilyScrapeResponse): t.GenericScrapeMetadata {
    if (!response.success || !response.data) {
      return {};
    }

    const metadata: t.GenericScrapeMetadata = {
      images_count: response.data.images?.length ?? 0,
    };
    if (response.data.favicon != null) {
      metadata.favicon = response.data.favicon;
    }
    return metadata;
  }
}

export const createTavilyScraper = (
  config: t.TavilyScraperConfig = {}
): TavilyScraper => {
  return new TavilyScraper(config);
};
