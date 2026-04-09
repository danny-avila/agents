import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_TIMEOUT = 15000;
const MAX_BATCH_SIZE = 20;

export class TavilyScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private extractDepth: 'basic' | 'advanced';
  private includeImages: boolean;
  private includeFavicon: boolean;
  private chunksPerSource: number | undefined;
  private format: 'markdown' | 'text' | undefined;

  constructor(config: t.TavilyScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.TAVILY_API_KEY ?? '';
    this.apiUrl =
      config.apiUrl ??
      process.env.TAVILY_EXTRACT_URL ??
      'https://api.tavily.com/extract';
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
    this.extractDepth = config.extractDepth ?? 'basic';
    this.includeImages = config.includeImages ?? false;
    this.includeFavicon = config.includeFavicon ?? false;
    this.chunksPerSource = config.chunksPerSource;
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
      for (const entry of batchResults) {
        allResults.push(entry);
      }
    }

    return allResults;
  }

  private async extractBatch(
    urls: string[],
    options: t.TavilyScrapeOptions = {}
  ): Promise<Array<[string, t.TavilyScrapeResponse]>> {
    try {
      const payload: Record<string, unknown> = {
        urls,
        extract_depth: options.extractDepth ?? this.extractDepth,
        include_images: options.includeImages ?? this.includeImages,
      };

      if (this.includeFavicon) {
        payload.include_favicon = true;
      }
      if (this.chunksPerSource != null) {
        payload.chunks_per_source = this.chunksPerSource;
      }
      if (this.format != null) {
        payload.format = this.format;
      }

      const effectiveTimeout = options.timeout ?? this.timeout;
      payload.timeout = Math.min(Math.max(effectiveTimeout / 1000, 1), 60);

      const response = await axios.post(this.apiUrl, payload, {
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
        successMap.set(result.url, result);
      }
      for (const result of data.failed_results ?? []) {
        failedMap.set(result.url, result);
      }

      return urls.map((url): [string, t.TavilyScrapeResponse] => {
        const success = successMap.get(url);
        if (success && success.error == null) {
          return [
            url,
            {
              success: true,
              data: {
                rawContent: success.raw_content ?? '',
                images: success.images ?? [],
              },
            },
          ];
        }

        const failed = failedMap.get(url);
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

  extractMetadata(
    response: t.TavilyScrapeResponse
  ): Record<string, string | number | boolean | null | undefined> {
    if (!response.success || !response.data) {
      return {};
    }

    return {
      images_count: response.data.images?.length ?? 0,
    };
  }
}

export const createTavilyScraper = (
  config: t.TavilyScraperConfig = {}
): TavilyScraper => {
  return new TavilyScraper(config);
};
