import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_EXA_SCRAPER_TIMEOUT = 30000;
const EXA_DEFAULT_BASE_URL = 'https://api.exa.ai';
const MAX_BATCH_SIZE = 20;

/**
 * Exa scraper. Batches URLs through Exa's contents endpoint
 * (POST {base}/contents), which returns full page text per URL plus a
 * per-URL `statuses` array for partial failures.
 */
export class ExaScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private maxAgeHours?: number;
  private livecrawlTimeout?: number;

  constructor(config: t.ExaScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.EXA_API_KEY ?? '';

    const baseUrl =
      config.apiUrl ?? process.env.EXA_API_URL ?? EXA_DEFAULT_BASE_URL;
    this.apiUrl = `${baseUrl.replace(/\/+$/, '')}/contents`;

    this.timeout = config.timeout ?? DEFAULT_EXA_SCRAPER_TIMEOUT;
    this.maxAgeHours = config.maxAgeHours;
    this.livecrawlTimeout = config.livecrawlTimeout;
    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('EXA_API_KEY is not set. Scraping will not work.');
    }
  }

  async scrapeUrl(
    url: string,
    options: t.ExaScrapeOptions = {}
  ): Promise<[string, t.ExaScrapeResponse]> {
    const results = await this.scrapeUrls([url], options);
    return results[0];
  }

  async scrapeUrls(
    urls: string[],
    options: t.ExaScrapeOptions = {}
  ): Promise<Array<[string, t.ExaScrapeResponse]>> {
    if (!this.apiKey) {
      return urls.map((url) => [
        url,
        { success: false, error: 'EXA_API_KEY is not set' },
      ]);
    }

    const batches: string[][] = [];
    for (let i = 0; i < urls.length; i += MAX_BATCH_SIZE) {
      batches.push(urls.slice(i, i + MAX_BATCH_SIZE));
    }

    const allResults: Array<[string, t.ExaScrapeResponse]> = [];

    for (const batch of batches) {
      const batchResults = await this.fetchContentsBatch(batch, options);
      allResults.push(...batchResults);
    }

    return allResults;
  }

  private async fetchContentsBatch(
    urls: string[],
    options: t.ExaScrapeOptions = {}
  ): Promise<Array<[string, t.ExaScrapeResponse]>> {
    try {
      const payload: t.ExaContentsPayload = {
        urls,
        text: true,
      };

      const maxAgeHours = options.maxAgeHours ?? this.maxAgeHours;
      if (maxAgeHours != null) {
        payload.maxAgeHours = maxAgeHours;
      }
      const livecrawlTimeout =
        options.livecrawlTimeout ?? this.livecrawlTimeout;
      if (livecrawlTimeout != null) {
        payload.livecrawlTimeout = livecrawlTimeout;
      }

      const response = await axios.post<t.ExaContentsResponse>(
        this.apiUrl,
        payload,
        {
          headers: {
            'x-api-key': this.apiKey,
            'Content-Type': 'application/json',
          },
          timeout: options.timeout ?? this.timeout,
        }
      );

      const data = response.data;
      const resultMap = new Map<string, t.ExaContentsResult>();
      for (const result of data.results ?? []) {
        if (result.id != null) {
          resultMap.set(result.id, result);
        }
        if (result.url != null) {
          resultMap.set(result.url, result);
        }
      }
      const statusMap = new Map<string, t.ExaContentsStatus>();
      for (const status of data.statuses ?? []) {
        statusMap.set(status.id, status);
      }

      return urls.map((url): [string, t.ExaScrapeResponse] => {
        const result = resultMap.get(url);
        const status = statusMap.get(url);
        if (result != null && status?.status !== 'error') {
          return [
            url,
            {
              success: true,
              data: {
                text: result.text ?? '',
                title: result.title ?? undefined,
                author: result.author ?? undefined,
                publishedDate: result.publishedDate,
                image: result.image,
                favicon: result.favicon,
              },
            },
          ];
        }

        const error =
          status?.error?.tag ??
          status?.status ??
          'URL not found in Exa contents response';
        return [url, { success: false, error }];
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return urls.map((url) => [
        url,
        {
          success: false,
          error: `Exa contents API request failed: ${errorMessage}`,
        },
      ]);
    }
  }

  extractContent(
    response: t.ExaScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    return [response.data.text ?? '', undefined];
  }

  extractMetadata(response: t.ExaScrapeResponse): t.ScrapeMetadata {
    if (!response.success || !response.data) {
      return {};
    }

    const { title, author, publishedDate, image, favicon } = response.data;
    const metadata: t.ScrapeMetadata = {};
    if (title != null) {
      metadata.title = title;
    }
    if (author != null) {
      metadata.author = author;
    }
    if (publishedDate != null) {
      metadata.publishedTime = publishedDate;
    }
    if (image != null) {
      metadata['og:image'] = image;
    }
    if (favicon != null) {
      metadata.favicon = favicon;
    }
    return metadata;
  }
}

export const createExaScraper = (config: t.ExaScraperConfig = {}): ExaScraper =>
  new ExaScraper(config);
