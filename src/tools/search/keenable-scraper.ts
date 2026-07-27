import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_KEENABLE_SCRAPE_TIMEOUT = 15000;

/** Keyed and keyless fetch endpoints. Keenable reads any URL as clean markdown
 * without a key via the public endpoint; a key only lifts rate limits. */
const KEENABLE_FETCH_API_URL = 'https://api.keenable.ai/v1/fetch';
const KEENABLE_FETCH_PUBLIC_URL = 'https://api.keenable.ai/v1/fetch/public';

export class KeenableScraper implements t.BaseScraper {
  private apiKey: string | undefined;
  private apiUrl: string;
  private timeout: number;
  private attributionTitle: string;
  private logger: t.Logger;

  constructor(config: t.KeenableScraperConfig = {}) {
    const resolvedKey = config.apiKey ?? process.env.KEENABLE_API_KEY;
    this.apiKey =
      resolvedKey != null && resolvedKey !== '' ? resolvedKey : undefined;
    this.apiUrl =
      config.apiUrl ??
      process.env.KEENABLE_FETCH_URL ??
      (this.apiKey != null
        ? KEENABLE_FETCH_API_URL
        : KEENABLE_FETCH_PUBLIC_URL);
    this.timeout = config.timeout ?? DEFAULT_KEENABLE_SCRAPE_TIMEOUT;
    this.attributionTitle = config.attributionTitle ?? 'LibreChat';
    this.logger = config.logger || createDefaultLogger();
  }

  private buildHeaders(): Record<string, string> {
    /** X-Keenable-Title is used for traffic attribution and required on the
     * keyless endpoint; the key only lifts rate limits, so it is sent only when
     * present. */
    const headers: Record<string, string> = {
      'X-Keenable-Title': this.attributionTitle,
    };
    if (this.apiKey != null) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  async scrapeUrl(
    url: string,
    options: t.KeenableScrapeOptions = {}
  ): Promise<[string, t.KeenableScrapeResponse]> {
    if (!url || !url.trim()) {
      return [url, { success: false, error: 'URL cannot be empty' }];
    }

    try {
      const response = await axios.get<t.KeenableFetchResult>(this.apiUrl, {
        params: { url },
        headers: this.buildHeaders(),
        timeout: options.timeout ?? this.timeout,
      });

      const data = response.data;
      const content = data.content ?? '';
      if (!content) {
        return [
          url,
          { success: false, error: 'Keenable Fetch returned no content' },
        ];
      }

      return [
        url,
        {
          success: true,
          data: {
            content,
            title: data.title,
            description: data.description,
            url: data.url ?? url,
          },
        },
      ];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Keenable Fetch API request failed: ${errorMessage}`,
        },
      ];
    }
  }

  async scrapeUrls(
    urls: string[],
    options: t.KeenableScrapeOptions = {}
  ): Promise<Array<[string, t.KeenableScrapeResponse]>> {
    /** Keenable fetch is single-URL; run the batch concurrently. */
    return Promise.all(urls.map((url) => this.scrapeUrl(url, options)));
  }

  extractContent(
    response: t.KeenableScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }
    /** Keenable returns clean markdown with no separate media arrays, so there
     * are no structured references to surface. */
    return [response.data.content, undefined];
  }

  extractMetadata(response: t.KeenableScrapeResponse): t.GenericScrapeMetadata {
    if (!response.success || !response.data) {
      return {};
    }
    const metadata: t.GenericScrapeMetadata = {};
    if (response.data.title != null) {
      metadata.title = response.data.title;
    }
    if (response.data.description != null && response.data.description !== '') {
      metadata.description = response.data.description;
    }
    if (response.data.url != null) {
      metadata.url = response.data.url;
    }
    return metadata;
  }
}

export const createKeenableScraper = (
  config: t.KeenableScraperConfig = {}
): KeenableScraper => {
  return new KeenableScraper(config);
};
