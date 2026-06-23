import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_BASE_URL = 'https://api.microsoft.ai/v3';
const DEFAULT_TIMEOUT = 15000;
const DEFAULT_MAX_LENGTH = 10000;

/**
 * Microsoft Web IQ Browse scraper.
 *
 * Calls `POST /v3/browse` with `liveCrawl: 'none'` so only indexed content is
 * returned. A 202 (on-demand crawl in progress) or 404 (URL not indexed) is
 * treated as a scrape failure — the source is skipped by the upstream pipeline
 * rather than retried.
 */
export class MicrosoftScraper implements t.BaseScraper {
  private apiKey: string;
  private baseUrl: string;
  private timeout: number;
  private maxLength: number;
  private contentFormat: 'text' | 'html' | 'markdown';
  private logger: t.Logger;

  constructor(config: t.MicrosoftScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.MICROSOFT_WEBIQ_API_KEY ?? '';
    this.baseUrl =
      config.baseUrl ??
      process.env.MICROSOFT_WEBIQ_BASE_URL ??
      DEFAULT_BASE_URL;
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
    this.maxLength = config.maxLength ?? DEFAULT_MAX_LENGTH;
    this.contentFormat = config.contentFormat ?? 'markdown';
    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn(
        'MICROSOFT_WEBIQ_API_KEY is not set. Scraping will not work.'
      );
    }
  }

  async scrapeUrl(
    url: string,
    options: t.MicrosoftScrapeOptions = {}
  ): Promise<[string, t.MicrosoftBrowseResponse]> {
    if (!this.apiKey) {
      return [
        url,
        { success: false, error: 'MICROSOFT_WEBIQ_API_KEY is not set' },
      ];
    }

    try {
      const payload = {
        url,
        contentFormat: options.contentFormat ?? this.contentFormat,
        maxLength: options.maxLength ?? this.maxLength,
        liveCrawl: 'none',
      };

      const response = await axios.post(`${this.baseUrl}/browse`, payload, {
        headers: {
          'x-apikey': this.apiKey,
          'content-type': 'application/json',
        },
        timeout: options.timeout ?? this.timeout,
        validateStatus: (status) => status >= 200 && status < 500,
      });

      if (response.status === 200) {
        const data = response.data;
        return [
          url,
          {
            success: true,
            data: {
              url: data.url ?? url,
              title: data.title,
              content: data.content,
              language: data.language,
              lastUpdatedAt: data.lastUpdatedAt,
              crawledAt: data.crawledAt,
              isAdult: data.isAdult,
            },
          },
        ];
      }

      return [
        url,
        {
          success: false,
          error: `Microsoft Browse returned status ${response.status}`,
        },
      ];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Microsoft Browse API request failed: ${errorMessage}`,
        },
      ];
    }
  }

  extractContent(
    response: t.MicrosoftBrowseResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }
    return [response.data.content ?? '', undefined];
  }

  extractMetadata(
    response: t.MicrosoftBrowseResponse
  ): t.GenericScrapeMetadata {
    if (!response.success || !response.data) {
      return {};
    }

    const { url, title, language, lastUpdatedAt, crawledAt } = response.data;
    const metadata: t.GenericScrapeMetadata = {};
    if (url != null) {
      metadata.url = url;
    }
    if (title != null) {
      metadata.title = title;
    }
    if (language != null) {
      metadata.language = language;
    }
    if (lastUpdatedAt != null) {
      metadata.modifiedTime = lastUpdatedAt;
    }
    if (crawledAt != null) {
      metadata.crawledAt = crawledAt;
    }
    return metadata;
  }
}

export const createMicrosoftScraper = (
  config: t.MicrosoftScraperConfig = {}
): MicrosoftScraper => {
  return new MicrosoftScraper(config);
};
