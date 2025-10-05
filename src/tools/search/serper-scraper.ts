import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

/**
 * Serper scraper implementation
 * Uses the Serper Scrape API (https://scrape.serper.dev) to scrape web pages
 *
 * Features:
 * - Simple API with single endpoint
 * - Returns both text and markdown content
 * - Includes metadata from scraped pages
 * - Credits-based pricing model
 *
 * @example
 * ```typescript
 * const scraper = createSerperScraper({
 *   apiKey: 'your-serper-api-key',
 *   includeMarkdown: true,
 *   timeout: 10000
 * });
 *
 * const [url, response] = await scraper.scrapeUrl('https://example.com');
 * if (response.success) {
 *   const [content] = scraper.extractContent(response);
 *   console.log(content);
 * }
 * ```
 */
export class SerperScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private includeMarkdown: boolean;

  constructor(config: t.SerperScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.SERPER_API_KEY ?? '';

    this.apiUrl =
      config.apiUrl ??
      process.env.SERPER_SCRAPE_URL ??
      'https://scrape.serper.dev';

    this.timeout = config.timeout ?? 7500;
    this.includeMarkdown = config.includeMarkdown ?? true;

    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('SERPER_API_KEY is not set. Scraping will not work.');
    }

    this.logger.debug(
      `Serper scraper initialized with API URL: ${this.apiUrl}`
    );
  }

  /**
   * Scrape a single URL
   * @param url URL to scrape
   * @param options Scrape options
   * @returns Scrape response
   */
  async scrapeUrl(
    url: string,
    options: t.SerperScrapeOptions = {}
  ): Promise<[string, t.SerperScrapeResponse]> {
    if (!this.apiKey) {
      return [
        url,
        {
          success: false,
          error: 'SERPER_API_KEY is not set',
        },
      ];
    }

    try {
      const payload = {
        url,
        includeMarkdown: options.includeMarkdown ?? this.includeMarkdown,
      };

      const response = await axios.post(this.apiUrl, payload, {
        headers: {
          'X-API-KEY': this.apiKey,
          'Content-Type': 'application/json',
        },
        timeout: options.timeout ?? this.timeout,
      });

      return [url, { success: true, data: response.data }];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Serper Scrape API request failed: ${errorMessage}`,
        },
      ];
    }
  }

  /**
   * Extract content from scrape response
   * @param response Scrape response
   * @returns Extracted content or empty string if not available
   */
  extractContent(
    response: t.SerperScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    if (response.data.markdown != null) {
      return [response.data.markdown, undefined];
    }

    if (response.data.text != null) {
      return [response.data.text, undefined];
    }

    return ['', undefined];
  }

  /**
   * Extract metadata from scrape response
   * @param response Scrape response
   * @returns Metadata object
   */
  extractMetadata(
    response: t.SerperScrapeResponse
  ): Record<string, string | number | boolean | null | undefined> {
    if (!response.success || !response.data || !response.data.metadata) {
      return {};
    }

    return response.data.metadata;
  }
}

/**
 * Create a Serper scraper instance
 * @param config Scraper configuration
 * @returns Serper scraper instance
 */
export const createSerperScraper = (
  config: t.SerperScraperConfig = {}
): SerperScraper => {
  return new SerperScraper(config);
};
