import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

/**
 * Crawl4AI scraper implementation
 * Uses the Crawl4AI API to scrape web pages with advanced extraction capabilities
 *
 * Features:
 * - Purpose-built for content extraction
 * - Multiple extraction strategies (cosine, LLM, etc.)
 * - Chunking strategies for large content
 * - Returns markdown and text content
 * - Includes metadata from scraped pages
 *
 * @example
 * ```typescript
 * const scraper = createCrawl4AIScraper({
 *   apiKey: 'your-crawl4ai-api-key',
 *   extractionStrategy: 'cosine',
 *   chunkingStrategy: 'sliding_window',
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
export class Crawl4AIScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private extractionStrategy?: string;
  private chunkingStrategy?: string;

  constructor(config: t.Crawl4AIScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.CRAWL4AI_API_KEY ?? '';

    this.apiUrl =
      config.apiUrl ??
      process.env.CRAWL4AI_API_URL ??
      'https://api.crawl4ai.com';

    this.timeout = config.timeout ?? 10000;
    this.extractionStrategy = config.extractionStrategy;
    this.chunkingStrategy = config.chunkingStrategy;

    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('CRAWL4AI_API_KEY is not set. Scraping will not work.');
    }

    this.logger.debug(
      `Crawl4AI scraper initialized with API URL: ${this.apiUrl}`
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
    options: t.Crawl4AIScrapeOptions = {}
  ): Promise<[string, t.Crawl4AIScrapeResponse]> {
    if (!this.apiKey) {
      return [
        url,
        {
          success: false,
          error: 'CRAWL4AI_API_KEY is not set',
        },
      ];
    }

    try {
      const payload: Record<string, unknown> = {
        url,
      };

      // Add extraction strategy if provided
      if (options.extractionStrategy ?? this.extractionStrategy) {
        payload.extractionStrategy = options.extractionStrategy ?? this.extractionStrategy;
      }

      // Add chunking strategy if provided
      if (options.chunkingStrategy ?? this.chunkingStrategy) {
        payload.chunkingStrategy = options.chunkingStrategy ?? this.chunkingStrategy;
      }

      const response = await axios.post(this.apiUrl, payload, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: options.timeout ?? this.timeout,
      });

      return [url, { success: true, data: response.data }];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      this.logger.error(`Crawl4AI scrape failed for ${url}:`, errorMessage);
      return [
        url,
        {
          success: false,
          error: `Crawl4AI API request failed: ${errorMessage}`,
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
    response: t.Crawl4AIScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    // Prefer markdown over text
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
    response: t.Crawl4AIScrapeResponse
  ): Record<string, string | number | boolean | null | undefined> {
    if (!response.success || !response.data || !response.data.metadata) {
      return {};
    }

    return response.data.metadata;
  }
}

/**
 * Create a Crawl4AI scraper instance
 * @param config Scraper configuration
 * @returns Crawl4AI scraper instance
 */
export const createCrawl4AIScraper = (
  config: t.Crawl4AIScraperConfig = {}
): Crawl4AIScraper => {
  return new Crawl4AIScraper(config);
};
