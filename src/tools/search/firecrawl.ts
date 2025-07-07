import axios from 'axios';
import { processContent } from './content';
import type * as t from './types';
import { createDefaultLogger } from './utils';

/**
 * Firecrawl scraper implementation
 * Uses the Firecrawl API to scrape web pages
 */
export class FirecrawlScraper {
  private apiKey: string;
  private apiUrl: string;
  private defaultFormats: string[];
  private timeout: number;
  private logger: t.Logger;

  constructor(config: t.FirecrawlScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.FIRECRAWL_API_KEY ?? '';

    const baseUrl =
      config.apiUrl ??
      process.env.FIRECRAWL_BASE_URL ??
      'https://api.firecrawl.dev';
    this.apiUrl = `${baseUrl.replace(/\/+$/, '')}/v1/scrape`;

    this.defaultFormats = config.formats ?? ['markdown', 'rawHtml'];
    this.timeout = config.timeout ?? 7500;

    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('FIRECRAWL_API_KEY is not set. Scraping will not work.');
    }

    this.logger.debug(
      `Firecrawl scraper initialized with API URL: ${this.apiUrl}`
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
    options: t.FirecrawlScrapeOptions = {}
  ): Promise<[string, t.FirecrawlScrapeResponse]> {
    if (!this.apiKey) {
      return [
        url,
        {
          success: false,
          error: 'FIRECRAWL_API_KEY is not set',
        },
      ];
    }

    try {
      const response = await axios.post(
        this.apiUrl,
        {
          url,
          formats: options.formats || this.defaultFormats,
          includeTags: options.includeTags,
          excludeTags: options.excludeTags,
          headers: options.headers,
          waitFor: options.waitFor,
          timeout: options.timeout ?? this.timeout,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
        }
      );

      return [url, response.data];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Firecrawl API request failed: ${errorMessage}`,
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
    response: t.FirecrawlScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    if (response.data.markdown != null && response.data.html != null) {
      try {
        const { markdown, ...rest } = processContent(
          response.data.html,
          response.data.markdown
        );
        return [markdown, rest];
      } catch (error) {
        this.logger.error('Error processing content:', error);
        return [response.data.markdown, undefined];
      }
    } else if (response.data.markdown != null) {
      return [response.data.markdown, undefined];
    }

    // Fall back to HTML content
    if (response.data.html != null) {
      return [response.data.html, undefined];
    }

    // Fall back to raw HTML content
    if (response.data.rawHtml != null) {
      return [response.data.rawHtml, undefined];
    }

    return ['', undefined];
  }

  /**
   * Extract metadata from scrape response
   * @param response Scrape response
   * @returns Metadata object
   */
  extractMetadata(response: t.FirecrawlScrapeResponse): t.ScrapeMetadata {
    if (!response.success || !response.data || !response.data.metadata) {
      return {};
    }

    return response.data.metadata;
  }
}

/**
 * Create a Firecrawl scraper instance
 * @param config Scraper configuration
 * @returns Firecrawl scraper instance
 */
export const createFirecrawlScraper = (
  config: t.FirecrawlScraperConfig = {}
): FirecrawlScraper => {
  return new FirecrawlScraper(config);
};
