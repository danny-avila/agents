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
  private fitStrategy?: string;

  constructor(config: t.Crawl4AIScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.CRAWL4AI_API_KEY ?? '';

    this.apiUrl =
      config.apiUrl ??
      process.env.CRAWL4AI_API_URL ??
      'https://api.crawl4ai.com';

    this.timeout = config.timeout ?? 10000;
    this.extractionStrategy = config.extractionStrategy;
    this.chunkingStrategy = config.chunkingStrategy;
    this.fitStrategy = config.fitStrategy;

    // crawl4ai has ways to filter raw markdown,
    // by default, we'll assume a fit (pruning) strategy
    // to process raw markdown before passing it back
    this.fitStrategy = config.fitStrategy === "raw" ? "raw" : "fit"; 

    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.info(
        'CRAWL4AI_API_KEY is not set. Using public/unauthenticated mode.'
      );
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
    try {
      // Crawl4AI /md endpoint for simple markdown extraction
      const payload: Record<string, unknown> = {
        url,
        cache: '0', // Bypass cache by default
        f: this.fitStrategy
      };

      // Build headers - only include Authorization if API key is provided
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await axios.post(`${this.apiUrl}/md`, payload, {
        headers,
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

    // Crawl4AI /md endpoint returns markdown directly at root level
    if (response.data.markdown != null) {
      return [response.data.markdown, undefined];
    }

    // Fallback for /crawl endpoint which returns results array
    if (
      response.data.results &&
      Array.isArray(response.data.results) &&
      response.data.results.length > 0
    ) {
      const result = response.data.results[0];

      // If there's fit markdown from /crawl, try that first
      if (result.markdown?.fit_markdown != null) {
        return [result.markdown.fit_markdown, undefined]
      }

      // Extract from markdown object (Crawl4AI /crawl structure)
      if (result.markdown?.raw_markdown != null) {
        return [result.markdown.raw_markdown, undefined];
      }

      // Fallback to markdown_with_citations if raw_markdown not available
      if (result.markdown?.markdown_with_citations != null) {
        return [result.markdown.markdown_with_citations, undefined];
      }

      // Fallback to HTML if no markdown
      if (result.html != null) {
        return [result.html, undefined];
      }
    }

    // Fallback to text field
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
    if (!response.success || !response.data) {
      return {};
    }

    // Crawl4AI returns results array
    if (
      response.data.results &&
      Array.isArray(response.data.results) &&
      response.data.results.length > 0
    ) {
      const result = response.data.results[0];
      if (result.metadata) {
        return result.metadata;
      }
    }

    // Legacy format support (if data has metadata directly)
    if (response.data.metadata) {
      return response.data.metadata;
    }

    return {};
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
