import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

/**
 * Tavily scraper implementation
 * Uses the Tavily Extract API to scrape web pages
 *
 * Features:
 * - Extract content from up to 20 URLs in a single API call
 * - Supports basic and advanced extraction depths
 * - Returns raw content in markdown format
 * - Includes image extraction support
 *
 * @example
 * ```typescript
 * const scraper = createTavilyScraper({
 *   apiKey: 'your-tavily-api-key',
 *   extractDepth: 'basic',
 *   timeout: 60000
 * });
 *
 * const [url, response] = await scraper.scrapeUrl('https://example.com');
 * if (response.success) {
 *   const [content] = scraper.extractContent(response);
 *   console.log(content);
 * }
 * ```
 */
export class TavilyScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private extractDepth: 'basic' | 'advanced';
  private includeImages: boolean;

  constructor(config: t.TavilyScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.TAVILY_API_KEY ?? '';

    this.apiUrl =
      config.apiUrl ??
      process.env.TAVILY_API_URL ??
      'https://api.tavily.com/extract';

    this.timeout = config.timeout ?? 60000;
    this.extractDepth = config.extractDepth ?? 'basic';
    this.includeImages = config.includeImages ?? false;

    this.logger = config.logger || createDefaultLogger();

    if (!this.apiKey) {
      this.logger.warn('TAVILY_API_KEY is not set. Scraping will not work.');
    }

    this.logger.debug(
      `Tavily scraper initialized with API URL: ${this.apiUrl}`
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
    options: t.TavilyScrapeOptions = {}
  ): Promise<[string, t.TavilyScrapeResponse]> {
    if (!this.apiKey) {
      return [
        url,
        {
          success: false,
          error: 'TAVILY_API_KEY is not set',
        },
      ];
    }

    try {
      const payload = {
        urls: [url],
        extract_depth: options.extractDepth ?? this.extractDepth,
        include_images: options.includeImages ?? this.includeImages,
      };

      const response = await axios.post(this.apiUrl, payload, {
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        timeout: options.timeout ?? this.timeout,
      });

      const data = response.data;
      const result = data.results?.[0];

      if (result && result.error == null) {
        return [
          url,
          {
            success: true,
            data: {
              raw_content: result.raw_content ?? '',
              images: result.images ?? [],
            },
          },
        ];
      } else {
        return [
          url,
          {
            success: false,
            error: result?.error ?? 'Unknown error from Tavily API',
          },
        ];
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `Tavily Extract API request failed: ${errorMessage}`,
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
    response: t.TavilyScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success || !response.data) {
      return ['', undefined];
    }

    const content = response.data.raw_content ?? '';
    const images = response.data.images ?? [];

    const references: t.References | undefined =
      images.length > 0
        ? {
          links: [],
          images: images.map((imageUrl) => ({
            originalUrl: imageUrl,
          })),
          videos: [],
        }
        : undefined;

    return [content, references];
  }

  /**
   * Extract metadata from scrape response
   * @param response Scrape response
   * @returns Metadata object
   */
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

/**
 * Create a Tavily scraper instance
 * @param config Scraper configuration
 * @returns Tavily scraper instance
 */
export const createTavilyScraper = (
  config: t.TavilyScraperConfig = {}
): TavilyScraper => {
  return new TavilyScraper(config);
};
