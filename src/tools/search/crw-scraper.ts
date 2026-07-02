import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';
import { processContent } from './content';

/** HTTP headroom over the payload render budget: fastCRW's queue/verification
 * overhead is not counted against `timeout`, so the client must wait longer. */
const CRW_TIMEOUT_BUFFER = 5000;

/**
 * fastCRW scraper. Firecrawl-compatible web scraper; single binary;
 * self-host or cloud. Posts to {base}/v1/scrape.
 */
export class CrwScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private defaultFormats: string[];
  private timeout: number;
  private logger: t.Logger;
  private onlyMainContent?: boolean;
  private includeTags?: string[];
  private excludeTags?: string[];
  private waitFor?: number;
  private headers?: Record<string, string>;
  private renderJs?: boolean | null;
  private cssSelector?: string;
  private xpath?: string;
  private proxy?: string;
  private stealth?: boolean;
  private jsonSchema?: object;

  constructor(config: t.CrwScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.CRW_API_KEY ?? '';

    const baseUrl =
      config.apiUrl ?? process.env.CRW_API_URL ?? 'https://fastcrw.com/api';
    this.apiUrl = `${baseUrl.replace(/\/+$/, '')}/v1/scrape`;

    this.defaultFormats = config.formats ?? ['markdown', 'rawHtml'];
    this.timeout = config.timeout ?? 7500;
    this.logger = config.logger || createDefaultLogger();

    this.onlyMainContent = config.onlyMainContent;
    this.includeTags = config.includeTags;
    this.excludeTags = config.excludeTags;
    this.waitFor = config.waitFor;
    this.headers = config.headers;
    this.renderJs = config.renderJs;
    this.cssSelector = config.cssSelector;
    this.xpath = config.xpath;
    this.proxy = config.proxy;
    this.stealth = config.stealth;
    this.jsonSchema = config.jsonSchema;

    // Self-host fastCRW may run without auth, so a missing key is only a
    // warning — unlike Firecrawl/Tavily, scrapeUrl does NOT early-return on it.
    if (!this.apiKey) {
      this.logger.warn('CRW_API_KEY is not set. Scraping will not work.');
    }
    this.logger.debug(`CRW scraper initialized with API URL: ${this.apiUrl}`);
  }

  async scrapeUrl(
    url: string,
    options: t.CrwScrapeOptions = {}
  ): Promise<[string, t.CrwScrapeResponse]> {
    try {
      const payloadTimeout = options.timeout ?? this.timeout;
      const payload = omitUndefined({
        url,
        formats: options.formats ?? this.defaultFormats,
        onlyMainContent: options.onlyMainContent ?? this.onlyMainContent,
        includeTags: options.includeTags ?? this.includeTags,
        excludeTags: options.excludeTags ?? this.excludeTags,
        waitFor: options.waitFor ?? this.waitFor,
        headers: options.headers ?? this.headers,
        renderJs: options.renderJs ?? this.renderJs,
        cssSelector: options.cssSelector ?? this.cssSelector,
        xpath: options.xpath ?? this.xpath,
        proxy: options.proxy ?? this.proxy,
        stealth: options.stealth ?? this.stealth,
        jsonSchema: options.jsonSchema ?? this.jsonSchema,
        timeout: payloadTimeout,
      });

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (this.apiKey) {
        headers.Authorization = `Bearer ${this.apiKey}`;
      }

      const response = await axios.post<t.CrwRawScrapeResponse>(
        this.apiUrl,
        payload,
        {
          headers,
          timeout: payloadTimeout + CRW_TIMEOUT_BUFFER,
        }
      );

      return [url, normalizeCrwResponse(response.data)];
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return [
        url,
        {
          success: false,
          error: `fastCRW API request failed: ${errorMessage}`,
        },
      ];
    }
  }

  /**
   * Extract content from scrape response. Mirrors FirecrawlScraper — reads
   * response.data.*, which normalizeCrwResponse guarantees, preserving the
   * processContent ref-markers used by the reranker. Parameter is typed as the
   * NARROWER t.CrwScrapeResponse, exactly like TavilyScraper/FirecrawlScraper:
   * TS class-method bivariance accepts this against BaseScraper's
   * AnyScraperResponse, AND it lets us read response.data.plainText (a
   * CrwScrapeResponse-only field) with no cast.
   */
  extractContent(
    response: t.CrwScrapeResponse
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

    // CRW-only fallback (no Firecrawl equivalent): plain-text body.
    if (response.data.plainText != null) {
      return [response.data.plainText, undefined];
    }

    return ['', undefined];
  }

  extractMetadata(response: t.CrwScrapeResponse): t.ScrapeMetadata {
    if (!response.success || !response.data || !response.data.metadata) {
      return {};
    }

    return response.data.metadata;
  }
}

/**
 * Create a fastCRW scraper instance
 * @param config Scraper configuration
 * @returns fastCRW scraper instance
 */
export const createCrwScraper = (config: t.CrwScraperConfig = {}): CrwScraper =>
  new CrwScraper(config);

/**
 * fastCRW cloud nests scrape fields under `data` ({success, data: {markdown,
 * ...}}, live-verified 2026-07-02), matching Firecrawl. Prefer the nested
 * container and fall back to top-level fields for self-host/legacy responses.
 */
function normalizeCrwResponse(
  raw: t.CrwRawScrapeResponse | null | undefined
): t.CrwScrapeResponse {
  if (raw == null) {
    return { success: false, error: 'Empty fastCRW response' };
  }
  if (raw.success === false) {
    return {
      success: false,
      error:
        raw.error_code != null
          ? `[${raw.error_code}] ${raw.error ?? 'Unknown error'}`
          : (raw.error ?? 'fastCRW scrape failed'),
      error_code: raw.error_code,
    };
  }
  const data = raw.data ?? raw;
  return {
    success: true,
    data: {
      markdown: data.markdown,
      html: data.html,
      rawHtml: data.rawHtml,
      plainText: data.plainText,
      screenshot: data.screenshot,
      links: data.links,
      metadata: data.metadata,
    },
  };
}

// Helper function to clean up payload for fastCRW
function omitUndefined<T extends object>(obj: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v !== undefined)
  ) as Partial<T>;
}
