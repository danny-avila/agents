import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

const DEFAULT_PARALLEL_EXTRACT_TIMEOUT = 60_000;
const DEFAULT_PARALLEL_EXTRACT_URL = 'https://api.parallel.ai/v1/extract';
const PARALLEL_EXTRACT_MAX_BATCH = 20;
const PARALLEL_MIN_EXCERPT_CHARS = 1_000;

const normalizeUrlKey = (url: string): string => {
  try {
    const parsedUrl = new URL(url);
    parsedUrl.hash = '';
    if (parsedUrl.pathname.length > 1) {
      parsedUrl.pathname = parsedUrl.pathname.replace(/\/+$/, '');
    }
    return parsedUrl.toString();
  } catch {
    return url;
  }
};

const setResult = <T extends { url: string }>(
  map: Map<string, T>,
  result: T
): void => {
  map.set(result.url, result);
  const normalized = normalizeUrlKey(result.url);
  if (!map.has(normalized)) {
    map.set(normalized, result);
  }
};

const buildExcerptContent = (result: t.ParallelExtractResult): string => {
  const fullContent = result.full_content?.trim();
  if (fullContent != null && fullContent.length > 0) {
    return fullContent;
  }
  const excerpts = Array.isArray(result.excerpts) ? result.excerpts : [];
  return excerpts.join('\n\n');
};

/**
 * Parallel Web Systems Extract API scraper.
 *
 * Docs: https://docs.parallel.ai/api-reference/extract/extract
 *
 * Notes:
 * - Auth header is `x-api-key`, not `Authorization: Bearer`.
 * - The Extract API caps batches at 20 URLs; larger sets are chunked.
 * - Defaults to excerpts-only (already markdown, LLM-ready). Set
 *   `fullContent: true` in the config to also request the full page body.
 */
export class ParallelScraper implements t.BaseScraper {
  private apiKey: string;
  private apiUrl: string;
  private timeout: number;
  private logger: t.Logger;
  private fullContent: boolean;
  private objective?: string;
  private searchQueries?: string[];
  private maxCharsTotal?: number;
  private maxCharsPerResult?: number;
  private clientModel?: string;

  constructor(config: t.ParallelScraperConfig = {}) {
    this.apiKey = config.apiKey ?? process.env.PARALLEL_API_KEY ?? '';
    this.apiUrl =
      config.apiUrl ??
      process.env.PARALLEL_EXTRACT_URL ??
      DEFAULT_PARALLEL_EXTRACT_URL;
    this.timeout = config.timeout ?? DEFAULT_PARALLEL_EXTRACT_TIMEOUT;
    this.fullContent = config.fullContent ?? false;
    this.objective = config.objective;
    this.searchQueries = config.searchQueries;
    this.maxCharsTotal = config.maxCharsTotal;
    this.maxCharsPerResult = config.maxCharsPerResult;
    this.clientModel = config.clientModel;
    this.logger = config.logger || createDefaultLogger();

    if (this.apiKey === '') {
      this.logger.warn(
        'PARALLEL_API_KEY is not set. Parallel Extract scraping will not work.'
      );
    }
  }

  async scrapeUrl(
    url: string,
    options: t.ParallelScrapeOptions = {}
  ): Promise<[string, t.ParallelScrapeResponse]> {
    const [result] = await this.scrapeUrls([url], options);
    return result;
  }

  async scrapeUrls(
    urls: string[],
    options: t.ParallelScrapeOptions = {}
  ): Promise<Array<[string, t.ParallelScrapeResponse]>> {
    if (this.apiKey === '') {
      return urls.map((url) => [
        url,
        { success: false, error: 'PARALLEL_API_KEY is not set' },
      ]);
    }

    const batches: string[][] = [];
    for (let i = 0; i < urls.length; i += PARALLEL_EXTRACT_MAX_BATCH) {
      batches.push(urls.slice(i, i + PARALLEL_EXTRACT_MAX_BATCH));
    }

    const aggregated: Array<[string, t.ParallelScrapeResponse]> = [];
    for (const batch of batches) {
      const batchResults = await this.extractBatch(batch, options);
      aggregated.push(...batchResults);
    }
    return aggregated;
  }

  private async extractBatch(
    urls: string[],
    options: t.ParallelScrapeOptions
  ): Promise<Array<[string, t.ParallelScrapeResponse]>> {
    const fullContent = options.fullContent ?? this.fullContent;
    const objective = options.objective ?? this.objective;
    const searchQueries = options.searchQueries ?? this.searchQueries;
    const maxCharsTotal = options.maxCharsTotal ?? this.maxCharsTotal;
    const maxCharsPerResult =
      options.maxCharsPerResult ?? this.maxCharsPerResult;
    const clientModel = options.clientModel ?? this.clientModel;
    const timeout = options.timeout ?? this.timeout;

    const payload: t.ParallelExtractPayload = { urls };
    if (objective != null && objective.trim().length > 0) {
      payload.objective = objective;
    }
    if (Array.isArray(searchQueries) && searchQueries.length > 0) {
      payload.search_queries = searchQueries;
    }
    if (maxCharsTotal != null) {
      payload.max_chars_total = maxCharsTotal;
    }
    if (clientModel != null) {
      payload.client_model = clientModel;
    }
    if (options.sessionId != null && options.sessionId.length > 0) {
      payload.session_id = options.sessionId;
    }
    const advanced: NonNullable<t.ParallelExtractPayload['advanced_settings']> =
      {};
    if (maxCharsPerResult != null) {
      advanced.excerpt_settings = {
        max_chars_per_result: Math.max(
          PARALLEL_MIN_EXCERPT_CHARS,
          maxCharsPerResult
        ),
      };
    }
    if (fullContent) {
      advanced.full_content_settings = { enabled: true };
    }
    if (Object.keys(advanced).length > 0) {
      payload.advanced_settings = advanced;
    }

    try {
      const response = await axios.post<t.ParallelExtractResponse>(
        this.apiUrl,
        payload,
        {
          headers: {
            'x-api-key': this.apiKey,
            'Content-Type': 'application/json',
          },
          timeout,
        }
      );

      const successMap = new Map<string, t.ParallelExtractResult>();
      const errorMap = new Map<string, t.ParallelExtractError>();
      for (const r of response.data.results ?? []) {
        setResult(successMap, r);
      }
      for (const e of response.data.errors ?? []) {
        setResult(errorMap, e);
      }

      return urls.map((url): [string, t.ParallelScrapeResponse] => {
        const success =
          successMap.get(url) ?? successMap.get(normalizeUrlKey(url));
        if (success) {
          const rawContent = buildExcerptContent(success);
          return [
            url,
            {
              success: true,
              data: {
                rawContent,
                excerpts: success.excerpts ?? [],
                ...(success.title != null && { title: success.title }),
                ...(success.publish_date != null && {
                  publishDate: success.publish_date,
                }),
              },
            },
          ];
        }
        const failure = errorMap.get(url) ?? errorMap.get(normalizeUrlKey(url));
        const detail =
          failure?.content ??
          failure?.error_type ??
          'URL not found in Parallel Extract response';
        return [url, { success: false, error: detail }];
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return urls.map((url) => [
        url,
        {
          success: false,
          error: `Parallel Extract API request failed: ${errorMessage}`,
        },
      ]);
    }
  }

  extractContent(
    response: t.ParallelScrapeResponse
  ): [string, undefined | t.References] {
    if (!response.success) {
      return ['', undefined];
    }
    return [response.data.rawContent, undefined];
  }

  extractMetadata(response: t.ParallelScrapeResponse): t.GenericScrapeMetadata {
    if (!response.success) {
      return {};
    }
    const metadata: t.GenericScrapeMetadata = {
      excerpts_count: response.data.excerpts.length,
    };
    if (response.data.title != null) {
      metadata.title = response.data.title;
    }
    if (response.data.publishDate != null) {
      metadata.publish_date = response.data.publishDate;
    }
    return metadata;
  }
}

export const createParallelScraper = (
  config: t.ParallelScraperConfig = {}
): ParallelScraper => new ParallelScraper(config);
