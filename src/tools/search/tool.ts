import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from './types';
import {
  DATE_RANGE,
  querySchema,
  dateSchema,
  countrySchema,
  imagesSchema,
  videosSchema,
  newsSchema,
} from './schema';
import { createSearchAPI, createSourceProcessor } from './search';
import { createSerperScraper } from './serper-scraper';
import { createFirecrawlScraper } from './firecrawl';
import { expandHighlights } from './highlights';
import { formatResultsForLLM } from './format';
import { createDefaultLogger } from './utils';
import { createReranker } from './rerankers';
import { Constants } from '@/common';

/**
 * Executes parallel searches and merges the results
 */
async function executeParallelSearches({
  searchAPI,
  query,
  date,
  country,
  safeSearch,
  images,
  videos,
  news,
  logger,
}: {
  searchAPI: ReturnType<typeof createSearchAPI>;
  query: string;
  date?: DATE_RANGE;
  country?: string;
  safeSearch: t.SearchToolConfig['safeSearch'];
  images: boolean;
  videos: boolean;
  news: boolean;
  logger: t.Logger;
}): Promise<t.SearchResult> {
  // Prepare all search tasks to run in parallel
  const searchTasks: Promise<t.SearchResult>[] = [
    // Main search
    searchAPI.getSources({
      query,
      date,
      country,
      safeSearch,
    }),
  ];

  if (images) {
    searchTasks.push(
      searchAPI
        .getSources({
          query,
          date,
          country,
          safeSearch,
          type: 'images',
        })
        .catch((error) => {
          logger.error('Error fetching images:', error);
          return {
            success: false,
            error: `Images search failed: ${error instanceof Error ? error.message : String(error)}`,
          };
        })
    );
  }
  if (videos) {
    searchTasks.push(
      searchAPI
        .getSources({
          query,
          date,
          country,
          safeSearch,
          type: 'videos',
        })
        .catch((error) => {
          logger.error('Error fetching videos:', error);
          return {
            success: false,
            error: `Videos search failed: ${error instanceof Error ? error.message : String(error)}`,
          };
        })
    );
  }
  if (news) {
    searchTasks.push(
      searchAPI
        .getSources({
          query,
          date,
          country,
          safeSearch,
          type: 'news',
        })
        .catch((error) => {
          logger.error('Error fetching news:', error);
          return {
            success: false,
            error: `News search failed: ${error instanceof Error ? error.message : String(error)}`,
          };
        })
    );
  }

  // Run all searches in parallel
  const results = await Promise.all(searchTasks);

  // Get the main search result (first result)
  const mainResult = results[0];
  if (!mainResult.success) {
    throw new Error(mainResult.error ?? 'Search failed');
  }

  // Merge additional results with the main results
  const mergedResults = { ...mainResult.data };

  // Convert existing news to topStories if present
  if (mergedResults.news !== undefined && mergedResults.news.length > 0) {
    const existingNewsAsTopStories = mergedResults.news
      .filter((newsItem) => newsItem.link !== undefined && newsItem.link !== '')
      .map((newsItem) => ({
        title: newsItem.title ?? '',
        link: newsItem.link ?? '',
        source: newsItem.source ?? '',
        date: newsItem.date ?? '',
        imageUrl: newsItem.imageUrl ?? '',
        processed: false,
      }));
    mergedResults.topStories = [
      ...(mergedResults.topStories ?? []),
      ...existingNewsAsTopStories,
    ];
    delete mergedResults.news;
  }

  results.slice(1).forEach((result) => {
    if (result.success && result.data !== undefined) {
      if (result.data.images !== undefined && result.data.images.length > 0) {
        mergedResults.images = [
          ...(mergedResults.images ?? []),
          ...result.data.images,
        ];
      }
      if (result.data.videos !== undefined && result.data.videos.length > 0) {
        mergedResults.videos = [
          ...(mergedResults.videos ?? []),
          ...result.data.videos,
        ];
      }
      if (result.data.news !== undefined && result.data.news.length > 0) {
        const newsAsTopStories = result.data.news.map((newsItem) => ({
          ...newsItem,
          link: newsItem.link ?? '',
        }));
        mergedResults.topStories = [
          ...(mergedResults.topStories ?? []),
          ...newsAsTopStories,
        ];
      }
    }
  });

  return { success: true, data: mergedResults };
}

function createSearchProcessor({
  searchAPI,
  safeSearch,
  sourceProcessor,
  onGetHighlights,
  logger,
}: {
  safeSearch: t.SearchToolConfig['safeSearch'];
  searchAPI: ReturnType<typeof createSearchAPI>;
  sourceProcessor: ReturnType<typeof createSourceProcessor>;
  onGetHighlights: t.SearchToolConfig['onGetHighlights'];
  logger: t.Logger;
}) {
  return async function ({
    query,
    date,
    country,
    proMode = true,
    maxSources = 5,
    onSearchResults,
    images = false,
    videos = false,
    news = false,
  }: {
    query: string;
    country?: string;
    date?: DATE_RANGE;
    proMode?: boolean;
    maxSources?: number;
    onSearchResults: t.SearchToolConfig['onSearchResults'];
    images?: boolean;
    videos?: boolean;
    news?: boolean;
  }): Promise<t.SearchResultData> {
    try {
      // Execute parallel searches and merge results
      const searchResult = await executeParallelSearches({
        searchAPI,
        query,
        date,
        country,
        safeSearch,
        images,
        videos,
        news,
        logger,
      });

      onSearchResults?.(searchResult);

      const processedSources = await sourceProcessor.processSources({
        query,
        news,
        result: searchResult,
        proMode,
        onGetHighlights,
        numElements: maxSources,
      });

      return expandHighlights(processedSources);
    } catch (error) {
      logger.error('Error in search:', error);
      return {
        organic: [],
        topStories: [],
        images: [],
        videos: [],
        news: [],
        relatedSearches: [],
        error: error instanceof Error ? error.message : String(error),
      };
    }
  };
}

function createOnSearchResults({
  runnableConfig,
  onSearchResults,
}: {
  runnableConfig: RunnableConfig;
  onSearchResults: t.SearchToolConfig['onSearchResults'];
}) {
  return function (results: t.SearchResult): void {
    if (!onSearchResults) {
      return;
    }
    onSearchResults(results, runnableConfig);
  };
}

function createTool({
  schema,
  search,
  onSearchResults: _onSearchResults,
}: {
  schema: Record<string, unknown>;
  search: ReturnType<typeof createSearchProcessor>;
  onSearchResults: t.SearchToolConfig['onSearchResults'];
}): DynamicStructuredTool {
  return tool(
    async (rawParams, runnableConfig) => {
      const params = rawParams as SearchToolParams;
      const { query, date, country: _c, images, videos, news } = params;
      const country = typeof _c === 'string' && _c ? _c : undefined;
      const searchResult = await search({
        query,
        date,
        country,
        images,
        videos,
        news,
        onSearchResults: createOnSearchResults({
          runnableConfig,
          onSearchResults: _onSearchResults,
        }),
      });
      const turn = runnableConfig.toolCall?.turn ?? 0;
      const { output, references } = formatResultsForLLM(turn, searchResult);
      const data: t.SearchResultData = { turn, ...searchResult, references };
      return [output, { [Constants.WEB_SEARCH]: data }];
    },
    {
      name: Constants.WEB_SEARCH,
      description: `Real-time search. Results have required citation anchors.

Note: Use ONCE per reply unless instructed otherwise.

Anchors:
- \\ue202turnXtypeY
- X = turn idx, type = 'search' | 'news' | 'image' | 'ref', Y = item idx

Special Markers:
- \\ue203...\\ue204 — highlight start/end of cited text (for Standalone or Group citations)
- \\ue200...\\ue201 — group block (e.g. \\ue200\\ue202turn0search1\\ue202turn0news2\\ue201)

**CITE EVERY NON-OBVIOUS FACT/QUOTE:**
Use anchor marker(s) immediately after the statement:
- Standalone: "Pure functions produce same output. \\ue202turn0search0"
- Standalone (multiple): "Today's News \\ue202turn0search0\\ue202turn0news0"
- Highlight: "\\ue203Highlight text.\\ue204\\ue202turn0news1"
- Group: "Sources. \\ue200\\ue202turn0search0\\ue202turn0news1\\ue201"
- Group Highlight: "\\ue203Highlight for group.\\ue204 \\ue200\\ue202turn0search0\\ue202turn0news1\\ue201"
- Image: "See photo \\ue202turn0image0."

**NEVER use markdown links, [1], or footnotes. CITE ONLY with anchors provided.**
`.trim(),
      schema: schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

/**
 * Creates a search tool with a schema that dynamically includes the country field
 * only when the searchProvider is 'serper'.
 *
 * Supports multiple scraper providers:
 * - Firecrawl (default): Full-featured web scraping with multiple formats
 * - Serper: Lightweight scraping using Serper's scrape API
 *
 * @example
 * ```typescript
 * // Using Firecrawl scraper (default)
 * const searchTool = createSearchTool({
 *   searchProvider: 'serper',
 *   scraperProvider: 'firecrawl',
 *   firecrawlApiKey: 'your-firecrawl-key'
 * });
 *
 * // Using Serper scraper
 * const searchTool = createSearchTool({
 *   searchProvider: 'serper',
 *   scraperProvider: 'serper',
 *   serperApiKey: 'your-serper-key'
 * });
 * ```
 *
 * @param config - The search tool configuration
 * @returns A DynamicStructuredTool with a schema that depends on the searchProvider
 */
/** Input params type for search tool */
interface SearchToolParams {
  query: string;
  date?: DATE_RANGE;
  country?: string;
  images?: boolean;
  videos?: boolean;
  news?: boolean;
}

export const createSearchTool = (
  config: t.SearchToolConfig = {}
): DynamicStructuredTool => {
  const {
    searchProvider = 'serper',
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
    rerankerType = 'cohere',
    topResults = 5,
    strategies = ['no_extraction'],
    filterContent = true,
    safeSearch = 1,
    scraperProvider = 'firecrawl',
    firecrawlApiKey,
    firecrawlApiUrl,
    firecrawlVersion,
    firecrawlOptions,
    serperScraperOptions,
    scraperTimeout,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    onSearchResults: _onSearchResults,
    onGetHighlights,
  } = config;

  const logger = config.logger || createDefaultLogger();

  const schemaProperties: Record<string, unknown> = {
    query: querySchema,
    date: dateSchema,
    images: imagesSchema,
    videos: videosSchema,
    news: newsSchema,
  };

  if (searchProvider === 'serper') {
    schemaProperties.country = countrySchema;
  }

  const toolSchema = {
    type: 'object',
    properties: schemaProperties,
    required: ['query'],
  };

  const searchAPI = createSearchAPI({
    searchProvider,
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
  });

  /** Create scraper based on scraperProvider */
  let scraperInstance: t.BaseScraper;

  if (scraperProvider === 'serper') {
    scraperInstance = createSerperScraper({
      ...serperScraperOptions,
      apiKey: serperApiKey,
      timeout: scraperTimeout ?? serperScraperOptions?.timeout,
      logger,
    });
  } else {
    scraperInstance = createFirecrawlScraper({
      ...firecrawlOptions,
      apiKey: firecrawlApiKey ?? process.env.FIRECRAWL_API_KEY,
      apiUrl: firecrawlApiUrl,
      version: firecrawlVersion,
      timeout: scraperTimeout ?? firecrawlOptions?.timeout,
      formats: firecrawlOptions?.formats ?? ['markdown', 'rawHtml'],
      logger,
    });
  }

  const selectedReranker = createReranker({
    rerankerType,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    logger,
  });

  if (!selectedReranker) {
    logger.warn('No reranker selected. Using default ranking.');
  }

  const sourceProcessor = createSourceProcessor(
    {
      reranker: selectedReranker,
      topResults,
      strategies,
      filterContent,
      logger,
    },
    scraperInstance
  );

  const search = createSearchProcessor({
    searchAPI,
    safeSearch,
    sourceProcessor,
    onGetHighlights,
    logger,
  });

  return createTool({
    search,
    schema: toolSchema,
    onSearchResults: _onSearchResults,
  });
};
