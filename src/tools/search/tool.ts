import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from './types';
import {
  WebSearchToolDescription,
  WebSearchToolName,
  countrySchema,
  imagesSchema,
  videosSchema,
  querySchema,
  dateSchema,
  newsSchema,
  DATE_RANGE,
} from './schema';
import { createSearchAPI, createSourceProcessor } from './search';
import { createKeenableScraper } from './keenable-scraper';
import { createSerperScraper } from './serper-scraper';
import { createTavilyScraper } from './tavily-scraper';
import { createFirecrawlScraper } from './firecrawl';
import { createExaScraper } from './exa-scraper';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { createCrwScraper } from './crw-scraper';
import { expandHighlights } from './highlights';
import { formatResultsForLLM } from './format';
import { createDefaultLogger } from './utils';
import { createReranker } from './rerankers';
import { Constants } from '@/common';

/**
 * Settled label for a `web_search` call's intent (see `intentArg.ts`).
 *
 * Counts the result kinds `formatResultsForLLM` actually renders —
 * `references` only tracks links embedded in extracted highlights, so it
 * undercounts ordinary results and can overcount when one highlight embeds
 * several links.
 *
 * A caught provider or processing failure is reported through `data.error`
 * while the tool still returns NORMALLY, so that case must author its own
 * label: the `ToolMessage` carries success status, so without an authored
 * outcome the in-flight intent ("Searching…") would stand as the settled
 * label and present a failed search as an ordinary one.
 *
 * Returns undefined for a genuine zero-result search, leaving the
 * model-authored intent to stand unchanged as the label.
 */
export function resolveSearchOutcome(
  data: t.SearchResultData,
  query: string
): string | undefined {
  if (data.error != null && data.error !== '') {
    return `Search failed for "${query}"`;
  }
  const count =
    (data.organic?.length ?? 0) +
    (data.topStories?.length ?? 0) +
    (data.news?.length ?? 0) +
    (data.images?.length ?? 0) +
    (data.videos?.length ?? 0) +
    (data.places?.length ?? 0) +
    (data.peopleAlsoAsk?.length ?? 0) +
    (data.knowledgeGraph != null ? 1 : 0) +
    (data.answerBox != null ? 1 : 0);
  if (count === 0) {
    return undefined;
  }
  return `Found ${count} result${count === 1 ? '' : 's'} for "${query}"`;
}

/**
 * Executes parallel searches and merges the results,
 * deduplicating top stories by link
 */
export async function executeParallelSearches({
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

  if (
    mergedResults.topStories !== undefined &&
    mergedResults.topStories.length > 1
  ) {
    /** The main search's own news results and the parallel news sub-search
     * frequently return the same stories — keep the first occurrence of each
     * link so duplicates aren't scraped, reranked, and formatted repeatedly */
    const seenLinks = new Set<string>();
    mergedResults.topStories = mergedResults.topStories.filter((story) => {
      if (!story.link || seenLinks.has(story.link)) {
        return false;
      }
      seenLinks.add(story.link);
      return true;
    });
  }

  return { success: true, data: mergedResults };
}

function createSearchProcessor({
  searchAPI,
  safeSearch,
  supportsImages,
  supportsVideos,
  supportsNews,
  sourceProcessor,
  onGetHighlights,
  mainExpandBy,
  separatorExpandBy,
  logger,
}: {
  safeSearch: t.SearchToolConfig['safeSearch'];
  supportsImages: boolean;
  supportsVideos: boolean;
  supportsNews: boolean;
  searchAPI: ReturnType<typeof createSearchAPI>;
  sourceProcessor: ReturnType<typeof createSourceProcessor>;
  onGetHighlights: t.SearchToolConfig['onGetHighlights'];
  mainExpandBy: t.SearchToolConfig['mainExpandBy'];
  separatorExpandBy: t.SearchToolConfig['separatorExpandBy'];
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
        images: supportsImages && images,
        videos: supportsVideos && videos,
        news: supportsNews && news,
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

      return expandHighlights(
        processedSources,
        mainExpandBy,
        separatorExpandBy
      );
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
  maxOutputChars,
  onSearchResults: _onSearchResults,
}: {
  schema: Record<string, unknown>;
  search: ReturnType<typeof createSearchProcessor>;
  maxOutputChars?: number;
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
      const { output, references } = formatResultsForLLM(
        turn,
        searchResult,
        maxOutputChars
      );
      const data: t.SearchResultData = { turn, ...searchResult, references };
      const outcome = resolveSearchOutcome(data, query);
      return [
        output,
        { [Constants.WEB_SEARCH]: data, ...(outcome != null && { outcome }) },
      ];
    },
    {
      name: WebSearchToolName,
      description: WebSearchToolDescription,
      schema: schema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

/**
 * Creates a search tool with configurable search and scraper providers.
 *
 * Search providers: Serper (Google results), SearXNG (self-hosted meta-search), Tavily (AI-optimized), fastCRW (Firecrawl-compatible, self-host or cloud).
 * Scraper providers: Firecrawl (default, full-featured), Serper (lightweight), Tavily (batch extraction), fastCRW (Firecrawl-compatible, self-host or cloud).
 *
 * The country schema field is exposed to the LLM for providers that support localized results.
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
    tavilyApiKey,
    tavilySearchUrl,
    tavilyExtractUrl,
    tavilySearchOptions,
    keenableApiKey,
    keenableApiUrl,
    keenableSearchOptions,
    keenableScraperOptions,
    exaApiKey,
    exaApiUrl,
    exaSearchOptions,
    exaScraperOptions,
    rerankerType = 'cohere',
    rerankerTimeout,
    topResults = 5,
    maxContentLength,
    chunkSize,
    chunkOverlap,
    mainExpandBy,
    separatorExpandBy,
    maxOutputChars,
    strategies = ['no_extraction'],
    filterContent = true,
    safeSearch = 1,
    scraperProvider = 'firecrawl',
    firecrawlApiKey,
    firecrawlApiUrl,
    firecrawlVersion,
    firecrawlOptions,
    serperScraperOptions,
    tavilyScraperOptions,
    crwApiKey,
    crwApiUrl,
    crwSearchOptions,
    crwScraperOptions,
    scraperTimeout,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    ragApiUrl,
    ragApiTokenSupplier,
    ragApiProfile,
    httpAgent,
    httpsAgent,
    onSearchResults: _onSearchResults,
    onGetHighlights,
  } = config;

  const logger = config.logger || createDefaultLogger();
  const effectiveTavilySearchOptions =
    searchProvider === 'tavily' && config.safeSearch != null
      ? {
        ...tavilySearchOptions,
        safeSearch: config.safeSearch !== 0,
      }
      : tavilySearchOptions;

  const schemaProperties: Record<string, unknown> = {
    intent: { ...INTENT_PROPERTY },
    query: querySchema,
    date: dateSchema,
    images: imagesSchema,
    videos: videosSchema,
    news: newsSchema,
  };

  if (
    searchProvider === 'serper' ||
    searchProvider === 'tavily' ||
    searchProvider === 'exa'
  ) {
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
    tavilyApiKey,
    tavilySearchUrl,
    tavilySearchOptions: effectiveTavilySearchOptions,
    keenableApiKey,
    keenableApiUrl,
    keenableSearchOptions,
    exaApiKey,
    exaApiUrl,
    exaSearchOptions,
    crwApiKey,
    crwApiUrl,
    crwSearchOptions,
    httpAgent,
    httpsAgent,
  });

  /** Create scraper based on scraperProvider */
  let scraperInstance: t.BaseScraper;

  if (scraperProvider === 'serper') {
    scraperInstance = createSerperScraper({
      ...serperScraperOptions,
      apiKey: serperApiKey,
      timeout: scraperTimeout ?? serperScraperOptions?.timeout,
      httpAgent: httpAgent ?? serperScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? serperScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'tavily') {
    scraperInstance = createTavilyScraper({
      ...tavilyScraperOptions,
      apiKey:
        tavilyScraperOptions?.apiKey ??
        tavilyApiKey ??
        process.env.TAVILY_API_KEY,
      apiUrl: tavilyScraperOptions?.apiUrl ?? tavilyExtractUrl,
      timeout: scraperTimeout ?? tavilyScraperOptions?.timeout,
      httpAgent: httpAgent ?? tavilyScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? tavilyScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'exa') {
    scraperInstance = createExaScraper({
      ...exaScraperOptions,
      apiKey: exaScraperOptions?.apiKey ?? exaApiKey ?? process.env.EXA_API_KEY,
      apiUrl: exaScraperOptions?.apiUrl ?? exaApiUrl,
      timeout: scraperTimeout ?? exaScraperOptions?.timeout,
      logger,
    });
  } else if (scraperProvider === 'crw') {
    scraperInstance = createCrwScraper({
      ...crwScraperOptions,
      apiKey: crwScraperOptions?.apiKey ?? crwApiKey ?? process.env.CRW_API_KEY,
      apiUrl: crwScraperOptions?.apiUrl ?? crwApiUrl,
      timeout: scraperTimeout ?? crwScraperOptions?.timeout,
      formats: crwScraperOptions?.formats ?? ['markdown', 'rawHtml'],
      httpAgent: httpAgent ?? crwScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? crwScraperOptions?.httpsAgent,
      logger,
    });
  } else if (scraperProvider === 'keenable') {
    scraperInstance = createKeenableScraper({
      ...keenableScraperOptions,
      apiKey: keenableScraperOptions?.apiKey ?? keenableApiKey,
      timeout: scraperTimeout ?? keenableScraperOptions?.timeout,
      attributionTitle:
        keenableScraperOptions?.attributionTitle ??
        keenableSearchOptions?.attributionTitle,
      httpAgent: httpAgent ?? keenableScraperOptions?.httpAgent,
      httpsAgent: httpsAgent ?? keenableScraperOptions?.httpsAgent,
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
      httpAgent: httpAgent ?? firecrawlOptions?.httpAgent,
      httpsAgent: httpsAgent ?? firecrawlOptions?.httpsAgent,
      logger,
    });
  }

  const selectedReranker = createReranker({
    rerankerType,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    ragApiUrl,
    ragApiTokenSupplier,
    ragApiProfile,
    rerankerTimeout,
    httpAgent,
    httpsAgent,
    logger,
  });

  if (!selectedReranker) {
    logger.warn('No reranker selected. Using default ranking.');
  }

  const sourceProcessor = createSourceProcessor(
    {
      reranker: selectedReranker,
      topResults,
      maxContentLength,
      chunkSize,
      chunkOverlap,
      strategies,
      filterContent,
      logger,
    },
    scraperInstance
  );

  const search = createSearchProcessor({
    searchAPI,
    safeSearch,
    // Keenable is organic-only: its API ignores `type`, so image/news
    // sub-searches would spend rate limit and merge nothing.
    // Exa has no image or video search, but supports news via its
    // dedicated `news` category.
    supportsImages: searchProvider !== 'keenable' && searchProvider !== 'exa',
    supportsVideos:
      searchProvider !== 'tavily' &&
      searchProvider !== 'keenable' &&
      searchProvider !== 'crw' &&
      searchProvider !== 'exa',
    supportsNews: searchProvider !== 'keenable',
    sourceProcessor,
    onGetHighlights,
    mainExpandBy,
    separatorExpandBy,
    logger,
  });

  return createTool({
    search,
    schema: toolSchema,
    maxOutputChars,
    onSearchResults: _onSearchResults,
  });
};
