import axios from 'axios';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import type * as t from './types';
import { getAttribution, createDefaultLogger } from './utils';
import { BaseReranker } from './rerankers';

const chunker = {
  cleanText: (text: string): string => {
    if (!text) return '';

    /** Normalized all line endings to '\n' */
    const normalizedText = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    /** Handle multiple backslashes followed by newlines
     * This replaces patterns like '\\\\\\n' with a single newline */
    const fixedBackslashes = normalizedText.replace(/\\+\n/g, '\n');

    /** Cleaned up consecutive newlines, tabs, and spaces around newlines */
    const cleanedNewlines = fixedBackslashes.replace(/[\t ]*\n[\t \n]*/g, '\n');

    /** Cleaned up excessive spaces and tabs */
    const cleanedSpaces = cleanedNewlines.replace(/[ \t]+/g, ' ');

    return cleanedSpaces.trim();
  },
  splitText: async (
    text: string,
    options?: {
      chunkSize?: number;
      chunkOverlap?: number;
      separators?: string[];
    }
  ): Promise<string[]> => {
    const chunkSize = options?.chunkSize ?? 150;
    const chunkOverlap = options?.chunkOverlap ?? 50;
    const separators = options?.separators || ['\n\n', '\n'];

    const splitter = new RecursiveCharacterTextSplitter({
      separators,
      chunkSize,
      chunkOverlap,
    });

    return await splitter.splitText(text);
  },

  splitTexts: async (
    texts: string[],
    options?: {
      chunkSize?: number;
      chunkOverlap?: number;
      separators?: string[];
    },
    logger?: t.Logger
  ): Promise<string[][]> => {
    // Split multiple texts
    const logger_ = logger || createDefaultLogger();
    const promises = texts.map((text) =>
      chunker.splitText(text, options).catch((error) => {
        logger_.error('Error splitting text:', error);
        return [text];
      })
    );
    return Promise.all(promises);
  },
};

function createSourceUpdateCallback(sourceMap: Map<string, t.ValidSource>) {
  return (link: string, update?: Partial<t.ValidSource>): void => {
    const source = sourceMap.get(link);
    if (source) {
      sourceMap.set(link, {
        ...source,
        ...update,
      });
    }
  };
}

const getHighlights = async ({
  query,
  content,
  reranker,
  topResults = 5,
  logger,
}: {
  content: string;
  query: string;
  reranker?: BaseReranker;
  topResults?: number;
  logger?: t.Logger;
}): Promise<t.Highlight[] | undefined> => {
  const logger_ = logger || createDefaultLogger();

  if (!content) {
    logger_.warn('No content provided for highlights');
    return;
  }
  if (!reranker) {
    logger_.warn('No reranker provided for highlights');
    return;
  }

  try {
    const documents = await chunker.splitText(content);
    if (Array.isArray(documents)) {
      return await reranker.rerank(query, documents, topResults);
    } else {
      logger_.error(
        'Expected documents to be an array, got:',
        typeof documents
      );
      return;
    }
  } catch (error) {
    logger_.error('Error in content processing:', error);
    return;
  }
};

const createSerperAPI = (
  apiKey?: string
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.SERPER_API_KEY,
    apiUrl: 'https://google.serper.dev/search',
    timeout: 10000,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('SERPER_API_KEY is required for SerperAPI');
  }

  const getSources = async ({
    query,
    date,
    country,
    safeSearch,
    numResults = 8,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const safe = ['off', 'moderate', 'active'] as const;
      const payload: t.SerperSearchPayload = {
        q: query,
        safe: safe[safeSearch ?? 1],
        num: Math.min(Math.max(1, numResults), 10),
      };

      // Set the search type if provided
      if (type) {
        payload.type = type;
      }

      if (date != null) {
        payload.tbs = `qdr:${date}`;
      }

      if (country != null && country !== '') {
        payload['gl'] = country.toLowerCase();
      }

      // Determine the API endpoint based on the search type
      let apiEndpoint = config.apiUrl;
      if (type === 'images') {
        apiEndpoint = 'https://google.serper.dev/images';
      } else if (type === 'videos') {
        apiEndpoint = 'https://google.serper.dev/videos';
      } else if (type === 'news') {
        apiEndpoint = 'https://google.serper.dev/news';
      }

      const response = await axios.post<t.SerperResultData>(
        apiEndpoint,
        payload,
        {
          headers: {
            'X-API-KEY': config.apiKey,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
        }
      );

      const data = response.data;
      const results: t.SearchResultData = {
        organic: data.organic,
        images: data.images ?? [],
        answerBox: data.answerBox,
        topStories: data.topStories ?? [],
        peopleAlsoAsk: data.peopleAlsoAsk,
        knowledgeGraph: data.knowledgeGraph,
        relatedSearches: data.relatedSearches,
        videos: data.videos ?? [],
        news: data.news ?? [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return { success: false, error: `API request failed: ${errorMessage}` };
    }
  };

  return { getSources };
};

const createSearXNGAPI = (
  instanceUrl?: string,
  apiKey?: string
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    instanceUrl: instanceUrl ?? process.env.SEARXNG_INSTANCE_URL,
    apiKey: apiKey ?? process.env.SEARXNG_API_KEY,
    defaultLocation: 'all',
    timeout: 10000,
  };

  if (config.instanceUrl == null || config.instanceUrl === '') {
    throw new Error('SEARXNG_INSTANCE_URL is required for SearXNG API');
  }

  const getSources = async ({
    query,
    numResults = 8,
    safeSearch,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      // Ensure the instance URL ends with /search
      if (config.instanceUrl == null || config.instanceUrl === '') {
        return { success: false, error: 'Instance URL is not defined' };
      }

      let searchUrl = config.instanceUrl;
      if (!searchUrl.endsWith('/search')) {
        searchUrl = searchUrl.replace(/\/$/, '') + '/search';
      }

      // Determine the search category based on the type
      let category = 'general';
      if (type === 'images') {
        category = 'images';
      } else if (type === 'videos') {
        category = 'videos';
      } else if (type === 'news') {
        category = 'news';
      }

      // Prepare parameters for SearXNG
      const params: t.SearxNGSearchPayload = {
        q: query,
        format: 'json',
        pageno: 1,
        categories: category,
        language: 'all',
        safesearch: safeSearch,
        engines: 'google,bing,duckduckgo',
      };

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (config.apiKey != null && config.apiKey !== '') {
        headers['X-API-Key'] = config.apiKey;
      }

      const response = await axios.get(searchUrl, {
        headers,
        params,
        timeout: config.timeout,
      });

      const data = response.data;

      // Helper function to identify news results since SearXNG doesn't provide that classification by default
      const isNewsResult = (result: t.SearXNGResult): boolean => {
        const url = result.url?.toLowerCase() ?? '';
        const title = result.title?.toLowerCase() ?? '';

        // News-related keywords in title/content
        const newsKeywords = [
          'breaking news',
          'latest news',
          'top stories',
          'news today',
          'developing story',
          'trending news',
          'news',
        ];

        // Check if title/content contains news keywords
        const hasNewsKeywords = newsKeywords.some(
          (keyword) => title.toLowerCase().includes(keyword) // just title probably fine, content parsing is overkill for what we need: || content.includes(keyword)
        );

        // Check if URL contains news-related paths
        const hasNewsPath =
          url.includes('/news/') ||
          url.includes('/world/') ||
          url.includes('/politics/') ||
          url.includes('/breaking/');

        return hasNewsKeywords || hasNewsPath;
      };

      // Transform SearXNG results to match SerperAPI format
      const organicResults = (data.results ?? [])
        .slice(0, numResults)
        .map((result: t.SearXNGResult, index: number) => {
          let attribution = '';
          try {
            attribution = new URL(result.url ?? '').hostname;
          } catch {
            attribution = '';
          }

          return {
            position: index + 1,
            title: result.title ?? '',
            link: result.url ?? '',
            snippet: result.content ?? '',
            date: result.publishedDate ?? '',
            attribution,
          };
        });

      const imageResults = (data.results ?? [])
        .filter((result: t.SearXNGResult) => result.img_src)
        .slice(0, 6)
        .map((result: t.SearXNGResult, index: number) => ({
          title: result.title ?? '',
          imageUrl: result.img_src ?? '',
          position: index + 1,
          source: new URL(result.url ?? '').hostname,
          domain: new URL(result.url ?? '').hostname,
          link: result.url ?? '',
        }));

      // Extract news results from organic results
      const newsResults = (data.results ?? [])
        .filter(isNewsResult)
        .map((result: t.SearXNGResult, index: number) => {
          let attribution = '';
          try {
            attribution = new URL(result.url ?? '').hostname;
          } catch {
            attribution = '';
          }

          return {
            title: result.title ?? '',
            link: result.url ?? '',
            snippet: result.content ?? '',
            date: result.publishedDate ?? '',
            source: attribution,
            imageUrl: result.img_src ?? '',
            position: index + 1,
          };
        });

      const topStories = newsResults.slice(0, 5);

      const relatedSearches = Array.isArray(data.suggestions)
        ? data.suggestions.map((suggestion: string) => ({ query: suggestion }))
        : [];

      const results: t.SearchResultData = {
        organic: organicResults,
        images: imageResults,
        topStories: topStories, // Use first 5 extracted news as top stories
        relatedSearches,
        videos: [],
        news: newsResults,
        // Add empty arrays for other Serper fields to maintain parity
        places: [],
        shopping: [],
        peopleAlsoAsk: [],
        knowledgeGraph: undefined,
        answerBox: undefined,
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `SearXNG API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};

const TAVILY_TIME_RANGE_MAP: Record<string, string> = {
  h: 'day',
  d: 'day',
  w: 'week',
  m: 'month',
  y: 'year',
} as const;

const DEFAULT_TAVILY_TIMEOUT = 15000;

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return link;
  }
};

const createTavilyAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.TavilySearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.TAVILY_API_KEY,
    apiUrl:
      apiUrl ??
      process.env.TAVILY_SEARCH_URL ??
      'https://api.tavily.com/search',
    timeout: options?.timeout ?? DEFAULT_TAVILY_TIMEOUT,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('TAVILY_API_KEY is required for Tavily API');
  }

  const getSources = async ({
    query,
    date,
    numResults = 8,
    type,
    news,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const timeRange =
        options?.timeRange ??
        (date != null ? (TAVILY_TIME_RANGE_MAP[date] ?? 'day') : undefined);
      const topic =
        options?.topic ??
        (news === true || type === 'news' ? 'news' : 'general');
      const maxResults = options?.maxResults ?? numResults;
      const searchDepth = options?.searchDepth ?? 'basic';

      const payload: Record<string, unknown> = {
        query,
        search_depth: searchDepth,
        topic,
        max_results: Math.min(Math.max(1, maxResults), 20),
      };

      if (timeRange != null) {
        payload.time_range = timeRange;
      }
      if (type === 'images' || options?.includeImages) {
        payload.include_images = true;
      }
      if (options?.includeAnswer != null) {
        payload.include_answer = options.includeAnswer;
      }
      if (options?.includeRawContent != null) {
        payload.include_raw_content = options.includeRawContent;
      }
      if (
        options?.includeDomains != null &&
        options.includeDomains.length > 0
      ) {
        payload.include_domains = options.includeDomains;
      }
      if (
        options?.excludeDomains != null &&
        options.excludeDomains.length > 0
      ) {
        payload.exclude_domains = options.excludeDomains;
      }
      if (options?.includeImageDescriptions != null) {
        payload.include_image_descriptions = options.includeImageDescriptions;
      }
      if (options?.includeFavicon != null) {
        payload.include_favicon = options.includeFavicon;
      }
      if (options?.chunksPerSource != null) {
        payload.chunks_per_source = options.chunksPerSource;
      }

      const response = await axios.post<t.TavilySearchResponse>(
        config.apiUrl,
        payload,
        {
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
        }
      );

      const data = response.data;

      const organicResults: t.OrganicResult[] = (data.results ?? []).map(
        (result: t.TavilySearchResult) => ({
          title: result.title ?? '',
          link: result.url ?? '',
          snippet: result.content ?? '',
          date: result.published_date,
        })
      );

      const imageResults: t.ImageResult[] = Array.isArray(data.images)
        ? data.images
            .slice(0, 6)
            .reduce<t.ImageResult[]>((acc, image, index) => {
              const imageUrl = typeof image === 'string' ? image : image.url;
              if (imageUrl == null || imageUrl === '') {
                return acc;
              }
              acc.push({
                imageUrl,
                title:
                  typeof image === 'string' ? undefined : image.description,
                position: index + 1,
              });
              return acc;
            }, [])
        : [];

      const newsResults: t.NewsResult[] =
        topic === 'news'
          ? organicResults.map((r) => ({
              title: r.title,
              link: r.link,
              snippet: r.snippet,
              date: r.date,
              source: getHostname(r.link),
            }))
          : [];

      const results: t.SearchResultData = {
        organic: organicResults,
        images: imageResults,
        topStories: [],
        videos: [],
        news: newsResults,
        answerBox: data.answer != null ? { snippet: data.answer } : undefined,
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Tavily API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};

export const createSearchAPI = (
  config: t.SearchConfig
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const {
    searchProvider = 'serper',
    serperApiKey,
    searxngInstanceUrl,
    searxngApiKey,
    tavilyApiKey,
    tavilySearchUrl,
    tavilySearchOptions,
  } = config;

  if (searchProvider.toLowerCase() === 'serper') {
    return createSerperAPI(serperApiKey);
  } else if (searchProvider.toLowerCase() === 'searxng') {
    return createSearXNGAPI(searxngInstanceUrl, searxngApiKey);
  } else if (searchProvider.toLowerCase() === 'tavily') {
    return createTavilyAPI(tavilyApiKey, tavilySearchUrl, tavilySearchOptions);
  } else {
    throw new Error(
      `Invalid search provider: ${searchProvider}. Must be 'serper', 'searxng', or 'tavily'`
    );
  }
};

export const createSourceProcessor = (
  config: t.ProcessSourcesConfig = {},
  scraperInstance?: t.BaseScraper
): {
  processSources: (
    fields: t.ProcessSourcesFields
  ) => Promise<t.SearchResultData>;
  topResults: number;
} => {
  if (!scraperInstance) {
    throw new Error('Scraper instance is required');
  }
  const {
    topResults = 5,
    // strategies = ['no_extraction'],
    // filterContent = true,
    reranker,
    logger,
  } = config;

  const logger_ = logger || createDefaultLogger();
  const scraper = scraperInstance;

  const processResponse = (
    url: string,
    response: t.AnyScraperResponse
  ): t.ScrapeResult => {
    const rawMetadata = scraper.extractMetadata(response);
    const metadata =
      Object.keys(rawMetadata).length > 0
        ? (rawMetadata as t.ScrapeMetadata)
        : undefined;
    const attribution = getAttribution(url, metadata, logger_);

    if (response.success && response.data) {
      const [content, references] = scraper.extractContent(response);
      return {
        url,
        references,
        attribution,
        content: chunker.cleanText(content),
      };
    }

    logger_.error(
      `Error scraping ${url}: ${response.error ?? 'Unknown error'}`
    );
    return { url, attribution, error: true, content: '' };
  };

  const addHighlights = async (
    result: t.ScrapeResult,
    query: string,
    onGetHighlights: t.SearchToolConfig['onGetHighlights']
  ): Promise<t.ScrapeResult> => {
    if (result.error != null) {
      return result;
    }
    try {
      const highlights = await getHighlights({
        query,
        reranker,
        content: result.content,
        logger: logger_,
      });
      if (onGetHighlights) {
        onGetHighlights(result.url);
      }
      return { ...result, highlights };
    } catch (error) {
      logger_.error('Error processing scraped content:', error);
      return result;
    }
  };

  const webScraper = {
    scrapeMany: async ({
      query,
      links,
      onGetHighlights,
    }: {
      query: string;
      links: string[];
      onGetHighlights: t.SearchToolConfig['onGetHighlights'];
    }): Promise<Array<t.ScrapeResult>> => {
      logger_.debug(`Scraping ${links.length} links`);
      try {
        let responses: Array<[string, t.AnyScraperResponse]>;

        if (scraper.scrapeUrls) {
          responses = await scraper.scrapeUrls(links);
        } else {
          responses = await Promise.all(
            links.map((link) =>
              scraper
                .scrapeUrl(link, {})
                .catch((error): [string, t.AnyScraperResponse] => {
                  logger_.error(`Error scraping ${link}:`, error);
                  return [link, { success: false, error: String(error) }];
                })
            )
          );
        }

        const results = responses.map(([url, response]) =>
          processResponse(url, response)
        );
        const withHighlights = await Promise.all(
          results.map((result) => addHighlights(result, query, onGetHighlights))
        );
        return withHighlights;
      } catch (error) {
        logger_.error('Error in scrapeMany:', error);
        return [];
      }
    },
  };

  const fetchContents = async ({
    links,
    query,
    target,
    onGetHighlights,
    onContentScraped,
  }: {
    links: string[];
    query: string;
    target: number;
    onGetHighlights: t.SearchToolConfig['onGetHighlights'];
    onContentScraped?: (link: string, update?: Partial<t.ValidSource>) => void;
  }): Promise<void> => {
    const initialLinks = links.slice(0, target);
    // const remainingLinks = links.slice(target).reverse();
    const results = await webScraper.scrapeMany({
      query,
      links: initialLinks,
      onGetHighlights,
    });
    for (const result of results) {
      if (result.error === true) {
        continue;
      }
      const { url, content, attribution, references, highlights } = result;
      onContentScraped?.(url, {
        content,
        attribution,
        references,
        highlights,
      });
    }
  };

  const processSources = async ({
    result,
    numElements,
    query,
    news,
    proMode = true,
    onGetHighlights,
  }: t.ProcessSourcesFields): Promise<t.SearchResultData> => {
    try {
      if (!result.data) {
        return {
          organic: [],
          topStories: [],
          images: [],
          relatedSearches: [],
        };
      } else if (!result.data.organic) {
        return result.data;
      }

      if (!proMode) {
        const wikiSources = result.data.organic.filter((source) =>
          source.link.includes('wikipedia.org')
        );

        if (!wikiSources.length) {
          return result.data;
        }

        const wikiSourceMap = new Map<string, t.ValidSource>();
        wikiSourceMap.set(wikiSources[0].link, wikiSources[0]);
        const onContentScraped = createSourceUpdateCallback(wikiSourceMap);
        await fetchContents({
          query,
          target: 1,
          onGetHighlights,
          onContentScraped,
          links: [wikiSources[0].link],
        });

        for (let i = 0; i < result.data.organic.length; i++) {
          const source = result.data.organic[i];
          const updatedSource = wikiSourceMap.get(source.link);
          if (updatedSource) {
            result.data.organic[i] = {
              ...source,
              ...updatedSource,
            };
          }
        }

        return result.data;
      }

      const sourceMap = new Map<string, t.ValidSource>();
      const organicLinksSet = new Set<string>();

      // Collect organic links
      const organicLinks = collectLinks(
        result.data.organic,
        sourceMap,
        organicLinksSet
      );

      // Collect top story links, excluding any that are already in organic links
      const topStories = result.data.topStories ?? [];
      const topStoryLinks = collectLinks(
        topStories,
        sourceMap,
        organicLinksSet
      );

      if (organicLinks.length === 0 && (topStoryLinks.length === 0 || !news)) {
        return result.data;
      }

      const onContentScraped = createSourceUpdateCallback(sourceMap);
      const promises: Promise<void>[] = [];

      // Process organic links
      if (organicLinks.length > 0) {
        promises.push(
          fetchContents({
            query,
            onGetHighlights,
            onContentScraped,
            links: organicLinks,
            target: numElements,
          })
        );
      }

      // Process top story links
      if (news && topStoryLinks.length > 0) {
        promises.push(
          fetchContents({
            query,
            onGetHighlights,
            onContentScraped,
            links: topStoryLinks,
            target: numElements,
          })
        );
      }

      await Promise.all(promises);

      if (result.data.organic.length > 0) {
        updateSourcesWithContent(result.data.organic, sourceMap);
      }

      if (news && topStories.length > 0) {
        updateSourcesWithContent(topStories, sourceMap);
      }

      return result.data;
    } catch (error) {
      logger_.error('Error in processSources:', error);
      return {
        organic: [],
        topStories: [],
        images: [],
        relatedSearches: [],
        ...result.data,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  };

  return {
    processSources,
    topResults,
  };
};

/** Helper function to collect links and update sourceMap */
function collectLinks(
  sources: Array<t.OrganicResult | t.TopStoryResult>,
  sourceMap: Map<string, t.ValidSource>,
  existingLinksSet?: Set<string>
): string[] {
  const links: string[] = [];

  for (const source of sources) {
    if (source.link) {
      // For topStories, only add if not already in organic links
      if (existingLinksSet && existingLinksSet.has(source.link)) {
        continue;
      }

      links.push(source.link);
      if (existingLinksSet) {
        existingLinksSet.add(source.link);
      }
      sourceMap.set(source.link, source as t.ValidSource);
    }
  }

  return links;
}

/** Helper function to update sources with scraped content */
function updateSourcesWithContent<T extends t.ValidSource>(
  sources: T[],
  sourceMap: Map<string, t.ValidSource>
): void {
  for (let i = 0; i < sources.length; i++) {
    const source = sources[i];
    const updatedSource = sourceMap.get(source.link);
    if (updatedSource) {
      sources[i] = {
        ...source,
        ...updatedSource,
      } as T;
    }
  }
}
