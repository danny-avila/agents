import type * as t from './types';
import { executeParallelSearches } from './tool';
import { createSourceProcessor } from './search';
import { createSearchMetrics } from './metrics';
import { expandHighlights } from './highlights';
import { BaseReranker } from './rerankers';

const noopLog = (..._args: unknown[]): void => {};
const silentLogger = {
  error: noopLog,
  warn: noopLog,
  info: noopLog,
  debug: noopLog,
} as t.Logger;

class RecordingReranker extends BaseReranker {
  readonly provider = 'recording';
  public rerankCalls: string[][] = [];
  public topKCalls: number[] = [];

  constructor() {
    super(silentLogger);
  }

  async rerank(
    _query: string,
    documents: string[],
    topK: number = 5,
    metrics?: t.SearchMetrics
  ): Promise<t.Highlight[]> {
    this.rerankCalls.push(documents);
    this.topKCalls.push(topK);
    return this.complete(
      this.beginRerank(documents, topK, metrics),
      this.getDefaultRanking(documents, topK)
    );
  }
}

const createFakeScraper = (
  contentByUrl: Record<string, string>,
  scrapedUrls: string[] = []
): t.BaseScraper => ({
  scrapeUrl: async (url: string): Promise<[string, t.AnyScraperResponse]> => {
    scrapedUrls.push(url);
    return [
      url,
      { success: true, data: { markdown: contentByUrl[url] ?? '' } },
    ];
  },
  extractContent: (
    response: t.AnyScraperResponse
  ): [string, undefined | t.References] => [
    (response as t.FirecrawlScrapeResponse).data?.markdown ?? '',
    undefined,
  ],
  extractMetadata: (): t.GenericScrapeMetadata => ({}),
});

const makeOrganic = (link: string): t.ProcessedOrganic => ({
  link,
  title: `Title for ${link}`,
  snippet: `Snippet for ${link}`,
});

const makeStory = (link: string): t.ProcessedTopStory => ({
  link,
  title: `Story for ${link}`,
});

const makeLongContent = (chars: number): string => {
  const line =
    'lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod\n';
  return line.repeat(Math.ceil(chars / line.length)).slice(0, chars);
};

describe('expandHighlights content stripping', () => {
  test('strips content and references from sources without highlights', () => {
    const references: t.References = { links: [], images: [], videos: [] };
    const data: t.SearchResultData = {
      organic: [
        {
          ...makeOrganic('https://a.com'),
          content: 'X'.repeat(100000),
          references,
        },
      ],
      topStories: [
        { ...makeStory('https://news.com'), content: 'Y'.repeat(100000) },
      ],
    };

    const result = expandHighlights(data);

    expect(result.organic?.[0].content).toBeUndefined();
    expect(result.organic?.[0].references).toBeUndefined();
    expect(result.organic?.[0].title).toBe('Title for https://a.com');
    expect(result.organic?.[0].snippet).toBe('Snippet for https://a.com');
    expect(result.topStories?.[0].content).toBeUndefined();
    expect(result.topStories?.[0].title).toBe('Story for https://news.com');
    expect(data.organic?.[0].content).toBeDefined();
  });

  test('strips content when highlights array is empty', () => {
    const data: t.SearchResultData = {
      organic: [
        {
          ...makeOrganic('https://a.com'),
          content: 'X'.repeat(5000),
          highlights: [],
        },
      ],
    };

    const result = expandHighlights(data);

    expect(result.organic?.[0].content).toBeUndefined();
    expect(result.organic?.[0].highlights).toEqual([]);
  });

  test('returns sources without content or references unchanged', () => {
    const source = makeOrganic('https://a.com');
    const result = expandHighlights({ organic: [source] });

    expect(result.organic?.[0]).toBe(source);
  });

  test('still expands highlights and strips content on the normal path', () => {
    const highlightText = 'THE KEY FACT OF THIS PAGE IS RIGHT HERE.';
    const content = `${makeLongContent(2000)} ${highlightText} ${makeLongContent(2000)}`;
    const data: t.SearchResultData = {
      organic: [
        {
          ...makeOrganic('https://a.com'),
          content,
          highlights: [{ text: highlightText, score: 0.9 }],
        },
      ],
    };

    const result = expandHighlights(data);
    const highlight = result.organic?.[0].highlights?.[0];

    expect(highlight?.text).toContain(highlightText);
    expect(highlight?.text.length).toBeGreaterThan(highlightText.length);
    expect(highlight?.score).toBe(0.9);
    expect(result.organic?.[0].content).toBeUndefined();
    expect(result.organic?.[0].references).toBeUndefined();
  });

  test('honors a custom mainExpandBy when expanding highlights', () => {
    const highlightText = 'KEYFACT';
    // Boundary-free filler so expansion is governed purely by mainExpandBy,
    // not by where natural separators happen to fall.
    const content = `${'x'.repeat(1000)}${highlightText}${'y'.repeat(1000)}`;
    const makeData = (): t.SearchResultData => ({
      organic: [
        {
          ...makeOrganic('https://a.com'),
          content,
          highlights: [{ text: highlightText, score: 0.9 }],
        },
      ],
    });

    // separatorExpandBy 0 isolates the mainExpandBy effect.
    const narrow = expandHighlights(makeData(), 50, 0).organic?.[0]
      .highlights?.[0];
    const wide = expandHighlights(makeData(), 500, 0).organic?.[0]
      .highlights?.[0];

    expect(narrow?.text).toContain(highlightText);
    expect(wide?.text).toContain(highlightText);
    // 50 chars of context each side vs. 500 → wide must be markedly longer.
    expect(wide!.text.length).toBeGreaterThan(narrow!.text.length);
    expect(narrow!.text.length).toBe(highlightText.length + 100);
    expect(wide!.text.length).toBe(highlightText.length + 1000);
  });

  test('honors a custom separatorExpandBy when seeking boundaries', () => {
    const highlightText = 'KEYFACT';
    // The main window lands inside a boundary-free 'y' run; the only natural
    // boundary ('. ') sits 250 chars past the main window's end. A small
    // separator range can't reach it; a large one can.
    const content = `${'x'.repeat(100)}${highlightText}${'y'.repeat(300)}. ${'z'.repeat(300)}`;
    const makeData = (): t.SearchResultData => ({
      organic: [
        {
          ...makeOrganic('https://a.com'),
          content,
          highlights: [{ text: highlightText, score: 0.9 }],
        },
      ],
    });

    // Same mainExpandBy; only the separator search range differs.
    const narrow = expandHighlights(makeData(), 50, 100).organic?.[0]
      .highlights?.[0];
    const wide = expandHighlights(makeData(), 50, 400).organic?.[0]
      .highlights?.[0];

    expect(narrow?.text).toContain(highlightText);
    expect(wide?.text).toContain(highlightText);
    // Only the wider separator range reaches the trailing sentence boundary.
    expect(wide!.text.length).toBeGreaterThan(narrow!.text.length);
  });
});

describe('createSourceProcessor content capping', () => {
  const link = 'https://a.com';
  const baseFields = {
    query: 'test query',
    proMode: true,
    onGetHighlights: undefined,
  };

  test('caps stored content and reranker input at maxContentLength', async () => {
    const reranker = new RecordingReranker();
    const scraper = createFakeScraper({ [link]: makeLongContent(200000) });
    const processor = createSourceProcessor(
      { reranker, maxContentLength: 1000, logger: silentLogger },
      scraper
    );

    const data = await processor.processSources({
      ...baseFields,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    expect(data.organic?.[0].content?.length).toBe(1000);
    const rerankedChars = reranker.rerankCalls
      .flat()
      .reduce((sum, doc) => sum + doc.length, 0);
    expect(rerankedChars).toBeGreaterThan(0);
    expect(rerankedChars).toBeLessThan(2000);
  });

  test('respects SEARCH_MAX_CONTENT_LENGTH env var when config is not set', async () => {
    process.env.SEARCH_MAX_CONTENT_LENGTH = '500';
    try {
      const reranker = new RecordingReranker();
      const scraper = createFakeScraper({ [link]: makeLongContent(10000) });
      const processor = createSourceProcessor(
        { reranker, logger: silentLogger },
        scraper
      );

      const data = await processor.processSources({
        ...baseFields,
        news: false,
        numElements: 5,
        result: { success: true, data: { organic: [makeOrganic(link)] } },
      });

      expect(data.organic?.[0].content?.length).toBe(500);
    } finally {
      delete process.env.SEARCH_MAX_CONTENT_LENGTH;
    }
  });

  test('caps content at 50,000 chars by default', async () => {
    const reranker = new RecordingReranker();
    const scraper = createFakeScraper({ [link]: makeLongContent(60000) });
    const processor = createSourceProcessor(
      { reranker, logger: silentLogger },
      scraper
    );

    const data = await processor.processSources({
      ...baseFields,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    expect(data.organic?.[0].content?.length).toBe(50000);
  });

  test('passes configured topResults through to the reranker', async () => {
    const reranker = new RecordingReranker();
    const scraper = createFakeScraper({ [link]: makeLongContent(3000) });
    const processor = createSourceProcessor(
      { reranker, topResults: 2, logger: silentLogger },
      scraper
    );

    await processor.processSources({
      ...baseFields,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    expect(reranker.topKCalls).toEqual([2]);
  });
});

describe('createSourceProcessor reranker chunking', () => {
  const link = 'https://a.com';
  const baseFields = {
    query: 'test query',
    proMode: true,
    onGetHighlights: undefined,
  };

  const runWithConfig = async (
    config: Partial<t.ProcessSourcesConfig>
  ): Promise<string[]> => {
    const reranker = new RecordingReranker();
    const scraper = createFakeScraper({ [link]: makeLongContent(10000) });
    const processor = createSourceProcessor(
      { reranker, logger: silentLogger, ...config },
      scraper
    );

    await processor.processSources({
      ...baseFields,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    return reranker.rerankCalls[0] ?? [];
  };

  test('configured chunkSize produces fewer, larger chunks', async () => {
    const defaultDocs = await runWithConfig({});
    const largeDocs = await runWithConfig({
      chunkSize: 500,
      chunkOverlap: 100,
    });

    expect(defaultDocs.length).toBeGreaterThan(0);
    expect(largeDocs.length).toBeGreaterThan(0);
    expect(largeDocs.length).toBeLessThan(defaultDocs.length);
    expect(Math.max(...defaultDocs.map((d) => d.length))).toBeLessThanOrEqual(
      150
    );
    expect(Math.max(...largeDocs.map((d) => d.length))).toBeLessThanOrEqual(
      500
    );
  });

  test('respects SEARCH_CHUNK_SIZE env vars when config is not set', async () => {
    process.env.SEARCH_CHUNK_SIZE = '500';
    process.env.SEARCH_CHUNK_OVERLAP = '100';
    try {
      const docs = await runWithConfig({});
      expect(docs.length).toBeGreaterThan(0);
      expect(Math.max(...docs.map((d) => d.length))).toBeLessThanOrEqual(500);
    } finally {
      delete process.env.SEARCH_CHUNK_SIZE;
      delete process.env.SEARCH_CHUNK_OVERLAP;
    }
  });

  test('clamps overlap below chunk size instead of throwing', async () => {
    const docs = await runWithConfig({ chunkSize: 200, chunkOverlap: 300 });
    expect(docs.length).toBeGreaterThan(0);
    expect(Math.max(...docs.map((d) => d.length))).toBeLessThanOrEqual(200);
  });
});

describe('createSourceProcessor topStories capping', () => {
  const storyLinks = Array.from(
    { length: 8 },
    (_, i) => `https://news${i}.com`
  );

  const createProcessorWithStories = (
    scrapedUrls: string[]
  ): ReturnType<typeof createSourceProcessor> => {
    const contentByUrl = Object.fromEntries(
      storyLinks.map((storyLink) => [storyLink, makeLongContent(500)])
    );
    contentByUrl['https://a.com'] = makeLongContent(500);
    const scraper = createFakeScraper(contentByUrl, scrapedUrls);
    return createSourceProcessor(
      { reranker: new RecordingReranker(), logger: silentLogger },
      scraper
    );
  };

  test('caps topStories to numElements when news is enabled', async () => {
    const scrapedUrls: string[] = [];
    const processor = createProcessorWithStories(scrapedUrls);

    const data = await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: true,
      numElements: 3,
      result: {
        success: true,
        data: { organic: [], topStories: storyLinks.map(makeStory) },
      },
    });

    expect(data.topStories?.length).toBe(3);
    expect(data.topStories?.map((story) => story.link)).toEqual(
      storyLinks.slice(0, 3)
    );
    expect(scrapedUrls).toEqual(storyLinks.slice(0, 3));
    expect(data.topStories?.[0].content).toBeDefined();
  });

  test('caps topStories when organic results are missing', async () => {
    const scrapedUrls: string[] = [];
    const processor = createProcessorWithStories(scrapedUrls);

    const data = await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: true,
      numElements: 3,
      result: {
        success: true,
        data: { topStories: storyLinks.map(makeStory) },
      },
    });

    expect(data.topStories?.length).toBe(3);
    expect(scrapedUrls).toEqual([]);
  });

  test('caps topStories on the empty-links early return with news disabled', async () => {
    const scrapedUrls: string[] = [];
    const processor = createProcessorWithStories(scrapedUrls);

    const data = await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 3,
      result: {
        success: true,
        data: { organic: [], topStories: storyLinks.map(makeStory) },
      },
    });

    expect(data.topStories?.length).toBe(3);
    expect(scrapedUrls).toEqual([]);
  });

  test('caps topStories to numElements even when news is disabled', async () => {
    const scrapedUrls: string[] = [];
    const processor = createProcessorWithStories(scrapedUrls);

    const data = await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 3,
      result: {
        success: true,
        data: {
          organic: [makeOrganic('https://a.com')],
          topStories: storyLinks.map(makeStory),
        },
      },
    });

    expect(data.topStories?.length).toBe(3);
    expect(scrapedUrls).toEqual(['https://a.com']);
  });
});

describe('executeParallelSearches topStories dedupe', () => {
  test('dedupes topStories by link across main and news searches', async () => {
    const mainResult: t.SearchResult = {
      success: true,
      data: {
        organic: [makeOrganic('https://a.com')],
        topStories: [makeStory('https://n1.com')],
        news: [
          { title: 'N1 from main news', link: 'https://n1.com' },
          { title: 'N2 from main news', link: 'https://n2.com' },
        ],
      },
    };
    const newsResult: t.SearchResult = {
      success: true,
      data: {
        news: [
          { title: 'N2 from news search', link: 'https://n2.com' },
          { title: 'N3 from news search', link: 'https://n3.com' },
          { title: 'No link' },
        ],
      },
    };
    const searchAPI = {
      getSources: async (
        params: t.GetSourcesParams
      ): Promise<t.SearchResult> =>
        params.type === 'news' ? newsResult : mainResult,
    };

    const merged = await executeParallelSearches({
      searchAPI,
      query: 'test query',
      safeSearch: 1,
      images: false,
      videos: false,
      news: true,
      logger: silentLogger,
      provider: 'serper',
      metrics: createSearchMetrics(silentLogger),
    });

    expect(merged.data?.topStories?.map((story) => story.link)).toEqual([
      'https://n1.com',
      'https://n2.com',
      'https://n3.com',
    ]);
    expect(merged.data?.topStories?.[0].title).toBe('Story for https://n1.com');
    expect(merged.data?.news).toBeUndefined();
  });
});

describe('createSourceProcessor log volume', () => {
  const createCountingLogger = (): t.Logger =>
    ({
      error: jest.fn(),
      warn: jest.fn(),
      info: jest.fn(),
      debug: jest.fn(),
    }) as unknown as t.Logger;

  const createMixedScraper = (
    okLinks: string[],
    failingLinks: Map<string, string>
  ): t.BaseScraper => ({
    scrapeUrl: async (url: string): Promise<[string, t.AnyScraperResponse]> => {
      const failure = failingLinks.get(url);
      if (failure != null) {
        return [url, { success: false, error: failure }];
      }
      return [
        url,
        {
          success: true,
          data: { markdown: okLinks.includes(url) ? makeLongContent(4000) : '' },
        },
      ];
    },
    extractContent: (
      response: t.AnyScraperResponse
    ): [string, undefined | t.References] => [
      (response as t.FirecrawlScrapeResponse).data?.markdown ?? '',
      undefined,
    ],
    extractMetadata: (): t.GenericScrapeMetadata => ({}),
  });

  test('summarizes an eight-source search in one line per phase', async () => {
    const logger = createCountingLogger();
    const okLinks = Array.from({ length: 6 }, (_, i) => `https://ok${i}.com`);
    const failing = new Map([
      ['https://bad0.com', 'Request failed with status code 403'],
      ['https://bad1.com', 'timeout of 7500ms exceeded'],
    ]);
    const links = [...okLinks, ...failing.keys()];
    const processor = createSourceProcessor(
      { reranker: new RecordingReranker(), logger },
      createMixedScraper(okLinks, failing)
    );

    await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: links.length,
      result: {
        success: true,
        data: { organic: links.map((link) => makeOrganic(link)) },
      },
    });

    /** Eight links formerly logged a scrape line plus three reranker lines
     * each; the whole call must now cost one line per phase. */
    expect(logger.debug).toHaveBeenCalledTimes(1);
    expect(logger.error).toHaveBeenCalledTimes(1);
    expect(logger.warn).not.toHaveBeenCalled();

    const scrapeLine = String((logger.error as jest.Mock).mock.calls[0][0]);
    expect(scrapeLine).toContain('links=8 ok=6');
    expect(scrapeLine).toContain(
      'failed=2 reasons={http_403:1,timeout:1}'
    );

    const rerankLine = String((logger.debug as jest.Mock).mock.calls[0][0]);
    expect(rerankLine).toContain('rerank=recording calls=6');
  });

  test('attributes a rejecting reranker to the reranker, not the chunker', async () => {
    const logger = createCountingLogger();
    class RejectingReranker extends BaseReranker {
      readonly provider = 'custom';
      constructor() {
        super(silentLogger);
      }
      async rerank(): Promise<t.Highlight[]> {
        throw new Error('custom reranker exploded');
      }
    }
    const link = 'https://a.com';
    const processor = createSourceProcessor(
      { reranker: new RejectingReranker(), logger },
      createFakeScraper({ [link]: makeLongContent(2000) })
    );

    await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 1,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    /** Splitting succeeded, so the chunk count is real and the reason must
     * point at the reranker rather than the chunker. */
    const summary = (logger.error as jest.Mock).mock.calls
      .map((call) => String(call[0]))
      .find((line) => line.includes('[web_search] rerank'));
    expect(summary).toContain('rerank=custom calls=1');
    expect(summary).toContain('reasons={error:1}');
    expect(summary).not.toContain('chunk_error');
    expect(summary).not.toContain('chunks=0');
  });

  test('counts every link of a batch scrape that never returned responses', async () => {
    const logger = createCountingLogger();
    const links = ['https://a.com', 'https://b.com', 'https://c.com'];
    const scraper: t.BaseScraper = {
      scrapeUrl: async (url: string): Promise<[string, t.AnyScraperResponse]> => [
        url,
        { success: true, data: { markdown: 'unused' } },
      ],
      scrapeUrls: async (): Promise<Array<[string, t.AnyScraperResponse]>> => {
        throw new Error('connect ECONNREFUSED 127.0.0.1:3002');
      },
      extractContent: (): [string, undefined] => ['', undefined],
      extractMetadata: (): t.GenericScrapeMetadata => ({}),
    };
    const processor = createSourceProcessor(
      { reranker: new RecordingReranker(), logger },
      scraper
    );

    await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: links.length,
      result: {
        success: true,
        data: { organic: links.map((link) => makeOrganic(link)) },
      },
    });

    /** A wholly failed batch must not read like a search that scraped
     * nothing: the summary has to show the attempt and the outage. */
    const summary = (logger.error as jest.Mock).mock.calls
      .map((call) => String(call[0]))
      .find((line) => line.includes('[web_search] scrape'));
    expect(summary).toContain('links=3 ok=0');
    expect(summary).toContain('failed=3 reasons={connection:3}');
  });

  test('flushes to the owning collector instead of logging itself', async () => {
    const logger = createCountingLogger();
    const metrics = createSearchMetrics(logger);
    const link = 'https://a.com';
    const processor = createSourceProcessor(
      { reranker: new RecordingReranker(), logger },
      createFakeScraper({ [link]: makeLongContent(2000) })
    );

    await processor.processSources({
      query: 'test query',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 1,
      metrics,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    expect(logger.debug).not.toHaveBeenCalled();

    metrics.flush();
    expect(logger.debug).toHaveBeenCalledTimes(2);
  });
});

describe('executeParallelSearches call contract', () => {
  const createMockLogger = (): t.Logger =>
    ({
      error: jest.fn(),
      warn: jest.fn(),
      info: jest.fn(),
      debug: jest.fn(),
    }) as unknown as t.Logger;

  const okResult: t.SearchResult = {
    success: true,
    data: { organic: [makeOrganic('https://a.com')] },
  };

  test('summarizes itself when called with only the legacy arguments', async () => {
    const logger = createMockLogger();
    const searchAPI = { getSources: async (): Promise<t.SearchResult> => okResult };

    const merged = await executeParallelSearches({
      searchAPI,
      query: 'test query',
      safeSearch: 1,
      images: false,
      videos: false,
      news: false,
      logger,
    });

    expect(merged.success).toBe(true);
    /** No collector supplied, so this call owns and flushes its own. */
    expect(logger.debug).toHaveBeenCalledTimes(1);
    expect(String((logger.debug as jest.Mock).mock.calls[0][0])).toContain(
      'search=unknown queries=1'
    );
  });

  test('rethrows the main search error object rather than a wrapped copy', async () => {
    const logger = createMockLogger();
    class ProviderError extends Error {}
    const failure = new ProviderError('provider exploded');
    const searchAPI = {
      getSources: async (): Promise<t.SearchResult> => {
        throw failure;
      },
    };

    await expect(
      executeParallelSearches({
        searchAPI,
        query: 'test query',
        safeSearch: 1,
        images: false,
        videos: false,
        news: false,
        logger,
      })
    ).rejects.toBe(failure);

    /** A rejected query was error-level before it was aggregated. */
    expect(logger.warn).not.toHaveBeenCalled();
    const [line, detail] = (logger.error as jest.Mock).mock.calls[0];
    expect(String(line)).toContain('failed=1 reasons={web:other:1}');
    expect(detail).toBe('provider exploded');
  });

  test('records a slow sibling failure even when the main search rejects first', async () => {
    const logger = createMockLogger();
    const failure = new Error('main provider exploded');
    const searchAPI = {
      getSources: async (
        params: t.GetSourcesParams
      ): Promise<t.SearchResult> => {
        if (params.type !== 'images') {
          throw failure;
        }
        // Settles well after the main search has already rejected.
        await new Promise((resolve) => setTimeout(resolve, 20));
        return { success: false, error: 'timeout of 10000ms exceeded' };
      },
    };

    await expect(
      executeParallelSearches({
        searchAPI,
        query: 'test query',
        safeSearch: 1,
        images: true,
        videos: false,
        news: false,
        logger,
        provider: 'serper',
      })
    ).rejects.toBe(failure);

    /** The summary must not be flushed while a sibling is still in flight,
     * or that provider's failure is lost from the aggregate entirely. */
    const line = String((logger.error as jest.Mock).mock.calls[0][0]);
    expect(line).toContain('queries=2');
    expect(line).toContain('failed=2');
    expect(line).toContain('images:timeout:1');
    expect(line).toContain('web:other:1');
  });

  test('keeps a sub-search failure off the error channel', async () => {
    const logger = createMockLogger();
    const searchAPI = {
      getSources: async (params: t.GetSourcesParams): Promise<t.SearchResult> =>
        params.type === 'news'
          ? { success: false, error: 'timeout of 10000ms exceeded' }
          : okResult,
    };

    const merged = await executeParallelSearches({
      searchAPI,
      query: 'test query',
      safeSearch: 1,
      images: false,
      videos: false,
      news: true,
      logger,
      provider: 'serper',
    });

    expect(merged.success).toBe(true);
    expect(logger.error).not.toHaveBeenCalled();
    expect(String((logger.warn as jest.Mock).mock.calls[0][0])).toContain(
      'failed=1 reasons={news:timeout:1}'
    );
  });
});
