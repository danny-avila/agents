import axios from 'axios';
import type * as t from './types';
import { ParallelScraper, createParallelScraper } from './parallel-scraper';
import { createSearchAPI } from './search';
import { createSearchTool } from './tool';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockLogger = {
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  debug: jest.fn(),
} as unknown as t.Logger;

describe('Parallel Search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws when Parallel API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'parallel',
        parallelApiKey: '',
      })
    ).toThrow('PARALLEL_API_KEY is required for Parallel Search API');
  });

  it('returns an error for empty queries', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('returns an error when the request fails', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.success).toBe(false);
    expect(result.error).toBe(
      'Parallel Search API request failed: Network error'
    );
  });

  it('sends x-api-key auth and maps response excerpts into organic results', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        search_id: 'srch_123',
        results: [
          {
            url: 'https://example.com',
            title: 'Example',
            publish_date: '2026-01-02',
            excerpts: ['First excerpt.', 'Second excerpt.'],
          },
          {
            url: 'https://example.org',
            excerpts: ['Plain excerpt.'],
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });
    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    const [url, payload, requestConfig] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelSearchPayload,
      { headers: Record<string, string> },
    ];

    expect(url).toBe('https://api.parallel.ai/v1/search');
    expect(requestConfig.headers).toMatchObject({
      'x-api-key': 'test-key',
      'Content-Type': 'application/json',
    });
    expect(payload).toMatchObject({
      objective: 'example query',
      search_queries: ['example query'],
    });
    expect(payload).not.toHaveProperty('advanced_settings');

    expect(result.success).toBe(true);
    expect(result.data?.organic).toHaveLength(2);
    expect(result.data?.organic?.[0]).toMatchObject({
      title: 'Example',
      link: 'https://example.com',
      snippet: 'First excerpt.\n\nSecond excerpt.',
      date: '2026-01-02',
    });
    expect(result.data?.organic?.[1]).toMatchObject({
      title: '',
      link: 'https://example.org',
      snippet: 'Plain excerpt.',
    });
  });

  it('merges configured options into the request payload', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { search_id: 'srch_456', results: [] },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
      parallelSearchUrl: 'https://api.parallel.ai/v1/search',
      parallelSearchOptions: {
        objective: 'Focus on releases from the last month.',
        search_queries: ['bun runtime release'],
        mode: 'basic',
        client_model: 'claude-opus-4-7',
        max_chars_total: 25_000,
        max_chars_per_result: 4_000,
        include_domains: ['example.com'],
        exclude_domains: ['reddit.com'],
        max_results: 8,
        location: 'gb',
      },
    });

    await searchAPI.getSources({ query: 'latest stable bun version' });
    const [, payload] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelSearchPayload,
    ];

    expect(payload.objective).toBe(
      'Focus on releases from the last month.\n\nlatest stable bun version'
    );
    expect(payload.search_queries).toEqual([
      'latest stable bun version',
      'bun runtime release',
    ]);
    expect(payload.mode).toBe('basic');
    expect(payload.client_model).toBe('claude-opus-4-7');
    expect(payload.max_chars_total).toBe(25_000);
    expect(payload.advanced_settings).toEqual({
      source_policy: {
        include_domains: ['example.com'],
        exclude_domains: ['reddit.com'],
      },
      location: 'gb',
      max_results: 8,
      excerpt_settings: { max_chars_per_result: 4_000 },
    });
  });

  it('deduplicates queries and clamps to 5 entries', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { search_id: 'srch_789', results: [] },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
      parallelSearchOptions: {
        search_queries: [
          'Same Query',
          'Other 1',
          'Other 2',
          'Other 3',
          'Other 4',
          'Other 5',
        ],
      },
    });

    await searchAPI.getSources({ query: 'same query' });
    const [, payload] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelSearchPayload,
    ];

    expect(payload.search_queries).toEqual([
      'same query',
      'Other 1',
      'Other 2',
      'Other 3',
      'Other 4',
    ]);
  });

  it('clamps excerpt cap to the 1000-char minimum from the API', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { search_id: 'srch_abc', results: [] },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
      parallelSearchOptions: { max_chars_per_result: 500 },
    });

    await searchAPI.getSources({ query: 'q' });
    const [, payload] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelSearchPayload,
    ];

    expect(payload.advanced_settings?.excerpt_settings).toEqual({
      max_chars_per_result: 1_000,
    });
  });

  it('falls back to country from the LLM when location is unset', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { search_id: 'srch_def', results: [] },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'parallel',
      parallelApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q', country: 'JP' });
    const [, payload] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelSearchPayload,
    ];

    expect(payload.advanced_settings?.location).toBe('jp');
  });
});

describe('Parallel Extract scraper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('warns instead of throwing when the API key is missing', () => {
    const scraper = createParallelScraper({ apiKey: '', logger: mockLogger });
    expect(scraper).toBeInstanceOf(ParallelScraper);
    expect(mockLogger.warn).toHaveBeenCalledWith(
      expect.stringContaining('PARALLEL_API_KEY is not set')
    );
  });

  it('returns failures for every URL when the API key is missing', async () => {
    const scraper = createParallelScraper({ apiKey: '', logger: mockLogger });
    const results = await scraper.scrapeUrls(
      ['https://example.com', 'https://example.org'],
      {}
    );

    expect(results).toEqual([
      [
        'https://example.com',
        {
          success: false,
          error: 'PARALLEL_API_KEY is not set',
        },
      ],
      [
        'https://example.org',
        {
          success: false,
          error: 'PARALLEL_API_KEY is not set',
        },
      ],
    ]);
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('sends x-api-key auth, defaults to excerpts only, and maps response', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        extract_id: 'ext_123',
        results: [
          {
            url: 'https://example.com',
            title: 'Example',
            publish_date: '2026-01-02',
            excerpts: ['Para 1.', 'Para 2.'],
          },
        ],
      },
    });

    const scraper = createParallelScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const [[url, response]] = await scraper.scrapeUrls(
      ['https://example.com'],
      {}
    );

    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    const [endpoint, payload, requestConfig] = mockedAxios.post.mock
      .calls[0] as [
      string,
      t.ParallelExtractPayload,
      { headers: Record<string, string> },
    ];
    expect(endpoint).toBe('https://api.parallel.ai/v1/extract');
    expect(requestConfig.headers).toMatchObject({
      'x-api-key': 'test-key',
      'Content-Type': 'application/json',
    });
    expect(payload).toEqual({ urls: ['https://example.com'] });

    expect(url).toBe('https://example.com');
    expect(response.success).toBe(true);
    if (response.success) {
      expect(response.data.rawContent).toBe('Para 1.\n\nPara 2.');
      expect(response.data.title).toBe('Example');
      expect(response.data.publishDate).toBe('2026-01-02');
      expect(response.data.excerpts).toEqual(['Para 1.', 'Para 2.']);
    }

    const metadata = scraper.extractMetadata(response);
    expect(metadata).toMatchObject({
      excerpts_count: 2,
      title: 'Example',
      publish_date: '2026-01-02',
    });

    const [content, refs] = scraper.extractContent(response);
    expect(content).toBe('Para 1.\n\nPara 2.');
    expect(refs).toBeUndefined();
  });

  it('opts into full-content output when configured', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        extract_id: 'ext_full',
        results: [
          {
            url: 'https://example.com',
            full_content: '# Full page\n\nbody here',
            excerpts: ['ignored excerpt'],
          },
        ],
      },
    });

    const scraper = createParallelScraper({
      apiKey: 'test-key',
      fullContent: true,
      logger: mockLogger,
    });

    const [[, response]] = await scraper.scrapeUrls(
      ['https://example.com'],
      {}
    );
    const [, payload] = mockedAxios.post.mock.calls[0] as [
      string,
      t.ParallelExtractPayload,
    ];

    expect(payload.advanced_settings?.full_content_settings).toEqual({
      enabled: true,
    });
    expect(response.success).toBe(true);
    if (response.success) {
      expect(response.data.rawContent).toBe('# Full page\n\nbody here');
    }
  });

  it('reports per-URL errors from the extract response', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        extract_id: 'ext_err',
        results: [],
        errors: [
          {
            url: 'https://blocked.example.com',
            error_type: 'fetch_blocked',
            http_status_code: 403,
            content: 'Origin blocked the fetch',
          },
        ],
      },
    });

    const scraper = createParallelScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const [[, response]] = await scraper.scrapeUrls(
      ['https://blocked.example.com'],
      {}
    );

    expect(response.success).toBe(false);
    if (!response.success) {
      expect(response.error).toBe('Origin blocked the fetch');
    }
  });

  it('chunks requests larger than 20 URLs', async () => {
    mockedAxios.post.mockResolvedValue({
      data: { extract_id: 'ext_batch', results: [] },
    });

    const scraper = createParallelScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const urls = Array.from({ length: 25 }, (_, i) => `https://e${i}.com`);
    await scraper.scrapeUrls(urls, {});

    expect(mockedAxios.post).toHaveBeenCalledTimes(2);
    const firstBatch = (
      mockedAxios.post.mock.calls[0] as [string, t.ParallelExtractPayload]
    )[1].urls;
    const secondBatch = (
      mockedAxios.post.mock.calls[1] as [string, t.ParallelExtractPayload]
    )[1].urls;
    expect(firstBatch).toHaveLength(20);
    expect(secondBatch).toHaveLength(5);
  });
});

describe('createSearchTool with parallel provider', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('builds a tool that uses the Parallel scraper when both providers are set', () => {
    const tool = createSearchTool({
      searchProvider: 'parallel',
      scraperProvider: 'parallel',
      parallelApiKey: 'test-key',
      rerankerType: 'none',
      logger: mockLogger,
    });

    expect(tool).toBeDefined();
    expect(tool.name).toBeDefined();
  });
});
