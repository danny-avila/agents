import axios from 'axios';
import type * as t from './types';
import { CrwScraper, createCrwScraper } from './crw-scraper';
import { createSearchAPI } from './search';
import { createSearchTool } from './tool';
import { DATE_RANGE } from './schema';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockLogger = {
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  debug: jest.fn(),
} as unknown as t.Logger;

describe('CRW search API', () => {
  const originalKey = process.env.CRW_API_KEY;
  const originalUrl = process.env.CRW_API_URL;

  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.CRW_API_KEY;
    delete process.env.CRW_API_URL;
  });

  afterAll(() => {
    if (originalKey === undefined) {
      delete process.env.CRW_API_KEY;
    } else {
      process.env.CRW_API_KEY = originalKey;
    }
    if (originalUrl === undefined) {
      delete process.env.CRW_API_URL;
    } else {
      process.env.CRW_API_URL = originalUrl;
    }
  });

  it('does not throw when the CRW API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'crw',
        crwApiKey: '',
      })
    ).not.toThrow();
    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: '',
    });
    expect(typeof searchAPI.getSources).toBe('function');
  });

  it('throws for an invalid search provider with an updated message', () => {
    expect(() =>
      createSearchAPI({ searchProvider: 'bogus' as t.SearchProvider })
    ).toThrow(/'crw'/);
  });

  it('returns an error for empty CRW search queries', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('returns an error when the CRW search request fails', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('fastCRW search request failed: Network error');
  });

  it('maps the CRW search response fields', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          web: [
            { title: 'T', url: 'https://e.com', description: 'D', position: 1 },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    const [url, payload, config] = mockedAxios.post.mock.calls[0];
    expect(url).toBe('https://api.fastcrw.com/v1/search');
    expect(payload).toEqual({
      query: 'example query',
      limit: 8,
      sources: ['web'],
    });
    expect(
      (config as { headers: Record<string, string> }).headers.Authorization
    ).toBe('Bearer test-key');
    expect(result.success).toBe(true);
    expect(result.data?.organic?.[0]).toEqual({
      title: 'T',
      link: 'https://e.com',
      snippet: 'D',
      position: 1,
    });
  });

  it('uses the images source and maps image results', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          images: [
            {
              title: 'I',
              url: 'https://e.com/page',
              imageUrl: 'https://e.com/i.png',
              position: 1,
            },
            { title: 'no-image-url', url: 'https://e.com/x' },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      type: 'images',
    });

    const [, payload] = mockedAxios.post.mock.calls[0];
    expect((payload as t.CrwSearchPayload).sources).toEqual(['images']);
    expect(result.data?.images).toEqual([
      {
        title: 'I',
        imageUrl: 'https://e.com/i.png',
        link: 'https://e.com/page',
        position: 1,
      },
    ]);
  });

  it('adds images to a web search when includeImages is set', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
      crwSearchOptions: { includeImages: true },
    });

    await searchAPI.getSources({ query: 'example query' });

    const [, payload] = mockedAxios.post.mock.calls[0];
    expect((payload as t.CrwSearchPayload).sources).toEqual(['web', 'images']);
  });

  it('uses the news source and maps news results', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          news: [
            {
              title: 'N',
              url: 'https://news.example.com/story',
              description: 'D',
              publishedDate: '2026-07-01T20:40:00',
              position: 1,
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      type: 'news',
    });

    const [, payload] = mockedAxios.post.mock.calls[0];
    expect((payload as t.CrwSearchPayload).sources).toEqual(['news']);
    expect(result.data?.news).toEqual([
      {
        title: 'N',
        link: 'https://news.example.com/story',
        snippet: 'D',
        date: '2026-07-01T20:40:00',
        source: 'news.example.com',
        position: 1,
      },
    ]);
  });

  it('routes a flat data array by category (documented OpenAPI shape)', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: [
          {
            title: 'W',
            url: 'https://e.com/w',
            snippet: 'S',
            category: 'general',
          },
          {
            title: 'N',
            url: 'https://e.com/n',
            snippet: 'S',
            category: 'news',
          },
          { title: 'I', url: 'https://e.com/i.png', category: 'images' },
          { title: 'U', url: 'https://e.com/u', snippet: 'S' },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.data?.organic?.map((r) => r.link)).toEqual([
      'https://e.com/w',
      'https://e.com/u',
    ]);
    expect(result.data?.news?.[0].link).toBe('https://e.com/n');
    expect(result.data?.images?.[0].imageUrl).toBe('https://e.com/i.png');
  });

  it('unwraps a data.results wrapper (self-host shape)', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          results: {
            web: [{ title: 'T', url: 'https://e.com', description: 'D' }],
          },
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.data?.organic?.[0].link).toBe('https://e.com');
  });

  it('maps the date parameter to a qdr tbs filter', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    await searchAPI.getSources({
      query: 'example query',
      date: DATE_RANGE.PAST_24_HOURS,
    });

    const [, payload] = mockedAxios.post.mock.calls[0];
    expect((payload as t.CrwSearchPayload).tbs).toBe('qdr:d');
  });

  it('omits tbs when no date is given', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'example query' });

    const [, payload] = mockedAxios.post.mock.calls[0];
    expect(payload as Record<string, unknown>).not.toHaveProperty('tbs');
  });

  it('surfaces an envelope failure with the error code', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: false,
        error: 'rate limited',
        error_code: 'RATE_LIMIT',
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result).toEqual({
      success: false,
      error: 'fastCRW search failed: [RATE_LIMIT] rate limited',
    });
  });

  it('honors the base-URL override for search', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
      crwApiUrl: 'http://localhost:3000',
    });

    await searchAPI.getSources({ query: 'example query' });

    const [url] = mockedAxios.post.mock.calls[0];
    expect(url).toBe('http://localhost:3000/v1/search');
  });

  it('omits the Authorization header when keyless', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: '',
      crwApiUrl: 'http://localhost:3000',
    });

    await searchAPI.getSources({ query: 'example query' });

    const [, , config] = mockedAxios.post.mock.calls[0];
    expect(
      (config as { headers: Record<string, string> }).headers.Authorization
    ).toBeUndefined();
  });

  it('drops results with an empty url', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          web: [
            { title: 'A', url: '', description: 'x' },
            { title: 'B', url: 'https://ok.com', description: 'y' },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'crw',
      crwApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.data?.organic).toHaveLength(1);
    expect(result.data?.organic?.[0].link).toBe('https://ok.com');
  });
});

describe('CrwScraper', () => {
  const originalUrl = process.env.CRW_API_URL;

  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.CRW_API_URL;
  });

  afterAll(() => {
    if (originalUrl === undefined) {
      delete process.env.CRW_API_URL;
    } else {
      process.env.CRW_API_URL = originalUrl;
    }
  });

  describe('constructor', () => {
    it('warns when CRW_API_KEY is not set', () => {
      const logger = { ...mockLogger, warn: jest.fn() } as unknown as t.Logger;
      new CrwScraper({ apiKey: '', logger });
      expect(logger.warn).toHaveBeenCalledWith(
        'CRW_API_KEY is not set. Scraping will not work.'
      );
    });

    it('uses the default base URL', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { success: true, markdown: '# Hi' },
      });
      const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
      await scraper.scrapeUrl('https://example.com');
      const [url] = mockedAxios.post.mock.calls[0];
      expect(url).toBe('https://api.fastcrw.com/v1/scrape');
    });

    it('honors the CRW_API_URL env override', async () => {
      process.env.CRW_API_URL = 'http://localhost:3000';
      mockedAxios.post.mockResolvedValueOnce({
        data: { success: true, markdown: '# Hi' },
      });
      const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
      await scraper.scrapeUrl('https://example.com');
      const [url] = mockedAxios.post.mock.calls[0];
      expect(url).toBe('http://localhost:3000/v1/scrape');
    });

    it('sends the render budget and pads the HTTP timeout', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { success: true, markdown: '# Hi' },
      });
      const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
      await scraper.scrapeUrl('https://example.com');
      const [, payload, config] = mockedAxios.post.mock.calls[0];
      expect((payload as { timeout: number }).timeout).toBe(7500);
      expect((payload as { deadlineMs: number }).deadlineMs).toBe(7500);
      expect((config as { timeout: number }).timeout).toBe(12500);
    });

    it('clamps deadlineMs to the documented 60000 max', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { success: true, markdown: '# Hi' },
      });
      const scraper = createCrwScraper({
        apiKey: 'k',
        timeout: 90000,
        logger: mockLogger,
      });
      await scraper.scrapeUrl('https://example.com');
      const [, payload] = mockedAxios.post.mock.calls[0];
      expect((payload as { timeout: number }).timeout).toBe(90000);
      expect((payload as { deadlineMs: number }).deadlineMs).toBe(60000);
    });

    it('requests markdown + html by default', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { success: true, markdown: '# Hi' },
      });
      const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
      await scraper.scrapeUrl('https://example.com');
      const [, payload] = mockedAxios.post.mock.calls[0];
      expect((payload as { formats: string[] }).formats).toEqual([
        'markdown',
        'html',
      ]);
    });
  });

  it('does not early-return on a missing key', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, markdown: '# Hi' },
    });
    const scraper = createCrwScraper({ apiKey: '', logger: mockLogger });
    await scraper.scrapeUrl('https://example.com');
    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
  });

  it('attaches the Bearer header only when a key is present', async () => {
    mockedAxios.post.mockResolvedValue({
      data: { success: true, markdown: '# Hi' },
    });

    const withKey = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    await withKey.scrapeUrl('https://example.com');
    const [, , withKeyConfig] = mockedAxios.post.mock.calls[0];
    expect(
      (withKeyConfig as { headers: Record<string, string> }).headers
        .Authorization
    ).toBe('Bearer k');

    mockedAxios.post.mockClear();

    const keyless = createCrwScraper({ apiKey: '', logger: mockLogger });
    await keyless.scrapeUrl('https://example.com');
    const [, , keylessConfig] = mockedAxios.post.mock.calls[0];
    expect(
      (keylessConfig as { headers: Record<string, string> }).headers
        .Authorization
    ).toBeUndefined();
  });

  it('reads the nested data container from cloud responses', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          markdown: '# Hi',
          html: '<p>x</p>',
          metadata: { title: 't' },
          links: [],
        },
      },
    });
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    const result = await scraper.scrapeUrl('https://example.com');
    expect(result).toEqual([
      'https://example.com',
      {
        success: true,
        data: {
          markdown: '# Hi',
          html: '<p>x</p>',
          metadata: { title: 't' },
          links: [],
          rawHtml: undefined,
          plainText: undefined,
          screenshot: undefined,
        },
      },
    ]);
  });

  it('normalizes top-level fields into the nested data shape', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        markdown: '# Hi',
        html: '<p>x</p>',
        metadata: { title: 't' },
        links: [],
      },
    });
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    const result = await scraper.scrapeUrl('https://example.com');
    expect(result).toEqual([
      'https://example.com',
      {
        success: true,
        data: {
          markdown: '# Hi',
          html: '<p>x</p>',
          metadata: { title: 't' },
          links: [],
          rawHtml: undefined,
          plainText: undefined,
          screenshot: undefined,
        },
      },
    ]);
  });

  describe('extractContent', () => {
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });

    it('processes markdown + html into ref markers', () => {
      const [content, references] = scraper.extractContent({
        success: true,
        data: { markdown: '# Hi', html: '<p>x</p>' },
      });
      expect(typeof content).toBe('string');
      expect(references).toHaveProperty('links');
      expect(references).toHaveProperty('images');
      expect(references).toHaveProperty('videos');
    });

    it('runs the ref-marker pass from rawHtml when html is absent', () => {
      const [content, references] = scraper.extractContent({
        success: true,
        data: { markdown: '# Hi', rawHtml: '<p>x</p>' },
      });
      expect(typeof content).toBe('string');
      expect(references).toHaveProperty('links');
    });

    it('returns empty on a failed response', () => {
      expect(scraper.extractContent({ success: false })).toEqual([
        '',
        undefined,
      ]);
    });

    it('falls back to plainText when no markdown/html/rawHtml', () => {
      expect(
        scraper.extractContent({
          success: true,
          data: {
            plainText: 'plain body',
            markdown: undefined,
            html: undefined,
            rawHtml: undefined,
          },
        })
      ).toEqual(['plain body', undefined]);
    });

    it('falls back to rawHtml when no markdown/html', () => {
      expect(
        scraper.extractContent({
          success: true,
          data: { rawHtml: '<x>', markdown: undefined, html: undefined },
        })
      ).toEqual(['<x>', undefined]);
    });
  });

  describe('extractMetadata', () => {
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });

    it('returns the metadata object', () => {
      expect(
        scraper.extractMetadata({
          success: true,
          data: { metadata: { title: 't' } },
        })
      ).toEqual({ title: 't' });
    });

    it('returns {} on a failed response', () => {
      expect(scraper.extractMetadata({ success: false })).toEqual({});
    });

    it('returns {} when metadata is missing', () => {
      expect(scraper.extractMetadata({ success: true, data: {} })).toEqual({});
    });
  });

  it('normalizes an envelope failure with an error code', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: false, error: 'blocked', error_code: 'X' },
    });
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    const result = await scraper.scrapeUrl('https://example.com');
    expect(result).toEqual([
      'https://example.com',
      { success: false, error: '[X] blocked', error_code: 'X' },
    ]);
  });

  it('normalizes an envelope failure without an error code', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: false, error: 'oops' },
    });
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    const result = await scraper.scrapeUrl('https://example.com');
    expect(result).toEqual([
      'https://example.com',
      { success: false, error: 'oops', error_code: undefined },
    ]);
  });

  it('returns a failure tuple on a network error', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    const [url, response] = await scraper.scrapeUrl('https://example.com');
    expect(url).toBe('https://example.com');
    expect(response.success).toBe(false);
    expect(response.error).toBe('fastCRW API request failed: Network error');
  });

  it('assembles the payload via omitUndefined', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, markdown: '# Hi' },
    });
    const scraper = createCrwScraper({ apiKey: 'k', logger: mockLogger });
    await scraper.scrapeUrl('https://example.com', {
      onlyMainContent: true,
      waitFor: 1000,
      formats: ['markdown'],
      renderJs: true,
    });
    const [, payload] = mockedAxios.post.mock.calls[0];
    expect(payload).toMatchObject({
      url: 'https://example.com',
      onlyMainContent: true,
      waitFor: 1000,
      formats: ['markdown'],
      renderJs: true,
    });
    expect(payload as Record<string, unknown>).not.toHaveProperty('proxy');
    expect(payload as Record<string, unknown>).not.toHaveProperty(
      'cssSelector'
    );
  });
});

describe('CRW search tool wiring', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('runs the search then scrape sequence', async () => {
    mockedAxios.post
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            web: [{ title: 'T', url: 'https://ex.com/a', description: 'D' }],
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            markdown: '# A',
            html: '<p>a</p>',
            metadata: { title: 'T' },
            links: [],
          },
        },
      });

    const searchTool = createSearchTool({
      searchProvider: 'crw',
      scraperProvider: 'crw',
      crwApiKey: 'k',
      topResults: 1,
      rerankerType: 'none',
      logger: mockLogger,
    });

    await searchTool.invoke({ query: 'hello' });

    expect(mockedAxios.post).toHaveBeenCalledTimes(2);
    const [searchUrl, , searchConfig] = mockedAxios.post.mock.calls[0];
    const [scrapeUrl, , scrapeConfig] = mockedAxios.post.mock.calls[1];
    expect((searchUrl as string).endsWith('/v1/search')).toBe(true);
    expect(
      (searchConfig as { headers: Record<string, string> }).headers
        .Authorization
    ).toBe('Bearer k');
    expect((scrapeUrl as string).endsWith('/v1/scrape')).toBe(true);
    expect(
      (scrapeConfig as { headers: Record<string, string> }).headers
        .Authorization
    ).toBe('Bearer k');
  });

  it('keeps the shared base and scraper override independent', async () => {
    mockedAxios.post
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            web: [{ title: 'T', url: 'https://ex.com/a', description: 'D' }],
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            markdown: '# A',
            html: '<p>a</p>',
            metadata: { title: 'T' },
            links: [],
          },
        },
      });

    const searchTool = createSearchTool({
      searchProvider: 'crw',
      scraperProvider: 'crw',
      crwApiKey: 'k',
      crwScraperOptions: { apiUrl: 'http://localhost:3000' },
      topResults: 1,
      rerankerType: 'none',
      logger: mockLogger,
    });

    await searchTool.invoke({ query: 'hello' });

    const [searchUrl] = mockedAxios.post.mock.calls[0];
    const [scrapeUrl] = mockedAxios.post.mock.calls[1];
    expect(searchUrl).toBe('https://api.fastcrw.com/v1/search');
    expect(scrapeUrl).toBe('http://localhost:3000/v1/scrape');
  });
});
