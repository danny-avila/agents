import axios from 'axios';
import { createSearchAPI } from './search';
import { TavilyScraper, createTavilyScraper } from './tavily-scraper';
import type * as t from './types';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockLogger = {
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  debug: jest.fn(),
} as unknown as t.Logger;

describe('Tavily search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes string-mode options and maps Tavily response fields', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        answer: 'A concise answer.',
        images: [
          {
            url: 'https://example.com/image.png',
            description: 'Example image',
          },
          'https://example.com/second.png',
        ],
        results: [
          {
            title: 'Example',
            url: 'https://example.com',
            content: 'Example summary',
            published_date: '2026-01-02',
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
      tavilySearchOptions: {
        includeAnswer: 'advanced',
        includeRawContent: 'markdown',
        includeImages: true,
        includeImageDescriptions: true,
        safeSearch: true,
      },
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      country: 'US',
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      country: 'united states',
      safe_search: true,
      include_answer: 'advanced',
      include_raw_content: 'markdown',
      include_images: true,
      include_image_descriptions: true,
    });
    expect(result.success).toBe(true);
    expect(result.data?.answerBox?.snippet).toBe('A concise answer.');
    expect(result.data?.organic?.[0]).toMatchObject({
      title: 'Example',
      link: 'https://example.com',
      snippet: 'Example summary',
      date: '2026-01-02',
    });
    expect(result.data?.images?.[0]).toMatchObject({
      imageUrl: 'https://example.com/image.png',
      title: 'Example image',
      position: 1,
    });
    expect(result.data?.images?.[1]).toMatchObject({
      imageUrl: 'https://example.com/second.png',
      position: 2,
    });
  });

  it('omits country for Tavily news searches', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
    });

    await searchAPI.getSources({
      query: 'example query',
      country: 'US',
      news: true,
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      topic: 'news',
    });
    expect(payload).not.toHaveProperty('country');
    expect(payload).not.toHaveProperty('safe_search');
  });

  it('maps ISO country codes to Tavily country enum values', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
    });

    await searchAPI.getSources({
      query: 'example query',
      country: 'CZ',
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      country: 'czech republic',
    });
    expect(payload).not.toHaveProperty('safe_search');
  });

  it('omits safe search for unsupported Tavily search depths', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
      tavilySearchOptions: {
        searchDepth: 'fast',
        safeSearch: true,
      },
    });

    await searchAPI.getSources({
      query: 'example query',
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      search_depth: 'fast',
    });
    expect(payload).not.toHaveProperty('safe_search');
  });

  it('only sends chunks per source for advanced Tavily searches', async () => {
    mockedAxios.post.mockResolvedValue({
      data: {
        results: [],
      },
    });

    const basicSearchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
      tavilySearchOptions: {
        chunksPerSource: 2,
      },
    });

    await basicSearchAPI.getSources({
      query: 'example query',
    });
    const [, basicPayload] = mockedAxios.post.mock.calls[0];

    expect(basicPayload).toMatchObject({
      query: 'example query',
      search_depth: 'basic',
    });
    expect(basicPayload).not.toHaveProperty('chunks_per_source');

    const advancedSearchAPI = createSearchAPI({
      searchProvider: 'tavily',
      tavilyApiKey: 'test-key',
      tavilySearchOptions: {
        searchDepth: 'advanced',
        chunksPerSource: 2,
      },
    });

    await advancedSearchAPI.getSources({
      query: 'example query',
    });
    const [, advancedPayload] = mockedAxios.post.mock.calls[1];

    expect(advancedPayload).toMatchObject({
      query: 'example query',
      search_depth: 'advanced',
      chunks_per_source: 2,
    });
  });
});

describe('TavilyScraper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('warns when TAVILY_API_KEY is not set', () => {
      const logger = { ...mockLogger, warn: jest.fn() } as unknown as t.Logger;
      new TavilyScraper({ apiKey: '', logger });
      expect(logger.warn).toHaveBeenCalledWith(
        'TAVILY_API_KEY is not set. Scraping will not work.'
      );
    });

    it('uses TAVILY_EXTRACT_URL env var for apiUrl', () => {
      const original = process.env.TAVILY_EXTRACT_URL;
      process.env.TAVILY_EXTRACT_URL =
        'https://custom-proxy.example.com/extract';
      const scraper = new TavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      expect(scraper['apiUrl']).toBe(
        'https://custom-proxy.example.com/extract'
      );
      if (original !== undefined) {
        process.env.TAVILY_EXTRACT_URL = original;
      } else {
        delete process.env.TAVILY_EXTRACT_URL;
      }
    });

    it('defaults to https://api.tavily.com/extract', () => {
      const original = process.env.TAVILY_EXTRACT_URL;
      delete process.env.TAVILY_EXTRACT_URL;
      const scraper = new TavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      expect(scraper['apiUrl']).toBe('https://api.tavily.com/extract');
      if (original !== undefined) {
        process.env.TAVILY_EXTRACT_URL = original;
      }
    });

    it('defaults timeout to 15000ms', () => {
      const scraper = new TavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      expect(scraper['timeout']).toBe(15000);
    });
  });

  describe('scrapeUrl', () => {
    it('returns error when API key is not set', async () => {
      const scraper = createTavilyScraper({ apiKey: '', logger: mockLogger });
      const [url, response] = await scraper.scrapeUrl('https://example.com');
      expect(url).toBe('https://example.com');
      expect(response.success).toBe(false);
      expect(response.error).toBe('TAVILY_API_KEY is not set');
    });

    it('returns scraped content on success', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: [
            {
              url: 'https://example.com',
              raw_content: '# Hello World\nSome content here.',
              images: ['https://example.com/img.png'],
              favicon: 'https://example.com/favicon.ico',
            },
          ],
          failed_results: [],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [url, response] = await scraper.scrapeUrl('https://example.com');

      expect(url).toBe('https://example.com');
      expect(response.success).toBe(true);
      expect(response.data?.rawContent).toBe(
        '# Hello World\nSome content here.'
      );
      expect(response.data?.images).toEqual(['https://example.com/img.png']);
      expect(response.data?.favicon).toBe('https://example.com/favicon.ico');
    });

    it('applies per-call extract options to the Tavily payload', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: [
            {
              url: 'https://example.com',
              raw_content: 'Content',
              images: [],
            },
          ],
          failed_results: [],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      await scraper.scrapeUrl('https://example.com', {
        includeFavicon: true,
        format: 'text',
        timeout: 2000,
      });
      const [, payload, config] = mockedAxios.post.mock.calls[0];

      expect(payload).toMatchObject({
        urls: ['https://example.com'],
        include_favicon: true,
        format: 'text',
        timeout: 2,
      });
      expect(payload).not.toHaveProperty('chunks_per_source');
      expect(config).toMatchObject({ timeout: 2000 });
    });

    it('omits extract timeout from the payload when using Tavily defaults', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: [
            {
              url: 'https://example.com',
              raw_content: 'Content',
              images: [],
            },
          ],
          failed_results: [],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
        extractDepth: 'advanced',
      });
      await scraper.scrapeUrl('https://example.com');
      const [, payload, config] = mockedAxios.post.mock.calls[0];

      expect(payload).toMatchObject({
        urls: ['https://example.com'],
        extract_depth: 'advanced',
      });
      expect(payload).not.toHaveProperty('timeout');
      expect(config).toMatchObject({ timeout: 15000 });
    });

    it('handles API failure gracefully', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [url, response] = await scraper.scrapeUrl('https://example.com');

      expect(url).toBe('https://example.com');
      expect(response.success).toBe(false);
      expect(response.error).toContain('Tavily Extract API request failed');
      expect(response.error).toContain('Network error');
    });

    it('reads error from failed_results when results is empty', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: [],
          failed_results: [
            {
              url: 'https://example.com',
              error: 'Page is behind a paywall',
            },
          ],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [url, response] = await scraper.scrapeUrl('https://example.com');

      expect(url).toBe('https://example.com');
      expect(response.success).toBe(false);
      expect(response.error).toBe('Page is behind a paywall');
    });

    it('returns descriptive error when URL not in results or failed_results', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { results: [], failed_results: [] },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [, response] = await scraper.scrapeUrl('https://missing.com');

      expect(response.success).toBe(false);
      expect(response.error).toBe('URL not found in Tavily Extract response');
    });
  });

  describe('scrapeUrls (batch)', () => {
    it('batches multiple URLs into a single API call', async () => {
      const urls = [
        'https://example.com/1',
        'https://example.com/2',
        'https://example.com/3',
      ];

      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: urls.map((url) => ({
            url,
            raw_content: `Content for ${url}`,
            images: [],
          })),
          failed_results: [],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const results = await scraper.scrapeUrls(urls);

      expect(mockedAxios.post).toHaveBeenCalledTimes(1);
      expect(results).toHaveLength(3);

      for (let i = 0; i < results.length; i++) {
        const [url, response] = results[i];
        expect(url).toBe(urls[i]);
        expect(response.success).toBe(true);
        expect(response.data?.rawContent).toBe(`Content for ${urls[i]}`);
      }
    });

    it('handles mixed success and failure results', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          results: [
            {
              url: 'https://example.com/ok',
              raw_content: 'Good content',
              images: [],
            },
          ],
          failed_results: [
            {
              url: 'https://example.com/fail',
              error: 'Access denied',
            },
          ],
        },
      });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const results = await scraper.scrapeUrls([
        'https://example.com/ok',
        'https://example.com/fail',
      ]);

      expect(results).toHaveLength(2);
      expect(results[0][1].success).toBe(true);
      expect(results[1][1].success).toBe(false);
      expect(results[1][1].error).toBe('Access denied');
    });

    it('splits large batches into chunks of 20', async () => {
      const urls = Array.from(
        { length: 25 },
        (_, i) => `https://example.com/${i}`
      );

      mockedAxios.post
        .mockResolvedValueOnce({
          data: {
            results: urls.slice(0, 20).map((url) => ({
              url,
              raw_content: 'content',
              images: [],
            })),
            failed_results: [],
          },
        })
        .mockResolvedValueOnce({
          data: {
            results: urls.slice(20).map((url) => ({
              url,
              raw_content: 'content',
              images: [],
            })),
            failed_results: [],
          },
        });

      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const results = await scraper.scrapeUrls(urls);

      expect(mockedAxios.post).toHaveBeenCalledTimes(2);
      expect(results).toHaveLength(25);
    });

    it('returns errors for all URLs when API key is missing', async () => {
      const scraper = createTavilyScraper({ apiKey: '', logger: mockLogger });
      const results = await scraper.scrapeUrls([
        'https://a.com',
        'https://b.com',
      ]);

      expect(results).toHaveLength(2);
      for (const [, response] of results) {
        expect(response.success).toBe(false);
        expect(response.error).toBe('TAVILY_API_KEY is not set');
      }
    });
  });

  describe('extractContent', () => {
    it('returns content and image references', () => {
      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [content, references] = scraper.extractContent({
        success: true,
        data: {
          rawContent: 'Hello world',
          images: ['https://img.example.com/1.png'],
        },
      });

      expect(content).toBe('Hello world');
      expect(references).toBeDefined();
      expect(references?.images).toHaveLength(1);
      expect(references?.images[0].originalUrl).toBe(
        'https://img.example.com/1.png'
      );
    });

    it('returns empty content for failed response', () => {
      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [content, references] = scraper.extractContent({
        success: false,
        error: 'Failed',
      });

      expect(content).toBe('');
      expect(references).toBeUndefined();
    });

    it('returns undefined references when no images', () => {
      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const [content, references] = scraper.extractContent({
        success: true,
        data: { rawContent: 'No images here', images: [] },
      });

      expect(content).toBe('No images here');
      expect(references).toBeUndefined();
    });
  });

  describe('extractMetadata', () => {
    it('returns images_count for successful response', () => {
      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const metadata = scraper.extractMetadata({
        success: true,
        data: {
          rawContent: 'content',
          images: ['a', 'b', 'c'],
          favicon: 'https://example.com/favicon.ico',
        },
      });

      expect(metadata).toEqual({
        favicon: 'https://example.com/favicon.ico',
        images_count: 3,
      });
    });

    it('returns empty object for failed response', () => {
      const scraper = createTavilyScraper({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      const metadata = scraper.extractMetadata({
        success: false,
        error: 'Failed',
      });

      expect(metadata).toEqual({});
    });
  });
});
