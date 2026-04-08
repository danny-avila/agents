import axios from 'axios';
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
        data: { rawContent: 'content', images: ['a', 'b', 'c'] },
      });

      expect(metadata).toEqual({ images_count: 3 });
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
