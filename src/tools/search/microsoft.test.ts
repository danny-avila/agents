import axios from 'axios';
import type * as t from './types';
import { MicrosoftScraper, createMicrosoftScraper } from './microsoft-scraper';
import { createSearchAPI } from './search';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockLogger = {
  error: jest.fn(),
  warn: jest.fn(),
  info: jest.fn(),
  debug: jest.fn(),
} as unknown as t.Logger;

describe('Microsoft Web IQ search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws when the API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'microsoftWebIQ',
        microsoftWebIQApiKey: '',
      })
    ).toThrow(
      'MICROSOFT_WEBIQ_API_KEY is required for Microsoft Web IQ search'
    );
  });

  it('returns an error for empty queries without calling the API', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'microsoftWebIQ',
      microsoftWebIQApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({ success: false, error: 'Query cannot be empty' });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('hits the web endpoint with x-apikey and passage format, mapping webResults to organic', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 200,
      data: {
        webResults: [
          {
            title: 'Result One',
            url: 'https://example.com/one',
            content: 'Relevant passage about the query.',
            lastUpdatedAt: '2026-01-02T00:00:00Z',
          },
          {
            title: 'No URL',
            url: '',
            content: 'should be filtered out',
          },
        ],
        traceId: 'trace-123',
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'microsoftWebIQ',
      microsoftWebIQApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    const [url, payload, requestConfig] = mockedAxios.post.mock.calls[0];
    expect(url).toBe('https://api.microsoft.ai/v3/search/web');
    expect(payload).toMatchObject({
      query: 'example query',
      contentFormat: 'passage',
    });
    expect(requestConfig?.headers).toMatchObject({ 'x-apikey': 'test-key' });

    expect(result.success).toBe(true);
    expect(result.data?.organic).toEqual([
      {
        position: 1,
        title: 'Result One',
        link: 'https://example.com/one',
        snippet: 'Relevant passage about the query.',
        date: '2026-01-02T00:00:00Z',
      },
    ]);
  });

  it('hits the news endpoint and maps newsResults to topStories when news is true', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 200,
      data: {
        newsResults: [
          {
            title: 'Breaking',
            url: 'https://news.example.com/a',
            snippet: 'terse news snippet',
            source: 'Example News',
            lastUpdatedAt: '2026-01-03T00:00:00Z',
            thumbnail: { url: 'https://img.example.com/a.jpg' },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'microsoftWebIQ',
      microsoftWebIQApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'tech funding',
      news: true,
    });

    const [url] = mockedAxios.post.mock.calls[0];
    expect(url).toBe('https://api.microsoft.ai/v3/search/news');
    expect(result.data?.topStories).toEqual([
      {
        title: 'Breaking',
        link: 'https://news.example.com/a',
        source: 'Example News',
        date: '2026-01-03T00:00:00Z',
        imageUrl: 'https://img.example.com/a.jpg',
      },
    ]);
    expect(result.data?.news?.[0]?.snippet).toBe('terse news snippet');
  });

  it('returns a failure result when the request throws', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'microsoftWebIQ',
      microsoftWebIQApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.success).toBe(false);
    expect(result.error).toBe(
      'Microsoft Web IQ search request failed: Network error'
    );
  });
});

describe('Microsoft Web IQ Browse scraper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns a failure response when the API key is missing', async () => {
    const scraper = new MicrosoftScraper({ apiKey: '', logger: mockLogger });

    const [url, response] = await scraper.scrapeUrl('https://example.com');

    expect(url).toBe('https://example.com');
    expect(response).toEqual({
      success: false,
      error: 'MICROSOFT_WEBIQ_API_KEY is not set',
    });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('maps a 200 response to content and metadata', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 200,
      data: {
        url: 'https://example.com/page',
        title: 'Example Page',
        content: '# Heading\n\nBody text.',
        language: 'en',
        lastUpdatedAt: '2026-01-04T00:00:00Z',
        crawledAt: '2026-01-05T00:00:00Z',
        isAdult: false,
      },
    });

    const scraper = createMicrosoftScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const [, response] = await scraper.scrapeUrl('https://example.com/page');

    const [requestUrl, payload, requestConfig] = mockedAxios.post.mock.calls[0];
    expect(requestUrl).toBe('https://api.microsoft.ai/v3/browse');
    expect(payload).toMatchObject({
      url: 'https://example.com/page',
      contentFormat: 'markdown',
      liveCrawl: 'none',
    });
    expect(requestConfig?.headers).toMatchObject({ 'x-apikey': 'test-key' });

    expect(response.success).toBe(true);
    const [content] = scraper.extractContent(response);
    expect(content).toBe('# Heading\n\nBody text.');
    const metadata = scraper.extractMetadata(response);
    expect(metadata).toMatchObject({
      url: 'https://example.com/page',
      title: 'Example Page',
      modifiedTime: '2026-01-04T00:00:00Z',
    });
  });

  it('treats a 202 (live crawl in progress) as a scrape failure', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 202,
      data: { retryAfter: '60s' },
    });

    const scraper = createMicrosoftScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const [, response] = await scraper.scrapeUrl('https://example.com/slow');

    expect(response.success).toBe(false);
    expect(response.error).toBe('Microsoft Browse returned status 202');
    expect(scraper.extractContent(response)).toEqual(['', undefined]);
  });

  it('treats a 404 (URL not indexed) as a scrape failure', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      status: 404,
      data: {},
    });

    const scraper = createMicrosoftScraper({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const [, response] = await scraper.scrapeUrl('https://example.com/missing');

    expect(response.success).toBe(false);
    expect(response.error).toBe('Microsoft Browse returned status 404');
  });
});
