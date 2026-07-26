import axios from 'axios';
import { createExaScraper } from './exa-scraper';
import { createSearchAPI } from './search';
import { createSearchTool } from './tool';
import { DATE_RANGE } from './schema';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const sampleResponse = {
  data: {
    results: [
      {
        id: 'https://example.com/ts',
        title: 'TypeScript Best Practices 2026',
        url: 'https://example.com/ts',
        publishedDate: '2026-01-15T10:30:00.000Z',
        author: 'Jane Doe',
        highlights: ['First highlight.', 'Second highlight.'],
      },
      {
        id: 'https://example.com/second',
        title: 'Second result',
        url: 'https://example.com/second',
        text: 'Full text fallback when highlights are absent.',
      },
    ],
  },
};

describe('Exa search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.EXA_API_KEY;
    delete process.env.EXA_API_URL;
  });

  it('throws when no API key is configured', () => {
    expect(() => createSearchAPI({ searchProvider: 'exa' })).toThrow(
      'EXA_API_KEY is required for Exa API'
    );
  });

  it('returns an error for empty queries without calling the API', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({ success: false, error: 'Query cannot be empty' });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('sends auto search with nested highlights contents and the API key header', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://api.exa.ai/search',
      {
        query: 'typescript',
        type: 'auto',
        numResults: 8,
        contents: { highlights: true },
      },
      expect.objectContaining({
        headers: expect.objectContaining({ 'x-api-key': 'test-key' }),
      })
    );
    expect(result.success).toBe(true);
  });

  it('maps results into organic sources (highlights and text fallback)', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.data?.organic).toEqual([
      {
        title: 'TypeScript Best Practices 2026',
        link: 'https://example.com/ts',
        snippet: 'First highlight.\n...\nSecond highlight.',
        date: '2026-01-15T10:30:00.000Z',
      },
      {
        title: 'Second result',
        link: 'https://example.com/second',
        snippet: 'Full text fallback when highlights are absent.',
        date: undefined,
      },
    ]);
  });

  it('applies domain filters, result limits, and a custom base URL', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
      exaApiUrl: 'https://exa.example.com/',
      exaSearchOptions: {
        maxResults: 3,
        includeDomains: ['github.com'],
        excludeDomains: ['example.org'],
      },
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://exa.example.com/search',
      expect.objectContaining({
        numResults: 3,
        includeDomains: ['github.com'],
        excludeDomains: ['example.org'],
      }),
      expect.any(Object)
    );
  });

  it('maps date ranges to startPublishedDate', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);
    const now = Date.now();
    jest.spyOn(Date, 'now').mockReturnValue(now);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    await searchAPI.getSources({
      query: 'typescript',
      date: DATE_RANGE.PAST_WEEK,
    });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        startPublishedDate: new Date(now - 7 * 24 * 3600000).toISOString(),
      }),
      expect.any(Object)
    );
  });

  it('maps country to a two-letter userLocation and ignores invalid values', async () => {
    mockedAxios.post.mockResolvedValue(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    await searchAPI.getSources({ query: 'typescript', country: 'US' });
    await searchAPI.getSources({
      query: 'typescript',
      country: 'United States',
    });

    expect(mockedAxios.post.mock.calls[0][1]).toEqual(
      expect.objectContaining({ userLocation: 'us' })
    );
    expect(mockedAxios.post.mock.calls[1][1]).not.toHaveProperty(
      'userLocation'
    );
  });

  it('maps news sub-searches to the news category with source attribution', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({
      query: 'typescript',
      type: 'news',
    });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ category: 'news' }),
      expect.any(Object)
    );
    expect(result.data?.news?.[0]).toEqual(
      expect.objectContaining({
        link: 'https://example.com/ts',
        source: 'example.com',
      })
    );
  });

  it('surfaces request failures as a structured error', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('Exa API request failed: Network error');
  });
});

describe('Exa capability gating', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.EXA_API_KEY;
    delete process.env.EXA_API_URL;
  });

  it('skips image/video sub-searches but runs the news sub-search', async () => {
    mockedAxios.post.mockResolvedValue(sampleResponse);

    const searchTool = createSearchTool({
      searchProvider: 'exa',
      exaApiKey: 'test-key',
      scraperProvider: 'firecrawl',
      firecrawlApiKey: 'k',
      topResults: 1,
      rerankerType: 'none',
    });

    await searchTool.invoke({
      query: 'typescript',
      images: true,
      news: true,
      videos: true,
    });

    const searchCalls = mockedAxios.post.mock.calls.filter(([url]) =>
      (url as string).includes('api.exa.ai/search')
    );
    expect(searchCalls).toHaveLength(2);
    expect(searchCalls[1][1]).toEqual(
      expect.objectContaining({ category: 'news' })
    );
  });
});

describe('Exa scraper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.EXA_API_KEY;
    delete process.env.EXA_API_URL;
  });

  const contentsResponse = {
    data: {
      results: [
        {
          id: 'https://example.com/a',
          url: 'https://example.com/a',
          title: 'Page A',
          author: 'Author A',
          publishedDate: '2026-01-15T10:30:00.000Z',
          text: 'Content of page A.',
          image: 'https://example.com/a.png',
          favicon: 'https://example.com/favicon.ico',
        },
      ],
      statuses: [
        { id: 'https://example.com/a', status: 'success' },
        {
          id: 'https://example.com/broken',
          status: 'error',
          error: { httpStatusCode: 500, tag: 'CRAWL_UNKNOWN_ERROR' },
        },
      ],
    },
  };

  it('returns structured errors for all URLs when no API key is set', async () => {
    const scraper = createExaScraper();
    const results = await scraper.scrapeUrls(['https://example.com/a']);

    expect(results).toEqual([
      [
        'https://example.com/a',
        { success: false, error: 'EXA_API_KEY is not set' },
      ],
    ]);
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('batches URLs through the contents endpoint and handles mixed success/failure', async () => {
    mockedAxios.post.mockResolvedValueOnce(contentsResponse);

    const scraper = createExaScraper({ apiKey: 'test-key' });
    const results = await scraper.scrapeUrls([
      'https://example.com/a',
      'https://example.com/broken',
    ]);

    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://api.exa.ai/contents',
      {
        urls: ['https://example.com/a', 'https://example.com/broken'],
        text: true,
      },
      expect.objectContaining({
        headers: expect.objectContaining({ 'x-api-key': 'test-key' }),
      })
    );

    const [okUrl, okResponse] = results[0];
    expect(okUrl).toBe('https://example.com/a');
    expect(okResponse.success).toBe(true);

    const [failedUrl, failedResponse] = results[1];
    expect(failedUrl).toBe('https://example.com/broken');
    expect(failedResponse).toEqual({
      success: false,
      error: 'CRAWL_UNKNOWN_ERROR',
    });
  });

  it('splits large URL lists into batches of 20', async () => {
    mockedAxios.post.mockResolvedValue({ data: { results: [], statuses: [] } });

    const scraper = createExaScraper({ apiKey: 'test-key' });
    const urls = Array.from(
      { length: 25 },
      (_, i) => `https://example.com/${i}`
    );
    await scraper.scrapeUrls(urls);

    expect(mockedAxios.post).toHaveBeenCalledTimes(2);
    expect(
      (mockedAxios.post.mock.calls[0][1] as { urls: string[] }).urls
    ).toHaveLength(20);
    expect(
      (mockedAxios.post.mock.calls[1][1] as { urls: string[] }).urls
    ).toHaveLength(5);
  });

  it('delegates scrapeUrl to a single-URL batch', async () => {
    mockedAxios.post.mockResolvedValueOnce(contentsResponse);

    const scraper = createExaScraper({ apiKey: 'test-key' });
    const [url, response] = await scraper.scrapeUrl('https://example.com/a');

    expect(url).toBe('https://example.com/a');
    expect(response.success).toBe(true);
  });

  it('returns structured errors for every URL when the request fails', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const scraper = createExaScraper({ apiKey: 'test-key' });
    const results = await scraper.scrapeUrls(['https://example.com/a']);

    expect(results).toEqual([
      [
        'https://example.com/a',
        {
          success: false,
          error: 'Exa contents API request failed: Network error',
        },
      ],
    ]);
  });

  it('extracts content and metadata from a successful response', async () => {
    mockedAxios.post.mockResolvedValueOnce(contentsResponse);

    const scraper = createExaScraper({ apiKey: 'test-key' });
    const [, response] = await scraper.scrapeUrl('https://example.com/a');

    const [content, references] = scraper.extractContent(response);
    expect(content).toBe('Content of page A.');
    expect(references).toBeUndefined();

    expect(scraper.extractMetadata(response)).toEqual({
      title: 'Page A',
      author: 'Author A',
      publishedTime: '2026-01-15T10:30:00.000Z',
      'og:image': 'https://example.com/a.png',
      favicon: 'https://example.com/favicon.ico',
    });
  });

  it('extracts empty content and metadata from failed responses', () => {
    const scraper = createExaScraper({ apiKey: 'test-key' });
    const failed = { success: false, error: 'nope' };

    expect(scraper.extractContent(failed)).toEqual(['', undefined]);
    expect(scraper.extractMetadata(failed)).toEqual({});
  });
});
