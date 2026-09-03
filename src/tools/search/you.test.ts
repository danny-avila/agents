import axios from 'axios';
import { createSearchAPI } from './search';
import { DATE_RANGE } from './schema';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const YOU_PUBLIC_URL = 'https://api.you.com/v1/agents/search';
const YOU_KEYED_URL = 'https://api.you.com/v1/search';

const sampleResponse = {
  data: {
    results: {
      web: [
        {
          title: 'TypeScript Best Practices 2026',
          url: 'https://example.com/ts',
          description: 'A comprehensive guide to TypeScript.',
          snippets: ['First passage.', 'Second passage.'],
          page_age: '2026-01-15T10:30:00Z',
        },
        {
          title: 'Second result',
          url: 'https://example.com/second',
          description: 'Description fallback when snippets are absent.',
        },
      ],
      news: [
        {
          title: 'TypeScript 8 released',
          url: 'https://news.example.com/ts8',
          description: 'The release landed today.',
          page_age: '2026-01-16T09:00:00Z',
          thumbnail_url: 'https://news.example.com/ts8.jpg',
        },
      ],
    },
  },
};

describe('You.com search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.YDC_API_KEY;
    delete process.env.YDC_API_URL;
  });

  it('returns an error for empty queries without calling the API', async () => {
    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({ success: false, error: 'Query cannot be empty' });
    expect(mockedAxios.get).not.toHaveBeenCalled();
  });

  it('hits the public endpoint and omits the API key header when keyless', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      YOU_PUBLIC_URL,
      expect.objectContaining({
        params: expect.objectContaining({ query: 'typescript' }),
      })
    );
    const headers = mockedAxios.get.mock.calls[0][1]?.headers as Record<
      string,
      string
    >;
    expect(headers['X-API-Key']).toBeUndefined();
    expect(result.success).toBe(true);
  });

  it('hits the authenticated endpoint and sends the API key when a key is set', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'you',
      youApiKey: 'secret-key',
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      YOU_KEYED_URL,
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'secret-key' }),
      })
    );
  });

  it('reads the key from YDC_API_KEY when no explicit key is passed', async () => {
    process.env.YDC_API_KEY = 'env-key';
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      YOU_KEYED_URL,
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'env-key' }),
      })
    );
  });

  it.each([
    ['keyless', undefined],
    ['keyed', 'secret-key'],
  ])(
    'identifies the host application on the %s endpoint',
    async (_label, key) => {
      mockedAxios.get.mockResolvedValueOnce(sampleResponse);

      const searchAPI = createSearchAPI({
        searchProvider: 'you',
        youApiKey: key,
      });
      await searchAPI.getSources({ query: 'typescript' });

      const headers = mockedAxios.get.mock.calls[0][1]?.headers as Record<
        string,
        string
      >;
      expect(headers['User-Agent']).toBe(
        'LibreChat youdotcom-integration/danny-avila-agents'
      );
    }
  );

  it('honours a custom attribution title', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'you',
      youSearchOptions: { attributionTitle: 'MyApp' },
    });
    await searchAPI.getSources({ query: 'typescript' });

    const headers = mockedAxios.get.mock.calls[0][1]?.headers as Record<
      string,
      string
    >;
    expect(headers['User-Agent']).toBe(
      'MyApp youdotcom-integration/danny-avila-agents'
    );
  });

  it('joins extracted passages and falls back to the description', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.data?.organic).toEqual([
      {
        position: 1,
        title: 'TypeScript Best Practices 2026',
        link: 'https://example.com/ts',
        snippet: 'First passage.\nSecond passage.',
        date: '2026-01-15T10:30:00Z',
      },
      {
        position: 2,
        title: 'Second result',
        link: 'https://example.com/second',
        snippet: 'Description fallback when snippets are absent.',
        date: undefined,
      },
    ]);
  });

  it('maps the news section into top stories', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.data?.topStories).toEqual([
      {
        title: 'TypeScript 8 released',
        link: 'https://news.example.com/ts8',
        date: '2026-01-16T09:00:00Z',
        imageUrl: 'https://news.example.com/ts8.jpg',
      },
    ]);
  });

  it('returns empty sections when the response carries no results', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: {} });

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result).toEqual({
      success: true,
      data: { organic: [], topStories: [] },
    });
  });

  it('caps the requested count and applies it per section', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: {
          web: [{ url: '1' }, { url: '2' }, { url: '3' }],
          news: [{ url: 'n1' }, { url: 'n2' }],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'you',
      youSearchOptions: { maxResults: 2 },
    });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        params: expect.objectContaining({ count: 2 }),
      })
    );
    expect(result.data?.organic).toHaveLength(2);
    expect(result.data?.topStories).toHaveLength(2);
  });

  it('clamps the count to the API maximum', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'you',
      youSearchOptions: { maxResults: 500 },
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        params: expect.objectContaining({ count: 100 }),
      })
    );
  });

  it.each([
    [DATE_RANGE.PAST_HOUR, 'day'],
    [DATE_RANGE.PAST_24_HOURS, 'day'],
    [DATE_RANGE.PAST_WEEK, 'week'],
    [DATE_RANGE.PAST_MONTH, 'month'],
    [DATE_RANGE.PAST_YEAR, 'year'],
  ])('maps date range %s to freshness %s', async (date, freshness) => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    await searchAPI.getSources({ query: 'typescript', date });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        params: expect.objectContaining({ freshness }),
      })
    );
  });

  it('upper-cases the country code and maps safe search', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    await searchAPI.getSources({
      query: 'typescript',
      country: 'de',
      safeSearch: 2,
    });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        params: expect.objectContaining({
          country: 'DE',
          safesearch: 'strict',
        }),
      })
    );
  });

  it('surfaces request failures as a structured error', async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({ searchProvider: 'you' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('You.com API request failed: Network error');
  });

  it('respects an explicit API URL override', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'you',
      youApiUrl: 'https://proxy.internal/v1/search',
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'https://proxy.internal/v1/search',
      expect.any(Object)
    );
  });
});
