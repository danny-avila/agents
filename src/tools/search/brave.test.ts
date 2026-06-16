import axios from 'axios';
import type * as t from './types';
import { DATE_RANGE } from './schema';
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

describe('Brave search API', () => {
  beforeEach(jest.clearAllMocks);

  afterEach(() => {
    delete process.env.BRAVE_API_KEY;
    delete process.env.BRAVE_API_URL;
  });

  it('throws when Brave API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'brave',
        braveApiKey: '',
      })
    ).toThrow('BRAVE_API_KEY is required for Brave API');
  });

  it('falls back to the BRAVE_API_KEY env var', () => {
    process.env.BRAVE_API_KEY = 'env-key';
    expect(() =>
      createSearchAPI({
        searchProvider: 'brave',
      })
    ).not.toThrow();
  });

  it('defaults the braveApiUrl to https://api.search.brave.com/res', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });
    await searchAPI.getSources({ query: 'q' });

    const [url] = mockedAxios.get.mock.calls[0];
    expect(url).toBe('https://api.search.brave.com/res/v1/web/search');
  });

  it('uses the braveApiUrl override when provided', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveApiUrl: 'https://internal.example.com/brave/res',
    });

    await searchAPI.getSources({ query: 'q' });
    const [url] = mockedAxios.get.mock.calls[0];
    expect(url).toBe('https://internal.example.com/brave/res/v1/web/search');
  });

  it('uses the BRAVE_API_URL env var when provided', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });
    process.env.BRAVE_API_URL = 'https://internal.example.com/brave/res';

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q' });
    const [url] = mockedAxios.get.mock.calls[0];
    expect(url).toBe('https://internal.example.com/brave/res/v1/web/search');
  });

  it('returns an error for empty Brave search queries', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(mockedAxios.get).not.toHaveBeenCalled();
  });

  it('returns an error when the Brave search request fails', async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });
    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('Brave API request failed: Network error');
  });

  it('maps web search response data to organic, news, and videos', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        web: {
          results: [
            {
              title: 'Example',
              url: 'https://example.com',
              description: 'Example summary',
              page_age: '2026-01-02',
            },
          ],
        },
        news: {
          results: [
            {
              title: 'Breaking story',
              url: 'https://news.example.com/story',
              description: 'A news result',
              age: '1 hour ago',
              source: 'Example News',
              thumbnail: { src: 'https://news.example.com/thumb.png' },
            },
          ],
        },
        videos: {
          results: [
            {
              title: 'A video',
              url: 'https://video.example.com/clip',
              description: 'Watch this',
              video: { duration: '3:14', publisher: 'Example TV' },
              thumbnail: { src: 'https://video.example.com/thumb.png' },
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      country: 'GB',
      date: DATE_RANGE.PAST_24_HOURS,
    });
    const [url, options] = mockedAxios.get.mock.calls[0];

    expect(url).toBe('https://api.search.brave.com/res/v1/web/search');
    expect(options?.headers).toMatchObject({
      'X-Subscription-Token': 'test-key',
    });
    expect(options?.params).toMatchObject({
      q: 'example query',
      country: 'GB',
      freshness: 'pd',
      safesearch: 'moderate',
    });

    expect(result.success).toBe(true);
    expect(result.data?.organic?.[0]).toMatchObject({
      title: 'Example',
      link: 'https://example.com',
      snippet: 'Example summary',
      date: '2026-01-02',
    });
    expect(result.data?.news?.[0]).toMatchObject({
      title: 'Breaking story',
      link: 'https://news.example.com/story',
      source: 'Example News',
      date: '1 hour ago',
    });
    expect(result.data?.videos?.[0]).toMatchObject({
      title: 'A video',
      link: 'https://video.example.com/clip',
      duration: '3:14',
      source: 'Example TV',
    });
  });

  it.each([
    [DATE_RANGE.PAST_HOUR, 'pd'],
    [DATE_RANGE.PAST_24_HOURS, 'pd'],
    [DATE_RANGE.PAST_WEEK, 'pw'],
    [DATE_RANGE.PAST_MONTH, 'pm'],
    [DATE_RANGE.PAST_YEAR, 'py'],
  ])('maps DATE_RANGE %s to Brave freshness %s', async (date, expected) => {
    mockedAxios.get.mockResolvedValueOnce({
      data: { web: { results: [] } },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q', date });
    expect(mockedAxios.get.mock.calls[0][1]?.params).toMatchObject({
      freshness: expected,
    });
  });

  it.each(['United States', 'ZZ'])(
    'omits malformed country code "%s"',
    async (country) => {
      mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

      const searchAPI = createSearchAPI({
        searchProvider: 'brave',
        braveApiKey: 'test-key',
      });

      await searchAPI.getSources({ query: 'q', country });
      const params = mockedAxios.get.mock.calls[0][1]?.params;
      expect(params.country).toBeUndefined();
    }
  );

  it('uppercases supported country codes', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q', country: 'us' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.country).toBe('US');
  });

  it('routes news searches to /v1/news/search', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: [
          {
            title: 'News headline',
            url: 'https://news.example.com/a',
            description: 'A news snippet',
            age: '2 hours ago',
            meta_url: { hostname: 'news.example.com' },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'q',
      type: 'news',
    });
    const [url] = mockedAxios.get.mock.calls[0];

    expect(url).toBe('https://api.search.brave.com/res/v1/news/search');
    expect(result.data?.news?.[0]).toMatchObject({
      title: 'News headline',
      link: 'https://news.example.com/a',
      source: 'news.example.com',
    });
  });

  it('routes images searches to /v1/images/search', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: [
          {
            title: 'Cute cat',
            url: 'https://example.com/cats',
            source: 'example.com',
            thumbnail: {
              src: 'https://example.com/thumb.jpg',
              width: 200,
              height: 100,
            },
            properties: { url: 'https://example.com/full.jpg' },
          },
          {
            title: 'No image',
            url: 'https://example.com/empty',
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'q',
      type: 'images',
    });
    const [url] = mockedAxios.get.mock.calls[0];

    expect(url).toBe('https://api.search.brave.com/res/v1/images/search');
    expect(result.data?.images).toHaveLength(1);
    expect(result.data?.images?.[0]).toMatchObject({
      imageUrl: 'https://example.com/full.jpg',
      thumbnailUrl: 'https://example.com/thumb.jpg',
      thumbnailWidth: 200,
      thumbnailHeight: 100,
      source: 'example.com',
      link: 'https://example.com/cats',
    });
  });

  it('routes videos searches to /v1/videos/search', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: [
          {
            title: 'A clip',
            url: 'https://video.example.com/x',
            description: 'desc',
            video: { duration: '1:23', publisher: 'pub' },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'q',
      type: 'videos',
    });
    const [url] = mockedAxios.get.mock.calls[0];

    expect(url).toBe('https://api.search.brave.com/res/v1/videos/search');
    expect(result.data?.videos?.[0]).toMatchObject({
      title: 'A clip',
      link: 'https://video.example.com/x',
      duration: '1:23',
      source: 'pub',
    });
  });

  it('lets braveSearchOptions.count override the getSources numResults argument', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: {
        count: 15,
      },
    });

    await searchAPI.getSources({ query: 'q', numResults: 3 });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.count).toBe(15);
  });

  it.each([
    { endpoint: 'web', type: undefined, expectedMaximum: 20 },
    { endpoint: 'news', type: 'news', expectedMaximum: 50 },
    { endpoint: 'images', type: 'images', expectedMaximum: 200 },
    { endpoint: 'videos', type: 'videos', expectedMaximum: 50 },
  ] as const)(
    'caps braveSearchOptions.count for the $endpoint endpoint at its maximum: $expectedMaximum',
    async ({ type, expectedMaximum }) => {
      mockedAxios.get.mockResolvedValueOnce({
        data: type == null ? { web: { results: [] } } : { results: [] },
      });

      const searchAPI = createSearchAPI({
        searchProvider: 'brave',
        braveApiKey: 'test-key',
        braveSearchOptions: { count: 999 },
      });

      await searchAPI.getSources({ query: 'q', type });
      const params = mockedAxios.get.mock.calls[0][1]?.params;
      expect(params.count).toBe(expectedMaximum);
    }
  );

  it('lets braveSearchOptions.freshness override the getSources argument', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: {
        freshness: 'py',
      },
    });

    await searchAPI.getSources({ query: 'q', date: DATE_RANGE.PAST_HOUR });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.freshness).toBe('py');
  });

  it('lets braveSearchOptions.safesearch override the getSources argument', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: {
        safesearch: 'strict',
      },
    });

    await searchAPI.getSources({ query: 'q', safeSearch: 1 });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.safesearch).toBe('strict');
  });

  it('falls back to braveSearchOptions.safesearch when no getSources argument is provided', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: {
        safesearch: 'strict',
      },
    });

    await searchAPI.getSources({ query: 'q' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.safesearch).toBe('strict');
  });

  it.each([
    [0, 'off'],
    [1, 'moderate'],
    [2, 'strict'],
  ] as const)(
    'maps SafeSearchLevel %i to Brave safesearch %s',
    async (level, expected) => {
      mockedAxios.get.mockResolvedValueOnce({
        data: { web: { results: [] } },
      });

      const searchAPI = createSearchAPI({
        searchProvider: 'brave',
        braveApiKey: 'test-key',
      });

      await searchAPI.getSources({ query: 'q', safeSearch: level });
      const params = mockedAxios.get.mock.calls[0][1]?.params;
      expect(params.safesearch).toBe(expected);
    }
  );

  it('defaults safesearch to moderate when no value is provided', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.safesearch).toBe('moderate');
  });

  it('defaults safesearch to `strict` on the images endpoint', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { results: [] } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'q', type: 'images', safeSearch: 1 });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.safesearch).toBe('strict');
  });

  it('maps the explicit tool safeSearch argument to the API safesearch value', async () => {
    mockedAxios.get.mockResolvedValue({ data: { web: { results: [] } } });

    const searchTool = createSearchTool({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      scraperProvider: 'tavily',
      tavilyApiKey: 'test-key',
      rerankerType: 'none',
      logger: mockLogger,
      safeSearch: 2,
    });

    await searchTool.invoke({ query: 'q' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.safesearch).toBe('strict');
  });

  it('forwards braveSearchOptions.goggles verbatim to the web endpoint', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { web: { results: [] } } });

    const goggles =
      'https://raw.githubusercontent.com/brave/goggles-quickstart/main/goggles/tech_blogs.goggle';
    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: { goggles },
    });

    await searchAPI.getSources({ query: 'q' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params.goggles).toBe(goggles);
  });

  it.each([
    { name: 'undefined', goggles: undefined },
    { name: 'empty', goggles: '' },
  ])('omits goggles when the option is $name', async ({ goggles }) => {
    mockedAxios.get.mockResolvedValue({ data: { web: { results: [] } } });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
      braveSearchOptions: { goggles },
    });
    await searchAPI.getSources({ query: 'q' });
    const params = mockedAxios.get.mock.calls[0][1]?.params;
    expect(params).not.toHaveProperty('goggles');
  });

  it('falls back to extra_snippets[0] when an organic result has no description', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        web: {
          results: [
            {
              title: 'No description',
              url: 'https://example.com',
              extra_snippets: ['First snippet', 'Second snippet'],
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'q' });
    expect(result.data?.organic?.[0]?.snippet).toBe('First snippet');
  });

  it('prefers age over page_age for organic result date', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        web: {
          results: [
            {
              title: 'Aged result',
              url: 'https://example.com',
              age: '2 weeks ago',
              page_age: '2026-06-01T00:00:00',
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'q' });
    expect(result.data?.organic?.[0]?.date).toBe('2 weeks ago');
  });

  it('uses news profile.name as a source fallback', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: [
          {
            title: 'Headline',
            url: 'https://apnews.com/story',
            description: 'Body',
            profile: { name: 'AP News', url: 'https://apnews.com/story' },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'q', type: 'news' });
    expect(result.data?.news?.[0]?.source).toBe('AP News');
  });

  it('falls back to video.author.name when video.creator is missing', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        results: [
          {
            title: 'A clip',
            url: 'https://video.example.com/x',
            video: {
              duration: '1:23',
              author: { name: 'Author Channel' },
            },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'brave',
      braveApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'q', type: 'videos' });
    expect(result.data?.videos?.[0]?.channel).toBe('Author Channel');
  });
});
