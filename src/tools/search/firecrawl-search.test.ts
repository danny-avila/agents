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

describe('Firecrawl search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('throws when Firecrawl API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'firecrawl',
        firecrawlApiKey: '',
      })
    ).toThrow('FIRECRAWL_API_KEY is required for Firecrawl API');
  });

  it('returns an error for empty Firecrawl search queries', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('passes Firecrawl v2 options and maps web results', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          web: [
            {
              title: 'Example',
              url: 'https://example.com',
              description: 'Example summary',
              markdown: '# Example',
            },
            {
              metadata: {
                title: 'Meta title',
                url: 'https://example.com/from-metadata-url',
              },
              description: 'Meta summary',
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
      firecrawlApiUrl: 'https://proxy.example.com/',
      firecrawlVersion: 'v2',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      country: 'de',
      date: DATE_RANGE.PAST_WEEK,
      numResults: 3,
    });
    const [requestUrl, payload, requestConfig] = mockedAxios.post.mock.calls[0];

    expect(requestUrl).toBe('https://proxy.example.com/v2/search');
    expect(payload).toMatchObject({
      query: 'example query',
      limit: 3,
      sources: ['web'],
      country: 'DE',
      tbs: 'qdr:w',
    });
    expect(payload).not.toHaveProperty('scrapeOptions');
    expect(requestConfig).toMatchObject({
      headers: {
        Authorization: 'Bearer test-key',
      },
    });
    expect(result).toMatchObject({
      success: true,
      data: {
        organic: [
          {
            title: 'Example',
            link: 'https://example.com',
            snippet: 'Example summary',
            content: '# Example',
          },
          {
            title: 'Meta title',
            link: 'https://example.com/from-metadata-url',
            snippet: 'Meta summary',
          },
        ],
      },
    });
  });

  it('uses configured Firecrawl v1 endpoint and markdown scrape options', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: [
          {
            title: 'V1 result',
            url: 'https://example.com/v1',
            description: 'V1 summary',
            markdown: '# V1',
          },
          {
            metadata: {
              title: 'V1 metadata title',
              sourceURL: 'https://example.com/v1-metadata',
              description: 'V1 metadata summary',
            },
          },
        ],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
      firecrawlApiUrl: 'https://proxy.example.com',
      firecrawlVersion: 'v1',
      firecrawlOptions: {
        formats: ['markdown', 'rawHtml'],
        onlyMainContent: true,
        headers: {
          'X-Test': 'yes',
        },
      },
    });

    const result = await searchAPI.getSources({
      query: 'example query',
    });
    const [requestUrl, payload] = mockedAxios.post.mock.calls[0];

    expect(requestUrl).toBe('https://proxy.example.com/v1/search');
    expect(payload).toMatchObject({
      query: 'example query',
      scrapeOptions: {
        formats: [{ type: 'markdown' }, { type: 'rawHtml' }],
        onlyMainContent: true,
        headers: {
          'X-Test': 'yes',
        },
      },
    });
    expect(result).toMatchObject({
      success: true,
      data: {
        organic: [
          {
            position: 1,
            title: 'V1 result',
            link: 'https://example.com/v1',
            snippet: 'V1 summary',
            content: '# V1',
          },
          {
            position: 2,
            title: 'V1 metadata title',
            link: 'https://example.com/v1-metadata',
            snippet: 'V1 metadata summary',
          },
        ],
      },
    });
  });

  it('maps Firecrawl image results without applying date filters', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          images: [
            {
              title: 'Example image',
              imageUrl: 'https://example.com/image.png',
              imageWidth: 640,
              imageHeight: 480,
              url: 'https://example.com/page',
              position: 7,
            },
            {
              title: 'Missing URL image',
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      type: 'images',
      date: DATE_RANGE.PAST_MONTH,
      country: 'us',
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      sources: ['images'],
      country: 'US',
    });
    expect(payload).not.toHaveProperty('tbs');
    expect(result.data?.images).toEqual([
      {
        title: 'Example image',
        imageUrl: 'https://example.com/image.png',
        imageWidth: 640,
        imageHeight: 480,
        link: 'https://example.com/page',
        source: 'example.com',
        domain: 'example.com',
        position: 7,
      },
    ]);
  });

  it('maps Firecrawl news results using metadata sourceURL fallback', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        data: {
          news: [
            {
              title: 'Example news',
              url: 'https://news.example.com/story',
              snippet: 'News summary',
              date: '2026-05-20',
              imageUrl: 'https://news.example.com/image.png',
              position: 2,
              markdown: '# News',
            },
            {
              metadata: {
                title: 'Metadata news',
                sourceURL: 'https://news.example.com/meta',
                description: 'Metadata summary',
              },
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({
      query: 'example query',
      type: 'news',
      date: DATE_RANGE.PAST_WEEK,
    });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({
      query: 'example query',
      sources: ['news'],
    });
    expect(payload).not.toHaveProperty('tbs');
    expect(result.data?.news?.[0]).toMatchObject({
      title: 'Example news',
      link: 'https://news.example.com/story',
      snippet: 'News summary',
      date: '2026-05-20',
      source: 'news.example.com',
      imageUrl: 'https://news.example.com/image.png',
      position: 2,
      content: '# News',
    });
    expect(result.data?.news?.[1]).toMatchObject({
      title: 'Metadata news',
      link: 'https://news.example.com/meta',
      snippet: 'Metadata summary',
    });
  });

  it('returns an error when the Firecrawl request fails', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result).toEqual({
      success: false,
      error: 'Firecrawl API request failed: Network error',
    });
  });

  it('passes Firecrawl search config through the search tool', async () => {
    mockedAxios.post
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            web: [
              {
                title: 'Example',
                url: 'https://example.com',
                description: 'Example summary',
              },
            ],
          },
        },
      })
      .mockResolvedValueOnce({
        data: {
          success: true,
          data: {
            markdown: 'Scraped content',
            metadata: {
              title: 'Example',
            },
          },
        },
      });

    const searchTool = createSearchTool({
      searchProvider: 'firecrawl',
      firecrawlApiKey: 'search-key',
      firecrawlApiUrl: 'https://proxy.example.com',
      firecrawlVersion: 'v2',
      firecrawlOptions: {
        formats: ['markdown', 'rawHtml'],
        onlyMainContent: true,
      },
      rerankerType: 'none',
      logger: mockLogger,
    });

    await searchTool.invoke({
      query: 'example query',
      country: 'us',
      videos: true,
    });
    const [searchUrl, searchPayload, searchConfig] =
      mockedAxios.post.mock.calls[0];
    const [scrapeUrl, , scrapeConfig] = mockedAxios.post.mock.calls[1];

    expect(mockedAxios.post).toHaveBeenCalledTimes(2);
    expect(searchUrl).toBe('https://proxy.example.com/v2/search');
    expect(searchPayload).toMatchObject({
      sources: ['web'],
      scrapeOptions: {
        formats: [{ type: 'markdown' }, { type: 'rawHtml' }],
        onlyMainContent: true,
      },
    });
    expect(searchConfig).toMatchObject({
      headers: {
        Authorization: 'Bearer search-key',
      },
    });
    expect(scrapeUrl).toBe('https://proxy.example.com/v2/scrape');
    expect(scrapeConfig).toMatchObject({
      headers: {
        Authorization: 'Bearer search-key',
      },
    });
  });
});
