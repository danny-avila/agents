import axios from 'axios';
import { createSearchAPI } from './search';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('Volcengine search API', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('throws when Volcengine API key is missing', () => {
    expect(() =>
      createSearchAPI({
        searchProvider: 'volcengine',
        volcengineApiKey: '',
      })
    ).toThrow('VOLCENGINE_API_KEY is required for VolcEngine API');
  });

  it('returns an error for empty search queries', async () => {
    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({
      success: false,
      error: 'Query cannot be empty',
    });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('returns an error when the API request fails', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
    });

    const result = await searchAPI.getSources({ query: 'example query' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('VolcEngine API request failed: Network error');
  });

  it('clamps numResults between 1 and 20', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        ResponseMetadata: {},
        Result: {
          ResultCount: 1,
          WebResults: [
            {
              Title: 'Result',
              Url: 'https://example.com',
              Snippet: 'A snippet',
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchType: 'web',
    });

    await searchAPI.getSources({ query: 'test', numResults: 50 });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({ Count: 20 });
  });

  it('defaults numResults to 8 when not provided', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        ResponseMetadata: {},
        Result: { ResultCount: 0, WebResults: [] },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'test' });
    const [, payload] = mockedAxios.post.mock.calls[0];

    expect(payload).toMatchObject({ Count: 8 });
  });

  it('builds correct payload with default web_summary search type', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        ResponseMetadata: {},
        Result: { ResultCount: 0, WebResults: [] },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
    });

    await searchAPI.getSources({ query: 'hello world', numResults: 5 });
    const [url, payload, config] = mockedAxios.post.mock.calls[0];

    expect(url).toBe('https://open.feedcoopapi.com/search_api/web_search');
    expect(payload).toMatchObject({
      Query: 'hello world',
      SearchType: 'web_summary',
      Count: 5,
      Filter: { NeedContent: true, NeedUrl: true },
      NeedSummary: true,
    });
    expect(config).toMatchObject({
      headers: {
        Authorization: 'Bearer test-key',
        'Content-Type': 'application/json',
      },
      timeout: 15000,
      responseType: 'text',
    });
  });

  it('parses SSE response from web_summary search type', async () => {
    const sseBody = [
      'data:{"ResponseMetadata":{"RequestId":"123"},"Result":{"ResultCount":2,"WebResults":[{"Id":"1","SortId":1,"Title":"SSE Result 1","SiteName":"Site A","Url":"https://sse1.example.com","Snippet":"SSE snippet 1","PublishTime":"2026-05-16T12:00:00+08:00"},{"Id":"2","SortId":2,"Title":"SSE Result 2","SiteName":"Site B","Url":"https://sse2.example.com","Snippet":"SSE snippet 2"}]}}',
      'data:{"Result":{"ResultCount":0}}',
      'data:[DONE]',
    ].join('\n\n');

    mockedAxios.post.mockResolvedValueOnce({ data: sseBody });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchType: 'web_summary',
    });

    const result = await searchAPI.getSources({ query: 'test' });

    expect(result.success).toBe(true);
    expect(result.data?.organic).toHaveLength(2);
    expect(result.data?.organic?.[0]).toMatchObject({
      position: 1,
      title: 'SSE Result 1',
      link: 'https://sse1.example.com',
      snippet: 'SSE snippet 1',
      date: '2026-05-16T12:00:00+08:00',
    });
    expect(result.data?.organic?.[1]).toMatchObject({
      position: 2,
      title: 'SSE Result 2',
      link: 'https://sse2.example.com',
      snippet: 'SSE snippet 2',
    });
  });

  it('parses JSON response from web search type', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        ResponseMetadata: { RequestId: '456' },
        Result: {
          ResultCount: 2,
          WebResults: [
            {
              Id: '1',
              SortId: 1,
              Title: 'JSON Result 1',
              SiteName: 'Site A',
              Url: 'https://json1.example.com',
              Snippet: 'JSON snippet 1',
              PublishTime: '2026-05-16T08:00:00+08:00',
            },
            {
              Id: '2',
              SortId: 2,
              Title: 'JSON Result 2',
              SiteName: 'Site B',
              Url: 'https://json2.example.com',
              Snippet: 'JSON snippet 2',
            },
          ],
        },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchType: 'web',
    });

    const result = await searchAPI.getSources({ query: 'test' });

    expect(result.success).toBe(true);
    expect(result.data?.organic).toHaveLength(2);
    expect(result.data?.organic?.[0]).toMatchObject({
      position: 1,
      title: 'JSON Result 1',
      link: 'https://json1.example.com',
      snippet: 'JSON snippet 1',
      date: '2026-05-16T08:00:00+08:00',
    });
  });

  it('returns error when no results found in SSE response', async () => {
    const sseBody = ['data:{"Result":{"ResultCount":0}}', 'data:[DONE]'].join(
      '\n\n'
    );

    mockedAxios.post.mockResolvedValueOnce({ data: sseBody });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchType: 'web_summary',
    });

    const result = await searchAPI.getSources({ query: 'no results' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('No results found');
  });

  it('uses custom API URL when configured', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        ResponseMetadata: {},
        Result: { ResultCount: 0, WebResults: [] },
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchUrl: 'https://custom.example.com/search',
      volcengineSearchType: 'web',
    });

    await searchAPI.getSources({ query: 'test' });
    const [url] = mockedAxios.post.mock.calls[0];

    expect(url).toBe('https://custom.example.com/search');
  });

  it('prioritizes environment variables for API key and URL', () => {
    const originalKey = process.env.VOLCENGINE_API_KEY;
    const originalUrl = process.env.VOLCENGINE_SEARCH_URL;

    try {
      process.env.VOLCENGINE_API_KEY = 'env-key';
      process.env.VOLCENGINE_SEARCH_URL = 'https://env.example.com/search';

      const searchAPI = createSearchAPI({
        searchProvider: 'volcengine',
      });

      mockedAxios.post.mockResolvedValueOnce({
        data: {
          ResponseMetadata: {},
          Result: { ResultCount: 0, WebResults: [] },
        },
      });

      expect(() => searchAPI).not.toThrow();
    } finally {
      if (originalKey !== undefined) {
        process.env.VOLCENGINE_API_KEY = originalKey;
      } else {
        delete process.env.VOLCENGINE_API_KEY;
      }
      if (originalUrl !== undefined) {
        process.env.VOLCENGINE_SEARCH_URL = originalUrl;
      } else {
        delete process.env.VOLCENGINE_SEARCH_URL;
      }
    }
  });

  it('maps empty fields to empty strings for missing optional data', async () => {
    const sseBody = [
      'data:{"ResponseMetadata":{},"Result":{"ResultCount":1,"WebResults":[{"Url":"https://minimal.example.com"}]}}',
      'data:[DONE]',
    ].join('\n\n');

    mockedAxios.post.mockResolvedValueOnce({ data: sseBody });

    const searchAPI = createSearchAPI({
      searchProvider: 'volcengine',
      volcengineApiKey: 'test-key',
      volcengineSearchType: 'web_summary',
    });

    const result = await searchAPI.getSources({ query: 'test' });

    expect(result.data?.organic?.[0]).toMatchObject({
      title: '',
      link: 'https://minimal.example.com',
      snippet: '',
    });
  });
});
