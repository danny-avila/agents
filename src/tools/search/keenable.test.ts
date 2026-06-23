import axios from 'axios';
import { createSearchAPI } from './search';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const sampleResponse = {
  data: {
    results: [
      {
        title: 'TypeScript Best Practices 2026',
        url: 'https://example.com/ts',
        description: 'A comprehensive guide to TypeScript.',
        published_at: '2026-01-15T10:30:00Z',
      },
      {
        title: 'Second result',
        url: 'https://example.com/second',
        snippet: 'Snippet fallback when description is absent.',
      },
    ],
  },
};

describe('Keenable search API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.KEENABLE_API_KEY;
    delete process.env.KEENABLE_API_URL;
  });

  it('returns an error for empty queries without calling the API', async () => {
    const searchAPI = createSearchAPI({ searchProvider: 'keenable' });
    const result = await searchAPI.getSources({ query: '   ' });

    expect(result).toEqual({ success: false, error: 'Query cannot be empty' });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  it('hits the public endpoint and omits the API key header when keyless', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'keenable' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://api.keenable.ai/v1/search/public',
      { query: 'typescript' },
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-Keenable-Title': 'LibreChat' }),
      })
    );
    const headers = mockedAxios.post.mock.calls[0][2]?.headers as Record<
      string,
      string
    >;
    expect(headers['X-API-Key']).toBeUndefined();
    expect(result.success).toBe(true);
  });

  it('hits the authenticated endpoint and sends the API key when a key is set', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({
      searchProvider: 'keenable',
      keenableApiKey: 'secret-key',
    });
    await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://api.keenable.ai/v1/search',
      { query: 'typescript' },
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'secret-key' }),
      })
    );
  });

  it('maps results into organic sources (description and snippet fallback)', async () => {
    mockedAxios.post.mockResolvedValueOnce(sampleResponse);

    const searchAPI = createSearchAPI({ searchProvider: 'keenable' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.data?.organic).toEqual([
      {
        title: 'TypeScript Best Practices 2026',
        link: 'https://example.com/ts',
        snippet: 'A comprehensive guide to TypeScript.',
        date: '2026-01-15T10:30:00Z',
      },
      {
        title: 'Second result',
        link: 'https://example.com/second',
        snippet: 'Snippet fallback when description is absent.',
        date: undefined,
      },
    ]);
  });

  it('applies the site filter and limits results client-side', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [{ url: '1' }, { url: '2' }, { url: '3' }],
      },
    });

    const searchAPI = createSearchAPI({
      searchProvider: 'keenable',
      keenableSearchOptions: { site: 'github.com', maxResults: 2 },
    });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      { query: 'typescript', site: 'github.com' },
      expect.any(Object)
    );
    expect(result.data?.organic).toHaveLength(2);
  });

  it('surfaces request failures as a structured error', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

    const searchAPI = createSearchAPI({ searchProvider: 'keenable' });
    const result = await searchAPI.getSources({ query: 'typescript' });

    expect(result.success).toBe(false);
    expect(result.error).toBe('Keenable API request failed: Network error');
  });
});
