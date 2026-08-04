import axios from 'axios';
import { Agent as HttpAgent } from 'http';
import { Agent as HttpsAgent } from 'https';
import { createFirecrawlScraper } from './firecrawl';
import { createReranker } from './rerankers';
import { createCrwAPI } from './crw-search';
import { createSearchAPI } from './search';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const httpAgent = new HttpAgent();
const httpsAgent = new HttpsAgent();

describe('injected http(s) agent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('passes both agents to the firecrawl scraper request config', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { markdown: 'x' } },
    });

    const scraper = createFirecrawlScraper({
      apiKey: 'test-key',
      httpAgent,
      httpsAgent,
    });
    await scraper.scrapeUrl('https://example.com');

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });

  it('omits agents from the firecrawl request config when none are provided', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { markdown: 'x' } },
    });

    const scraper = createFirecrawlScraper({ apiKey: 'test-key' });
    await scraper.scrapeUrl('https://example.com');

    const requestConfig = mockedAxios.post.mock.calls[0][2];
    expect(requestConfig?.httpAgent).toBeUndefined();
    expect(requestConfig?.httpsAgent).toBeUndefined();
  });

  it('passes both agents to the Jina reranker request config', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { results: [] } });

    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
      httpAgent,
      httpsAgent,
    });
    await reranker?.rerank('query', ['a', 'b'], 2);

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });

  it('passes agents supplied through the crw search options bag', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { success: true, data: { web: [] } },
    });

    const api = createCrwAPI('test-key', undefined, { httpAgent, httpsAgent });
    await api.getSources({ query: 'query' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });

  it('threads agents from the tool config into the serper search request', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { organic: [] } });

    const api = createSearchAPI({
      searchProvider: 'serper',
      serperApiKey: 'test-key',
      httpAgent,
      httpsAgent,
    });
    await api.getSources({ query: 'query' });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.any(String),
      expect.anything(),
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });

  it('threads agents from the tool config into the searxng search request', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { results: [] } });

    const api = createSearchAPI({
      searchProvider: 'searxng',
      searxngInstanceUrl: 'https://searxng.example.com',
      httpAgent,
      httpsAgent,
    });
    await api.getSources({ query: 'query' });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ httpAgent, httpsAgent })
    );
  });

  it('omits agents from the serper search request when none are provided', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { organic: [] } });

    const api = createSearchAPI({
      searchProvider: 'serper',
      serperApiKey: 'test-key',
    });
    await api.getSources({ query: 'query' });

    const requestConfig = mockedAxios.post.mock.calls[0][2];
    expect(requestConfig?.httpAgent).toBeUndefined();
    expect(requestConfig?.httpsAgent).toBeUndefined();
  });
});
