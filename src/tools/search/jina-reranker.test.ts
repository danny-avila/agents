import axios from 'axios';
import { createReranker, JinaReranker, ZeroEntropyReranker } from './rerankers';
import { createDefaultLogger } from './utils';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('JinaReranker', () => {
  const mockLogger = createDefaultLogger();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('constructor', () => {
    it('should use default API URL when no apiUrl is provided', () => {
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      const apiUrl = reranker.getApiUrl();
      expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
    });

    it('should use custom API URL when provided', () => {
      const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = reranker.getApiUrl();
      expect(apiUrl).toBe(customUrl);
    });

    it('should use environment variable JINA_API_URL when available', () => {
      const originalEnv = process.env.JINA_API_URL;
      process.env.JINA_API_URL = 'https://env-jina-endpoint.com/v1/rerank';

      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      const apiUrl = reranker.getApiUrl();
      expect(apiUrl).toBe('https://env-jina-endpoint.com/v1/rerank');

      if (originalEnv !== undefined) {
        process.env.JINA_API_URL = originalEnv;
      } else {
        delete process.env.JINA_API_URL;
      }
    });

    it('should prioritize explicit apiUrl over environment variable', () => {
      const originalEnv = process.env.JINA_API_URL;
      process.env.JINA_API_URL = 'https://env-jina-endpoint.com/v1/rerank';

      const customUrl = 'https://explicit-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = reranker.getApiUrl();
      expect(apiUrl).toBe(customUrl);

      if (originalEnv !== undefined) {
        process.env.JINA_API_URL = originalEnv;
      } else {
        delete process.env.JINA_API_URL;
      }
    });
  });

  describe('rerank method', () => {
    it('should log the API URL being used', async () => {
      const customUrl = 'https://test-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const logSpy = jest.spyOn(mockLogger, 'debug');
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: {
            total_tokens: 1,
          },
          results: [],
        },
      });

      await reranker.rerank('test query', ['document1', 'document2'], 2);

      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          `Reranking 2 chunks with Jina using API URL: ${customUrl}`
        )
      );

      logSpy.mockRestore();
    });
  });
});

describe('ZeroEntropyReranker', () => {
  const mockLogger = createDefaultLogger();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should use default API URL and model when not provided', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [
          {
            index: 1,
            relevance_score: 0.91,
          },
        ],
      },
    });

    const reranker = new ZeroEntropyReranker({
      apiKey: 'test-key',
      logger: mockLogger,
    });

    const highlights = await reranker.rerank(
      'test query',
      ['document1', 'document2'],
      1
    );
    const [requestUrl, requestData, requestConfig] =
      mockedAxios.post.mock.calls[0];

    expect(requestUrl).toBe('https://api.zeroentropy.dev/v1/models/rerank');
    expect(requestData).toMatchObject({
      model: 'zerank-2',
      query: 'test query',
      documents: ['document1', 'document2'],
      top_n: 1,
    });
    expect(requestConfig).toMatchObject({
      headers: {
        Authorization: 'Bearer test-key',
      },
    });
    expect(highlights).toEqual([
      {
        text: 'document2',
        score: 0.91,
      },
    ]);
  });

  it('should use custom API URL and model when provided', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        results: [
          {
            index: 0,
            relevance_score: 0.88,
          },
        ],
      },
    });

    const reranker = new ZeroEntropyReranker({
      apiKey: 'test-key',
      apiUrl: 'https://proxy.example.com/rerank',
      model: 'zerank-1-small',
      logger: mockLogger,
    });

    await reranker.rerank('test query', ['document1', 'document2'], 1);
    const [requestUrl, requestData] = mockedAxios.post.mock.calls[0];

    expect(requestUrl).toBe('https://proxy.example.com/rerank');
    expect(requestData).toMatchObject({
      model: 'zerank-1-small',
    });
  });
});

describe('createReranker', () => {
  it('should create JinaReranker with jinaApiUrl when provided', () => {
    const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
      jinaApiUrl: customUrl,
    });

    expect(reranker).toBeInstanceOf(JinaReranker);
    const apiUrl =
      reranker instanceof JinaReranker ? reranker.getApiUrl() : undefined;
    expect(apiUrl).toBe(customUrl);
  });

  it('should create JinaReranker with default URL when jinaApiUrl is not provided', () => {
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
    });

    expect(reranker).toBeInstanceOf(JinaReranker);
    const apiUrl =
      reranker instanceof JinaReranker ? reranker.getApiUrl() : undefined;
    expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
  });

  it('should create ZeroEntropyReranker with explicit config', () => {
    const reranker = createReranker({
      rerankerType: 'zeroentropy',
      zeroEntropyApiKey: 'test-key',
      zeroEntropyApiUrl: 'https://proxy.example.com/rerank',
      zeroEntropyModel: 'zerank-1',
    });

    expect(reranker).toBeInstanceOf(ZeroEntropyReranker);
  });
});
