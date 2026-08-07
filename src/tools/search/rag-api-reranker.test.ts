import axios from 'axios';
import type * as t from './types';
import { createReranker, RagApiReranker } from './rerankers';
import { createDefaultLogger } from './utils';

describe('RagApiReranker', () => {
  const mockLogger = createDefaultLogger();
  const baseUrl = 'https://rag.example.com';

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('constructor', () => {
    it('should use RAG_API_URL when no baseUrl is provided', async () => {
      const originalEnv = process.env.RAG_API_URL;
      process.env.RAG_API_URL = 'https://env-rag-endpoint.com';

      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          profile: 'fast-v1',
          model: 'embed-blend-v0',
          results: [{ id: '0', index: 0, score: 0.9 }],
        },
      });

      await reranker.rerank('query', ['doc1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        'https://env-rag-endpoint.com/v1/rerank',
        expect.any(Object),
        expect.any(Object)
      );

      if (typeof originalEnv === 'string') {
        process.env.RAG_API_URL = originalEnv;
      } else {
        delete process.env.RAG_API_URL;
      }
    });

    it('should prioritize an explicit baseUrl over RAG_API_URL', async () => {
      const originalEnv = process.env.RAG_API_URL;
      process.env.RAG_API_URL = 'https://env-rag-endpoint.com';

      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ id: '0', index: 0, score: 0.9 }] },
      });

      await reranker.rerank('query', ['doc1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        `${baseUrl}/v1/rerank`,
        expect.any(Object),
        expect.any(Object)
      );

      if (typeof originalEnv === 'string') {
        process.env.RAG_API_URL = originalEnv;
      } else {
        delete process.env.RAG_API_URL;
      }
    });
  });

  describe('rerank method', () => {
    it('should return [] without a network call when there are no documents', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      const postSpy = jest.spyOn(axios, 'post');

      const result = await reranker.rerank('query', [], 5);

      expect(result).toEqual([]);
      expect(postSpy).not.toHaveBeenCalled();
      expect(tokenSupplier).not.toHaveBeenCalled();
    });

    it('should fall back to default ranking without a network call when baseUrl is missing', async () => {
      const originalEnv = process.env.RAG_API_URL;
      delete process.env.RAG_API_URL;

      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({ tokenSupplier, logger: mockLogger });
      const warnSpy = jest
        .spyOn(mockLogger, 'warn')
        .mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post');

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
      ]);
      expect(postSpy).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(
        'RAG_API_URL is not set. Using default ranking.'
      );

      if (typeof originalEnv === 'string') {
        process.env.RAG_API_URL = originalEnv;
      }
    });

    it('should fall back to default ranking without a network call when no token supplier is configured', async () => {
      const reranker = new RagApiReranker({ baseUrl, logger: mockLogger });
      const warnSpy = jest
        .spyOn(mockLogger, 'warn')
        .mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post');

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
      ]);
      expect(postSpy).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(
        'No rag_api token supplier configured. Using default ranking.'
      );
    });

    it('should call the token supplier per request and send it as a bearer token', async () => {
      const tokenSupplier = jest
        .fn<Promise<string>, []>()
        .mockResolvedValueOnce('jwt-1')
        .mockResolvedValueOnce('jwt-2');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValue({
        data: { results: [{ id: '0', index: 0, score: 0.9 }] },
      });

      await reranker.rerank('query', ['document1'], 1);
      await reranker.rerank('query', ['document1'], 1);

      expect(tokenSupplier).toHaveBeenCalledTimes(2);
      expect(postSpy).toHaveBeenNthCalledWith(
        1,
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer jwt-1' }),
        })
      );
      expect(postSpy).toHaveBeenNthCalledWith(
        2,
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({
          headers: expect.objectContaining({ Authorization: 'Bearer jwt-2' }),
        })
      );
    });

    it('should send the fast-v1 candidates/top_n contract shape', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          results: [
            { id: '0', index: 0, score: 0.5 },
            { id: '1', index: 1, score: 0.9 },
          ],
        },
      });

      await reranker.rerank('search query', ['document1', 'document2'], 2);

      const [, requestBody] = postSpy.mock.calls[0];
      expect(requestBody).toEqual({
        profile: 'fast-v1',
        query: 'search query',
        candidates: [
          { id: '0', text: 'document1', base_score: 0 },
          { id: '1', text: 'document2', base_score: 0 },
        ],
        top_n: 2,
      });
    });

    it('should use a custom profile when provided', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        profile: 'custom-profile',
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ id: '0', index: 0, score: 0.9 }] },
      });

      await reranker.rerank('query', ['document1'], 1);

      const [, requestBody] = postSpy.mock.calls[0];
      expect((requestBody as t.RagApiRerankRequestBody).profile).toBe(
        'custom-profile'
      );
    });

    it('should bound the rerank request with the default timeout', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ id: '0', index: 0, score: 0.9 }] },
      });

      await reranker.rerank('query', ['document1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({ timeout: 10000 })
      );
    });

    it('should bound the rerank request with a custom timeout', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        timeout: 3000,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ id: '0', index: 0, score: 0.9 }] },
      });

      await reranker.rerank('query', ['document1'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        expect.objectContaining({ timeout: 3000 })
      );
    });

    it('should map results back onto the original document text by index', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          results: [
            { id: '2', index: 2, score: 0.95 },
            { id: '0', index: 0, score: 0.4 },
          ],
        },
      });

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2', 'document3'],
        2
      );

      expect(result).toEqual([
        { text: 'document3', score: 0.95 },
        { text: 'document1', score: 0.4 },
      ]);
    });

    it('should order results deterministically by index on tied scores, regardless of response order', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          // Deliberately out of index order, all tied at the same score.
          results: [
            { id: '2', index: 2, score: 0.5 },
            { id: '0', index: 0, score: 0.5 },
            { id: '1', index: 1, score: 0.5 },
          ],
        },
      });

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2', 'document3'],
        3
      );

      expect(result).toEqual([
        { text: 'document1', score: 0.5 },
        { text: 'document2', score: 0.5 },
        { text: 'document3', score: 0.5 },
      ]);
    });

    it('should truncate candidates beyond the 50-candidate contract limit with a debug log', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      const debugSpy = jest
        .spyOn(mockLogger, 'debug')
        .mockImplementation(() => mockLogger);
      const documents = Array.from({ length: 60 }, (_, i) => `document${i}`);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          results: Array.from({ length: 25 }, (_, i) => ({
            id: String(i),
            index: i,
            score: 1 - i / 100,
          })),
        },
      });

      await reranker.rerank('query', documents, 25);

      const [, requestBody] = postSpy.mock.calls[0];
      expect(
        (requestBody as t.RagApiRerankRequestBody).candidates
      ).toHaveLength(50);
      expect(debugSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          'accepts at most 50 candidates; truncating 60 to 50'
        )
      );
    });

    it('should clamp top_n beyond the 25-result contract limit with a debug log', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      const debugSpy = jest
        .spyOn(mockLogger, 'debug')
        .mockImplementation(() => mockLogger);
      const documents = Array.from({ length: 30 }, (_, i) => `document${i}`);
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          results: Array.from({ length: 25 }, (_, i) => ({
            id: String(i),
            index: i,
            score: 1 - i / 100,
          })),
        },
      });

      const result = await reranker.rerank('query', documents, 40);

      const [, requestBody] = postSpy.mock.calls[0];
      expect((requestBody as t.RagApiRerankRequestBody).top_n).toBe(25);
      expect(result).toHaveLength(25);
      expect(debugSpy).toHaveBeenCalledWith(
        expect.stringContaining('accepts top_n <= 25; clamping 40 to 25')
      );
    });

    it('should fall back to the candidates original order on a request timeout', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const errorSpy = jest
        .spyOn(mockLogger, 'error')
        .mockImplementation(() => mockLogger);
      const timeoutError = Object.assign(new Error('timeout of 10000ms exceeded'), {
        isAxiosError: true,
        code: 'ECONNABORTED',
      });
      jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);
      jest.spyOn(axios, 'post').mockRejectedValueOnce(timeoutError);

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2', 'document3'],
        3
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
        { text: 'document3', score: 0 },
      ]);
      expect(errorSpy).toHaveBeenCalledWith(
        'Error using rag_api reranker',
        expect.objectContaining({ code: 'ECONNABORTED' })
      );
    });

    it('should fall back to the candidates original order on a non-2xx response', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const errorSpy = jest
        .spyOn(mockLogger, 'error')
        .mockImplementation(() => mockLogger);
      const httpError = Object.assign(new Error('Request failed with status code 500'), {
        isAxiosError: true,
        code: 'ERR_BAD_RESPONSE',
        response: { status: 500, data: { message: 'upstream failed' } },
      });
      jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);
      jest.spyOn(axios, 'post').mockRejectedValueOnce(httpError);

      const result = await reranker.rerank('query', ['document1'], 1);

      expect(result).toEqual([{ text: 'document1', score: 0 }]);
      expect(errorSpy).toHaveBeenCalledWith(
        'Error using rag_api reranker',
        expect.objectContaining({ status: 500 })
      );

      const metadata = errorSpy.mock.calls.flat()[1];
      expect(JSON.stringify(metadata)).not.toContain('upstream failed');
    });

    it('should fall back to the candidates original order when the response has no results array', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const warnSpy = jest
        .spyOn(mockLogger, 'warn')
        .mockImplementation(() => mockLogger);
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { profile: 'fast-v1', model: 'embed-blend-v0' },
      });

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
      ]);
      expect(warnSpy).toHaveBeenCalledWith(
        'Unexpected response format from rag_api rerank. Using default ranking.'
      );
    });

    it('should fall back to the candidates original order when results carry out-of-range indices', async () => {
      const tokenSupplier = jest.fn().mockResolvedValue('token');
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      const warnSpy = jest
        .spyOn(mockLogger, 'warn')
        .mockImplementation(() => mockLogger);
      jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: {
          results: [
            { id: '5', index: 5, score: 0.9 },
            { id: '-1', index: -1, score: 0.8 },
          ],
        },
      });

      const result = await reranker.rerank(
        'query',
        ['document1', 'document2'],
        2
      );

      expect(result).toEqual([
        { text: 'document1', score: 0 },
        { text: 'document2', score: 0 },
      ]);
      expect(warnSpy).toHaveBeenCalledWith(
        'rag_api rerank response contained no valid results. Using default ranking.'
      );
    });

    it('should fall back to the candidates original order when the token supplier rejects', async () => {
      const tokenSupplier = jest
        .fn()
        .mockRejectedValue(new Error('token mint failed'));
      const reranker = new RagApiReranker({
        baseUrl,
        tokenSupplier,
        logger: mockLogger,
      });
      jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
      jest.spyOn(mockLogger, 'error').mockImplementation(() => mockLogger);
      const postSpy = jest.spyOn(axios, 'post');

      const result = await reranker.rerank('query', ['document1'], 1);

      expect(result).toEqual([{ text: 'document1', score: 0 }]);
      expect(postSpy).not.toHaveBeenCalled();
    });
  });
});

describe('createReranker (rag-api)', () => {
  it('should create a RagApiReranker for rerankerType "rag-api"', () => {
    const tokenSupplier = jest.fn().mockResolvedValue('token');
    const reranker = createReranker({
      rerankerType: 'rag-api',
      ragApiUrl: 'https://rag.example.com',
      ragApiTokenSupplier: tokenSupplier,
    });

    expect(reranker).toBeInstanceOf(RagApiReranker);
  });

  it('should pass ragApiProfile and rerankerTimeout through to the rerank request', async () => {
    const tokenSupplier = jest.fn().mockResolvedValue('token');
    const reranker = createReranker({
      rerankerType: 'rag-api',
      ragApiUrl: 'https://rag.example.com',
      ragApiTokenSupplier: tokenSupplier,
      ragApiProfile: 'custom-profile',
      rerankerTimeout: 5000,
      logger: createDefaultLogger(),
    });
    if (!(reranker instanceof RagApiReranker)) {
      throw new Error('Expected createReranker to return a RagApiReranker.');
    }
    const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { results: [{ id: '0', index: 0, score: 0.9 }] },
    });

    await reranker.rerank('query', ['document1'], 1);

    expect(postSpy).toHaveBeenCalledWith(
      'https://rag.example.com/v1/rerank',
      expect.objectContaining({ profile: 'custom-profile' }),
      expect.objectContaining({ timeout: 5000 })
    );
  });
});
