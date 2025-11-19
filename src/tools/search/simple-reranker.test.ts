// simple-rerander.test.ts
import axios from 'axios';
import jwt from 'jsonwebtoken';
import { nanoid } from 'nanoid';
import type { Highlight } from './types';
import type { AxiosResponse, InternalAxiosRequestConfig } from 'axios';

import { SimpleReranker } from './rerankers';
import { createDefaultLogger } from './utils';

jest.mock('axios');
jest.mock('jsonwebtoken');
jest.mock('nanoid');

const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedNanoid = nanoid as unknown as jest.MockedFunction<typeof nanoid>;

type SimpleRerankerWithInstanceUrl = {
  instanceUrl?: string;
};

// Strongly typed sync sign function used in our code under test
type JwtSignFn = (
  payload: { nonce: string },
  secret: string,
  options: { expiresIn: string }
) => string;
const mockedJwtSign = jwt.sign as unknown as jest.MockedFunction<JwtSignFn>;

describe('SimpleReranker', () => {
  const mockLogger = createDefaultLogger();
  const originalEnv = process.env;

  const documents = ['doc1', 'doc2', 'doc3'];
  const mockHighlights: Highlight[] = [
    { score: 0.9, text: 'doc1' },
    { score: 0.8, text: 'doc2' },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.JWT_SECRET;
    delete process.env.RAG_API_URL;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('constructor', () => {
    it('sets instanceUrl from RAG_API_URL when it is defined and non-empty', () => {
      process.env.RAG_API_URL = 'https://rag-api.local';

      const reranker = new SimpleReranker({ logger: mockLogger });
      const instanceUrl = (reranker as unknown as SimpleRerankerWithInstanceUrl)
        .instanceUrl;

      expect(instanceUrl).toBe('https://rag-api.local/rerank');
    });

    it('leaves instanceUrl undefined when RAG_API_URL is not set', () => {
      delete process.env.RAG_API_URL;

      const reranker = new SimpleReranker({ logger: mockLogger });
      const instanceUrl = (reranker as unknown as SimpleRerankerWithInstanceUrl)
        .instanceUrl;

      expect(instanceUrl).toBeUndefined();
    });

    it('leaves instanceUrl undefined when RAG_API_URL is an empty string', () => {
      process.env.RAG_API_URL = '';

      const reranker = new SimpleReranker({ logger: mockLogger });
      const instanceUrl = (reranker as unknown as SimpleRerankerWithInstanceUrl)
        .instanceUrl;

      expect(instanceUrl).toBeUndefined();
    });
  });

  describe('rerank', () => {
    type SimpleRerankerPrivate = {
      getDefaultRanking: (docs: string[], topK: number) => Highlight[];
    };

    it('should log the number of chunks being reranked', async () => {
      process.env.JWT_SECRET = 'test-secret';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const logSpy = jest.spyOn(mockLogger, 'debug');

      const axiosResponse: AxiosResponse<Highlight[]> = {
        data: mockHighlights,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };

      mockedNanoid.mockReturnValue('nonce-123');
      mockedJwtSign.mockReturnValue('signed-token');
      mockedAxios.post.mockResolvedValue(axiosResponse);

      await reranker.rerank('test query', documents, 2);

      expect(logSpy).toHaveBeenCalledWith(
        `Reranking ${documents.length} chunks with SimpleReranker`
      );

      logSpy.mockRestore();
    });

    it('falls back to default ranking when RAG_API_URL is not set', async () => {
      process.env.JWT_SECRET = 'test-secret';
      delete process.env.RAG_API_URL; // ensure it is really unset

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const warnSpy = jest.spyOn(mockLogger, 'warn');

      const fallback: Highlight[] = [{ score: 0.1, text: 'fallback' }];
      const rerankerWithPrivate = reranker as unknown as SimpleRerankerPrivate;
      const getDefaultRankingSpy = jest
        .spyOn(rerankerWithPrivate, 'getDefaultRanking')
        .mockReturnValue(fallback);

      const result = await reranker.rerank('test query', documents, 3);

      expect(warnSpy).toHaveBeenCalledWith(
        'RAG_API_URL is not set. Using default ranking.'
      );
      expect(getDefaultRankingSpy).toHaveBeenCalledWith(documents, 3);
      expect(result).toEqual(fallback);

      // No network or JWT usage in this case
      expect(mockedAxios.post).not.toHaveBeenCalled();
      expect(mockedJwtSign).not.toHaveBeenCalled();

      warnSpy.mockRestore();
    });

    it('falls back to default ranking when JWT_SECRET is not set', async () => {
      process.env.RAG_API_URL = 'https://rag-api.local';
      delete process.env.JWT_SECRET; // explicit

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const warnSpy = jest.spyOn(mockLogger, 'warn');

      const fallback: Highlight[] = [{ score: 0.1, text: 'fallback' }];
      const rerankerWithPrivate = reranker as unknown as SimpleRerankerPrivate;
      const getDefaultRankingSpy = jest
        .spyOn(rerankerWithPrivate, 'getDefaultRanking')
        .mockReturnValue(fallback);

      const result = await reranker.rerank('test query', documents, 2);

      expect(warnSpy).toHaveBeenCalledWith(
        'JWT_SECRET is not set. Using default ranking.'
      );
      expect(getDefaultRankingSpy).toHaveBeenCalledWith(documents, 2);
      expect(result).toEqual(fallback);

      // Should not attempt JWT signing or network calls
      expect(mockedJwtSign).not.toHaveBeenCalled();
      expect(mockedAxios.post).not.toHaveBeenCalled();

      warnSpy.mockRestore();
    });

    it('falls back to default ranking when response format is invalid', async () => {
      process.env.JWT_SECRET = 'test-secret';
      process.env.RAG_API_URL = 'https://rag-api.local';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      // Invalid response: score should be number but is a string
      const invalidResponse: AxiosResponse<Highlight[]> = {
        // cast only the data, not the entire response
        data: [
          { text: 'doc1', score: 'not-a-number' },
        ] as unknown as Highlight[],
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };

      mockedNanoid.mockReturnValue('bad-nonce');
      mockedJwtSign.mockReturnValue('signed-token');
      mockedAxios.post.mockResolvedValue(invalidResponse);

      const fallback: Highlight[] = [{ score: 0.1, text: 'fallback' }];
      const rerankerWithPrivate = reranker as unknown as SimpleRerankerPrivate;
      const getDefaultRankingSpy = jest
        .spyOn(rerankerWithPrivate, 'getDefaultRanking')
        .mockReturnValue(fallback);

      const warnSpy = jest.spyOn(mockLogger, 'warn');

      const result = await reranker.rerank('test query', documents, 2);

      expect(warnSpy).toHaveBeenCalledWith(
        'Unexpected response format from Simple reranker. Using default ranking.'
      );
      expect(getDefaultRankingSpy).toHaveBeenCalledWith(documents, 2);
      expect(result).toEqual(fallback);

      warnSpy.mockRestore();
    });

    it('falls back to default ranking when response is an empty array', async () => {
      process.env.JWT_SECRET = 'test-secret';
      process.env.RAG_API_URL = 'https://rag-api.local';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const emptyResponse: AxiosResponse<Highlight[]> = {
        data: [], // empty
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };

      mockedNanoid.mockReturnValue('nonce-empty');
      mockedJwtSign.mockReturnValue('signed-token');
      mockedAxios.post.mockResolvedValue(emptyResponse);

      const fallback: Highlight[] = [{ score: 0.1, text: 'fallback' }];
      const rerankerWithPrivate = reranker as unknown as SimpleRerankerPrivate;
      const getDefaultRankingSpy = jest
        .spyOn(rerankerWithPrivate, 'getDefaultRanking')
        .mockReturnValue(fallback);

      const warnSpy = jest.spyOn(mockLogger, 'warn');

      const result = await reranker.rerank('test query', documents, 2);

      // no special warning in this branch, but we can assert fallback usage
      expect(getDefaultRankingSpy).toHaveBeenCalledWith(documents, 2);
      expect(result).toEqual(fallback);

      // optional: ensure we did NOT log the "Unexpected response format" warning
      expect(
        warnSpy.mock.calls.some(([msg]) =>
          String(msg).includes(
            'Unexpected response format from Simple reranker'
          )
        )
      ).toBe(false);

      warnSpy.mockRestore();
    });

    it('should successfully rerank with a valid API response', async () => {
      process.env.JWT_SECRET = 'test-secret';
      process.env.RAG_API_URL = 'https://rag-api.local';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const axiosResponse: AxiosResponse<Highlight[]> = {
        data: mockHighlights,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };

      mockedNanoid.mockReturnValue('test-nonce');
      mockedJwtSign.mockReturnValue('signed-jwt-token');
      mockedAxios.post.mockResolvedValue(axiosResponse);

      const result = await reranker.rerank('test query', documents, 2);

      // JWT token generated
      expect(mockedJwtSign).toHaveBeenCalledWith(
        { nonce: 'test-nonce' },
        'test-secret',
        { expiresIn: '10m' }
      );

      const expectedUrl = `${process.env.RAG_API_URL}/rerank`; // match your implementation

      // Request sent to correct endpoint with proper payload and headers
      expect(mockedAxios.post).toHaveBeenCalledWith(
        expectedUrl,
        {
          query: 'test query',
          docs: documents,
          k: 2,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: 'Bearer signed-jwt-token',
          },
        }
      );

      // Response returned as-is
      expect(result).toEqual(mockHighlights);
    });

    it('should handle errors from axios.post and fall back to default ranking', async () => {
      process.env.JWT_SECRET = 'test-secret';
      process.env.RAG_API_URL = 'https://simple-reranker.local';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const errorSpy = jest.spyOn(mockLogger, 'error');

      mockedNanoid.mockReturnValue('test-nonce');
      mockedJwtSign.mockReturnValue('signed-jwt-token');
      mockedAxios.post.mockRejectedValue(new Error('network error'));

      const fallback: Highlight[] = [{ score: 0.1, text: 'fallback' }];

      const rerankerWithPrivate = reranker as unknown as SimpleRerankerPrivate;
      const getDefaultRankingSpy = jest
        .spyOn(rerankerWithPrivate, 'getDefaultRanking')
        .mockReturnValue(fallback);

      const result = await reranker.rerank('test query', documents, 5);

      // axios.post should have been attempted once and failed:
      expect(mockedAxios.post).toHaveBeenCalled();

      // Error logged and fallback used
      expect(errorSpy).toHaveBeenCalled();
      expect(getDefaultRankingSpy).toHaveBeenCalledWith(documents, 5);
      expect(result).toEqual(fallback);
    });

    it('should generate JWT with nonce and set Authorization header correctly', async () => {
      process.env.JWT_SECRET = 'jwt-secret';
      process.env.RAG_API_URL = 'https://simple-reranker.local';

      const reranker = new SimpleReranker({
        logger: mockLogger,
      });

      const axiosResponse: AxiosResponse<Highlight[]> = {
        data: mockHighlights,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };

      mockedNanoid.mockReturnValue('nonce-xyz');
      mockedJwtSign.mockReturnValue('jwt-token'); // typed mock
      mockedAxios.post.mockResolvedValue(axiosResponse);

      await reranker.rerank('query', documents, 1);

      // Nonce generation
      expect(mockedNanoid).toHaveBeenCalled();

      // JWT sign call
      expect(mockedJwtSign).toHaveBeenCalledWith(
        { nonce: 'nonce-xyz' },
        'jwt-secret',
        { expiresIn: '10m' }
      );

      const axiosCall = mockedAxios.post.mock.calls[0];
      const url = axiosCall[0];
      const data = axiosCall[1];
      const config = axiosCall[2];

      expect(url).toBe(`${process.env.RAG_API_URL}/rerank`);
      expect(data).toEqual({
        query: 'query',
        docs: documents,
        k: 1,
      });

      expect(config).toBeDefined();
      const headers = (config as { headers?: Record<string, string> }).headers!;
      expect(headers.Authorization).toBe('Bearer jwt-token');
      expect(headers['Content-Type']).toBe('application/json');
    });
  });
});
