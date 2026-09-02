import axios from 'axios';

import { createReranker, CohereReranker } from './rerankers';
import { createDefaultLogger } from './utils';

const getApiUrl = (reranker: CohereReranker): string => {
  const descriptor = Object.getOwnPropertyDescriptor(reranker, 'apiUrl');
  if (typeof descriptor?.value === 'string') {
    return descriptor.value;
  }
  throw new Error('Expected CohereReranker apiUrl to be initialized.');
};

describe('CohereReranker', () => {
  const mockLogger = createDefaultLogger();

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const spyOnLogger = (): {
    warn: jest.SpyInstance;
    error: jest.SpyInstance;
  } => {
    jest.spyOn(mockLogger, 'debug').mockImplementation(() => mockLogger);
    return {
      warn: jest.spyOn(mockLogger, 'warn').mockImplementation(() => mockLogger),
      error: jest
        .spyOn(mockLogger, 'error')
        .mockImplementation(() => mockLogger),
    };
  };

  it('should map results back to their source documents by index', async () => {
    const reranker = new CohereReranker({
      apiKey: 'test-key',
      logger: mockLogger,
    });
    const spies = spyOnLogger();
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        id: 'req-1',
        meta: { billed_units: { search_units: 1 } },
        results: [
          { index: 2, relevance_score: 0.91 },
          { index: 0, relevance_score: 0.42 },
        ],
      },
    });

    const result = await reranker.rerank('query', ['a', 'b', 'c'], 2);

    expect(result).toEqual([
      { text: 'c', score: 0.91 },
      { text: 'a', score: 0.42 },
    ]);
    expect(spies.warn).not.toHaveBeenCalled();
    expect(spies.error).not.toHaveBeenCalled();
  });

  it('should keep a ranking when the response omits the billing telemetry', async () => {
    const reranker = new CohereReranker({
      apiKey: 'test-key',
      logger: mockLogger,
    });
    const spies = spyOnLogger();
    // Telemetry is not part of the ranking; a response without it is usable.
    jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { id: 'req-1', results: [{ index: 1, relevance_score: 0.77 }] },
    });

    const result = await reranker.rerank('query', ['a', 'b'], 1);

    expect(result).toEqual([{ text: 'b', score: 0.77 }]);
    expect(spies.warn).not.toHaveBeenCalled();
    expect(spies.error).not.toHaveBeenCalled();
  });

  it('should report a payload with no results array as a bad response', async () => {
    const reranker = new CohereReranker({
      apiKey: 'test-key',
      logger: mockLogger,
    });
    const spies = spyOnLogger();
    jest.spyOn(axios, 'post').mockResolvedValueOnce({ data: { id: 'req-1' } });

    const result = await reranker.rerank('query', ['a', 'b'], 2);

    expect(result).toEqual([
      { text: 'a', score: 0 },
      { text: 'b', score: 0 },
    ]);
    /** A malformed payload is a bad response, not a thrown error. */
    expect(spies.error).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith(
      expect.stringContaining('fallbacks=1 reasons={bad_response:1}')
    );
  });

  it('should fall back to input order without a network call when no key is set', async () => {
    const reranker = new CohereReranker({ apiKey: '', logger: mockLogger });
    const spies = spyOnLogger();
    const postSpy = jest.spyOn(axios, 'post');

    const result = await reranker.rerank('query', ['a', 'b'], 2);

    expect(result).toEqual([
      { text: 'a', score: 0 },
      { text: 'b', score: 0 },
    ]);
    expect(postSpy).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith(
      expect.stringContaining('fallbacks=1 reasons={no_api_key:1}')
    );
  });

  describe('constructor', () => {
    it('should use the default API URL when no apiUrl is provided', () => {
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      expect(getApiUrl(reranker)).toBe('https://api.cohere.com/v2/rerank');
    });

    it('should use a custom API URL when provided', () => {
      const customUrl = 'https://litellm.internal/v2/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      expect(getApiUrl(reranker)).toBe(customUrl);
    });

    it('should use environment variable COHERE_API_URL when available', () => {
      const originalEnv = process.env.COHERE_API_URL;
      process.env.COHERE_API_URL = 'https://env-cohere-endpoint.com/v2/rerank';

      const reranker = new CohereReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      expect(getApiUrl(reranker)).toBe(
        'https://env-cohere-endpoint.com/v2/rerank'
      );

      if (typeof originalEnv === 'string') {
        process.env.COHERE_API_URL = originalEnv;
      } else {
        delete process.env.COHERE_API_URL;
      }
    });
  });

  describe('rerank method', () => {
    it('should post to the configured API URL', async () => {
      const customUrl = 'https://litellm.internal/v2/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });
      const spies = spyOnLogger();
      const postSpy = jest.spyOn(axios, 'post').mockResolvedValueOnce({
        data: { results: [{ index: 0, relevance_score: 0.9 }] },
      });

      await reranker.rerank('query', ['a'], 1);

      expect(postSpy).toHaveBeenCalledWith(
        customUrl,
        expect.any(Object),
        expect.anything()
      );
      expect(spies.error).not.toHaveBeenCalled();
    });
  });
});

describe('createReranker', () => {
  it('should create CohereReranker with cohereApiUrl when provided', () => {
    const customUrl = 'https://litellm.internal/v2/rerank';
    const reranker = createReranker({
      rerankerType: 'cohere',
      cohereApiKey: 'test-key',
      cohereApiUrl: customUrl,
    });

    expect(reranker).toBeInstanceOf(CohereReranker);
    if (!(reranker instanceof CohereReranker)) {
      throw new Error('Expected createReranker to return a CohereReranker.');
    }
    expect(getApiUrl(reranker)).toBe(customUrl);
  });

  it('should create CohereReranker with default URL when cohereApiUrl is not provided', () => {
    const reranker = createReranker({
      rerankerType: 'cohere',
      cohereApiKey: 'test-key',
    });

    expect(reranker).toBeInstanceOf(CohereReranker);
    if (!(reranker instanceof CohereReranker)) {
      throw new Error('Expected createReranker to return a CohereReranker.');
    }
    expect(getApiUrl(reranker)).toBe('https://api.cohere.com/v2/rerank');
  });
});
