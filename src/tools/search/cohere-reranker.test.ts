import { CohereReranker, createReranker } from './rerankers';
import { createDefaultLogger } from './utils';

// Helper to access private apiUrl property for testing
const getApiUrl = (reranker: CohereReranker): string =>
  (reranker as unknown as { apiUrl: string }).apiUrl;

describe('CohereReranker', () => {
  const mockLogger = createDefaultLogger();

  describe('constructor', () => {
    it('should use default API URL when no apiUrl is provided', () => {
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      expect(getApiUrl(reranker)).toBe('https://api.cohere.com/v2/rerank');
    });

    it('should use custom API URL when provided', () => {
      const customUrl = 'https://custom-cohere-endpoint.com/v2/rerank';
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

      // Restore original environment
      if (originalEnv !== undefined) {
        process.env.COHERE_API_URL = originalEnv;
      } else {
        delete process.env.COHERE_API_URL;
      }
    });

    it('should prioritize explicit apiUrl over environment variable', () => {
      const originalEnv = process.env.COHERE_API_URL;
      process.env.COHERE_API_URL = 'https://env-cohere-endpoint.com/v2/rerank';

      const customUrl = 'https://explicit-cohere-endpoint.com/v2/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      expect(getApiUrl(reranker)).toBe(customUrl);

      // Restore original environment
      if (originalEnv !== undefined) {
        process.env.COHERE_API_URL = originalEnv;
      } else {
        delete process.env.COHERE_API_URL;
      }
    });
  });

  describe('rerank method', () => {
    it('should log the API URL being used', async () => {
      const customUrl = 'https://test-cohere-endpoint.com/v2/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const logSpy = jest.spyOn(mockLogger, 'debug');

      try {
        await reranker.rerank('test query', ['document1', 'document2'], 2);
      } catch (_error) {
        // Expected to fail due to network error, but we can check the log
      }

      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining(
          `Reranking 2 chunks with Cohere using API URL: ${customUrl}`
        )
      );

      logSpy.mockRestore();
    });
  });
});

describe('createReranker for Cohere', () => {
  it('should create CohereReranker with cohereApiUrl when provided', () => {
    const customUrl = 'https://custom-cohere-endpoint.com/v2/rerank';
    const reranker = createReranker({
      rerankerType: 'cohere',
      cohereApiKey: 'test-key',
      cohereApiUrl: customUrl,
    });

    expect(reranker).toBeInstanceOf(CohereReranker);
    expect(getApiUrl(reranker as CohereReranker)).toBe(customUrl);
  });

  it('should create CohereReranker with default URL when cohereApiUrl is not provided', () => {
    const reranker = createReranker({
      rerankerType: 'cohere',
      cohereApiKey: 'test-key',
    });

    expect(reranker).toBeInstanceOf(CohereReranker);
    expect(getApiUrl(reranker as CohereReranker)).toBe(
      'https://api.cohere.com/v2/rerank'
    );
  });
});
