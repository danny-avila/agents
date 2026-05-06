import { CohereReranker } from './rerankers';
import { createDefaultLogger } from './utils';

describe('CohereReranker', () => {
  const mockLogger = createDefaultLogger();

  describe('constructor', () => {
    it('should use default API URL when no apiUrl is provided', () => {
      const originalEnv = process.env.COHERE_API_URL;
      delete process.env.COHERE_API_URL;

      const reranker = new CohereReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      // Access private property for testing
      const apiUrl = (reranker as unknown as { apiUrl: string }).apiUrl;
      expect(apiUrl).toBe('https://api.cohere.com/v2/rerank');

      if (originalEnv) {
        process.env.COHERE_API_URL = originalEnv;
      }
    });

    it('should use custom API URL when provided', () => {
      const customUrl = 'https://my-azure.endpoint.com/v1/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = (reranker as unknown as { apiUrl: string }).apiUrl;
      expect(apiUrl).toBe(customUrl);
    });

    it('should use environment variable COHERE_API_URL when available', () => {
      const originalEnv = process.env.COHERE_API_URL;
      process.env.COHERE_API_URL = 'https://env-cohere.example.com/v2/rerank';

      const reranker = new CohereReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });

      const apiUrl = (reranker as unknown as { apiUrl: string }).apiUrl;
      expect(apiUrl).toBe('https://env-cohere.example.com/v2/rerank');

      if (originalEnv) {
        process.env.COHERE_API_URL = originalEnv;
      } else {
        delete process.env.COHERE_API_URL;
      }
    });

    it('should prioritize explicit apiUrl over environment variable', () => {
      const originalEnv = process.env.COHERE_API_URL;
      process.env.COHERE_API_URL = 'https://env-cohere.example.com/v2/rerank';

      const customUrl = 'https://explicit-cohere.example.com/v2/rerank';
      const reranker = new CohereReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });

      const apiUrl = (reranker as unknown as { apiUrl: string }).apiUrl;
      expect(apiUrl).toBe(customUrl);

      if (originalEnv) {
        process.env.COHERE_API_URL = originalEnv;
      } else {
        delete process.env.COHERE_API_URL;
      }
    });
  });
});
