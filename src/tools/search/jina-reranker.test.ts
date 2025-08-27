import { JinaReranker } from './rerankers';
import { createDefaultLogger } from './utils';

describe('JinaReranker', () => {
  const mockLogger = createDefaultLogger();
  
  describe('constructor', () => {
    it('should use default API URL when no apiUrl is provided', () => {
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      
      // Access private property for testing
      const apiUrl = (reranker as any).apiUrl;
      expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
    });

    it('should use custom API URL when provided', () => {
      const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        apiUrl: customUrl,
        logger: mockLogger,
      });
      
      const apiUrl = (reranker as any).apiUrl;
      expect(apiUrl).toBe(customUrl);
    });

    it('should use environment variable JINA_API_URL when available', () => {
      const originalEnv = process.env.JINA_API_URL;
      process.env.JINA_API_URL = 'https://env-jina-endpoint.com/v1/rerank';
      
      const reranker = new JinaReranker({
        apiKey: 'test-key',
        logger: mockLogger,
      });
      
      const apiUrl = (reranker as any).apiUrl;
      expect(apiUrl).toBe('https://env-jina-endpoint.com/v1/rerank');
      
      // Restore original environment
      if (originalEnv) {
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
      
      const apiUrl = (reranker as any).apiUrl;
      expect(apiUrl).toBe(customUrl);
      
      // Restore original environment
      if (originalEnv) {
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
      
      try {
        await reranker.rerank('test query', ['document1', 'document2'], 2);
      } catch (error) {
        // Expected to fail due to missing API key, but we can check the log
      }
      
      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining(`Reranking 2 chunks with Jina using API URL: ${customUrl}`)
      );
      
      logSpy.mockRestore();
    });
  });
});

describe('createReranker', () => {
  const { createReranker } = require('./rerankers');
  
  it('should create JinaReranker with jinaApiUrl when provided', () => {
    const customUrl = 'https://custom-jina-endpoint.com/v1/rerank';
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
      jinaApiUrl: customUrl,
    });
    
    expect(reranker).toBeInstanceOf(JinaReranker);
    const apiUrl = (reranker as any).apiUrl;
    expect(apiUrl).toBe(customUrl);
  });

  it('should create JinaReranker with default URL when jinaApiUrl is not provided', () => {
    const reranker = createReranker({
      rerankerType: 'jina',
      jinaApiKey: 'test-key',
    });
    
    expect(reranker).toBeInstanceOf(JinaReranker);
    const apiUrl = (reranker as any).apiUrl;
    expect(apiUrl).toBe('https://api.jina.ai/v1/rerank');
  });
});
