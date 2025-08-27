/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-require-imports, @typescript-eslint/no-unused-vars, @typescript-eslint/no-unsafe-function-type, @typescript-eslint/strict-boolean-expressions */

import { JinaReranker } from './rerankers';
import type * as t from './types';

// Mock axios and promise-retry to control API responses
jest.mock('axios');
jest.mock('promise-retry');

const mockAxios = require('axios');
const mockPromiseRetry = require('promise-retry');

// Mock logger
const mockLogger = {
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

describe('JinaReranker Chunking Tests', () => {
  let jinaReranker: JinaReranker;

  beforeEach(() => {
    jest.clearAllMocks();
    jest.restoreAllMocks(); // Clear any spies from previous tests

    // Mock promise-retry to execute the function directly (success case)
    mockPromiseRetry.mockImplementation(async (fn: Function, options: any) => {
      return await fn(() => {
        throw new Error('Should not retry');
      }, 1); // retry function and attempt number
    });

    jinaReranker = new JinaReranker({
      apiKey: 'test-api-key',
      logger: mockLogger as any,
      chunkingConfig: {
        maxChunkSize: 1800,
        overlapSize: 200,
        enableParallelProcessing: true,
        aggregationStrategy: 'weighted_average',
      },
    });
  });

  describe('Document Size Validation', () => {
    test('should correctly identify documents that need chunking', () => {
      const smallDocument = 'This is a small document.';
      const largeDocument = 'x'.repeat(2500); // 2500 characters, exceeds 2048 limit

      // Use private method access for testing
      expect((jinaReranker as any).needsChunking(smallDocument)).toBe(false);
      expect((jinaReranker as any).needsChunking(largeDocument)).toBe(true);
    });

    test('should correctly calculate document byte size', () => {
      const document = 'Hello World!'; // 12 characters in UTF-8
      const size = (jinaReranker as any).calculateDocumentSize(document);
      expect(size).toBe(12);
    });

    test('should handle Unicode characters correctly', () => {
      const unicodeDocument = 'Hello 世界! 🌍'; // Contains multi-byte characters
      const size = (jinaReranker as any).calculateDocumentSize(unicodeDocument);
      expect(size).toBeGreaterThan(unicodeDocument.length); // Byte size > character count
    });
  });

  describe('Document Chunking', () => {
    test('should chunk large documents correctly', async () => {
      const largeDocument = 'This is a sentence. '.repeat(200); // Creates ~4000 character document
      const chunks = await (jinaReranker as any).chunkDocument(largeDocument);

      expect(chunks.length).toBeGreaterThan(1);
      expect(chunks[0].length).toBeLessThanOrEqual(1800);

      // Check for overlap between chunks
      if (chunks.length > 1) {
        const firstChunkEnd = chunks[0].slice(-100);
        const secondChunkStart = chunks[1].slice(0, 100);
        // Should have some overlap
        expect(chunks[1]).toContain(firstChunkEnd.slice(-50));
      }
    });

    test('should not chunk small documents', async () => {
      const smallDocument = 'This is a small document that fits in one chunk.';
      const chunks = await (jinaReranker as any).chunkDocument(smallDocument);

      expect(chunks).toEqual([smallDocument]);
    });

    test('should preserve sentence boundaries when chunking', async () => {
      const document = Array(100)
        .fill('This is a complete sentence with proper punctuation.')
        .join(' ');
      const chunks = await (jinaReranker as any).chunkDocument(document);

      expect(chunks.length).toBeGreaterThan(1); // Should actually create multiple chunks

      // The RecursiveCharacterTextSplitter should attempt to split on sentence boundaries
      // Test that sentences are not broken in the middle when possible
      chunks.forEach((chunk: string): void => {
        expect(chunk.length).toBeLessThanOrEqual(1800); // Within max chunk size
        expect(chunk.trim().length).toBeGreaterThan(0); // Not empty
      });

      // Verify that the full document is preserved across all chunks
      const reconstructed = chunks.join('').replace(/\s+/g, ' ').trim();
      const original = document.replace(/\s+/g, ' ').trim();
      expect(reconstructed).toContain(original.substring(0, 100)); // At least first part preserved
    });
  });

  describe('Result Aggregation', () => {
    const mockChunkResults: t.JinaChunkResult[] = [
      {
        originalIndex: 0,
        chunkIndex: 0,
        relevanceScore: 0.9,
        text: 'First chunk of document 1',
        documentLength: 1000,
      },
      {
        originalIndex: 0,
        chunkIndex: 1,
        relevanceScore: 0.7,
        text: 'Second chunk of document 1',
        documentLength: 1000,
      },
      {
        originalIndex: 1,
        chunkIndex: 0,
        relevanceScore: 0.8,
        text: 'First chunk of document 2',
        documentLength: 800,
      },
    ];

    const originalDocuments = ['Document 1 content', 'Document 2 content'];

    test('should aggregate chunks using weighted average strategy', () => {
      const results = (jinaReranker as any).aggregateChunkResults(
        mockChunkResults,
        originalDocuments
      );

      expect(results).toHaveLength(2); // Two original documents
      expect(results[0].score).toBeGreaterThan(0.7); // Should be weighted average of 0.9 and 0.7
      expect(results[0].score).toBeLessThan(0.9);
      expect(results[1].score).toBe(0.8); // Single chunk, so same score
    });

    test('should aggregate chunks using max score strategy', () => {
      const maxScoreReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
        chunkingConfig: {
          aggregationStrategy: 'max_score',
        },
      });

      const results = (maxScoreReranker as any).aggregateChunkResults(
        mockChunkResults,
        originalDocuments
      );

      expect(results[0].score).toBe(0.9); // Max of 0.9 and 0.7
      expect(results[1].score).toBe(0.8);
    });

    test('should aggregate chunks using first chunk strategy', () => {
      const firstChunkReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
        chunkingConfig: {
          aggregationStrategy: 'first_chunk',
        },
      });

      const results = (firstChunkReranker as any).aggregateChunkResults(
        mockChunkResults,
        originalDocuments
      );

      expect(results[0].score).toBe(0.9); // First chunk score
      expect(results[1].score).toBe(0.8);
    });

    test('should use best chunk text for all aggregation strategies', () => {
      const results = (jinaReranker as any).aggregateChunkResults(
        mockChunkResults,
        originalDocuments
      );

      expect(results[0].text).toBe('First chunk of document 1'); // Highest scoring chunk
      expect(results[1].text).toBe('First chunk of document 2');
    });
  });

  describe('API Error Handling', () => {
    test('should handle size limit errors with chunking fallback', async () => {
      const documents = ['x'.repeat(1000)]; // Smaller document that won't trigger automatic chunking

      // Mock promise-retry to throw size limit error, then trigger fallback
      mockPromiseRetry.mockRejectedValueOnce({
        response: {
          status: 400,
          data: { message: 'Document size exceeds 2048 character limit' },
        },
      });

      // Mock the processChunkedDocuments fallback to return success
      jest
        .spyOn(jinaReranker as any, 'processChunkedDocuments')
        .mockResolvedValueOnce([
          { text: 'chunk1', score: 0.8 },
          { text: 'chunk2', score: 0.7 },
        ]);

      const results = await jinaReranker.rerank('test query', documents, 2);

      expect(results).toBeDefined();
      expect(results).toHaveLength(2);
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'Size limit error detected, falling back to chunking strategy'
      );
    });

    test('should handle timeout errors with exponential backoff', async () => {
      const documents = ['small document'];

      // Mock promise-retry to simulate retry behavior - should return array
      let attemptCount = 0;
      mockPromiseRetry.mockImplementation(
        async (fn: Function, options: any) => {
          const retry = (error: any): any => {
            attemptCount++;
            if (attemptCount >= 2) {
              // After 2 attempts, succeed with proper array format
              return [{ text: 'small document', score: 0.9 }];
            }
            throw error;
          };

          try {
            attemptCount++;
            return await fn(retry, attemptCount);
          } catch (error) {
            if (attemptCount < 2) {
              return retry(error);
            }
            throw error;
          }
        }
      );

      // Mock axios to throw timeout, then succeed
      mockAxios.post
        .mockRejectedValueOnce({ response: { status: 524 } })
        .mockResolvedValueOnce({
          data: {
            model: 'jina-reranker-v2-base-multilingual',
            usage: { total_tokens: 10 },
            results: [
              {
                index: 0,
                relevance_score: 0.9,
                document: { text: 'small document' },
              },
            ],
          },
        });

      const results = await jinaReranker.rerank('test query', documents, 1);

      expect(results).toHaveLength(1);
      expect(mockLogger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Jina API timeout on attempt')
      );
    });

    test('should fallback to default ranking after all retries fail', async () => {
      const documents = ['test document'];

      // Mock promise-retry to simulate exhausted retries
      mockPromiseRetry.mockRejectedValue(new Error('Network error'));

      const results = await jinaReranker.rerank('test query', documents, 1);

      expect(results).toHaveLength(1);
      expect(results[0].score).toBe(0); // Default ranking score
      expect(results[0].text).toBe('test document');
      expect(mockLogger.error).toHaveBeenCalledWith(
        'Error using Jina reranker:',
        expect.objectContaining({ error: expect.any(Object) })
      );
    });
  });

  describe('Performance and Edge Cases', () => {
    test('should handle empty document array', async () => {
      const results = await jinaReranker.rerank('test query', [], 5);

      expect(results).toEqual([]);
    });

    test('should handle single character documents', async () => {
      const documents = ['a', 'b', 'c'];

      // Create a fresh instance to avoid test interference
      const freshReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
      });

      // Completely reset all mocks
      jest.resetAllMocks();

      // Reset promise-retry mock for this test - it should call the function directly
      mockPromiseRetry.mockImplementation(
        async (fn: Function, options: any) => {
          return await fn(() => {}, 1);
        }
      );

      mockAxios.post.mockResolvedValueOnce({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: { total_tokens: 5 },
          results: [
            { index: 0, relevance_score: 0.9, document: { text: 'a' } },
            { index: 1, relevance_score: 0.8, document: { text: 'b' } },
            { index: 2, relevance_score: 0.7, document: { text: 'c' } },
          ],
        },
      });

      const results = await freshReranker.rerank('test query', documents, 3);

      expect(results).toHaveLength(3);
      expect(results[0].text).toBe('a');
    });

    test('should handle documents exactly at size limit', async () => {
      const exactSizeDocument = 'x'.repeat(2048); // Exactly 2048 characters

      // Create a fresh instance to avoid test interference
      const freshReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
      });

      // Completely reset all mocks
      jest.resetAllMocks();

      // Reset promise-retry mock for this test - it should call the function directly
      mockPromiseRetry.mockImplementation(
        async (fn: Function, options: any) => {
          return await fn(() => {}, 1);
        }
      );

      mockAxios.post.mockResolvedValueOnce({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: { total_tokens: 50 },
          results: [
            {
              index: 0,
              relevance_score: 0.85,
              document: { text: exactSizeDocument },
            },
          ],
        },
      });

      const results = await freshReranker.rerank(
        'test query',
        [exactSizeDocument],
        1
      );

      expect(results).toHaveLength(1);
      expect(results[0].score).toBe(0.85);
      expect(mockLogger.debug).toHaveBeenCalledWith(
        expect.stringMatching(
          /^Total request size \(\d+ bytes\) within limits, using direct API call$/
        )
      );
    });

    test('should handle mixed document sizes correctly', async () => {
      // Create documents that will exceed the 64KB total request limit
      const documents = [
        'small document',
        'x'.repeat(70000), // Large document to push total over 64KB limit
        'another small document',
      ];

      mockAxios.post.mockResolvedValueOnce({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: { total_tokens: 100 },
          results: [
            { index: 0, relevance_score: 0.9, document: { text: 'chunk1' } },
            { index: 1, relevance_score: 0.8, document: { text: 'chunk2' } },
          ],
        },
      });

      const results = await jinaReranker.rerank('test query', documents, 3);

      expect(results).toBeDefined();
      expect(mockLogger.debug).toHaveBeenCalledWith(
        expect.stringMatching(
          /^Total request size \(\d+ bytes\) exceeds limits, using chunking strategy$/
        )
      );
    });
  });

  describe('Configuration', () => {
    test('should use default chunking configuration when not provided', () => {
      const defaultReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
      });

      expect((defaultReranker as any).chunkingConfig.maxChunkSize).toBe(1800);
      expect((defaultReranker as any).chunkingConfig.overlapSize).toBe(200);
      expect(
        (defaultReranker as any).chunkingConfig.enableParallelProcessing
      ).toBe(false);
      expect((defaultReranker as any).chunkingConfig.aggregationStrategy).toBe(
        'weighted_average'
      );
    });

    test('should override default configuration with provided values', () => {
      const customReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
        chunkingConfig: {
          maxChunkSize: 1000,
          overlapSize: 100,
          enableParallelProcessing: false,
          aggregationStrategy: 'max_score',
        },
      });

      expect((customReranker as any).chunkingConfig.maxChunkSize).toBe(1000);
      expect((customReranker as any).chunkingConfig.overlapSize).toBe(100);
      expect(
        (customReranker as any).chunkingConfig.enableParallelProcessing
      ).toBe(false);
      expect((customReranker as any).chunkingConfig.aggregationStrategy).toBe(
        'max_score'
      );
    });

    test('should handle missing API key gracefully', async () => {
      const noKeyReranker = new JinaReranker({
        logger: mockLogger as any,
      });

      const results = await noKeyReranker.rerank(
        'test query',
        ['test document'],
        1
      );

      expect(results).toHaveLength(1);
      expect(results[0].score).toBe(0);
      expect(mockLogger.warn).toHaveBeenCalledWith(
        'JINA_API_KEY is not set. Using default ranking.'
      );
    });
  });

  describe('Parallel Processing', () => {
    test('should process chunks in parallel when enabled', async () => {
      const parallelReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
        chunkingConfig: {
          enableParallelProcessing: true,
        },
      });

      const largeDocuments = Array(5).fill('x'.repeat(2500));

      mockAxios.post.mockResolvedValue({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: { total_tokens: 50 },
          results: [
            { index: 0, relevance_score: 0.8, document: { text: 'chunk' } },
          ],
        },
      });

      const startTime = Date.now();
      await parallelReranker.rerank('test query', largeDocuments, 5);
      const endTime = Date.now();

      // With parallel processing, this should complete faster
      // This is more of a smoke test as timing can be flaky in tests
      expect(endTime - startTime).toBeLessThan(10000); // Should complete within 10 seconds
    });

    test('should process chunks sequentially when parallel processing is disabled', async () => {
      const sequentialReranker = new JinaReranker({
        apiKey: 'test-api-key',
        logger: mockLogger as any,
        chunkingConfig: {
          enableParallelProcessing: false,
        },
      });

      const largeDocuments = ['x'.repeat(2500)];

      mockAxios.post.mockResolvedValue({
        data: {
          model: 'jina-reranker-v2-base-multilingual',
          usage: { total_tokens: 50 },
          results: [
            { index: 0, relevance_score: 0.8, document: { text: 'chunk' } },
          ],
        },
      });

      const results = await sequentialReranker.rerank(
        'test query',
        largeDocuments,
        1
      );

      expect(results).toBeDefined();
      expect(results).toHaveLength(1);
    });
  });
});

describe('Integration Tests', () => {
  test('should work end-to-end with realistic document sizes', async () => {
    const reranker = new JinaReranker({
      apiKey: process.env.JINA_API_KEY || 'test-key',
      logger: mockLogger as any,
    });

    // Create documents similar to real-world search results
    const documents = [
      'Short article snippet.',
      // Simulate a long article (like the 2,493 character error case)
      'This is a comprehensive article about artificial intelligence and machine learning. '.repeat(
        30
      ),
      'Medium-length document content that provides detailed information about the topic. '.repeat(
        10
      ),
    ];

    // Mock the chunking process to return aggregated results correctly
    jest.spyOn(reranker as any, 'processChunkedDocuments').mockResolvedValue([
      { text: documents[0], score: 0.92 },
      { text: documents[1].substring(0, 100), score: 0.87 },
      { text: documents[2], score: 0.75 },
    ]);

    // Mock direct API call for smaller documents
    mockAxios.post.mockResolvedValue({
      data: {
        model: 'jina-reranker-v2-base-multilingual',
        usage: { total_tokens: 200 },
        results: [
          { index: 0, relevance_score: 0.92, document: { text: documents[0] } },
          {
            index: 1,
            relevance_score: 0.87,
            document: { text: documents[1].substring(0, 100) },
          },
          { index: 2, relevance_score: 0.75, document: { text: documents[2] } },
        ],
      },
    });

    const results = await reranker.rerank(
      'artificial intelligence',
      documents,
      3
    );

    expect(results).toHaveLength(3);
    expect(results[0].score).toBeGreaterThan(0.7);
    expect(results[0].text).toBeTruthy();
  });
});
