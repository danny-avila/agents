import axios from 'axios';
import promiseRetry from 'promise-retry';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import type * as t from './types';
import { createDefaultLogger } from './utils';
import _omit from 'lodash/omit';

export abstract class BaseReranker {
  protected apiKey: string | undefined;
  protected logger: t.Logger;

  constructor(logger?: t.Logger) {
    // Each specific reranker will set its API key
    this.logger = logger || createDefaultLogger();
  }

  abstract rerank(
    query: string,
    documents: string[],
    topK?: number
  ): Promise<t.Highlight[]>;

  protected getDefaultRanking(
    documents: string[],
    topK: number
  ): t.Highlight[] {
    return documents
      .slice(0, Math.min(topK, documents.length))
      .map((doc) => ({ text: doc, score: 0 }));
  }
}

export class JinaReranker extends BaseReranker {
  private chunkingConfig: t.JinaChunkingConfig;

  constructor({
    apiKey = process.env.JINA_API_KEY,
    logger,
    chunkingConfig,
  }: {
    apiKey?: string;
    logger?: t.Logger;
    chunkingConfig?: Partial<t.JinaChunkingConfig>;
  }) {
    super(logger);
    this.apiKey = apiKey;
    this.chunkingConfig = {
      maxChunkSize: 1800,
      overlapSize: 200,
      enableParallelProcessing: false, // Disabled by default to prevent API overload
      aggregationStrategy: 'weighted_average',
      ...chunkingConfig,
    };
  }

  private async chunkDocument(document: string): Promise<string[]> {
    const splitter = new RecursiveCharacterTextSplitter({
      separators: ['\n\n', '\n', '. ', ' '],
      chunkSize: this.chunkingConfig.maxChunkSize,
      chunkOverlap: this.chunkingConfig.overlapSize,
    });

    return await splitter.splitText(document);
  }

  private calculateDocumentSize(document: string): number {
    return new TextEncoder().encode(document).length;
  }

  private needsChunking(document: string): boolean {
    return this.calculateDocumentSize(document) > 2048;
  }

  private calculateTotalRequestSize(
    query: string,
    documents: string[]
  ): number {
    const querySize = this.calculateDocumentSize(query);
    const documentsSize = documents.reduce(
      (total, doc) => total + this.calculateDocumentSize(doc),
      0
    );
    // Add some overhead for JSON structure, field names, etc.
    const overhead = 200;
    return querySize + documentsSize + overhead;
  }

  private needsChunkingForRequest(query: string, documents: string[]): boolean {
    // Jina API request limit - increased from 8KB to reduce unnecessary chunking
    // Previous 8KB was too conservative and causing context overflow issues
    const JINA_REQUEST_LIMIT = 64 * 1024; // 64KB
    return (
      this.calculateTotalRequestSize(query, documents) > JINA_REQUEST_LIMIT
    );
  }

  private aggregateChunkResults(
    chunkResults: t.JinaChunkResult[],
    _originalDocuments: string[]
  ): t.Highlight[] {
    const documentMap = new Map<number, t.JinaChunkResult[]>();

    // Group chunks by original document
    for (const result of chunkResults) {
      if (!documentMap.has(result.originalIndex)) {
        documentMap.set(result.originalIndex, []);
      }
      documentMap.get(result.originalIndex)!.push(result);
    }

    const highlights: t.Highlight[] = [];

    for (const [, chunks] of documentMap) {
      if (chunks.length === 0) continue;

      let aggregatedScore: number;
      let bestText: string;

      switch (this.chunkingConfig.aggregationStrategy) {
      case 'max_score': {
        const bestChunk = chunks.reduce((best, current) =>
          current.relevanceScore > best.relevanceScore ? current : best
        );
        aggregatedScore = bestChunk.relevanceScore;
        bestText = bestChunk.text;
        break;
      }

      case 'first_chunk': {
        aggregatedScore = chunks[0].relevanceScore;
        bestText = chunks[0].text;
        break;
      }

      case 'weighted_average':
      default: {
        // Weight chunks by their length and position
        const totalWeight = chunks.reduce((sum, chunk, index) => {
          const positionWeight = Math.max(0.1, 1 - index * 0.1);
          const lengthWeight =
              chunk.text.length / this.chunkingConfig.maxChunkSize;
          return sum + positionWeight * lengthWeight;
        }, 0);

        aggregatedScore = chunks.reduce((sum, chunk, index) => {
          const positionWeight = Math.max(0.1, 1 - index * 0.1);
          const lengthWeight =
              chunk.text.length / this.chunkingConfig.maxChunkSize;
          const weight = (positionWeight * lengthWeight) / totalWeight;
          return sum + chunk.relevanceScore * weight;
        }, 0);

        // Use the text from the highest-scoring chunk
        bestText = chunks.reduce((best, current) =>
          current.relevanceScore > best.relevanceScore ? current : best
        ).text;
        break;
      }
      }

      highlights.push({
        text: bestText,
        score: aggregatedScore,
      });
    }

    return highlights;
  }

  private async processChunkedDocuments(
    query: string,
    documents: string[],
    topK: number
  ): Promise<t.Highlight[]> {
    const chunkResults: t.JinaChunkResult[] = [];
    const allChunks: string[] = [];
    const chunkToDocumentMap: Array<{
      originalIndex: number;
      chunkIndex: number;
      documentLength: number;
    }> = [];

    // Process each document and create chunks
    for (let docIndex = 0; docIndex < documents.length; docIndex++) {
      const document = documents[docIndex];

      if (this.needsChunking(document)) {
        this.logger.debug(
          `Document ${docIndex} needs chunking (size: ${this.calculateDocumentSize(document)} bytes)`
        );
        const chunks = await this.chunkDocument(document);

        for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
          allChunks.push(chunks[chunkIndex]);
          chunkToDocumentMap.push({
            originalIndex: docIndex,
            chunkIndex,
            documentLength: document.length,
          });
        }
      } else {
        // Document is small enough, add it directly
        allChunks.push(document);
        chunkToDocumentMap.push({
          originalIndex: docIndex,
          chunkIndex: 0,
          documentLength: document.length,
        });
      }
    }

    this.logger.debug(
      `Processing ${allChunks.length} chunks from ${documents.length} documents`
    );

    // Process chunks in batches to respect API limits
    // Reduced from 50 to 5 to prevent overwhelming Jina API and causing 524 timeouts
    const batchSize = Math.min(5, topK);
    const batches: string[][] = [];

    for (let i = 0; i < allChunks.length; i += batchSize) {
      batches.push(allChunks.slice(i, i + batchSize));
    }

    // Process batches
    const batchPromises = batches.map(async (batch, batchIndex) => {
      try {
        // Add rate limiting delay between batches to prevent API overload
        if (batchIndex > 0) {
          const delay = Math.min(1000, batchIndex * 200); // 200ms per batch, max 1s
          await new Promise((resolve) => setTimeout(resolve, delay));
        }

        const response = await this.callJinaAPI(query, batch, batch.length);

        return response.map((result, resultIndex) => {
          const globalIndex = batchIndex * batchSize + resultIndex;
          const mapping = chunkToDocumentMap[globalIndex];

          return {
            originalIndex: mapping.originalIndex,
            chunkIndex: mapping.chunkIndex,
            relevanceScore: result.score,
            text: result.text,
            documentLength: mapping.documentLength,
          };
        });
      } catch (error) {
        this.logger.error(`Error processing batch ${batchIndex}:`, { error });
        return [];
      }
    });

    if (this.chunkingConfig.enableParallelProcessing) {
      const batchResults = await Promise.all(batchPromises);
      for (const results of batchResults) {
        chunkResults.push(...results);
      }
    } else {
      for (const promise of batchPromises) {
        const results = await promise;
        chunkResults.push(...results);
      }
    }

    return this.aggregateChunkResults(chunkResults, documents);
  }

  private async callJinaAPI(
    query: string,
    documents: string[],
    topK: number,
    retries: number = 3
  ): Promise<t.Highlight[]> {
    const requestData = {
      model: 'jina-reranker-v2-base-multilingual',
      query: query,
      top_n: topK,
      documents: documents,
      return_documents: true,
    };

    return promiseRetry<t.Highlight[]>(
      async (retry, attemptNumber) => {
        try {
          const timeout = Number(process.env.JINA_API_TIMEOUT) || 300000;
          const response = await axios.post<t.JinaRerankerResponse>(
            'https://api.jina.ai/v1/rerank',
            requestData,
            {
              headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.apiKey}`,
              },
              timeout,
            }
          );

          this.logger.debug('Jina API Model:', { model: response.data.model });
          this.logger.debug('Jina API Usage:', response.data.usage);

          if (response.data.results.length > 0) {
            return response.data.results.map((result) => {
              const docIndex = result.index;
              const score = result.relevance_score;
              let text = '';

              if (result.document != null) {
                const doc = result.document;
                if (typeof doc === 'object' && 'text' in doc) {
                  text = doc.text;
                } else if (typeof doc === 'string') {
                  text = doc;
                }
              } else {
                text = documents[docIndex];
              }

              return { text, score };
            });
          } else {
            this.logger.warn('Unexpected response format from Jina API');
            return [];
          }
        } catch (error: unknown) {
          const axiosError = error as {
            response?: { status?: number; data?: { message?: string } };
            code?: string;
          };
          this.logger.warn('Jina API error:', {
            error: _omit(axiosError, ['request']),
          });
          if (
            axiosError.response?.status === 400 &&
            Boolean(axiosError.response.data?.message?.includes('2048'))
          ) {
            // Size limit error - don't retry, re-throw immediately
            this.logger.error('Size limit error despite chunking:', {
              error: _omit(axiosError, ['request']),
            });
            throw error;
          } else if (
            axiosError.response?.status === 524 ||
            axiosError.code === 'ECONNABORTED'
          ) {
            // Timeout errors - retry with logging
            this.logger.warn(
              `Jina API timeout on attempt ${attemptNumber}/${retries}`
            );
            return retry(error);
          } else {
            // Other errors - don't retry
            this.logger.error('Jina API error:', {
              error: _omit(axiosError, ['request']),
            });
            throw error;
          }
        }
      },
      {
        retries,
        factor: 2,
        minTimeout: 1000,
        maxTimeout: 3000,
      }
    );
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(`Reranking ${documents.length} documents with Jina`);

    // Handle empty document array
    if (documents.length === 0) {
      return [];
    }

    try {
      if (this.apiKey == null || this.apiKey === '') {
        this.logger.warn('JINA_API_KEY is not set. Using default ranking.');
        return this.getDefaultRanking(documents, topK);
      }

      // Check if the total request size exceeds limits
      const needsChunking = this.needsChunkingForRequest(query, documents);

      if (needsChunking) {
        const totalSize = this.calculateTotalRequestSize(query, documents);
        this.logger.debug(
          `Total request size (${totalSize} bytes) exceeds limits, using chunking strategy`
        );
        return await this.processChunkedDocuments(query, documents, topK);
      } else {
        // Documents are small enough, use direct API call with enhanced error handling
        const totalSize = this.calculateTotalRequestSize(query, documents);
        this.logger.debug(
          `Total request size (${totalSize} bytes) within limits, using direct API call`
        );
        return await this.callJinaAPI(query, documents, topK);
      }
    } catch (error: unknown) {
      // Enhanced error handling with fallback strategies
      const axiosError = error as {
        response?: { status?: number; data?: { message?: string } };
      };
      this.logger.error('Error using Jina reranker:', {
        error: _omit(axiosError, ['request']),
      });
      if (
        axiosError.response?.status === 400 &&
        Boolean(axiosError.response.data?.message?.includes('2048'))
      ) {
        this.logger.warn(
          'Size limit error detected, falling back to chunking strategy'
        );
        try {
          return await this.processChunkedDocuments(query, documents, topK);
        } catch (chunkError) {
          this.logger.error('Chunking fallback also failed:', { chunkError });
          return this.getDefaultRanking(documents, topK);
        }
      }

      // Fallback to default ranking for other errors
      return this.getDefaultRanking(documents, topK);
    }
  }
}

export class CohereReranker extends BaseReranker {
  constructor({
    apiKey = process.env.COHERE_API_KEY,
    logger,
  }: {
    apiKey?: string;
    logger?: t.Logger;
  }) {
    super(logger);
    this.apiKey = apiKey;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(`Reranking ${documents.length} chunks with Cohere`);

    try {
      if (this.apiKey == null || this.apiKey === '') {
        this.logger.warn('COHERE_API_KEY is not set. Using default ranking.');
        return this.getDefaultRanking(documents, topK);
      }

      const requestData = {
        model: 'rerank-v3.5',
        query: query,
        top_n: topK,
        documents: documents,
      };

      const timeout = Number(process.env.COHERE_API_TIMEOUT) || 300000;
      const response = await axios.post<t.CohereRerankerResponse | undefined>(
        'https://api.cohere.com/v2/rerank',
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout,
        }
      );

      this.logger.debug('Cohere API ID:', response.data?.id);
      this.logger.debug('Cohere API Meta:', response.data?.meta);

      if (response.data?.results != null && response.data.results.length > 0) {
        return response.data.results.map((result) => {
          const docIndex = result.index;
          const score = result.relevance_score;
          const text = documents[docIndex];
          return { text, score };
        });
      } else {
        this.logger.warn(
          'Unexpected response format from Cohere API. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }
    } catch (error) {
      this.logger.error('Error using Cohere reranker:', { error });
      // Fallback to default ranking on error
      return this.getDefaultRanking(documents, topK);
    }
  }
}

export class InfinityReranker extends BaseReranker {
  constructor(logger?: t.Logger) {
    super(logger);
    // No API key needed for the placeholder implementation
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(
      `Reranking ${documents.length} chunks with Infinity (placeholder)`
    );
    // This would be replaced with actual Infinity reranker implementation
    return this.getDefaultRanking(documents, topK);
  }
}

/**
 * Creates the appropriate reranker based on type and configuration
 */
export const createReranker = (config: {
  rerankerType: t.RerankerType;
  jinaApiKey?: string;
  cohereApiKey?: string;
  logger?: t.Logger;
  jinaChunkingConfig?: Partial<t.JinaChunkingConfig>;
}): BaseReranker | undefined => {
  const { rerankerType, jinaApiKey, cohereApiKey, logger, jinaChunkingConfig } =
    config;

  // Create a default logger if none is provided
  const defaultLogger = logger || createDefaultLogger();

  switch (rerankerType.toLowerCase()) {
  case 'jina':
    return new JinaReranker({
      apiKey: jinaApiKey,
      logger: defaultLogger,
      chunkingConfig: jinaChunkingConfig,
    });
  case 'cohere':
    return new CohereReranker({
      apiKey: cohereApiKey,
      logger: defaultLogger,
    });
  case 'infinity':
    return new InfinityReranker(defaultLogger);
  case 'none':
    defaultLogger.debug('Skipping reranking as reranker is set to "none"');
    return undefined;
  default:
    defaultLogger.warn(
      `Unknown reranker type: ${rerankerType}. Defaulting to JinaReranker.`
    );
    return new JinaReranker({
      apiKey: jinaApiKey,
      logger: defaultLogger,
      chunkingConfig: jinaChunkingConfig,
    });
  }
};
