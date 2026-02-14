import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger } from './utils';

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
  private apiUrl: string;

  constructor({
    apiKey = process.env.JINA_API_KEY,
    apiUrl = process.env.JINA_API_URL || 'https://api.jina.ai/v1/rerank',
    logger,
  }: {
    apiKey?: string;
    apiUrl?: string;
    logger?: t.Logger;
  }) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(`Reranking ${documents.length} chunks with Jina using API URL: ${this.apiUrl}`);

    try {
      if (this.apiKey == null || this.apiKey === '') {
        this.logger.warn('JINA_API_KEY is not set. Using default ranking.');
        return this.getDefaultRanking(documents, topK);
      }

      const requestData = {
        model: 'jina-reranker-v2-base-multilingual',
        query: query,
        top_n: topK,
        documents: documents,
        return_documents: true,
      };

      const response = await axios.post<t.JinaRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
        }
      );

      this.logger.debug('Jina API Model:', response.data?.model);
      this.logger.debug('Jina API Usage:', response.data?.usage);

      if (response.data && response.data.results.length) {
        return response.data.results.map((result) => {
          const docIndex = result.index;
          const score = result.relevance_score;
          let text = '';

          // If return_documents is true, the document field will be present
          if (result.document != null) {
            const doc = result.document;
            if (typeof doc === 'object' && 'text' in doc) {
              text = doc.text;
            } else if (typeof doc === 'string') {
              text = doc;
            }
          } else {
            // Otherwise, use the index to get the document
            text = documents[docIndex];
          }

          return { text, score };
        });
      } else {
        this.logger.warn(
          'Unexpected response format from Jina API. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }
    } catch (error) {
      this.logger.error('Error using Jina reranker:', error);
      // Fallback to default ranking on error
      return this.getDefaultRanking(documents, topK);
    }
  }
}

export class CohereReranker extends BaseReranker {
  private apiUrl: string;

  constructor({
    apiKey = process.env.COHERE_API_KEY,
    apiUrl = process.env.COHERE_API_URL || 'https://api.cohere.com/v2/rerank',
    logger,
  }: {
    apiKey?: string;
    apiUrl?: string;
    logger?: t.Logger;
  }) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(`Reranking ${documents.length} chunks with Cohere using API URL: ${this.apiUrl}`);

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

      const response = await axios.post<t.CohereRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
        }
      );

      this.logger.debug('Cohere API ID:', response.data?.id);
      this.logger.debug('Cohere API Meta:', response.data?.meta);

      if (response.data && response.data.results.length) {
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
      this.logger.error('Error using Cohere reranker:', error);
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
  jinaApiUrl?: string;
  cohereApiKey?: string;
  cohereApiUrl?: string;
  logger?: t.Logger;
}): BaseReranker | undefined => {
  const { rerankerType, jinaApiKey, jinaApiUrl, cohereApiKey, cohereApiUrl, logger } = config;

  // Create a default logger if none is provided
  const defaultLogger = logger || createDefaultLogger();

  switch (rerankerType.toLowerCase()) {
  case 'jina':
    return new JinaReranker({ apiKey: jinaApiKey, apiUrl: jinaApiUrl, logger: defaultLogger });
  case 'cohere':
    return new CohereReranker({
      apiKey: cohereApiKey,
      apiUrl: cohereApiUrl,
      logger: defaultLogger,
    });
  case 'infinity':
    return new InfinityReranker(defaultLogger);
  case 'none':
    defaultLogger.debug('Skipping reranking as reranker is set to "none"');
    return undefined;
  default:
    defaultLogger.warn(
      `Unknown reranker type: ${rerankerType}. Defaulting to InfinityReranker.`
    );
    return new JinaReranker({ apiKey: jinaApiKey, apiUrl: jinaApiUrl, logger: defaultLogger });
  }
};

// Example usage:
// const jinaReranker = new JinaReranker();
// const cohereReranker = new CohereReranker();
// const infinityReranker = new InfinityReranker();
