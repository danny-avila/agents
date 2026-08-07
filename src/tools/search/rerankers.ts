import axios from 'axios';
import type * as t from './types';
import { createDefaultLogger, formatErrorForLog } from './utils';

const DEFAULT_JINA_API_URL = 'https://api.jina.ai/v1/rerank';

/** Every other network call in the search pipeline is bounded (scrapers,
 * search providers); rerank requests must be too, or a hung rerank API
 * stalls the whole tool. */
const DEFAULT_RERANKER_TIMEOUT = 10000;

const getDefaultJinaApiUrl = (): string =>
  process.env.JINA_API_URL != null && process.env.JINA_API_URL !== ''
    ? process.env.JINA_API_URL
    : DEFAULT_JINA_API_URL;

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
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiKey = process.env.JINA_API_KEY,
    apiUrl = getDefaultJinaApiUrl(),
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiKey?: string;
    apiUrl?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    this.logger.debug(
      `Reranking ${documents.length} chunks with Jina using API URL: ${this.apiUrl}`
    );

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
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
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
      this.logger.error('Error using Jina reranker', formatErrorForLog(error));
      // Fallback to default ranking on error
      return this.getDefaultRanking(documents, topK);
    }
  }
}

export class CohereReranker extends BaseReranker {
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiKey = process.env.COHERE_API_KEY,
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiKey?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
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

      const response = await axios.post<t.CohereRerankerResponse | undefined>(
        'https://api.cohere.com/v2/rerank',
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${this.apiKey}`,
          },
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
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
      this.logger.error(
        'Error using Cohere reranker',
        formatErrorForLog(error)
      );
      // Fallback to default ranking on error
      return this.getDefaultRanking(documents, topK);
    }
  }
}

/** rag_api's `/v1/rerank` contract caps candidates at 50 and `top_n` at 25;
 * violating either is a client bug, not a server error, so both are clamped
 * locally with a debug log instead of erroring. */
const RAG_API_MAX_CANDIDATES = 50;
const RAG_API_MAX_TOP_N = 25;
const RAG_API_DEFAULT_PROFILE = 'fast-v1';

const getDefaultRagApiUrl = (): string | undefined =>
  process.env.RAG_API_URL != null && process.env.RAG_API_URL !== ''
    ? process.env.RAG_API_URL
    : undefined;

const clampRagApiCandidates = (
  candidates: t.RagApiRerankCandidate[],
  logger: t.Logger
): t.RagApiRerankCandidate[] => {
  if (candidates.length <= RAG_API_MAX_CANDIDATES) {
    return candidates;
  }
  logger.debug(
    `rag_api fast-v1 accepts at most ${RAG_API_MAX_CANDIDATES} candidates; truncating ${candidates.length} to ${RAG_API_MAX_CANDIDATES}.`
  );
  return candidates.slice(0, RAG_API_MAX_CANDIDATES);
};

const clampRagApiTopN = (topN: number, logger: t.Logger): number => {
  if (topN <= RAG_API_MAX_TOP_N) {
    return topN;
  }
  logger.debug(
    `rag_api fast-v1 accepts top_n <= ${RAG_API_MAX_TOP_N}; clamping ${topN} to ${RAG_API_MAX_TOP_N}.`
  );
  return RAG_API_MAX_TOP_N;
};

/** Deterministic tie ordering: rank by score descending, breaking ties on
 * original candidate index rather than relying on rag_api's response order. */
const sortRagApiResults = (
  results: t.RagApiRerankResult[]
): t.RagApiRerankResult[] =>
  [...results].sort((a, b) => b.score - a.score || a.index - b.index);

const isValidRagApiResult = (
  result: t.RagApiRerankResult | undefined,
  documentCount: number
): result is t.RagApiRerankResult =>
  result != null &&
  typeof result.index === 'number' &&
  Number.isInteger(result.index) &&
  result.index >= 0 &&
  result.index < documentCount &&
  typeof result.score === 'number' &&
  Number.isFinite(result.score);

/**
 * Calls the public `danny-avila/rag_api` `/v1/rerank` endpoint (the
 * `fast-v1` profile). Auth is supplied per call by a token supplier (a
 * short-lived JWT minted by the host app) rather than a static key. A
 * reranker failure must never throw into the search flow: on a missing
 * base URL/token supplier, timeout, non-2xx response, or malformed payload,
 * this falls back to the candidates' original order via
 * {@link BaseReranker.getDefaultRanking}.
 */
export class RagApiReranker extends BaseReranker {
  private baseUrl?: string;
  private tokenSupplier?: t.RagApiTokenSupplier;
  private profile: string;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    baseUrl = getDefaultRagApiUrl(),
    tokenSupplier,
    profile = RAG_API_DEFAULT_PROFILE,
    timeout = DEFAULT_RERANKER_TIMEOUT,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    baseUrl?: string;
    tokenSupplier?: t.RagApiTokenSupplier;
    profile?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.baseUrl = baseUrl;
    this.tokenSupplier = tokenSupplier;
    this.profile = profile;
    this.timeout = timeout;
    this.httpAgent = httpAgent;
    this.httpsAgent = httpsAgent;
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number = 5
  ): Promise<t.Highlight[]> {
    if (documents.length === 0) {
      return [];
    }

    this.logger.debug(
      `Reranking ${documents.length} chunks with rag_api (${this.profile}) using base URL: ${this.baseUrl}`
    );

    if (this.baseUrl == null || this.baseUrl === '') {
      this.logger.warn('RAG_API_URL is not set. Using default ranking.');
      return this.getDefaultRanking(documents, topK);
    }

    if (this.tokenSupplier == null) {
      this.logger.warn(
        'No rag_api token supplier configured. Using default ranking.'
      );
      return this.getDefaultRanking(documents, topK);
    }

    const candidates = clampRagApiCandidates(
      documents.map((text, index) => ({ id: String(index), text, base_score: 0 })),
      this.logger
    );
    const topN = clampRagApiTopN(Math.max(0, topK), this.logger);

    try {
      const token = await this.tokenSupplier();
      const requestData: t.RagApiRerankRequestBody = {
        profile: this.profile,
        query,
        candidates,
        top_n: topN,
      };

      const response = await axios.post<t.RagApiRerankResponse | undefined>(
        `${this.baseUrl}/v1/rerank`,
        requestData,
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          timeout: this.timeout,
          httpAgent: this.httpAgent,
          httpsAgent: this.httpsAgent,
        }
      );

      this.logger.debug('rag_api rerank model:', response.data?.model);

      const rawResults = response.data?.results;
      if (!Array.isArray(rawResults) || rawResults.length === 0) {
        this.logger.warn(
          'Unexpected response format from rag_api rerank. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      const validResults = rawResults.filter((result) =>
        isValidRagApiResult(result, documents.length)
      );
      if (validResults.length === 0) {
        this.logger.warn(
          'rag_api rerank response contained no valid results. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      return sortRagApiResults(validResults)
        .slice(0, topN)
        .map((result) => ({
          text: documents[result.index],
          score: result.score,
        }));
    } catch (error) {
      this.logger.error(
        'Error using rag_api reranker',
        formatErrorForLog(error)
      );
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
export const createReranker = (
  config: {
    rerankerType: t.RerankerType;
    jinaApiKey?: string;
    jinaApiUrl?: string;
    cohereApiKey?: string;
    ragApiUrl?: string;
    ragApiTokenSupplier?: t.RagApiTokenSupplier;
    ragApiProfile?: string;
    rerankerTimeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig
): BaseReranker | undefined => {
  const {
    rerankerType,
    jinaApiKey,
    jinaApiUrl,
    cohereApiKey,
    ragApiUrl,
    ragApiTokenSupplier,
    ragApiProfile,
    rerankerTimeout,
    logger,
    httpAgent,
    httpsAgent,
  } = config;

  // Create a default logger if none is provided
  const defaultLogger = logger || createDefaultLogger();

  switch (rerankerType.toLowerCase()) {
  case 'jina':
    return new JinaReranker({
      apiKey: jinaApiKey,
      apiUrl: jinaApiUrl,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'cohere':
    return new CohereReranker({
      apiKey: cohereApiKey,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  case 'rag-api':
    return new RagApiReranker({
      baseUrl: ragApiUrl,
      tokenSupplier: ragApiTokenSupplier,
      profile: ragApiProfile,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
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
    return new JinaReranker({
      apiKey: jinaApiKey,
      apiUrl: jinaApiUrl,
      timeout: rerankerTimeout,
      logger: defaultLogger,
      httpAgent,
      httpsAgent,
    });
  }
};

// Example usage:
// const jinaReranker = new JinaReranker();
// const cohereReranker = new CohereReranker();
// const infinityReranker = new InfinityReranker();
