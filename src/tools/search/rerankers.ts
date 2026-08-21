import axios, { isAxiosError } from 'axios';
import type { AxiosResponse } from 'axios';
import type * as t from './types';
import { createDefaultLogger, formatErrorForLog } from './utils';

const DEFAULT_JINA_API_URL = 'https://api.jina.ai/v1/rerank';
const DEFAULT_VOYAGE_API_URL = 'https://api.voyageai.com/v1/rerank';
const DEFAULT_VOYAGE_MODEL = 'rerank-2.5';
const DEFAULT_VOYAGE_TIMEOUT_MS = 7000;
const DEFAULT_VOYAGE_RETRY_DELAY_MS = 250;
const MAX_VOYAGE_RETRY_DELAY_MS = 2000;
const VOYAGE_RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

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

/** Voyage's hosted rerank API. Search remains available if the provider is
 * unavailable: all provider, timeout, and response-shape failures fall back to
 * the original source ordering. */
export class VoyageReranker extends BaseReranker {
  private apiUrl: string;
  private model: string;
  private timeout: number;
  private httpAgent?: t.HttpAgent;
  private httpsAgent?: t.HttpsAgent;

  constructor({
    apiKey = process.env.VOYAGE_API_KEY,
    apiUrl = process.env.VOYAGE_API_URL != null &&
    process.env.VOYAGE_API_URL !== ''
      ? process.env.VOYAGE_API_URL
      : DEFAULT_VOYAGE_API_URL,
    model = process.env.VOYAGE_MODEL != null && process.env.VOYAGE_MODEL !== ''
      ? process.env.VOYAGE_MODEL
      : DEFAULT_VOYAGE_MODEL,
    timeout = DEFAULT_VOYAGE_TIMEOUT_MS,
    logger,
    httpAgent,
    httpsAgent,
  }: {
    apiKey?: string;
    apiUrl?: string;
    model?: string;
    timeout?: number;
    logger?: t.Logger;
  } & t.HttpAgentConfig) {
    super(logger);
    this.apiKey = apiKey;
    this.apiUrl = apiUrl;
    this.model = model;
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
      `Reranking ${documents.length} chunks with Voyage using API URL: ${this.apiUrl}`
    );

    try {
      if (this.apiKey == null || this.apiKey === '') {
        this.logger.warn('VOYAGE_API_KEY is not set. Using default ranking.');
        return this.getDefaultRanking(documents, topK);
      }

      const expectedResultCount = Math.max(
        0,
        Math.min(Math.floor(topK), documents.length)
      );
      if (expectedResultCount === 0) {
        return [];
      }

      const response = await this.request(
        query,
        documents,
        expectedResultCount
      );
      const results = response.data?.data;
      if (!Array.isArray(results) || results.length !== expectedResultCount) {
        this.logger.warn(
          'Unexpected response format from Voyage API. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      const seenIndices = new Set<number>();
      const validResults = results.every((result) => {
        if (
          !Number.isInteger(result.index) ||
          result.index < 0 ||
          result.index >= documents.length ||
          !Number.isFinite(result.relevance_score) ||
          seenIndices.has(result.index)
        ) {
          return false;
        }
        seenIndices.add(result.index);
        return true;
      });
      if (!validResults) {
        this.logger.warn(
          'Invalid result data from Voyage API. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      this.logger.debug('Voyage rerank completed', {
        model: this.model,
        resultCount: results.length,
        totalTokens: response.data?.usage?.total_tokens,
      });
      return results.map((result) => ({
        text: documents[result.index],
        score: result.relevance_score,
      }));
    } catch (error) {
      this.logger.error(
        'Error using Voyage reranker',
        formatErrorForLog(error)
      );
      return this.getDefaultRanking(documents, topK);
    }
  }

  private async request(
    query: string,
    documents: string[],
    topK: number
  ): Promise<AxiosResponse<t.VoyageRerankerResponse | undefined>> {
    const requestData = {
      model: this.model,
      query,
      documents,
      top_k: topK,
      return_documents: false,
      truncation: false,
    };
    const requestConfig = {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      timeout: this.timeout,
      httpAgent: this.httpAgent,
      httpsAgent: this.httpsAgent,
    };

    try {
      return await axios.post<t.VoyageRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        requestConfig
      );
    } catch (error) {
      if (!this.isRetryable(error)) {
        throw error;
      }
      await this.wait(this.getRetryDelay(error));
      return axios.post<t.VoyageRerankerResponse | undefined>(
        this.apiUrl,
        requestData,
        requestConfig
      );
    }
  }

  private isRetryable(error: unknown): boolean {
    if (!isAxiosError(error)) {
      return false;
    }
    return (
      error.code === 'ECONNABORTED' ||
      (error.response?.status != null &&
        VOYAGE_RETRYABLE_STATUSES.has(error.response.status))
    );
  }

  private getRetryDelay(error: unknown): number {
    if (!isAxiosError(error)) {
      return DEFAULT_VOYAGE_RETRY_DELAY_MS;
    }
    const headers = error.response?.headers;
    const retryAfterSeconds = Number(
      headers != null ? headers['retry-after'] : undefined
    );
    if (!Number.isFinite(retryAfterSeconds) || retryAfterSeconds < 0) {
      return DEFAULT_VOYAGE_RETRY_DELAY_MS;
    }
    return Math.min(retryAfterSeconds * 1000, MAX_VOYAGE_RETRY_DELAY_MS);
  }

  private wait(milliseconds: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
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

const toRagApiCandidate = (
  text: string,
  index: number
): t.RagApiRerankCandidate => ({ id: String(index), text, base_score: 0 });

/** A single source split into overlapping chunks routinely exceeds the
 * contract limit, so documents are capped before any candidate is built:
 * only the submittable window is ever allocated. */
const buildRagApiCandidates = (
  documents: string[],
  logger: t.Logger
): t.RagApiRerankCandidate[] => {
  if (documents.length <= RAG_API_MAX_CANDIDATES) {
    return documents.map(toRagApiCandidate);
  }
  logger.debug(
    `rag_api fast-v1 accepts at most ${RAG_API_MAX_CANDIDATES} candidates; truncating ${documents.length} to ${RAG_API_MAX_CANDIDATES}.`
  );
  return documents.slice(0, RAG_API_MAX_CANDIDATES).map(toRagApiCandidate);
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

/** Indices are only meaningful against the candidates actually submitted:
 * anything beyond `candidateCount` refers to text the server never saw. */
const isValidRagApiResult = (
  result: t.RagApiRerankResult | undefined,
  candidateCount: number
): result is t.RagApiRerankResult =>
  result != null &&
  typeof result.index === 'number' &&
  Number.isInteger(result.index) &&
  result.index >= 0 &&
  result.index < candidateCount &&
  typeof result.score === 'number' &&
  Number.isFinite(result.score);

/** A repeated index would map one document into several `top_n` slots and
 * silently drop distinct results, so a duplicate invalidates the batch just
 * like any other malformed row. Seen indices are tracked in the same pass
 * that validates each row. */
const isValidRagApiBatch = (
  results: t.RagApiRerankResult[],
  candidateCount: number
): boolean => {
  const seenIndices = new Set<number>();
  return results.every((result) => {
    if (!isValidRagApiResult(result, candidateCount)) {
      return false;
    }
    if (seenIndices.has(result.index)) {
      return false;
    }
    seenIndices.add(result.index);
    return true;
  });
};

/** Bounds the whole rerank round trip, token acquisition included: the
 * supplier mints its token over the network, so awaiting it before axios
 * starts would leave the search unbounded whenever an auth service stalls.
 * The signal reaches both legs, so returning a fallback also cancels whatever
 * is still in flight rather than leaving a request running past its caller.
 * A non-positive timeout keeps axios' "no timeout" semantics. */
const withRerankDeadline = <T>(
  operation: (signal?: AbortSignal) => Promise<T>,
  timeout: number
): Promise<T> => {
  if (timeout <= 0) {
    return operation();
  }

  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error(`rag_api rerank exceeded its ${timeout}ms timeout.`));
    }, timeout);
  });

  const pending = operation(controller.signal);
  pending.catch(() => undefined);

  return Promise.race([pending, deadline]).finally(() => clearTimeout(timer));
};

/**
 * Calls the public `danny-avila/rag_api` `/v1/rerank` endpoint (the
 * `fast-v1` profile). Auth is supplied per call by a token supplier (a
 * short-lived JWT minted by the host app) rather than a static key. A
 * reranker failure must never throw into the search flow: on a missing
 * base URL/token supplier, a timeout (covering token acquisition as well as
 * the request), a non-2xx response, or any malformed payload, this falls back
 * to the candidates' original order via {@link BaseReranker.getDefaultRanking}.
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
    this.baseUrl = baseUrl?.replace(/\/+$/, '');
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

    const baseUrl = this.baseUrl;
    if (baseUrl == null || baseUrl === '') {
      this.logger.warn('RAG_API_URL is not set. Using default ranking.');
      return this.getDefaultRanking(documents, topK);
    }

    const tokenSupplier = this.tokenSupplier;
    if (tokenSupplier == null) {
      this.logger.warn(
        'No rag_api token supplier configured. Using default ranking.'
      );
      return this.getDefaultRanking(documents, topK);
    }

    const candidates = buildRagApiCandidates(documents, this.logger);
    const topN = clampRagApiTopN(Math.max(0, topK), this.logger);
    const requestData: t.RagApiRerankRequestBody = {
      profile: this.profile,
      query,
      candidates,
      top_n: topN,
    };

    try {
      const data = await withRerankDeadline(async (signal) => {
        const token = await tokenSupplier(signal);
        const response = await axios.post<t.RagApiRerankResponse | undefined>(
          `${baseUrl}/v1/rerank`,
          requestData,
          {
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
            timeout: this.timeout,
            httpAgent: this.httpAgent,
            httpsAgent: this.httpsAgent,
            signal,
          }
        );
        return response.data;
      }, this.timeout);

      this.logger.debug('rag_api rerank model:', data?.model);

      const rawResults = data?.results;
      if (!Array.isArray(rawResults) || rawResults.length === 0) {
        this.logger.warn(
          'Unexpected response format from rag_api rerank. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      if (!isValidRagApiBatch(rawResults, candidates.length)) {
        this.logger.warn(
          'rag_api rerank response contained no valid results. Using default ranking.'
        );
        return this.getDefaultRanking(documents, topK);
      }

      return sortRagApiResults(rawResults)
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
    voyageApiKey?: string;
    voyageApiUrl?: string;
    voyageModel?: string;
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
    voyageApiKey,
    voyageApiUrl,
    voyageModel,
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
  case 'voyage':
    return new VoyageReranker({
      apiKey: voyageApiKey,
      apiUrl: voyageApiUrl,
      model: voyageModel,
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
