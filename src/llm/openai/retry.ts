import {
  AsyncCaller,
  parseRetryAfterMs,
} from '@langchain/core/utils/async_caller';

const MAX_RETRY_AFTER_MS = 60_000;
const NETWORK_ERROR_CODES = new Set([
  'EAI_AGAIN',
  'ECONNABORTED',
  'ECONNREFUSED',
  'ECONNRESET',
  'EHOSTUNREACH',
  'ENETUNREACH',
  'ENOTFOUND',
  'ETIMEDOUT',
  'UND_ERR_CONNECT_TIMEOUT',
  'UND_ERR_SOCKET',
]);
const NETWORK_ERROR_NAMES = new Set([
  'APIConnectionError',
  'APIConnectionTimeoutError',
]);

type RetryError = Error & {
  code?: string;
  status?: number;
  statusCode?: number;
  retryAfterMs?: number;
  headers?: Headers | Record<string, string | undefined>;
  response?: {
    status?: number;
    headers?: Headers | Record<string, string | undefined>;
  };
  error?: { code?: string };
};

function getStatus(error: RetryError): number | undefined {
  return error.status ?? error.statusCode ?? error.response?.status;
}

function getHeader(
  headers: RetryError['headers'],
  name: string
): string | undefined {
  if (headers == null) {
    return undefined;
  }
  if ('get' in headers && typeof headers.get === 'function') {
    return headers.get(name) ?? undefined;
  }
  const record = headers as Record<string, string | undefined>;
  return record[name] ?? record[name.toLowerCase()];
}

function getRetryAfterMs(error: RetryError): number | undefined {
  const headers = error.headers ?? error.response?.headers;
  const value = getHeader(headers, 'Retry-After');
  return value == null ? undefined : parseRetryAfterMs(value);
}

function isNetworkError(error: RetryError): boolean {
  return (
    (error.code != null && NETWORK_ERROR_CODES.has(error.code)) ||
    NETWORK_ERROR_NAMES.has(error.name) ||
    (error instanceof TypeError && error.message === 'fetch failed')
  );
}

export function handleOpenAIFailedAttempt(error: Error): void {
  const providerError = error as RetryError;
  if (providerError.name === 'AbortError') {
    throw error;
  }

  const status = getStatus(providerError);
  const code = providerError.code ?? providerError.error?.code;
  const retryableStatus =
    status === 408 || status === 429 || (status != null && status >= 500);
  if (
    (status == null && !isNetworkError(providerError)) ||
    (status != null && !retryableStatus) ||
    code === 'insufficient_quota'
  ) {
    throw error;
  }

  const retryAfterMs = getRetryAfterMs(providerError);
  if (retryAfterMs != null) {
    if (retryAfterMs > MAX_RETRY_AFTER_MS) {
      throw error;
    }
    providerError.retryAfterMs = retryAfterMs;
  }
}

export function createOpenAIRetryCaller(fields?: {
  maxConcurrency?: number;
  maxRetries?: number;
}): AsyncCaller {
  return new AsyncCaller({
    maxConcurrency: fields?.maxConcurrency,
    maxRetries: fields?.maxRetries,
    onFailedAttempt: handleOpenAIFailedAttempt,
  });
}
