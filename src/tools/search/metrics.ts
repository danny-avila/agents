import type * as t from './types';

/** Distinct failure reasons kept per phase before the tail folds into
 * `other`: enough to characterize a mixed failure without letting a
 * pathological run grow one map entry per source. */
const MAX_REASON_KEYS = 6;
/** Failing hosts named in a summary before the rest are counted only. */
const MAX_FAILURE_SAMPLES = 3;
/** Cap on the raw provider message carried alongside a summary. */
const MAX_DETAIL_LENGTH = 200;
const OTHER_REASON = 'other';

const HTTP_STATUS_REGEX = /\b(?:status(?:\scode)?|http)\D{0,3}(\d{3})\b/i;

/** Free-form provider messages have unbounded cardinality; the aggregate only
 * keeps a normalized class so the reason map stays small and comparable. */
const REASON_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  [/ECONNABORTED|ETIMEDOUT|timeout|timed out/i, 'timeout'],
  [/ENOTFOUND|EAI_AGAIN|getaddrinfo/i, 'dns'],
  [/ECONNREFUSED|ECONNRESET|EPIPE|socket hang up/i, 'connection'],
  [/CERT_|SSL|TLS|self[- ]signed/i, 'tls'],
  [/abort|cancel/i, 'aborted'],
];

export const classifyFailure = (message?: string): string => {
  if (message == null || message === '') {
    return 'unknown';
  }
  const status = HTTP_STATUS_REGEX.exec(message);
  if (status != null) {
    return `http_${status[1]}`;
  }
  for (const [pattern, reason] of REASON_PATTERNS) {
    if (pattern.test(message)) {
      return reason;
    }
  }
  return OTHER_REASON;
};

const bump = (counts: Map<string, number>, key: string): void => {
  const existing = counts.get(key);
  if (existing != null) {
    counts.set(key, existing + 1);
    return;
  }
  if (counts.size >= MAX_REASON_KEYS) {
    counts.set(OTHER_REASON, (counts.get(OTHER_REASON) ?? 0) + 1);
    return;
  }
  counts.set(key, 1);
};

const hostOf = (url: string): string => {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url.slice(0, 60);
  }
};

const formatCount = (value: number): string =>
  value < 10000 ? String(value) : `${(value / 1000).toFixed(1)}k`;

const formatMs = (ms: number): string =>
  ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;

const formatMap = (counts: Map<string, number>): string => {
  let out = '';
  for (const [key, count] of counts) {
    out += `${out === '' ? '' : ','}${key}:${count}`;
  }
  return `{${out}}`;
};

const truncateDetail = (detail: string): string =>
  detail.length <= MAX_DETAIL_LENGTH
    ? detail
    : `${detail.slice(0, MAX_DETAIL_LENGTH)}…`;

interface SearchPhase {
  provider: string;
  queries: number;
  failed: number;
  durationMs: number;
  types: Map<string, number>;
  reasons: Map<string, number>;
}

interface ScrapePhase {
  links: number;
  ok: number;
  empty: number;
  chars: number;
  highlights: number;
  failed: number;
  reasons: Map<string, number>;
  samples: string[];
  detail?: string;
}

interface RerankPhase {
  provider: string;
  model?: string;
  calls: number;
  fallbacks: number;
  errors: number;
  chunks: number;
  maxChunks: number;
  dropped: number;
  results: number;
  units: number;
  durationMs: number;
  maxDurationMs: number;
  reasons: Map<string, number>;
  error?: t.SafeErrorLog;
}

/**
 * Creates the counter set for one `web_search` call.
 *
 * The pipeline reranks once per scraped source and scrapes once per link, so
 * logging at those points scales the log with the result count. Every phase
 * records into fixed-width counters instead and {@link t.SearchMetrics.flush}
 * emits one line per phase, at the highest severity that phase reached.
 *
 * @param autoFlush emit on every record, for a caller with no enclosing search
 * to fold into (a reranker used directly). Recording is synchronous, so the
 * record/flush/reset cycle can never interleave with a concurrent call.
 */
export const createSearchMetrics = (
  logger: t.Logger,
  autoFlush = false
): t.SearchMetrics => {
  let search: SearchPhase | undefined;
  let scrape: ScrapePhase | undefined;
  let rerank: RerankPhase | undefined;

  const flushSearch = (phase: SearchPhase): void => {
    let line = `[web_search] search=${phase.provider} queries=${phase.queries}`;
    if (phase.types.size > 0) {
      line += ` results=${formatMap(phase.types)}`;
    }
    line += ` dur=${formatMs(phase.durationMs)}`;
    if (phase.failed === 0) {
      logger.debug(line);
      return;
    }
    logger.warn(
      `${line} failed=${phase.failed} reasons=${formatMap(phase.reasons)}`
    );
  };

  const flushScrape = (phase: ScrapePhase): void => {
    let line =
      `[web_search] scrape links=${phase.links} ok=${phase.ok}` +
      ` chars=${formatCount(phase.chars)} highlights=${phase.highlights}`;
    if (phase.empty > 0) {
      line += ` empty=${phase.empty}`;
    }
    if (phase.failed === 0) {
      logger.debug(line);
      return;
    }
    line += ` failed=${phase.failed} reasons=${formatMap(phase.reasons)}`;
    if (phase.samples.length > 0) {
      line += ` sample=${phase.samples.join(',')}`;
    }
    if (phase.detail == null) {
      logger.error(line);
      return;
    }
    logger.error(line, phase.detail);
  };

  const flushRerank = (phase: RerankPhase): void => {
    let line =
      `[web_search] rerank=${phase.provider} calls=${phase.calls}` +
      ` chunks=${phase.chunks} maxChunks=${phase.maxChunks}` +
      ` results=${phase.results}` +
      ` dur=${formatMs(phase.durationMs)}` +
      ` maxDur=${formatMs(phase.maxDurationMs)}`;
    if (phase.model != null) {
      line += ` model=${phase.model}`;
    }
    if (phase.units > 0) {
      line += ` units=${phase.units}`;
    }
    if (phase.dropped > 0) {
      line += ` dropped=${phase.dropped}`;
    }
    if (phase.fallbacks === 0) {
      logger.debug(line);
      return;
    }
    line += ` fallbacks=${phase.fallbacks} reasons=${formatMap(phase.reasons)}`;
    if (phase.errors === 0) {
      logger.warn(line);
      return;
    }
    if (phase.error == null) {
      logger.error(line);
      return;
    }
    logger.error(line, phase.error);
  };

  const flush = (): void => {
    if (search != null) {
      flushSearch(search);
      search = undefined;
    }
    if (scrape != null) {
      flushScrape(scrape);
      scrape = undefined;
    }
    if (rerank != null) {
      flushRerank(rerank);
      rerank = undefined;
    }
  };

  const recordSearch = (observation: t.SearchObservation): void => {
    search ??= {
      provider: observation.provider,
      queries: 0,
      failed: 0,
      durationMs: 0,
      types: new Map(),
      reasons: new Map(),
    };
    search.queries += 1;
    /** Sub-searches run concurrently, so the phase lasts as long as its
     * slowest query rather than the sum of all of them. */
    search.durationMs = Math.max(search.durationMs, observation.durationMs);
    if (observation.error != null) {
      search.failed += 1;
      bump(search.reasons, `${observation.type}:${observation.error}`);
    } else if (observation.results > 0) {
      const seen = search.types.get(observation.type) ?? 0;
      search.types.set(observation.type, seen + observation.results);
    }
    if (autoFlush) {
      flush();
    }
  };

  const recordScrape = (observation: t.ScrapeObservation): void => {
    scrape ??= {
      links: 0,
      ok: 0,
      empty: 0,
      chars: 0,
      highlights: 0,
      failed: 0,
      reasons: new Map(),
      samples: [],
      detail: undefined,
    };
    scrape.links += 1;
    if (observation.error != null) {
      scrape.failed += 1;
      const reason = classifyFailure(observation.error);
      bump(scrape.reasons, reason);
      if (scrape.samples.length < MAX_FAILURE_SAMPLES) {
        scrape.samples.push(`${hostOf(observation.url)}:${reason}`);
      }
      scrape.detail ??= truncateDetail(observation.error);
    } else {
      scrape.ok += 1;
      const chars = observation.chars ?? 0;
      scrape.chars += chars;
      scrape.highlights += observation.highlights ?? 0;
      if (chars === 0) {
        scrape.empty += 1;
      }
    }
    if (autoFlush) {
      flush();
    }
  };

  const recordRerank = (observation: t.RerankObservation): void => {
    rerank ??= {
      provider: observation.provider,
      calls: 0,
      fallbacks: 0,
      errors: 0,
      chunks: 0,
      maxChunks: 0,
      dropped: 0,
      results: 0,
      units: 0,
      durationMs: 0,
      maxDurationMs: 0,
      reasons: new Map(),
    };
    rerank.calls += 1;
    rerank.chunks += observation.chunks;
    rerank.maxChunks = Math.max(rerank.maxChunks, observation.chunks);
    rerank.results += observation.results;
    rerank.dropped += observation.dropped ?? 0;
    rerank.units += observation.units ?? 0;
    rerank.durationMs += observation.durationMs;
    rerank.maxDurationMs = Math.max(
      rerank.maxDurationMs,
      observation.durationMs
    );
    rerank.model ??= observation.model;
    if (observation.reason != null) {
      rerank.fallbacks += 1;
      bump(rerank.reasons, observation.reason);
    }
    if (observation.error != null) {
      rerank.errors += 1;
      rerank.error ??= observation.error;
    }
    if (autoFlush) {
      flush();
    }
  };

  return { recordSearch, recordScrape, recordRerank, flush };
};
