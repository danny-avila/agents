import type * as t from './types';
import { createSearchMetrics, classifyFailure } from './metrics';

const createMockLogger = (): t.Logger =>
  ({
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn(),
    debug: jest.fn(),
  }) as unknown as t.Logger;

const lineOf = (fn: unknown): string =>
  String((fn as jest.Mock).mock.calls[0][0]);

describe('classifyFailure', () => {
  it.each([
    ['Request failed with status code 403', 'http_403'],
    ['fastCRW API request failed: timeout of 7500ms exceeded', 'timeout'],
    ['connect ECONNREFUSED 127.0.0.1:8080', 'connection'],
    ['getaddrinfo ENOTFOUND example.com', 'dns'],
    ['unable to verify the first certificate SSL', 'tls'],
    ['canceled', 'aborted'],
    ['something entirely novel', 'other'],
    ['', 'unknown'],
    [undefined, 'unknown'],
  ])('maps %p to %p', (message, expected) => {
    expect(classifyFailure(message)).toBe(expected);
  });
});

describe('createSearchMetrics', () => {
  it('emits nothing until flushed, then one line per phase that ran', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordSearch({
      provider: 'serper',
      type: 'web',
      results: 8,
      durationMs: 120,
    });
    metrics.recordScrape({ url: 'https://a.com', chars: 400, highlights: 5 });
    metrics.recordRerank({
      provider: 'cohere',
      chunks: 40,
      results: 5,
      durationMs: 90,
    });

    expect(logger.debug).not.toHaveBeenCalled();

    metrics.flush();

    expect(logger.debug).toHaveBeenCalledTimes(3);
    expect(logger.warn).not.toHaveBeenCalled();
    expect(logger.error).not.toHaveBeenCalled();
  });

  it('folds every rerank call into a single summary', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    for (let i = 0; i < 8; i++) {
      metrics.recordRerank({
        provider: 'cohere',
        model: 'rerank-v3.5',
        chunks: 10 + i,
        results: 5,
        units: 1,
        durationMs: 100 + i,
      });
    }
    metrics.flush();

    expect(logger.debug).toHaveBeenCalledTimes(1);
    const line = lineOf(logger.debug);
    expect(line).toContain('rerank=cohere');
    expect(line).toContain('calls=8');
    expect(line).toContain('chunks=108 maxChunks=17');
    expect(line).toContain('results=40');
    expect(line).toContain('model="rerank-v3.5"');
    expect(line).toContain('units=8');
    expect(line).toContain('maxDur=107ms');
  });

  it('reports rerank fallbacks at warn and rerank errors at error', () => {
    const warnLogger = createMockLogger();
    const warned = createSearchMetrics(warnLogger);
    warned.recordRerank({
      provider: 'cohere',
      chunks: 4,
      results: 4,
      durationMs: 1,
      reason: 'no_api_key',
    });
    warned.flush();
    expect(warnLogger.debug).not.toHaveBeenCalled();
    expect(lineOf(warnLogger.warn)).toContain(
      'fallbacks=1 reasons={no_api_key:1}'
    );

    const errorLogger = createMockLogger();
    const errored = createSearchMetrics(errorLogger);
    errored.recordRerank({
      provider: 'jina',
      chunks: 4,
      results: 4,
      durationMs: 1,
      reason: 'error',
      error: { message: 'boom', status: 500 },
    });
    errored.flush();
    expect(errorLogger.warn).not.toHaveBeenCalled();
    expect(errorLogger.error).toHaveBeenCalledWith(
      expect.stringContaining('fallbacks=1 reasons={error:1}'),
      expect.objectContaining({ status: 500 })
    );
  });

  it('classifies scrape failures and names a bounded sample of hosts', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordScrape({ url: 'https://ok.com', chars: 900, highlights: 3 });
    metrics.recordScrape({ url: 'https://empty.com', chars: 0 });
    for (let i = 0; i < 6; i++) {
      metrics.recordScrape({
        url: `https://fail${i}.com/page`,
        error: 'Request failed with status code 403',
      });
    }
    metrics.flush();

    const [line, detail] = (logger.error as jest.Mock).mock.calls[0];
    expect(line).toContain('links=8 ok=2');
    expect(line).toContain('empty=1');
    expect(line).toContain('failed=6 reasons={http_403:6}');
    expect(line).toContain('sample=fail0.com:http_403');
    expect(line.match(/http_403/g)).toHaveLength(4);
    expect(detail).toBe('Request failed with status code 403');
  });

  it('bounds the reason map when failures never repeat a class', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    for (let i = 0; i < 40; i++) {
      metrics.recordScrape({
        url: `https://fail${i}.com`,
        error: `Request failed with status code ${400 + i}`,
      });
    }
    metrics.flush();

    const line = lineOf(logger.error);
    const reasons = /\{(.*?)\}/.exec(line)?.[1].split(',') ?? [];
    expect(reasons.length).toBeLessThanOrEqual(7);
    expect(line).toContain('other:');
    expect(line).toContain('failed=40');
  });

  it('takes the search phase duration from its slowest concurrent query', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordSearch({
      provider: 'serper',
      type: 'web',
      results: 8,
      durationMs: 300,
    });
    metrics.recordSearch({
      provider: 'serper',
      type: 'images',
      results: 10,
      durationMs: 900,
    });
    metrics.recordSearch({
      provider: 'serper',
      type: 'news',
      results: 0,
      durationMs: 200,
      error: 'Serper API request failed: timeout of 10000ms exceeded',
    });
    metrics.flush();

    const line = lineOf(logger.warn);
    expect(line).toContain('search=serper queries=3');
    expect(line).toContain('results={web:8,images:10}');
    expect(line).toContain('dur=900ms');
    /** Provider prose is classified, not copied into the summary verbatim. */
    expect(line).toContain('failed=1 reasons={news:timeout:1}');
    expect(line).not.toContain('Serper API request failed');
  });

  it('collapses equivalent search failures into one classified reason', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    for (const message of [
      'Images search failed: timeout of 10000ms exceeded',
      'Images search failed: ETIMEDOUT connecting to provider',
    ]) {
      metrics.recordSearch({
        provider: 'serper',
        type: 'images',
        results: 0,
        durationMs: 10,
        error: message,
      });
    }
    metrics.flush();

    expect(lineOf(logger.warn)).toContain('failed=2 reasons={images:timeout:2}');
  });

  it('raises the search phase to error only when a query rejected', () => {
    const soft = createMockLogger();
    const softMetrics = createSearchMetrics(soft);
    softMetrics.recordSearch({
      provider: 'serper',
      type: 'news',
      results: 0,
      durationMs: 5,
      error: 'Request failed with status code 429',
    });
    softMetrics.flush();
    expect(soft.error).not.toHaveBeenCalled();
    expect(lineOf(soft.warn)).toContain('failed=1 reasons={news:http_429:1}');

    const thrown = createMockLogger();
    const thrownMetrics = createSearchMetrics(thrown);
    thrownMetrics.recordSearch({
      provider: 'serper',
      type: 'images',
      results: 0,
      durationMs: 5,
      error: 'socket hang up',
      thrown: true,
    });
    thrownMetrics.flush();
    expect(thrown.warn).not.toHaveBeenCalled();
    expect(thrown.error).toHaveBeenCalledWith(
      expect.stringContaining('failed=1 reasons={images:connection:1}'),
      'socket hang up'
    );
  });

  it('strips control characters from a provider-supplied label', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordRerank({
      provider: 'jina',
      // A forged second entry, were the newline to survive into the message.
      model: 'evil\n[web_search] scrape links=0 ok=0 chars=0 highlights=0',
      chunks: 4,
      results: 4,
      durationMs: 1,
    });
    metrics.flush();

    /** The line it could have forged cannot exist: the record stays one
     * physical line, and the label is bounded within it. */
    const line = lineOf(logger.debug);
    expect(line.split(/\r?\n|[\u2028\u2029]/)).toHaveLength(1);
    /** Whatever survives truncation stays inside the delimited field. */
    expect(/ model="[^"]*"$/.test(line)).toBe(true);
  });

  it('attaches the thrown query detail, not an earlier soft failure', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordSearch({
      provider: 'serper',
      type: 'news',
      results: 0,
      durationMs: 5,
      error: 'Request failed with status code 429',
    });
    metrics.recordSearch({
      provider: 'serper',
      type: 'images',
      results: 0,
      durationMs: 5,
      error: 'socket hang up',
      thrown: true,
    });
    metrics.flush();

    /** The exception raised the severity, so its message is the one worth
     * carrying — not whichever failure happened to arrive first. */
    const [, detail] = (logger.error as jest.Mock).mock.calls[0];
    expect(detail).toBe('socket hang up');
  });

  it('labels a phase mixed rather than mislabeling it as the first provider', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordRerank({
      provider: 'chunker',
      chunks: 0,
      results: 0,
      durationMs: 1,
    });
    metrics.recordRerank({
      provider: 'jina',
      chunks: 10,
      results: 5,
      durationMs: 1,
    });
    metrics.flush();

    expect(lineOf(logger.debug)).toContain('rerank=mixed calls=2');
  });

  it('keeps a placeholder reranker at debug, and real fallbacks at warn', () => {
    const placeholder = createMockLogger();
    const placeholderMetrics = createSearchMetrics(placeholder);
    placeholderMetrics.recordRerank({
      provider: 'infinity',
      chunks: 4,
      results: 4,
      durationMs: 1,
      reason: 'placeholder',
    });
    placeholderMetrics.flush();
    /** `infinity` is a supported setting that returns input order by design;
     * choosing it must not read as a degradation. */
    expect(placeholder.warn).not.toHaveBeenCalled();
    expect(lineOf(placeholder.debug)).toContain('reasons={placeholder:1}');

    const degraded = createMockLogger();
    const degradedMetrics = createSearchMetrics(degraded);
    degradedMetrics.recordRerank({
      provider: 'infinity',
      chunks: 4,
      results: 4,
      durationMs: 1,
      reason: 'placeholder',
    });
    degradedMetrics.recordRerank({
      provider: 'infinity',
      chunks: 4,
      results: 4,
      durationMs: 1,
      reason: 'no_api_key',
    });
    degradedMetrics.flush();
    expect(degraded.debug).not.toHaveBeenCalled();
    expect(lineOf(degraded.warn)).toContain('fallbacks=2');
  });

  it('marks the model mixed rather than crediting the first one seen', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordRerank({
      provider: 'rag-api',
      model: 'fast-v1-a',
      chunks: 4,
      results: 4,
      durationMs: 1,
    });
    metrics.recordRerank({
      provider: 'rag-api',
      model: 'fast-v1-b',
      chunks: 4,
      results: 4,
      durationMs: 1,
    });
    metrics.flush();

    expect(lineOf(logger.debug)).toContain('model="mixed"');
  });

  it('reports a clamped result count so short highlights have an explanation', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordRerank({
      provider: 'rag-api',
      chunks: 40,
      results: 25,
      durationMs: 1,
      topK: 40,
      topKLimit: 25,
    });
    metrics.flush();

    expect(lineOf(logger.debug)).toContain('topK=40 topKLimit=25');
  });

  it('bounds a provider-supplied model name inside the summary line', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordRerank({
      provider: 'jina',
      model: 'm'.repeat(500),
      chunks: 4,
      results: 4,
      durationMs: 1,
    });
    metrics.flush();

    const line = lineOf(logger.debug);
    expect(line).toContain('model="');
    expect(line.length).toBeLessThan(200);
  });

  it('resets after a flush so a reused collector never double-counts', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger);

    metrics.recordScrape({ url: 'https://a.com', chars: 10, highlights: 1 });
    metrics.flush();
    metrics.flush();
    metrics.recordScrape({ url: 'https://b.com', chars: 20, highlights: 2 });
    metrics.flush();

    expect(logger.debug).toHaveBeenCalledTimes(2);
    expect(lineOf(logger.debug)).toContain('links=1 ok=1 chars=10');
    expect(String((logger.debug as jest.Mock).mock.calls[1][0])).toContain(
      'links=1 ok=1 chars=20'
    );
  });

  it('emits per record when auto-flushing for a caller with no run scope', () => {
    const logger = createMockLogger();
    const metrics = createSearchMetrics(logger, true);

    metrics.recordRerank({
      provider: 'jina',
      chunks: 4,
      results: 4,
      durationMs: 1,
    });
    metrics.recordRerank({
      provider: 'jina',
      chunks: 6,
      results: 6,
      durationMs: 1,
    });

    expect(logger.debug).toHaveBeenCalledTimes(2);
    expect(lineOf(logger.debug)).toContain('calls=1 chunks=4 maxChunks=4');
  });
});
