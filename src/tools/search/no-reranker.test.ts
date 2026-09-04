/**
 * `rerankerType: 'none'` is a schema-valid opt-out, but it used to make the
 * web search useless: `createReranker` returns `undefined` for it, so
 * `getHighlights` returned `undefined`, so `expandHighlights` found a source
 * with content but no highlights and stripped the content. The model received
 * links and no text.
 *
 * The scraped chunks now pass through unscored via `getDefaultRanking` — the
 * same ranking every reranker already falls back to when it fails.
 */
import type * as t from './types';
import { createSourceProcessor } from './search';
import { expandHighlights } from './highlights';
import { createSearchMetrics } from './metrics';
import { BaseReranker, getDefaultRanking } from './rerankers';

const noopLog = (..._args: unknown[]): void => {};
const silentLogger = {
  error: noopLog,
  warn: noopLog,
  info: noopLog,
  debug: noopLog,
} as t.Logger;

const link = 'https://example.com/article';
const MARKER = 'THE ANSWER THE USER ASKED FOR IS FORTY-TWO.';
const FILLER = 'lorem ipsum dolor sit amet consectetur adipiscing elit\n';

/** Marker at the very top, inside the first few chunks. */
const makeContent = (): string => `${MARKER}\n${FILLER.repeat(40)}`;

/** Marker far past `topResults * chunkSize` — used to pin down the cap. */
const makeContentWithDeepMarker = (): string =>
  `${FILLER.repeat(40)}${MARKER}\n${FILLER.repeat(40)}`;

const createFakeScraper = (content: string): t.BaseScraper => ({
  scrapeUrl: async (url: string): Promise<[string, t.AnyScraperResponse]> => [
    url,
    { success: true, data: { markdown: content } },
  ],
  extractContent: (
    response: t.AnyScraperResponse
  ): [string, undefined | t.References] => [
    (response as t.FirecrawlScrapeResponse).data?.markdown ?? '',
    undefined,
  ],
  extractMetadata: (): t.GenericScrapeMetadata => ({}),
});

const makeOrganic = (l: string): t.ProcessedOrganic => ({
  link: l,
  title: `Title for ${l}`,
  snippet: `Snippet for ${l}`,
});

const runSearch = async (
  reranker: BaseReranker | undefined,
  content = makeContent()
): Promise<t.SearchResultData> => {
  const processor = createSourceProcessor(
    { reranker, topResults: 5, logger: silentLogger },
    createFakeScraper(content)
  );
  return processor.processSources({
    query: 'what is the answer',
    proMode: true,
    onGetHighlights: undefined,
    news: false,
    numElements: 5,
    result: { success: true, data: { organic: [makeOrganic(link)] } },
  });
};

/** Stands in for a configured reranker that returns nothing usable — the case
 * the stripping rule exists for, and which must keep stripping. */
class EmptyReranker extends BaseReranker {
  readonly provider = 'empty';
  constructor() {
    super(silentLogger);
  }
  async rerank(): Promise<t.Highlight[]> {
    return [];
  }
}

describe('getHighlights when no reranker is configured', () => {
  test('passes the scraped text through instead of dropping it', async () => {
    const data = await runSearch(undefined);
    const expanded = expandHighlights(data);
    const source = expanded.organic?.[0];

    expect(source?.highlights?.length).toBeGreaterThan(0);
    /** The point of the fix: the scraped text reaches the model. */
    expect(source?.highlights?.map((h) => h.text).join('\n')).toContain(MARKER);
  });

  test('still strips the raw content itself, only highlights leave', async () => {
    const data = await runSearch(undefined);
    const expanded = expandHighlights(data);

    expect(expanded.organic?.[0].content).toBeUndefined();
    expect(expanded.organic?.[0].references).toBeUndefined();
  });

  test('keeps stripping when a reranker is configured but yields nothing', async () => {
    const data = await runSearch(new EmptyReranker());
    const expanded = expandHighlights(data);

    /** The guarantee the rule protects is unchanged for every other case. */
    expect(expanded.organic?.[0].content).toBeUndefined();
    expect(expanded.organic?.[0].highlights ?? []).toHaveLength(0);
  });

  /** ⚠️ Pass-through is not the whole page. `getDefaultRanking` keeps the
   * first `topK` candidates, so a source contributes about
   * `topResults * chunkSize` characters — with the defaults (5 x 150) roughly
   * 750, however long the article is. That is the same amount a failing
   * reranker delivers today through its fallback, and it is deliberate: the
   * chunk budget is what keeps a search from flooding the context. Anything
   * beyond it needs a reranker to decide *which* chunks are worth sending. */
  test('delivers the first chunks only, not the whole document', async () => {
    const data = await runSearch(undefined, makeContentWithDeepMarker());
    const expanded = expandHighlights(data);
    const delivered = expanded.organic?.[0].highlights
      ?.map((h) => h.text)
      .join('\n');

    expect(delivered).toBeDefined();
    expect(delivered).not.toContain(MARKER);
  });

  test('honors topResults, like every reranker fallback does', async () => {
    const processor = createSourceProcessor(
      { reranker: undefined, topResults: 2, logger: silentLogger },
      createFakeScraper(makeContent())
    );
    const data = await processor.processSources({
      query: 'what is the answer',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
    });

    expect(data.organic?.[0].highlights?.length).toBeLessThanOrEqual(2);
  });

  test('records the pass-through in metrics without marking it a failure', async () => {
    const recorded: t.RerankObservation[] = [];
    const metrics = createSearchMetrics(silentLogger);
    const original = metrics.recordRerank.bind(metrics);
    metrics.recordRerank = (observation: t.RerankObservation): void => {
      recorded.push(observation);
      original(observation);
    };

    const processor = createSourceProcessor(
      { reranker: undefined, topResults: 5, logger: silentLogger },
      createFakeScraper(makeContent())
    );
    await processor.processSources({
      query: 'what is the answer',
      proMode: true,
      onGetHighlights: undefined,
      news: false,
      numElements: 5,
      result: { success: true, data: { organic: [makeOrganic(link)] } },
      metrics,
    });

    const passThrough = recorded.find((o) => o.provider === 'none');
    expect(passThrough).toBeDefined();
    /** Nothing failed — a `reason` here would show up as a fallback in the
     * search summary and hide real reranker problems. */
    expect(passThrough?.reason).toBeUndefined();
    expect(passThrough?.results).toBeGreaterThan(0);
  });
});

describe('getDefaultRanking', () => {
  test('keeps candidate order and caps at topK', () => {
    const docs = ['a', 'b', 'c', 'd'];
    expect(getDefaultRanking(docs, 2)).toEqual([
      { text: 'a', score: 0 },
      { text: 'b', score: 0 },
    ]);
  });

  test('never returns more than it was given', () => {
    expect(getDefaultRanking(['a'], 10)).toHaveLength(1);
  });
});
