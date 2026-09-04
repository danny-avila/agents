import { describe, it, expect } from '@jest/globals';
import type * as t from './types';
import { normalizeSearchResultData, resolveSearchOutcome } from './tool';

const data = (partial: Partial<t.SearchResultData>): t.SearchResultData =>
  ({ turn: 0, ...partial }) as t.SearchResultData;

describe('resolveSearchOutcome', () => {
  test.each([null, undefined])('normalizes %p result data as a failure', (result) => {
    expect(resolveSearchOutcome(normalizeSearchResultData(result), 'oauth')).toBe(
      'Search failed for "oauth"'
    );
  });

  it('authors a FAILURE label when the processor caught an error', () => {
    expect(
      resolveSearchOutcome(
        data({ error: 'provider timeout', organic: [] }),
        'oauth handling'
      )
    ).toBe('Search failed for "oauth handling"');
  });

  it('prefers the failure label over any partial results', () => {
    expect(
      resolveSearchOutcome(
        data({
          error: 'partial failure',
          organic: [{ link: 'a' }] as t.SearchResultData['organic'],
        }),
        'oauth'
      )
    ).toBe('Search failed for "oauth"');
  });

  it('ignores an empty-string error', () => {
    expect(
      resolveSearchOutcome(
        data({
          error: '',
          organic: [{ link: 'a' }] as t.SearchResultData['organic'],
        }),
        'oauth'
      )
    ).toBe('Found 1 result for "oauth"');
  });

  it('counts every rendered collection kind', () => {
    expect(
      resolveSearchOutcome(
        data({
          organic: [
            { link: 'a' },
            { link: 'b' },
          ] as t.SearchResultData['organic'],
          topStories: [{ link: 'c' }] as t.SearchResultData['topStories'],
          images: [{ imageUrl: 'd' }] as t.SearchResultData['images'],
          videos: [{ link: 'e' }] as t.SearchResultData['videos'],
          places: [{ name: 'f' }] as t.SearchResultData['places'],
          peopleAlsoAsk: [
            { question: 'g' },
          ] as t.SearchResultData['peopleAlsoAsk'],
        }),
        'oauth'
      )
    ).toBe('Found 7 results for "oauth"');
  });

  it('counts singleton structured results (knowledge graph / answer box)', () => {
    expect(
      resolveSearchOutcome(
        data({
          knowledgeGraph: {
            title: 'OAuth',
          } as t.SearchResultData['knowledgeGraph'],
        }),
        'oauth'
      )
    ).toBe('Found 1 result for "oauth"');
    expect(
      resolveSearchOutcome(
        data({
          answerBox: { snippet: 'x' } as t.SearchResultData['answerBox'],
          knowledgeGraph: {
            title: 'OAuth',
          } as t.SearchResultData['knowledgeGraph'],
        }),
        'oauth'
      )
    ).toBe('Found 2 results for "oauth"');
  });

  it('leaves a genuine zero-result search unlabeled, so the intent stands', () => {
    expect(resolveSearchOutcome(data({ organic: [] }), 'oauth')).toBeUndefined();
  });
});
