import type * as t from './types';
import { formatResultsForLLM, resolveMaxLLMOutputChars } from './format';

const makeOrganic = (
  link: string,
  highlights: t.Highlight[]
): t.ProcessedOrganic => ({
  link,
  title: `Title for ${link}`,
  snippet: `Snippet for ${link}`,
  highlights,
});

const highlight = (text: string, score = 0.9): t.Highlight => ({ text, score });

const reference = (url: string, originalIndex = 0): t.UsedReferences[number] => ({
  type: 'link',
  originalIndex,
  reference: { originalUrl: url, title: 'Ref', text: 'ref' },
});

const countHighlightBlocks = (output: string): number =>
  (output.match(/### Highlight \d+/g) ?? []).length;

const OMISSION_MARKER = 'omitted to fit the context budget';

describe('resolveMaxLLMOutputChars', () => {
  const originalEnv = process.env.SEARCH_MAX_LLM_OUTPUT_CHARS;

  afterEach(() => {
    if (originalEnv == null) {
      delete process.env.SEARCH_MAX_LLM_OUTPUT_CHARS;
    } else {
      process.env.SEARCH_MAX_LLM_OUTPUT_CHARS = originalEnv;
    }
  });

  test('falls back to the 50,000 char default when nothing is configured', () => {
    delete process.env.SEARCH_MAX_LLM_OUTPUT_CHARS;
    expect(resolveMaxLLMOutputChars()).toBe(50000);
    expect(resolveMaxLLMOutputChars(0)).toBe(50000);
    expect(resolveMaxLLMOutputChars(-100)).toBe(50000);
  });

  test('honors the SEARCH_MAX_LLM_OUTPUT_CHARS env var', () => {
    process.env.SEARCH_MAX_LLM_OUTPUT_CHARS = '777';
    expect(resolveMaxLLMOutputChars()).toBe(777);
    expect(resolveMaxLLMOutputChars(0)).toBe(777);
  });

  test('an explicit positive config value wins over env and default', () => {
    process.env.SEARCH_MAX_LLM_OUTPUT_CHARS = '777';
    expect(resolveMaxLLMOutputChars(1234)).toBe(1234);
  });

  test('ignores a non-numeric env var', () => {
    process.env.SEARCH_MAX_LLM_OUTPUT_CHARS = 'not-a-number';
    expect(resolveMaxLLMOutputChars()).toBe(50000);
  });
});

describe('formatResultsForLLM highlight budget', () => {
  test('keeps whole highlights in relevance order until the budget is hit', () => {
    const results: t.SearchResultData = {
      organic: [
        makeOrganic('https://a.com', [highlight('A'.repeat(100))]),
        makeOrganic('https://b.com', [highlight('B'.repeat(100))]),
      ],
    };

    const { output } = formatResultsForLLM(0, results, 100);

    expect(output).toContain('A'.repeat(100));
    expect(output).not.toContain('B'.repeat(100));
    expect(countHighlightBlocks(output)).toBe(1);
    expect(output).toContain('_[1 additional highlight omitted to fit the context budget');
  });

  test('truncates the boundary highlight when meaningful room remains', () => {
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [highlight('A'.repeat(1000))])],
    };

    const { output } = formatResultsForLLM(0, results, 500);

    expect(output).toContain('…[truncated]');
    expect(output).toContain('A'.repeat(500));
    expect(output).not.toContain('A'.repeat(501));
    expect(output).toContain('_[1 additional highlight omitted to fit the context budget');
  });

  test('drops the boundary highlight entirely when too little room remains', () => {
    const results: t.SearchResultData = {
      organic: [
        makeOrganic('https://a.com', [highlight('A'.repeat(100))]),
        makeOrganic('https://b.com', [highlight('B'.repeat(100))]),
      ],
    };

    const { output } = formatResultsForLLM(0, results, 150);

    expect(output).toContain('A'.repeat(100));
    expect(output).not.toContain('…[truncated]');
    expect(output).not.toContain('B');
    expect(countHighlightBlocks(output)).toBe(1);
  });

  test('always keeps snippets, titles, and URLs even when all highlights are dropped', () => {
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [highlight('A'.repeat(100))])],
    };

    const { output } = formatResultsForLLM(0, results, 10);

    expect(output).toContain('URL: https://a.com');
    expect(output).toContain('Summary: Snippet for https://a.com');
    expect(output).toContain('"Title for https://a.com"');
    expect(countHighlightBlocks(output)).toBe(0);
    expect(output).toContain('_[1 additional highlight omitted to fit the context budget');
  });

  test('emits no omission marker when every highlight fits the budget', () => {
    const results: t.SearchResultData = {
      organic: [
        makeOrganic('https://a.com', [highlight('A'.repeat(100))]),
        makeOrganic('https://b.com', [highlight('B'.repeat(100))]),
      ],
    };

    const { output } = formatResultsForLLM(0, results, 50000);

    expect(output).toContain('A'.repeat(100));
    expect(output).toContain('B'.repeat(100));
    expect(countHighlightBlocks(output)).toBe(2);
    expect(output).not.toContain(OMISSION_MARKER);
  });

  test('drops references with no surviving marker when truncating', () => {
    const withRefs = highlight('A'.repeat(1000));
    withRefs.references = [reference('https://cited.example')];
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [withRefs])],
    };

    const { output, references } = formatResultsForLLM(0, results, 500);

    expect(output).toContain('…[truncated]');
    expect(output).not.toContain('Core References');
    expect(output).not.toContain('https://cited.example');
    expect(references).toHaveLength(0);
  });

  test('keeps references whose marker survives truncation and drops the rest', () => {
    const withRefs = highlight(`(link#1) ${'A'.repeat(1000)} (link#2)`);
    withRefs.references = [
      reference('https://one.example', 0),
      reference('https://two.example', 1),
    ];
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [withRefs])],
    };

    const { output, references } = formatResultsForLLM(0, results, 500);

    expect(output).toContain('…[truncated]');
    expect(output).toContain('https://one.example');
    expect(output).not.toContain('https://two.example');
    expect(references).toHaveLength(1);
    expect(references[0].link).toBe('https://one.example');
  });

  test('stops at the boundary highlight — no lower-ranked highlight slips in', () => {
    const results: t.SearchResultData = {
      organic: [
        makeOrganic('https://a.com', [
          highlight('A'.repeat(100), 0.9),
          highlight('B'.repeat(300), 0.8),
          highlight('C'.repeat(10), 0.7),
        ]),
      ],
    };

    const { output } = formatResultsForLLM(0, results, 150);

    expect(output).toContain('A'.repeat(100));
    expect(output).not.toContain('B'.repeat(300));
    expect(output).not.toContain('C'.repeat(10));
    expect(output).not.toContain('…[truncated]');
    expect(countHighlightBlocks(output)).toBe(1);
  });

  test('keeps references on a whole highlight that fits the budget', () => {
    const withRefs = highlight('A'.repeat(100));
    withRefs.references = [reference('https://cited.example')];
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [withRefs])],
    };

    const { output, references } = formatResultsForLLM(0, results, 50000);

    expect(output).toContain('Core References');
    expect(references).toHaveLength(1);
    expect(references[0].link).toBe('https://cited.example');
  });

  test('skips blank highlights instead of charging them against the budget', () => {
    const results: t.SearchResultData = {
      organic: [
        makeOrganic('https://a.com', [
          highlight('   \n\t  '),
          highlight('A'.repeat(100)),
        ]),
      ],
    };

    const { output } = formatResultsForLLM(0, results, 100);

    expect(output).toContain('A'.repeat(100));
    expect(output).not.toContain('…[truncated]');
    expect(countHighlightBlocks(output)).toBe(1);
    expect(output).not.toContain(OMISSION_MARKER);
  });

  test('spends the budget across organic results before news results', () => {
    const results: t.SearchResultData = {
      organic: [makeOrganic('https://a.com', [highlight('A'.repeat(100))])],
      topStories: [
        {
          link: 'https://news.com',
          title: 'Story',
          highlights: [highlight('N'.repeat(100))],
        },
      ],
    };

    const { output } = formatResultsForLLM(0, results, 100);

    expect(output).toContain('A'.repeat(100));
    expect(output).not.toContain('N'.repeat(100));
    expect(output).toContain('_[1 additional highlight omitted to fit the context budget');
  });
});
