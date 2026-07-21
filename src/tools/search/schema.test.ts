import {
  DEFAULT_DATE_DESCRIPTION,
  TAVILY_DATE_DESCRIPTION,
  tavilyDateSchema,
  dateSchema,
  DATE_RANGE,
} from './schema';

describe('web search schema', () => {
  it('warns models that date filtering can exclude undated results', () => {
    expect(dateSchema.description).toBe(DEFAULT_DATE_DESCRIPTION);
    expect(dateSchema.description).toContain(
      'Only provide this when the user explicitly requests recent results'
    );
    expect(dateSchema.description).toContain(
      'pages without a detectable publication or update date'
    );
  });

  it('does not advertise unsupported hourly filtering for Tavily', () => {
    expect(tavilyDateSchema.description).toBe(TAVILY_DATE_DESCRIPTION);
    expect(tavilyDateSchema.enum).not.toContain(DATE_RANGE.PAST_HOUR);
    expect(tavilyDateSchema.description).not.toContain('past hour');
  });
});
