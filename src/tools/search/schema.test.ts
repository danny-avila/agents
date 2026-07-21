import { DEFAULT_DATE_DESCRIPTION, dateSchema } from './schema';

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
});
