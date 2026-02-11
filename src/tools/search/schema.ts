export enum DATE_RANGE {
  PAST_HOUR = 'h',
  PAST_24_HOURS = 'd',
  PAST_WEEK = 'w',
  PAST_MONTH = 'm',
  PAST_YEAR = 'y',
}

export const DEFAULT_QUERY_DESCRIPTION = `
GUIDELINES:
- Start broad, then narrow: Begin with key concepts, then refine with specifics
- Think like sources: Use terminology experts would use in the field
- Consider perspective: Frame queries from different viewpoints for better results
- Quality over quantity: A precise 3-4 word query often beats lengthy sentences

TECHNIQUES (combine for power searches):
- EXACT PHRASES: Use quotes ("climate change report")
- EXCLUDE TERMS: Use minus to remove unwanted results (-wikipedia)
- SITE-SPECIFIC: Restrict to websites (site:edu research)
- FILETYPE: Find specific documents (filetype:pdf study)
- OR OPERATOR: Find alternatives (electric OR hybrid cars)
- DATE RANGE: Recent information (data after:2020)
- WILDCARDS: Use * for unknown terms (how to * bread)
- SPECIFIC QUESTIONS: Use who/what/when/where/why/how
- DOMAIN TERMS: Include technical terminology for specialized topics
- CONCISE TERMS: Prioritize keywords over sentences
`.trim();

export const DEFAULT_COUNTRY_DESCRIPTION =
  `Country code to localize search results.
Use standard 2-letter country codes: "us", "uk", "ca", "de", "fr", "jp", "br", etc.
Provide this when the search should return results specific to a particular country.
Examples:
- "us" for United States (default)
- "de" for Germany
- "in" for India
`.trim();

export const querySchema = {
  type: 'string',
  description: DEFAULT_QUERY_DESCRIPTION,
} as const;

export const dateSchema = {
  type: 'string',
  enum: Object.values(DATE_RANGE),
  description: 'Date range for search results.',
} as const;

export const countrySchema = {
  type: 'string',
  description: DEFAULT_COUNTRY_DESCRIPTION,
} as const;

export const imagesSchema = {
  type: 'boolean',
  description: 'Whether to also run an image search.',
} as const;

export const videosSchema = {
  type: 'boolean',
  description: 'Whether to also run a video search.',
} as const;

export const newsSchema = {
  type: 'boolean',
  description: 'Whether to also run a news search.',
} as const;
