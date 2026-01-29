import type { MessageContentComplex } from '../types';

// Unicode citation marker (U+E202 - Private Use Area)
const CITATION_MARKER = String.fromCharCode(0xe202);

/**
 * A single citation source in LibreChat format
 */
export interface CitationSource {
  link: string;
  title: string;
  snippet: string;
  date: string;
  position: number;
}

/**
 * Citation data format for web_search attachments (LibreChat frontend compatible)
 */
export interface CitationData {
  organic: CitationSource[];
  topStories: never[];
  images: never[];
  videos: never[];
  references: never[];
}

/**
 * Raw search result from Perplexity API
 */
export interface RawPerplexitySearchResult {
  url?: string;
  link?: string;
  title?: string;
  snippet?: string;
  date?: string;
}

/**
 * Processed citation data ready for LibreChat consumption
 */
export interface ProcessedCitations {
  /** Transformed to LibreChat's web_search attachment format */
  searchResults: CitationData | null;
  /** Raw citation URLs (for reference) */
  citations: string[] | null;
}

/**
 * Transform raw Perplexity citations/search_results to LibreChat's SearchResultData format
 *
 * @param citations - Raw citation URLs from Perplexity
 * @param searchResults - Detailed search results from Perplexity (preferred if available)
 * @returns SearchResultData object compatible with LibreChat's frontend
 */
export function transformCitations(
  citations: string[] | null,
  searchResults: unknown[] | null
): CitationData | null {
  // Prefer search_results if available (has richer data), fallback to citations
  const sources =
    searchResults && searchResults.length > 0 ? searchResults : citations;

  if (!sources || sources.length === 0) {
    return null;
  }

  return {
    organic: sources.map((source, index) => {
      // If source is a string (from citations array), create basic object
      if (typeof source === 'string') {
        return {
          link: source,
          title: `Source ${index + 1}`,
          snippet: '',
          date: new Date().toISOString().split('T')[0],
          position: index + 1,
        };
      }

      // If source is an object (from search_results array)
      const s = source as RawPerplexitySearchResult;
      return {
        link: s.url || s.link || '',
        title: s.title || `Source ${index + 1}`,
        snippet: s.snippet || '',
        date: s.date || new Date().toISOString().split('T')[0],
        position: index + 1,
      };
    }),
    topStories: [],
    images: [],
    videos: [],
    references: [],
  };
}

/**
 * Inject Unicode citation markers into text content.
 * Replaces [1], [2] etc with {U+E202}turn{N}search{index}
 *
 * The Unicode marker U+E202 (Private Use Area) is recognized by LibreChat's
 * frontend markdown parser to render as hoverable citation links.
 *
 * @param content - Text content with [N] style citation markers
 * @param turnNumber - Current conversation turn number (0-indexed)
 * @returns Content with Unicode citation markers injected
 */
export function injectCitationMarkers(
  content: string,
  turnNumber: number
): string {
  if (!content) return content;

  return content.replace(/\[(\d+)\]/g, (match, num) => {
    const index = parseInt(num, 10) - 1; // Convert 1-based to 0-based index
    if (index < 0) return match; // Keep original if invalid number

    // Space before marker prevents breaking markdown bold parsing
    // Without the space, `**bold**{U+E202}` is not recognized as valid bold
    // by remark-gfm because U+E202 is not classified as punctuation/whitespace
    return ` ${CITATION_MARKER}turn${turnNumber}search${index}`;
  });
}

/**
 * Process an array of content parts, injecting citation markers into text parts.
 *
 * @param contentParts - Array of MessageContentComplex parts
 * @param turnNumber - Current conversation turn number (0-indexed)
 * @returns New array with citation markers injected into text parts
 */
export function processContentParts(
  contentParts: MessageContentComplex[],
  turnNumber: number
): MessageContentComplex[] {
  return contentParts.map((part) => {
    if (
      part &&
      part.type === 'text' &&
      typeof (part as { text?: string }).text === 'string'
    ) {
      return {
        ...part,
        text: injectCitationMarkers(
          (part as { text: string }).text,
          turnNumber
        ),
      };
    }
    return part;
  });
}
