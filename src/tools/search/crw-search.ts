import axios from 'axios';
import type * as t from './types';

const DEFAULT_CRW_TIMEOUT = 15000;

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return link;
  }
};

/** The published OpenAPI (docs.fastcrw.com) documents `data` as a flat
 * SearchResult[] with a `category` field; route rows into source groups. */
const splitByCategory = (rows: t.CrwSearchResult[]): t.CrwSearchGroups => {
  const groups: Required<t.CrwSearchGroups> = { web: [], images: [], news: [] };
  for (const row of rows) {
    if (row.category === 'news') {
      groups.news.push(row);
    } else if (row.category === 'images') {
      groups.images.push({ ...row, imageUrl: row.url });
    } else {
      groups.web.push(row);
    }
  }
  return groups;
};

export const createCrwAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.CrwSearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  // NOTE: fastCRW /v1/search is cloud-only, but we still allow a base-URL
  // override for parity with the scraper. Self-host may have no auth, so —
  // unlike Tavily (createTavilyAPI throws on missing key) — we do NOT throw
  // here; we only attach the Authorization header when a key is present.
  const base = (
    apiUrl ??
    process.env.CRW_API_URL ??
    'https://api.fastcrw.com'
  ).replace(/\/+$/, '');
  const config = {
    apiKey: apiKey ?? process.env.CRW_API_KEY,
    apiUrl: `${base}/v1/search`,
    timeout: options?.timeout ?? DEFAULT_CRW_TIMEOUT,
  };

  const getSources = async ({
    query,
    date,
    numResults = 8,
    type,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const limit = Math.min(
        Math.max(1, options?.maxResults ?? numResults),
        20
      );
      // Mirror Serper's verticals: image/news requests hit only their native
      // fastCRW source; plain web optionally adds images via includeImages.
      let sources: t.CrwSearchSource[];
      if (type === 'images') {
        sources = ['images'];
      } else if (type === 'news') {
        sources = ['news'];
      } else {
        sources = options?.includeImages === true ? ['web', 'images'] : ['web'];
      }

      const payload: t.CrwSearchPayload = { query, limit, sources };
      if (date != null) {
        // Serper-style qdr filter; live-verified to constrain results.
        payload.tbs = `qdr:${date}`;
      }

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (config.apiKey != null && config.apiKey !== '') {
        headers.Authorization = `Bearer ${config.apiKey}`;
      }

      const response = await axios.post<t.CrwSearchResponse>(
        config.apiUrl,
        payload,
        { headers, timeout: config.timeout }
      );

      const body = response.data;
      if (body.success === false) {
        return {
          success: false,
          error: `fastCRW search failed: ${
            body.error_code != null ? `[${body.error_code}] ` : ''
          }${body.error ?? 'Unknown error'}`,
        };
      }

      // fastCRW cloud keys results by source: {data: {web: [...], images:
      // [...], news: [...]}} (live-verified 2026-07-02); the published
      // OpenAPI documents a flat SearchResult[] instead (self-host), and a
      // {data: {results: {...groups}}} wrapper also exists. Accept all three.
      // OrganicResult.link is a REQUIRED string (types.ts), so defend against
      // null/empty urls to never feed a broken '' link into the scraper.
      const container = body.data ?? {};
      const data: t.CrwSearchGroups = Array.isArray(container)
        ? splitByCategory(container)
        : (container.results ?? container);
      const organicResults: t.OrganicResult[] = (data.web ?? [])
        .filter((r) => r.url != null && r.url !== '')
        .map((r) => ({
          title: r.title ?? '',
          link: r.url as string,
          snippet: r.description ?? r.snippet ?? '',
          position: r.position,
        }));

      const imageResults: t.ImageResult[] = (data.images ?? [])
        .filter((r) => r.imageUrl != null && r.imageUrl !== '')
        .map((r) => ({
          title: r.title,
          imageUrl: r.imageUrl,
          link: r.url,
          position: r.position,
        }));

      const newsResults: t.NewsResult[] = (data.news ?? [])
        .filter((r) => r.url != null && r.url !== '')
        .map((r) => ({
          title: r.title ?? '',
          link: r.url as string,
          snippet: r.description ?? r.snippet ?? '',
          date: r.publishedDate,
          source: getHostname(r.url as string),
          position: r.position,
        }));

      const results: t.SearchResultData = {
        organic: organicResults,
        images: imageResults,
        topStories: [],
        videos: [],
        news: newsResults,
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `fastCRW search request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
