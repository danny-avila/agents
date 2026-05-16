import axios from 'axios';
import type * as t from './types';

const DEFAULT_VOLCENGINE_TIMEOUT = 15000;
const DEFAULT_VOLCENGINE_URL =
  'https://open.feedcoopapi.com/search_api/web_search';

interface VolcEngineWebResult {
  Id?: string;
  SortId?: number;
  Title?: string;
  SiteName?: string;
  Url?: string;
  Snippet?: string;
  Summary?: string;
  Content?: string;
  PublishTime?: string;
}

interface VolcEngineResult {
  ResultCount?: number;
  WebResults?: VolcEngineWebResult[];
}

interface VolcEngineResponse {
  ResponseMetadata?: Record<string, string>;
  Result?: VolcEngineResult;
  error?: string;
}

function parseSSEResponse(body: string): VolcEngineResponse | null {
  const events = body.split('\n\n').filter(Boolean);

  for (const event of events) {
    if (!event.startsWith('data:')) {
      continue;
    }

    const jsonStr = event.substring(5).trim();
    if (jsonStr === '[DONE]') {
      continue;
    }

    try {
      const parsed = JSON.parse(jsonStr) as VolcEngineResponse;
      if (parsed.Result?.WebResults?.length) {
        return parsed;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function mapWebResults(webResults: VolcEngineWebResult[]): t.OrganicResult[] {
  return webResults.map((result, index) => ({
    position: index + 1,
    title: result.Title ?? '',
    link: result.Url ?? '',
    snippet: result.Snippet ?? '',
    date: result.PublishTime,
  }));
}

export const createVolcEngineAPI = (
  apiKey?: string,
  apiUrl?: string,
  searchType?: 'web' | 'web_summary'
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.VOLCENGINE_API_KEY,
    apiUrl:
      apiUrl ?? process.env.VOLCENGINE_SEARCH_URL ?? DEFAULT_VOLCENGINE_URL,
    searchType: searchType ?? 'web_summary',
    timeout: DEFAULT_VOLCENGINE_TIMEOUT,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('VOLCENGINE_API_KEY is required for VolcEngine API');
  }

  const getSources = async ({
    query,
    numResults = 8,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const payload = {
        Query: query,
        SearchType: config.searchType,
        Count: Math.min(Math.max(1, numResults), 20),
        Filter: {
          NeedContent: true,
          NeedUrl: true,
        },
        NeedSummary: true,
      };

      const response = await axios.post<string | VolcEngineResponse>(
        config.apiUrl,
        payload,
        {
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
          responseType: config.searchType === 'web_summary' ? 'text' : 'json',
        }
      );

      const rawData = response.data;
      const data: VolcEngineResponse | null =
        typeof rawData === 'string'
          ? parseSSEResponse(rawData)
          : (rawData as VolcEngineResponse);

      if (!data?.Result?.WebResults?.length) {
        return { success: false, error: 'No results found' };
      }

      if (data.error) {
        return { success: false, error: data.error };
      }

      const organicResults = mapWebResults(data.Result.WebResults);

      const results: t.SearchResultData = {
        organic: organicResults,
        images: [],
        topStories: [],
        videos: [],
        news: [],
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `VolcEngine API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
