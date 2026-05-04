import axios from 'axios';
import type * as t from './types';

const DEFAULT_TAVILY_TIMEOUT = 15000;

const TAVILY_COUNTRY_ALIASES: Record<string, string> = {
  ba: 'bosnia and herzegovina',
  cg: 'congo',
  cz: 'czech republic',
  gb: 'united kingdom',
  mm: 'myanmar',
  tr: 'turkey',
  tt: 'trinidad and tobago',
  uk: 'united kingdom',
};

const TAVILY_SUPPORTED_COUNTRIES = new Set([
  'afghanistan',
  'albania',
  'algeria',
  'andorra',
  'angola',
  'argentina',
  'armenia',
  'australia',
  'austria',
  'azerbaijan',
  'bahamas',
  'bahrain',
  'bangladesh',
  'barbados',
  'belarus',
  'belgium',
  'belize',
  'benin',
  'bhutan',
  'bolivia',
  'bosnia and herzegovina',
  'botswana',
  'brazil',
  'brunei',
  'bulgaria',
  'burkina faso',
  'burundi',
  'cambodia',
  'cameroon',
  'canada',
  'cape verde',
  'central african republic',
  'chad',
  'chile',
  'china',
  'colombia',
  'comoros',
  'congo',
  'costa rica',
  'croatia',
  'cuba',
  'cyprus',
  'czech republic',
  'denmark',
  'djibouti',
  'dominican republic',
  'ecuador',
  'egypt',
  'el salvador',
  'equatorial guinea',
  'eritrea',
  'estonia',
  'ethiopia',
  'fiji',
  'finland',
  'france',
  'gabon',
  'gambia',
  'georgia',
  'germany',
  'ghana',
  'greece',
  'guatemala',
  'guinea',
  'haiti',
  'honduras',
  'hungary',
  'iceland',
  'india',
  'indonesia',
  'iran',
  'iraq',
  'ireland',
  'israel',
  'italy',
  'jamaica',
  'japan',
  'jordan',
  'kazakhstan',
  'kenya',
  'kuwait',
  'kyrgyzstan',
  'latvia',
  'lebanon',
  'lesotho',
  'liberia',
  'libya',
  'liechtenstein',
  'lithuania',
  'luxembourg',
  'madagascar',
  'malawi',
  'malaysia',
  'maldives',
  'mali',
  'malta',
  'mauritania',
  'mauritius',
  'mexico',
  'moldova',
  'monaco',
  'mongolia',
  'montenegro',
  'morocco',
  'mozambique',
  'myanmar',
  'namibia',
  'nepal',
  'netherlands',
  'new zealand',
  'nicaragua',
  'niger',
  'nigeria',
  'north korea',
  'north macedonia',
  'norway',
  'oman',
  'pakistan',
  'panama',
  'papua new guinea',
  'paraguay',
  'peru',
  'philippines',
  'poland',
  'portugal',
  'qatar',
  'romania',
  'russia',
  'rwanda',
  'saudi arabia',
  'senegal',
  'serbia',
  'singapore',
  'slovakia',
  'slovenia',
  'somalia',
  'south africa',
  'south korea',
  'south sudan',
  'spain',
  'sri lanka',
  'sudan',
  'sweden',
  'switzerland',
  'syria',
  'taiwan',
  'tajikistan',
  'tanzania',
  'thailand',
  'togo',
  'trinidad and tobago',
  'tunisia',
  'turkey',
  'turkmenistan',
  'uganda',
  'ukraine',
  'united arab emirates',
  'united kingdom',
  'united states',
  'uruguay',
  'uzbekistan',
  'venezuela',
  'vietnam',
  'yemen',
  'zambia',
  'zimbabwe',
]);

const TAVILY_REGION_NAMES = new Intl.DisplayNames(['en'], {
  type: 'region',
});

const normalizeTavilyCountryName = (country: string): string =>
  country
    .toLowerCase()
    .replace(/\s*\([^)]*\)/g, '')
    .replace(/\s*&\s*/g, ' and ')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '');

const normalizeTavilyCountry = (country?: string): string | undefined => {
  const normalizedCountry = country?.trim().toLowerCase();
  if (normalizedCountry == null || normalizedCountry === '') {
    return undefined;
  }

  const countryAlias = TAVILY_COUNTRY_ALIASES[normalizedCountry];
  if (countryAlias != null) {
    return TAVILY_SUPPORTED_COUNTRIES.has(countryAlias)
      ? countryAlias
      : undefined;
  }

  if (/^[a-z]{2}$/.test(normalizedCountry)) {
    const regionName = TAVILY_REGION_NAMES.of(normalizedCountry.toUpperCase());
    if (regionName == null) {
      return undefined;
    }
    const countryName = normalizeTavilyCountryName(regionName);
    return TAVILY_SUPPORTED_COUNTRIES.has(countryName)
      ? countryName
      : undefined;
  }

  const countryName = normalizeTavilyCountryName(normalizedCountry);
  return TAVILY_SUPPORTED_COUNTRIES.has(countryName) ? countryName : undefined;
};

const normalizeTavilyTimeRange = (
  timeRange?: t.TavilyTimeRangeInput
): t.TavilyTimeRange | undefined => {
  switch (timeRange) {
    case 'h':
    case 'd':
      return 'day';
    case 'w':
      return 'week';
    case 'm':
      return 'month';
    case 'y':
      return 'year';
    case 'day':
    case 'week':
    case 'month':
    case 'year':
      return timeRange;
    default:
      return undefined;
  }
};

const getHostname = (link: string): string => {
  try {
    return new URL(link).hostname;
  } catch {
    return link;
  }
};

export const createTavilyAPI = (
  apiKey?: string,
  apiUrl?: string,
  options?: t.TavilySearchOptions
): {
  getSources: (params: t.GetSourcesParams) => Promise<t.SearchResult>;
} => {
  const config = {
    apiKey: apiKey ?? process.env.TAVILY_API_KEY,
    apiUrl:
      apiUrl ??
      process.env.TAVILY_SEARCH_URL ??
      'https://api.tavily.com/search',
    timeout: options?.timeout ?? DEFAULT_TAVILY_TIMEOUT,
  };

  if (config.apiKey == null || config.apiKey === '') {
    throw new Error('TAVILY_API_KEY is required for Tavily API');
  }

  const getSources = async ({
    query,
    date,
    country,
    numResults = 8,
    type,
    news,
  }: t.GetSourcesParams): Promise<t.SearchResult> => {
    if (!query.trim()) {
      return { success: false, error: 'Query cannot be empty' };
    }

    try {
      const timeRange =
        normalizeTavilyTimeRange(options?.timeRange) ??
        (date != null ? (normalizeTavilyTimeRange(date) ?? 'day') : undefined);
      const topic =
        news === true || type === 'news'
          ? 'news'
          : (options?.topic ?? 'general');
      const maxResults = options?.maxResults ?? numResults;
      const searchDepth = options?.searchDepth ?? 'basic';

      const payload: t.TavilySearchPayload = {
        query,
        search_depth: searchDepth,
        topic,
        max_results: Math.min(Math.max(1, maxResults), 20),
      };

      if (
        options?.safeSearch != null &&
        searchDepth !== 'fast' &&
        searchDepth !== 'ultra-fast'
      ) {
        payload.safe_search = options.safeSearch;
      }
      if (timeRange != null) {
        payload.time_range = timeRange;
      }
      const tavilyCountry =
        topic === 'general' ? normalizeTavilyCountry(country) : undefined;
      if (tavilyCountry != null) {
        payload.country = tavilyCountry;
      }
      if (type === 'images' || options?.includeImages) {
        payload.include_images = true;
      }
      if (options?.includeAnswer != null) {
        payload.include_answer = options.includeAnswer;
      }
      if (options?.includeRawContent != null) {
        payload.include_raw_content = options.includeRawContent;
      }
      if (
        options?.includeDomains != null &&
        options.includeDomains.length > 0
      ) {
        payload.include_domains = options.includeDomains;
      }
      if (
        options?.excludeDomains != null &&
        options.excludeDomains.length > 0
      ) {
        payload.exclude_domains = options.excludeDomains;
      }
      if (options?.includeImageDescriptions != null) {
        payload.include_image_descriptions = options.includeImageDescriptions;
      }
      if (options?.includeFavicon != null) {
        payload.include_favicon = options.includeFavicon;
      }
      if (options?.chunksPerSource != null && searchDepth === 'advanced') {
        payload.chunks_per_source = options.chunksPerSource;
      }

      const response = await axios.post<t.TavilySearchResponse>(
        config.apiUrl,
        payload,
        {
          headers: {
            Authorization: `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
          },
          timeout: config.timeout,
        }
      );

      const data = response.data;

      const organicResults: t.OrganicResult[] = (data.results ?? []).map(
        (result: t.TavilySearchResult) => ({
          title: result.title ?? '',
          link: result.url ?? '',
          snippet: result.content ?? '',
          date: result.published_date,
        })
      );

      const imageResults: t.ImageResult[] = Array.isArray(data.images)
        ? data.images.slice(0, 6).reduce<t.ImageResult[]>((acc, image) => {
            const imageUrl = typeof image === 'string' ? image : image.url;
            if (imageUrl == null || imageUrl === '') {
              return acc;
            }
            acc.push({
              imageUrl,
              title: typeof image === 'string' ? undefined : image.description,
              position: acc.length + 1,
            });
            return acc;
          }, [])
        : [];

      const newsResults: t.NewsResult[] =
        topic === 'news'
          ? organicResults.map((r) => ({
              title: r.title,
              link: r.link,
              snippet: r.snippet,
              date: r.date,
              source: getHostname(r.link),
            }))
          : [];

      const results: t.SearchResultData = {
        organic: organicResults,
        images: imageResults,
        topStories: [],
        videos: [],
        news: newsResults,
        answerBox: data.answer != null ? { snippet: data.answer } : undefined,
        relatedSearches: [],
      };

      return { success: true, data: results };
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      return {
        success: false,
        error: `Tavily API request failed: ${errorMessage}`,
      };
    }
  };

  return { getSources };
};
