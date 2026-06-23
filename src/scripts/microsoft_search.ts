/* eslint-disable no-console */
import { config } from 'dotenv';
config();
import { createMicrosoftSearchAPI } from '@/tools/search/microsoft-search';
import { createMicrosoftScraper } from '@/tools/search/microsoft-scraper';
import { createSearchTool } from '@/tools/search';

const apiKey = process.env.MICROSOFT_WEBIQ_API_KEY;
const baseUrl = process.env.MICROSOFT_WEBIQ_BASE_URL;

const QUERY = process.env.SMOKE_QUERY ?? 'latest typescript release notes';
const SCRAPE_URL = process.env.SMOKE_URL ?? 'https://www.typescriptlang.org/';

function header(title: string): void {
  console.log(`\n========== ${title} ==========`);
}

async function testWebSearch(): Promise<void> {
  header('1. Web Search (/v3/search/web)');
  const api = createMicrosoftSearchAPI({
    searchProvider: 'microsoftWebIQ',
    microsoftWebIQApiKey: apiKey,
    microsoftWebIQBaseUrl: baseUrl,
  });
  const result = await api.getSources({ query: QUERY });
  console.log('success:', result.success);
  if (!result.success) {
    console.log('error:', result.error);
    return;
  }
  console.log('organic count:', result.data?.organic?.length ?? 0);
  console.dir(result.data?.organic?.slice(0, 3), { depth: null });
}

async function testNewsSearch(): Promise<void> {
  header('2. News Search (/v3/search/news)');
  const api = createMicrosoftSearchAPI({
    searchProvider: 'microsoftWebIQ',
    microsoftWebIQApiKey: apiKey,
    microsoftWebIQBaseUrl: baseUrl,
  });
  const result = await api.getSources({ query: QUERY, news: true });
  console.log('success:', result.success);
  if (!result.success) {
    console.log('error:', result.error);
    return;
  }
  console.log('topStories count:', result.data?.topStories?.length ?? 0);
  console.dir(result.data?.topStories?.slice(0, 3), { depth: null });
}

async function testBrowse(): Promise<void> {
  header('3. Browse scraper (/v3/browse)');
  const scraper = createMicrosoftScraper({
    apiKey,
    baseUrl,
  });
  const [url, response] = await scraper.scrapeUrl(SCRAPE_URL);
  console.log('url:', url);
  console.log('success:', response.success);
  if (!response.success) {
    console.log('error:', response.error);
    return;
  }
  const [content] = scraper.extractContent(response);
  const metadata = scraper.extractMetadata(response);
  console.log('content length:', content.length);
  console.log('content preview:', content.slice(0, 300));
  console.log('metadata:', metadata);
}

async function testFullTool(): Promise<void> {
  header(
    '4. Full search tool (search -> scrape -> rerank, no LLM, no reranker)'
  );
  const tool = createSearchTool({
    searchProvider: 'microsoftWebIQ',
    scraperProvider: 'microsoftWebIQ',
    microsoftWebIQApiKey: apiKey,
    microsoftWebIQBaseUrl: baseUrl,
    rerankerType: 'none',
    topResults: 3,
  });
  const output = await tool.invoke({ query: QUERY });
  console.dir(output, { depth: 4 });
}

async function main(): Promise<void> {
  if (apiKey == null || apiKey === '') {
    console.error('MICROSOFT_WEBIQ_API_KEY is not set. Add it to .env first.');
    process.exit(1);
  }
  console.log('baseUrl:', baseUrl ?? 'https://api.microsoft.ai/v3 (default)');
  console.log('query:', QUERY);
  console.log('scrape url:', SCRAPE_URL);

  await testWebSearch();
  await testNewsSearch();
  await testBrowse();
  await testFullTool();
}

main().catch((err) => {
  console.error('Smoke test failed:', err);
  process.exit(1);
});
