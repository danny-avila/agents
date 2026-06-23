/* eslint-disable no-console */
import { config } from 'dotenv';
config();
import { createMicrosoftSearchAPI } from '@/tools/search/microsoft-search';
import { createMicrosoftScraper } from '@/tools/search/microsoft-scraper';
import { createSearchTool } from '@/tools/search';
import { Constants } from '@/common';

const apiKey = process.env.MICROSOFT_WEBIQ_API_KEY;
const baseUrl = process.env.MICROSOFT_WEBIQ_BASE_URL;

const QUERY = process.env.SMOKE_QUERY ?? 'latest typescript release notes';
const SCRAPE_URL = process.env.SMOKE_URL ?? 'https://www.typescriptlang.org/';
/** Comma-separated subset of steps to run, e.g. SMOKE_ONLY=videos,images */
const ONLY = (process.env.SMOKE_ONLY ?? '')
  .split(',')
  .map((step) => step.trim().toLowerCase())
  .filter((step) => step !== '');

const createApi = (): ReturnType<typeof createMicrosoftSearchAPI> =>
  createMicrosoftSearchAPI({
    searchProvider: 'microsoftWebIQ',
    microsoftWebIQApiKey: apiKey,
    microsoftWebIQBaseUrl: baseUrl,
  });

function header(title: string): void {
  console.log(`\n========== ${title} ==========`);
}

function shouldRun(step: string): boolean {
  return ONLY.length === 0 || ONLY.includes(step);
}

async function testWebSearch(): Promise<void> {
  header('1. Web Search (/v3/search/web)');
  const result = await createApi().getSources({ query: QUERY });
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
  const result = await createApi().getSources({ query: QUERY, news: true });
  console.log('success:', result.success);
  if (!result.success) {
    console.log('error:', result.error);
    return;
  }
  console.log('topStories count:', result.data?.topStories?.length ?? 0);
  console.dir(result.data?.topStories?.slice(0, 3), { depth: null });
}

async function testVideoSearch(): Promise<void> {
  header('3. Video Search (/v3/search/videos)');
  const result = await createApi().getSources({ query: QUERY, type: 'videos' });
  console.log('success:', result.success);
  if (!result.success) {
    console.log('error:', result.error);
    return;
  }
  console.log('videos count:', result.data?.videos?.length ?? 0);
  console.dir(result.data?.videos?.slice(0, 3), { depth: null });
}

async function testImageSearch(): Promise<void> {
  header('4. Image Search (/v3/search/images)');
  const result = await createApi().getSources({ query: QUERY, type: 'images' });
  console.log('success:', result.success);
  if (!result.success) {
    console.log('error:', result.error);
    return;
  }
  console.log('images count:', result.data?.images?.length ?? 0);
  console.dir(result.data?.images?.slice(0, 3), { depth: null });
}

async function testBrowse(): Promise<void> {
  header('5. Browse scraper (/v3/browse)');
  const scraper = createMicrosoftScraper({ apiKey, baseUrl });
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
    '6. Full search tool (web + news + videos + images -> scrape, no LLM, no reranker)'
  );
  const tool = createSearchTool({
    searchProvider: 'microsoftWebIQ',
    scraperProvider: 'microsoftWebIQ',
    microsoftWebIQApiKey: apiKey,
    microsoftWebIQBaseUrl: baseUrl,
    rerankerType: 'none',
    topResults: 3,
  });
  const message = await tool.invoke({
    name: tool.name,
    args: { query: QUERY, news: true, videos: true, images: true },
    id: 'smoke-full-tool',
    type: 'tool_call',
  });
  const data = message.artifact?.[Constants.WEB_SEARCH];
  console.log('organic:', data?.organic?.length ?? 0);
  console.log('topStories:', data?.topStories?.length ?? 0);
  console.log('videos:', data?.videos?.length ?? 0);
  console.log('images:', data?.images?.length ?? 0);
  console.log('\n--- formatted content sent to the model ---');
  console.log(message.content);
}

async function main(): Promise<void> {
  if (apiKey == null || apiKey === '') {
    console.error('MICROSOFT_WEBIQ_API_KEY is not set. Add it to .env first.');
    process.exit(1);
  }
  console.log('baseUrl:', baseUrl ?? 'https://api.microsoft.ai/v3 (default)');
  console.log('query:', QUERY);
  console.log('scrape url:', SCRAPE_URL);
  if (ONLY.length > 0) {
    console.log('only:', ONLY.join(', '));
  }

  const steps: Array<[string, () => Promise<void>]> = [
    ['web', testWebSearch],
    ['news', testNewsSearch],
    ['videos', testVideoSearch],
    ['images', testImageSearch],
    ['browse', testBrowse],
    ['tool', testFullTool],
  ];

  for (const [name, run] of steps) {
    if (shouldRun(name)) {
      await run();
    }
  }
}

main().catch((err) => {
  console.error('Smoke test failed:', err);
  process.exit(1);
});
