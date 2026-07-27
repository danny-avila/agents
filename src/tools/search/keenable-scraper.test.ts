import axios from 'axios';
import { createKeenableScraper } from './keenable-scraper';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const sampleResponse = {
  data: {
    url: 'https://example.com/',
    title: 'Example Domain',
    content:
      '# Example Domain\n\nThis domain is for use in documentation examples.',
    description: 'An example page.',
  },
};

describe('Keenable scraper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    delete process.env.KEENABLE_API_KEY;
    delete process.env.KEENABLE_FETCH_URL;
  });

  it('returns an error for empty URLs without calling the API', async () => {
    const scraper = createKeenableScraper();
    const [url, response] = await scraper.scrapeUrl('   ');

    expect(url).toBe('   ');
    expect(response.success).toBe(false);
    expect(mockedAxios.get).not.toHaveBeenCalled();
  });

  it('hits the public endpoint and omits the API key header when keyless', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const scraper = createKeenableScraper();
    const [, response] = await scraper.scrapeUrl('https://example.com');

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'https://api.keenable.ai/v1/fetch/public',
      expect.objectContaining({
        params: { url: 'https://example.com' },
        headers: expect.objectContaining({ 'X-Keenable-Title': 'LibreChat' }),
      })
    );
    const headers = mockedAxios.get.mock.calls[0][1]?.headers as Record<
      string,
      string
    >;
    expect(headers['X-API-Key']).toBeUndefined();
    expect(response.success).toBe(true);
    expect(response.data?.content).toContain('# Example Domain');
  });

  it('hits the keyed endpoint and sends the API key header when a key is set', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const scraper = createKeenableScraper({ apiKey: 'secret-key' });
    await scraper.scrapeUrl('https://example.com');

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'https://api.keenable.ai/v1/fetch',
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-API-Key': 'secret-key' }),
      })
    );
  });

  it('honors a custom attribution title and fetch URL override', async () => {
    mockedAxios.get.mockResolvedValueOnce(sampleResponse);

    const scraper = createKeenableScraper({
      apiUrl: 'https://keenable.internal/v1/fetch/public',
      attributionTitle: 'MyApp',
    });
    await scraper.scrapeUrl('https://example.com');

    expect(mockedAxios.get).toHaveBeenCalledWith(
      'https://keenable.internal/v1/fetch/public',
      expect.objectContaining({
        headers: expect.objectContaining({ 'X-Keenable-Title': 'MyApp' }),
      })
    );
  });

  it('reports failure when the API returns no content', async () => {
    mockedAxios.get.mockResolvedValueOnce({
      data: { url: 'https://x.test', content: '' },
    });

    const scraper = createKeenableScraper();
    const [, response] = await scraper.scrapeUrl('https://x.test');

    expect(response.success).toBe(false);
    expect(response.error).toMatch(/no content/i);
  });

  it('reports failure with a message when the request throws', async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error('network down'));

    const scraper = createKeenableScraper();
    const [, response] = await scraper.scrapeUrl('https://x.test');

    expect(response.success).toBe(false);
    expect(response.error).toContain('network down');
  });

  it('scrapes multiple URLs concurrently', async () => {
    mockedAxios.get
      .mockResolvedValueOnce({ data: { content: 'a', url: 'https://a.test' } })
      .mockResolvedValueOnce({ data: { content: 'b', url: 'https://b.test' } });

    const scraper = createKeenableScraper();
    const results = await scraper.scrapeUrls([
      'https://a.test',
      'https://b.test',
    ]);

    expect(results).toHaveLength(2);
    expect(results[0][1].data?.content).toBe('a');
    expect(results[1][1].data?.content).toBe('b');
  });

  it('extractContent returns the markdown with no structured references', () => {
    const scraper = createKeenableScraper();
    const [content, references] = scraper.extractContent({
      success: true,
      data: { content: '# Title\n\nBody' },
    });

    expect(content).toBe('# Title\n\nBody');
    expect(references).toBeUndefined();
  });

  it('extractMetadata surfaces title, description and url', () => {
    const scraper = createKeenableScraper();
    const metadata = scraper.extractMetadata({
      success: true,
      data: {
        content: 'x',
        title: 'Example Domain',
        description: 'An example page.',
        url: 'https://example.com/',
      },
    });

    expect(metadata).toEqual({
      title: 'Example Domain',
      description: 'An example page.',
      url: 'https://example.com/',
    });
  });
});
