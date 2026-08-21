import axios from 'axios';
import { createDefaultLogger } from './utils';
import { VoyageReranker } from './rerankers';

describe('VoyageReranker', () => {
  const logger = createDefaultLogger();

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses the Voyage rerank contract with bounded latency', async () => {
    const post = jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: {
        data: [{ index: 1, relevance_score: 0.91 }],
        usage: { total_tokens: 55 },
      },
    });
    jest.spyOn(logger, 'debug').mockImplementation(() => logger);

    const reranker = new VoyageReranker({
      apiKey: 'test-key',
      logger,
    });

    await expect(
      reranker.rerank('test query', ['document1', 'document2'], 1)
    ).resolves.toEqual([{ text: 'document2', score: 0.91 }]);

    expect(post).toHaveBeenCalledWith(
      'https://api.voyageai.com/v1/rerank',
      {
        model: 'rerank-2.5',
        query: 'test query',
        documents: ['document1', 'document2'],
        top_k: 1,
        return_documents: false,
        truncation: false,
      },
      expect.objectContaining({
        timeout: 7000,
        headers: expect.objectContaining({
          Authorization: 'Bearer test-key',
        }),
      })
    );
  });

  it('uses explicit endpoint, model, and timeout values', async () => {
    const post = jest.spyOn(axios, 'post').mockResolvedValueOnce({
      data: { data: [{ index: 0, relevance_score: 0.88 }] },
    });

    const reranker = new VoyageReranker({
      apiKey: 'test-key',
      apiUrl: 'https://proxy.example.com/rerank',
      model: 'rerank-custom',
      timeout: 5000,
      logger,
    });

    await reranker.rerank('test query', ['document1'], 1);

    expect(post).toHaveBeenCalledWith(
      'https://proxy.example.com/rerank',
      expect.objectContaining({ model: 'rerank-custom' }),
      expect.objectContaining({ timeout: 5000 })
    );
  });

  it('does not call the provider without a key', async () => {
    const post = jest.spyOn(axios, 'post');
    jest.spyOn(logger, 'warn').mockImplementation(() => logger);

    const reranker = new VoyageReranker({ apiKey: '', logger });

    await expect(
      reranker.rerank('query', ['first', 'second'], 1)
    ).resolves.toEqual([{ text: 'first', score: 0 }]);
    expect(post).not.toHaveBeenCalled();
  });

  it('retries one rate-limited request before returning the provider result', async () => {
    jest.useFakeTimers();
    const rateLimitError = Object.assign(new Error('rate limited'), {
      isAxiosError: true,
      response: { status: 429, headers: { 'retry-after': '0' } },
    });
    jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);
    const post = jest
      .spyOn(axios, 'post')
      .mockRejectedValueOnce(rateLimitError)
      .mockResolvedValueOnce({
        data: { data: [{ index: 0, relevance_score: 0.9 }] },
      });

    const reranker = new VoyageReranker({ apiKey: 'test-key', logger });
    const result = reranker.rerank('query', ['document'], 1);
    await jest.runAllTimersAsync();

    await expect(result).resolves.toEqual([{ text: 'document', score: 0.9 }]);
    expect(post).toHaveBeenCalledTimes(2);
    jest.useRealTimers();
  });

  it.each([
    ['missing data', {}],
    ['partial data', { data: [{ index: 0, relevance_score: 0.9 }] }],
    [
      'duplicate indices',
      {
        data: [
          { index: 0, relevance_score: 0.9 },
          { index: 0, relevance_score: 0.8 },
        ],
      },
    ],
    [
      'out-of-range index',
      {
        data: [
          { index: 0, relevance_score: 0.9 },
          { index: 2, relevance_score: 0.8 },
        ],
      },
    ],
    [
      'non-finite score',
      {
        data: [
          { index: 0, relevance_score: 0.9 },
          { index: 1, relevance_score: Number.NaN },
        ],
      },
    ],
  ])('falls back deterministically for %s', async (_name, data) => {
    jest.spyOn(axios, 'post').mockResolvedValueOnce({ data });
    jest.spyOn(logger, 'warn').mockImplementation(() => logger);

    const reranker = new VoyageReranker({
      apiKey: 'test-key',
      logger,
    });

    await expect(
      reranker.rerank('query', ['first', 'second'], 2)
    ).resolves.toEqual([
      { text: 'first', score: 0 },
      { text: 'second', score: 0 },
    ]);
  });

  it('logs compact provider errors without request secrets or documents', async () => {
    const error = Object.assign(new Error('Request failed'), {
      isAxiosError: true,
      code: 'ECONNABORTED',
      config: {
        method: 'post',
        url: 'https://proxy.example.com/rerank?api_key=hidden',
        headers: { Authorization: 'Bearer test-key' },
        data: JSON.stringify({ documents: ['sensitive document'] }),
      },
      response: { status: 429, data: { message: 'too many requests' } },
    });
    jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);
    jest.spyOn(axios, 'post').mockRejectedValue(error);
    const errorLog = jest
      .spyOn(logger, 'error')
      .mockImplementation(() => logger);

    const reranker = new VoyageReranker({
      apiKey: 'test-key',
      logger,
    });

    await reranker.rerank('query', ['sensitive document'], 1);

    const serialized = JSON.stringify(errorLog.mock.calls);
    expect(serialized).toContain('ECONNABORTED');
    expect(serialized).toContain('429');
    expect(serialized).not.toContain('Authorization');
    expect(serialized).not.toContain('api_key');
    expect(serialized).not.toContain('test-key');
    expect(serialized).not.toContain('sensitive document');
  });
});
