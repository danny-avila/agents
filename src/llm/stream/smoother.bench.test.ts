import http from 'node:http';
import { HumanMessage } from '@langchain/core/messages';
import type { AddressInfo } from 'node:net';
import { ChatOpenAI } from '@/llm/openai';
import { SMOOTH_TARGET_LATENCY_MS } from './smoother';

/**
 * End-to-end cadence/lag benchmark: drives the real ChatOpenAI client against
 * a local SSE server that emits large chunks fast (the big-chunk gateway
 * profile smoothing exists for), asserting the properties that define the
 * feature: even cadence at the configured tick, and render lag pinned near
 * the target latency instead of growing with reply length.
 */
describe('adaptive smoothing benchmark (big-chunk provider)', () => {
  const WORDS =
    'ClickHouse stores data in columns so queries read only what they touch and compress well '.split(
      ' '
    );
  const CHUNK_CHARS = 110;
  const PROVIDER_INTERVAL_MS = 20;

  function buildChunks(totalWords: number): string[] {
    const out: string[] = [];
    let current = '';
    for (let i = 0; i < totalWords; i++) {
      current += WORDS[i % WORDS.length] + ' ';
      if (current.length >= CHUNK_CHARS) {
        out.push(current);
        current = '';
      }
    }
    if (current) {
      out.push(current);
    }
    return out;
  }

  function startServer(chunks: string[], stamp: { doneAt: number }): Promise<{
    server: http.Server;
    port: number;
  }> {
    return new Promise((resolve) => {
      const server = http.createServer((req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/event-stream' });
        const frame = (
          delta: Record<string, unknown>,
          finish: string | null
        ): string =>
          `data: ${JSON.stringify({
            id: 'bench',
            object: 'chat.completion.chunk',
            created: 1,
            model: 'bench-model',
            choices: [{ index: 0, delta, finish_reason: finish }],
          })}\n\n`;
        res.write(frame({ role: 'assistant', content: '' }, null));
        let i = 0;
        const send = (): void => {
          if (i < chunks.length) {
            res.write(frame({ content: chunks[i] }, null));
            i += 1;
            setTimeout(send, PROVIDER_INTERVAL_MS);
          } else {
            stamp.doneAt = Date.now();
            res.write(frame({}, 'stop'));
            res.write('data: [DONE]\n\n');
            res.end();
          }
        };
        send();
      });
      server.listen(0, '127.0.0.1', () => {
        resolve({ server, port: (server.address() as AddressInfo).port });
      });
    });
  }

  async function measure(
    totalWords: number,
    tick: number
  ): Promise<{
    meanGap: number;
    jitter: number;
    lag: number;
    totalText: string;
    expected: string;
  }> {
    const chunks = buildChunks(totalWords);
    const stamp = { doneAt: 0 };
    const { server, port } = await startServer(chunks, stamp);
    try {
      const model = new ChatOpenAI({
        model: 'bench-model',
        apiKey: 'bench',
        streaming: true,
        configuration: { baseURL: `http://127.0.0.1:${port}/v1` },
        _lc_stream_delay: tick,
      });

      const gaps: number[] = [];
      let totalText = '';
      let last = Date.now();
      for await (const piece of await model.stream([new HumanMessage('go')])) {
        if (typeof piece.content !== 'string' || piece.content === '') {
          continue;
        }
        const now = Date.now();
        gaps.push(now - last);
        last = now;
        totalText += piece.content;
      }
      const renderDoneAt = Date.now();

      const paint = gaps.slice(1);
      const mean = paint.reduce((a, b) => a + b, 0) / paint.length;
      const jitter = Math.sqrt(
        paint.reduce((a, b) => a + (b - mean) ** 2, 0) / paint.length
      );
      return {
        meanGap: mean,
        jitter,
        lag: renderDoneAt - stamp.doneAt,
        totalText,
        expected: chunks.join(''),
      };
    } finally {
      server.close();
    }
  }

  test('holds cadence at the tick with bounded lag on a medium reply', async () => {
    const tick = 15;
    const result = await measure(500, tick);

    expect(result.totalText).toBe(result.expected);
    expect(result.meanGap).toBeGreaterThan(tick * 0.6);
    expect(result.meanGap).toBeLessThan(tick * 1.6);
    expect(result.jitter).toBeLessThan(10);
    expect(result.lag).toBeLessThan(SMOOTH_TARGET_LATENCY_MS * 3);
  }, 60000);

  test('lag does not grow with reply length', async () => {
    const tick = 15;
    const short = await measure(250, tick);
    const long = await measure(1000, tick);

    expect(long.totalText).toBe(long.expected);
    expect(long.lag).toBeLessThan(SMOOTH_TARGET_LATENCY_MS * 3);
    expect(long.lag).toBeLessThan(short.lag + SMOOTH_TARGET_LATENCY_MS * 2);
  }, 60000);
});
