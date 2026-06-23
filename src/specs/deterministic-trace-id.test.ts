import { runWithTraceIdSeed, traceIdFromSeed } from '@/langfuseRuntimeContext';
import { initializeLangfuseTracing } from '@/instrumentation';

const langfuse = {
  publicKey: 'pk-lf-test',
  secretKey: 'sk-lf-test',
  baseUrl: 'http://localhost:3999',
};

describe('deterministic Langfuse trace ids', () => {
  it('derives the root trace id from the seed when run inside runWithTraceIdSeed', () => {
    const provider = initializeLangfuseTracing(langfuse);
    expect(provider).toBeDefined();
    const tracer = provider!.getTracer('deterministic-trace-id-test');

    const seed = 'response-message-id-1234';
    let traceId: string | undefined;
    runWithTraceIdSeed(seed, () => {
      const span = tracer.startSpan('AgentRun');
      traceId = span.spanContext().traceId;
      span.end();
    });

    expect(traceId).toBe(traceIdFromSeed(seed));
  });

  it('falls back to a random trace id when no seed is active', () => {
    const provider = initializeLangfuseTracing(langfuse);
    const tracer = provider!.getTracer('deterministic-trace-id-test');

    const span = tracer.startSpan('AgentRun');
    const traceId = span.spanContext().traceId;
    span.end();

    expect(traceId).toMatch(/^[0-9a-f]{32}$/);
    expect(traceId).not.toBe(traceIdFromSeed('response-message-id-1234'));
  });

  it('runWithTraceIdSeed is a passthrough when the seed is undefined', () => {
    const sentinel = {};
    expect(runWithTraceIdSeed(undefined, () => sentinel)).toBe(sentinel);
  });
});
