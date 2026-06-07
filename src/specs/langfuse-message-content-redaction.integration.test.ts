const mockLangfuseSpanProcessorInstance = {
  onStart: jest.fn(),
  onEnd: jest.fn(),
  forceFlush: jest.fn().mockResolvedValue(undefined),
  shutdown: jest.fn().mockResolvedValue(undefined),
};
const mockLangfuseSpanProcessor = jest.fn(
  () => mockLangfuseSpanProcessorInstance
);

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: mockLangfuseSpanProcessor,
  isDefaultExportSpan: jest.fn(() => false),
}));

import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import {
  createLangfuseSpanProcessor,
  withLangfuseToolOutputTracingConfig,
} from '@/langfuseToolOutputTracing';
import type { LangfuseConfig } from '@/types/graph';

function buildPipeline(runLangfuse: LangfuseConfig): {
  provider: BasicTracerProvider;
  exporter: InMemorySpanExporter;
} {
  const exporter = new InMemorySpanExporter();
  const redactingProcessor = createLangfuseSpanProcessor(
    { publicKey: 'pk-test', secretKey: 'sk-test' },
    runLangfuse
  );
  const provider = new BasicTracerProvider({
    spanProcessors: [redactingProcessor, new SimpleSpanProcessor(exporter)],
  });
  return { provider, exporter };
}

const PROMPT_WITH_CREDENTIAL =
  'CANARY-XYZ-9876 my key is sk-ant-api03-FAKE1234567890abcdefghij please help';

describe('Langfuse redacting span processor end-to-end', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_REDACT_MESSAGE_CONTENT;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('scrubs credentials from generation span input through the full processor pipeline', async () => {
    const runLangfuse: LangfuseConfig = {
      publicKey: 'pk-test',
      secretKey: 'sk-test',
      messageContentRedaction: { enabled: true },
    };
    const { provider, exporter } = buildPipeline(runLangfuse);
    const tracer = provider.getTracer('integration-test');

    withLangfuseToolOutputTracingConfig(runLangfuse, () => {
      const span = tracer.startSpan('chat-generation');
      span.setAttribute(
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        JSON.stringify([{ role: 'user', content: PROMPT_WITH_CREDENTIAL }])
      );
      span.end();
    });

    await provider.forceFlush();

    const captured = exporter.getFinishedSpans();
    expect(captured).toHaveLength(1);

    const input = captured[0].attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    const parsed = JSON.parse(input) as Array<{ content: string }>;

    expect(parsed[0].content).toContain('CANARY-XYZ-9876');
    expect(parsed[0].content).toContain('sk-ant-[REDACTED]');
    expect(parsed[0].content).not.toContain('FAKE1234567890');

    await provider.shutdown();
  });

  it('preserves tool inputs while scrubbing user-message credentials in the same span', async () => {
    const runLangfuse: LangfuseConfig = {
      publicKey: 'pk-test',
      secretKey: 'sk-test',
      messageContentRedaction: { enabled: true },
    };
    const { provider, exporter } = buildPipeline(runLangfuse);
    const tracer = provider.getTracer('integration-test');

    withLangfuseToolOutputTracingConfig(runLangfuse, () => {
      const span = tracer.startSpan('chat-generation');
      span.setAttribute(
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        JSON.stringify([
          { role: 'user', content: PROMPT_WITH_CREDENTIAL },
          {
            role: 'assistant',
            content: '',
            tool_calls: [
              {
                id: 'call_sql',
                name: 'execute_sql',
                args: { query: 'SELECT 1' },
              },
            ],
          },
        ])
      );
      span.end();
    });

    await provider.forceFlush();

    const captured = exporter.getFinishedSpans();
    const input = captured[0].attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    const parsed = JSON.parse(input) as Array<{
      role: string;
      content: string;
      tool_calls?: Array<{ args: { query: string } }>;
    }>;

    expect(parsed[0].content).toContain('sk-ant-[REDACTED]');
    expect(parsed[1].tool_calls?.[0].args.query).toBe('SELECT 1');

    await provider.shutdown();
  });

  it('does not redact when messageContentRedaction is disabled', async () => {
    const runLangfuse: LangfuseConfig = {
      publicKey: 'pk-test',
      secretKey: 'sk-test',
      messageContentRedaction: { enabled: false },
    };
    const { provider, exporter } = buildPipeline(runLangfuse);
    const tracer = provider.getTracer('integration-test');

    withLangfuseToolOutputTracingConfig(runLangfuse, () => {
      const span = tracer.startSpan('chat-generation');
      span.setAttribute(
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        JSON.stringify([{ role: 'user', content: PROMPT_WITH_CREDENTIAL }])
      );
      span.end();
    });

    await provider.forceFlush();

    const captured = exporter.getFinishedSpans();
    const input = captured[0].attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;

    expect(input).toContain('FAKE1234567890');
    expect(input).not.toContain('[REDACTED]');

    await provider.shutdown();
  });

  it('forwards the same span to the inner LangfuseSpanProcessor so real exports still fire', async () => {
    const runLangfuse: LangfuseConfig = {
      publicKey: 'pk-test',
      secretKey: 'sk-test',
      messageContentRedaction: { enabled: true },
    };
    const { provider } = buildPipeline(runLangfuse);
    const tracer = provider.getTracer('integration-test');

    withLangfuseToolOutputTracingConfig(runLangfuse, () => {
      const span = tracer.startSpan('chat-generation');
      span.setAttribute(
        LangfuseOtelSpanAttributes.OBSERVATION_INPUT,
        JSON.stringify([{ role: 'user', content: PROMPT_WITH_CREDENTIAL }])
      );
      span.end();
    });

    await provider.forceFlush();

    expect(mockLangfuseSpanProcessorInstance.onEnd).toHaveBeenCalledTimes(1);
    const forwardedSpan =
      mockLangfuseSpanProcessorInstance.onEnd.mock.calls[0][0];
    const forwardedInput = forwardedSpan.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(forwardedInput).toContain('sk-ant-[REDACTED]');
    expect(forwardedInput).not.toContain('FAKE1234567890');

    await provider.shutdown();
  });
});
