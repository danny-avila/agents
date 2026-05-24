import { LangfuseSpanProcessor } from '@langfuse/otel';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { createLangfuseHandler } from '@/langfuse';

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation(() => ({
    forceFlush: jest.fn(),
    getTracer: jest.fn(),
    shutdown: jest.fn(),
  })),
}));

describe('createLangfuseHandler', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('creates a handler when keys are provided and baseUrl is omitted', () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });

    expect(handler).toBeDefined();
    expect(LangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-test',
        secretKey: 'sk-test',
        exportMode: 'immediate',
      })
    );
    expect(
      (LangfuseSpanProcessor as jest.Mock).mock.calls[0][0].baseUrl
    ).toBeUndefined();
    expect(BasicTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('does not create a handler when a required key is missing', () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
      },
    });

    expect(handler).toBeUndefined();
    expect(LangfuseSpanProcessor).not.toHaveBeenCalled();
    expect(BasicTracerProvider).not.toHaveBeenCalled();
  });
});
