const mockLangfuseSpanProcessorInstance = {};
const mockLangfuseSpanProcessor = jest.fn(
  () => mockLangfuseSpanProcessorInstance
);
const mockSetLangfuseTracerProvider = jest.fn();
let mockContextHasActiveValue = false;
const mockContextKey = Symbol('mock-context-key');
const mockContextActive = jest.fn(() => ({
  getValue: jest.fn(() => mockContextHasActiveValue),
}));
const mockContextWith = jest.fn(
  (_contextValue: unknown, callback: () => boolean) => callback()
);
const mockSetGlobalContextManager = jest.fn(() => true);
const mockRootContext = {
  setValue: jest.fn(() => ({})),
};
const mockContextManager = {
  disable: jest.fn(),
  enable: jest.fn(),
};
const mockAsyncLocalStorageContextManager = jest.fn(() => mockContextManager);
const mockTracerProvider = {
  forceFlush: jest.fn(),
  getTracer: jest.fn(),
  shutdown: jest.fn(),
};
const mockBasicTracerProvider = jest.fn(() => mockTracerProvider);

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: mockLangfuseSpanProcessor,
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@langfuse/tracing', () => ({
  ...jest.requireActual('@langfuse/tracing'),
  setLangfuseTracerProvider: mockSetLangfuseTracerProvider,
}));

jest.mock('@opentelemetry/api', () => ({
  ...jest.requireActual('@opentelemetry/api'),
  context: {
    ...jest.requireActual('@opentelemetry/api').context,
    active: mockContextActive,
    setGlobalContextManager: mockSetGlobalContextManager,
    with: mockContextWith,
  },
  createContextKey: jest.fn(() => mockContextKey),
  ROOT_CONTEXT: mockRootContext,
}));

jest.mock('@opentelemetry/context-async-hooks', () => ({
  AsyncLocalStorageContextManager: mockAsyncLocalStorageContextManager,
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: mockBasicTracerProvider,
  SpanStatusCode: jest.requireActual('@opentelemetry/api').SpanStatusCode,
}));

describe('Langfuse instrumentation', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    mockContextHasActiveValue = false;
    mockSetGlobalContextManager.mockReturnValue(true);
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('does not initialize tracing when Langfuse env vars are missing', async () => {
    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );

    expect(initializeLangfuseTracingFromEnv()).toBeUndefined();
    expect(mockLangfuseSpanProcessor).not.toHaveBeenCalled();
    expect(mockBasicTracerProvider).not.toHaveBeenCalled();
    expect(mockAsyncLocalStorageContextManager).not.toHaveBeenCalled();
    expect(mockSetLangfuseTracerProvider).not.toHaveBeenCalled();
  });

  it('registers an isolated Langfuse tracer provider from env config', async () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );
    const provider = initializeLangfuseTracingFromEnv();

    expect(provider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(1);
    expect(mockBasicTracerProvider).toHaveBeenCalledWith({
      spanProcessors: [mockLangfuseSpanProcessorInstance],
    });
    expect(mockSetLangfuseTracerProvider).toHaveBeenCalledWith(
      mockTracerProvider
    );
    expect(mockAsyncLocalStorageContextManager).toHaveBeenCalledTimes(1);
    expect(mockContextManager.enable).toHaveBeenCalledTimes(1);
    expect(mockSetGlobalContextManager).toHaveBeenCalledWith(
      mockContextManager
    );
  });

  it('reuses the isolated provider after initialization', async () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    const { initializeLangfuseTracingFromEnv } = await import(
      '@/instrumentation'
    );
    const firstProvider = initializeLangfuseTracingFromEnv();
    const secondProvider = initializeLangfuseTracingFromEnv();

    expect(firstProvider).toBe(mockTracerProvider);
    expect(secondProvider).toBe(mockTracerProvider);
    expect(mockLangfuseSpanProcessor).toHaveBeenCalledTimes(1);
    expect(mockBasicTracerProvider).toHaveBeenCalledTimes(1);
    expect(mockSetLangfuseTracerProvider).toHaveBeenCalledTimes(1);
  });

  it('does not replace an existing active context manager', async () => {
    mockContextHasActiveValue = true;
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    await import('@/instrumentation');

    expect(mockAsyncLocalStorageContextManager).not.toHaveBeenCalled();
    expect(mockSetGlobalContextManager).not.toHaveBeenCalled();
  });

  it('disables the local context manager when registration is rejected', async () => {
    mockSetGlobalContextManager.mockReturnValue(false);
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    await import('@/instrumentation');

    expect(mockContextManager.enable).toHaveBeenCalledTimes(1);
    expect(mockSetGlobalContextManager).toHaveBeenCalledWith(
      mockContextManager
    );
    expect(mockContextManager.disable).toHaveBeenCalledTimes(1);
  });
});
