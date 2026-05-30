import { CallbackHandler } from '@langfuse/langchain';
import {
  createLangfuseHandler,
  disposeLangfuseHandler,
  hasLangfuseConfigCredentials,
  shouldCreateLangfuseHandler,
} from '@/langfuse';

const mockForceFlush = jest.fn();

jest.mock('@langfuse/langchain', () => ({
  CallbackHandler: jest.fn().mockImplementation((params) => ({ params })),
}));

jest.mock('@langfuse/tracing', () => ({
  getLangfuseTracerProvider: jest.fn(() => ({
    forceFlush: mockForceFlush,
  })),
}));

const MockedCallbackHandler = CallbackHandler as jest.MockedClass<
  typeof CallbackHandler
>;

describe('createLangfuseHandler', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
    delete process.env.LANGFUSE_FORCE_FLUSH_ON_DISPOSE;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('creates the official Langfuse callback handler when env keys are present', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.env';

    const handler = createLangfuseHandler({
      userId: 'user-1',
      sessionId: 'thread-1',
      traceMetadata: {
        messageId: 'message-1',
        agentId: 'agent-1',
        agentName: 'DWAINE',
      },
      tags: ['librechat', 'agent'],
    });

    expect(handler).toBeDefined();
    expect(MockedCallbackHandler).toHaveBeenCalledWith({
      userId: 'user-1',
      sessionId: 'thread-1',
      traceMetadata: {
        messageId: 'message-1',
        agentId: 'agent-1',
        agentName: 'DWAINE',
      },
      tags: ['librechat', 'agent'],
    });
  });

  it('creates a handler for explicit credentials supplied in config', () => {
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });

    expect(handler).toBeDefined();
    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
  });

  it('hydrates redaction-only config from env keys', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.env';

    const handler = createLangfuseHandler({
      langfuse: {
        toolOutputTracing: { enabled: false },
      },
    });

    expect(handler).toBeDefined();
    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
  });

  it('does not create a handler when Langfuse is disabled', () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: false,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });

    expect(handler).toBeUndefined();
    expect(MockedCallbackHandler).not.toHaveBeenCalled();
  });

  it('does not create a handler when credentials are unavailable', () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
      },
    });

    expect(handler).toBeUndefined();
    expect(MockedCallbackHandler).not.toHaveBeenCalled();
  });

  it('detects complete config credentials', () => {
    expect(
      hasLangfuseConfigCredentials({
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      })
    ).toBe(true);
    expect(
      hasLangfuseConfigCredentials({
        publicKey: 'pk-test',
      })
    ).toBe(false);
  });

  it('uses env credentials for redaction-only configs', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.env';

    expect(
      shouldCreateLangfuseHandler({
        toolOutputTracing: { enabled: false },
      })
    ).toBe(true);
  });

  it('uses env credentials with a config-provided baseUrl', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';

    expect(
      shouldCreateLangfuseHandler({
        baseUrl: 'https://langfuse.config',
        toolOutputTracing: { enabled: false },
      })
    ).toBe(true);
  });

  it('does not flush the shared Langfuse provider during per-chat cleanup', async () => {
    await expect(disposeLangfuseHandler({})).resolves.toBeUndefined();
    expect(mockForceFlush).not.toHaveBeenCalled();
  });

  it('force flushes during cleanup when explicitly enabled', async () => {
    process.env.LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'true';

    await expect(disposeLangfuseHandler({})).resolves.toBeUndefined();

    expect(mockForceFlush).toHaveBeenCalledTimes(1);
  });
});
