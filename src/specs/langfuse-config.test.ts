import { CallbackHandler } from '@langfuse/langchain';
import { AIMessageChunk } from '@langchain/core/messages';
import type { ChatGeneration, LLMResult } from '@langchain/core/outputs';
import {
  hasLangfuseConfigCredentials,
  shouldCreateLangfuseHandler,
  isExplicitLangfuseConfig,
  disposeLangfuseHandler,
  createLangfuseHandler,
} from '@/langfuse';
import { ContentTypes } from '@/common';

const mockForceFlush = jest.fn();
const mockHandleLLMEnd = jest.fn();

jest.mock('@langfuse/langchain', () => {
  const CallbackHandler = jest.fn(function (
    this: { params?: unknown; name?: string },
    params: unknown
  ) {
    this.params = params;
    this.name = 'LangfuseCallbackHandler';
  });
  CallbackHandler.prototype.handleLLMEnd = (...args: unknown[]) =>
    mockHandleLLMEnd(...args);
  return { CallbackHandler };
});

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
    mockHandleLLMEnd.mockResolvedValue(undefined);
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

  it('adds configured trace metadata and tags to the callback handler', () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';

    const handler = createLangfuseHandler({
      langfuse: {
        metadata: {
          tenantId: 'tenant-1',
          empty: '',
          skipped: null,
        },
        tags: ['tenant:tenant-1', 'agent'],
      },
      traceMetadata: {
        messageId: 'message-1',
        agentId: 'agent-1',
      },
      tags: ['librechat', 'agent'],
    });

    expect(handler).toBeDefined();
    expect(MockedCallbackHandler).toHaveBeenCalledWith({
      userId: undefined,
      sessionId: undefined,
      traceMetadata: {
        tenantId: 'tenant-1',
        messageId: 'message-1',
        agentId: 'agent-1',
      },
      tags: ['librechat', 'agent', 'tenant:tenant-1'],
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

  it('adds exposed reasoning text to Langfuse generation output without mutating the message', async () => {
    let tracedContent: AIMessageChunk['content'] | undefined;
    mockHandleLLMEnd.mockImplementation(async (llmOutput: LLMResult) => {
      const tracedMessage = llmOutput.generations[0][0] as unknown as {
        message: AIMessageChunk;
      };
      tracedContent = tracedMessage.message.content;
    });
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });
    const message = new AIMessageChunk({
      content: 'Visible answer.',
      additional_kwargs: {
        reasoning_content: 'Inspect the data first.',
      },
    });
    const output: LLMResult = {
      generations: [
        [
          {
            text: 'Visible answer.',
            message,
          } as ChatGeneration,
        ],
      ],
    };

    await handler?.handleLLMEnd(output, 'run-1');

    expect(mockHandleLLMEnd).toHaveBeenCalledTimes(1);
    expect(tracedContent).toEqual([
      {
        type: ContentTypes.THINK,
        think: 'Inspect the data first.',
      },
      {
        type: ContentTypes.TEXT,
        text: 'Visible answer.',
      },
    ]);
    expect(message.content).toBe('Visible answer.');
  });

  it('does not duplicate Bedrock reasoning_content blocks already in message content', async () => {
    let tracedContent: AIMessageChunk['content'] | undefined;
    mockHandleLLMEnd.mockImplementation(async (llmOutput: LLMResult) => {
      const tracedMessage = llmOutput.generations[0][0] as unknown as {
        message: AIMessageChunk;
      };
      tracedContent = tracedMessage.message.content;
    });
    const content = [
      {
        type: ContentTypes.REASONING_CONTENT,
        reasoningText: { text: 'Use Bedrock native reasoning.' },
      },
      { type: ContentTypes.TEXT, text: 'Visible answer.' },
    ];
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });
    const message = new AIMessageChunk({
      content,
      additional_kwargs: {
        reasoning_content: 'Use Bedrock native reasoning.',
      },
    });
    const output: LLMResult = {
      generations: [
        [
          {
            text: 'Visible answer.',
            message,
          } as ChatGeneration,
        ],
      ],
    };

    await handler?.handleLLMEnd(output, 'run-1');

    expect(mockHandleLLMEnd).toHaveBeenCalledTimes(1);
    expect(tracedContent).toBe(content);
    expect(message.content).toBe(content);
  });

  it('passes Anthropic thinking blocks already in message content unchanged', async () => {
    let tracedContent: AIMessageChunk['content'] | undefined;
    mockHandleLLMEnd.mockImplementation(async (llmOutput: LLMResult) => {
      const tracedMessage = llmOutput.generations[0][0] as unknown as {
        message: AIMessageChunk;
      };
      tracedContent = tracedMessage.message.content;
    });
    const content = [
      {
        type: ContentTypes.THINKING,
        thinking: 'Use Anthropic native thinking.',
        signature: 'sig',
      },
      { type: ContentTypes.TEXT, text: 'Visible answer.' },
    ];
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });
    const message = new AIMessageChunk({
      content,
    });
    const output: LLMResult = {
      generations: [
        [
          {
            text: 'Visible answer.',
            message,
          } as ChatGeneration,
        ],
      ],
    };

    await handler?.handleLLMEnd(output, 'run-1');

    expect(mockHandleLLMEnd).toHaveBeenCalledTimes(1);
    expect(tracedContent).toBe(content);
    expect(message.content).toBe(content);
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

  it('does not treat sanitized-away trace attributes as explicit config', () => {
    expect(
      isExplicitLangfuseConfig({
        metadata: {
          empty: '',
          whitespace: '   ',
          missing: null,
          tooLong: 'x'.repeat(201),
        },
        tags: ['', '   '],
      })
    ).toBe(false);
  });

  it('treats valid trace metadata or tags as explicit config', () => {
    expect(
      isExplicitLangfuseConfig({
        metadata: {
          tenantId: 'tenant-1',
        },
        tags: ['', '   '],
      })
    ).toBe(true);
    expect(
      isExplicitLangfuseConfig({
        metadata: {
          empty: '',
        },
        tags: ['tenant:tenant-1'],
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
