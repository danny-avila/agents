import { LangfuseSpanProcessor } from '@langfuse/otel';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { HumanMessage } from '@langchain/core/messages';
import type { Serialized } from '@langchain/core/load/serializable';
import { createLangfuseHandler } from '@/langfuse';

const mockSpan = {
  end: jest.fn(),
  setAttributes: jest.fn(),
  setStatus: jest.fn(),
};
const mockStartSpan = jest.fn(() => mockSpan);
const mockGetTracer = jest.fn(() => ({
  startSpan: mockStartSpan,
}));
const mockForceFlush = jest.fn();
const mockShutdown = jest.fn();

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation(() => ({})),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation(() => ({
    forceFlush: mockForceFlush,
    getTracer: mockGetTracer,
    shutdown: mockShutdown,
  })),
}));

describe('createLangfuseHandler', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...originalEnv };
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
  });

  afterEach(() => {
    process.env = originalEnv;
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

  it('starts per-agent spans with v5 trace attributes', async () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
      userId: 'user-1',
      sessionId: 'thread-1',
      traceMetadata: {
        messageId: 'message-1',
        agentId: 'agent-1',
        agentName: 'DWAINE',
      },
      tags: ['librechat', 'agent'],
    });

    await handler?.handleChatModelStart(
      {
        id: ['langchain', 'chat_models', 'ChatOpenAI'],
        kwargs: { model: 'gpt-4o' },
      } as unknown as Serialized,
      [[new HumanMessage('hello')]],
      'run-1'
    );

    expect(mockGetTracer).toHaveBeenCalledWith('langfuse-sdk');
    expect(mockStartSpan).toHaveBeenCalledWith(
      'gpt-4o',
      expect.objectContaining({
        attributes: expect.objectContaining({
          'langfuse.trace.name': 'LibreChat Agent: DWAINE',
          'langfuse.trace.metadata.agentId': 'agent-1',
          'langfuse.trace.metadata.messageId': 'message-1',
          'langfuse.observation.model.name': 'gpt-4o',
          'langfuse.observation.type': 'generation',
          'user.id': 'user-1',
          'session.id': 'thread-1',
          'langfuse.trace.tags': ['librechat', 'agent'],
        }),
      })
    );
  });

  it('starts tool spans with input and output attributes', async () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
      userId: 'user-1',
      sessionId: 'thread-1',
      traceMetadata: {
        messageId: 'message-1',
        agentId: 'agent-1',
        agentName: 'DWAINE',
      },
      tags: ['librechat', 'agent'],
      callbackScope: 'tools',
      useToolOutputTracingFallback: false,
    });

    await handler?.handleToolStart(
      {
        id: ['langchain', 'tools', 'DynamicStructuredTool'],
        kwargs: { name: 'mcp_clickhouse' },
      } as unknown as Serialized,
      '{"query":"select 1"}',
      'tool-run-1',
      undefined,
      undefined,
      { langgraph_node: 'tools=agent-1' },
      undefined,
      'call-1'
    );
    await handler?.handleToolEnd('secret rows', 'tool-run-1');

    expect(mockStartSpan).toHaveBeenCalledWith(
      'mcp_clickhouse',
      expect.objectContaining({
        attributes: expect.objectContaining({
          'langfuse.trace.name': 'LibreChat Agent: DWAINE',
          'langfuse.trace.metadata.agentId': 'agent-1',
          'langfuse.trace.metadata.messageId': 'message-1',
          'langfuse.observation.type': 'tool',
          'langfuse.observation.input': '{"query":"select 1"}',
          'langfuse.observation.metadata.toolCallId': 'call-1',
          'user.id': 'user-1',
          'session.id': 'thread-1',
          'langfuse.trace.tags': ['librechat', 'agent'],
        }),
      })
    );
    expect(mockSpan.setAttributes).toHaveBeenCalledWith(
      expect.objectContaining({
        'langfuse.observation.type': 'tool',
        'langfuse.observation.output': 'secret rows',
      })
    );
    expect(mockSpan.end).toHaveBeenCalledTimes(1);
    expect(mockForceFlush).toHaveBeenCalled();
  });

  it('does not start model spans for a tool-only handler', async () => {
    const handler = createLangfuseHandler({
      langfuse: {
        enabled: true,
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
      callbackScope: 'tools',
    });

    await handler?.handleChatModelStart(
      {
        id: ['langchain', 'chat_models', 'ChatOpenAI'],
        kwargs: { model: 'gpt-4o' },
      } as unknown as Serialized,
      [[new HumanMessage('hello')]],
      'run-1'
    );

    expect(mockStartSpan).not.toHaveBeenCalled();
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
    expect(LangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-env',
        secretKey: 'sk-env',
        baseUrl: 'https://langfuse.env',
        exportMode: 'immediate',
      })
    );
    expect(BasicTracerProvider).toHaveBeenCalledTimes(1);
  });
});
