import { CallbackManager } from '@langchain/core/callbacks/manager';
import { HumanMessage } from '@langchain/core/messages';
import { Providers } from '@/common';
import { Run } from '@/run';
import type * as t from '@/types';

const mockSpan = {
  end: jest.fn(),
  spanContext: jest.fn(() => ({
    traceId: 'trace-id',
    spanId: 'span-id',
    traceFlags: 1,
  })),
  setAttributes: jest.fn(),
  setStatus: jest.fn(),
};
const mockStartSpan = jest.fn(() => mockSpan);
const mockStartActiveSpan = jest.fn(
  (
    _name: string,
    _options: unknown,
    _context: unknown,
    callback: (span: typeof mockSpan) => unknown
  ) => callback(mockSpan)
);
const mockForceFlush = jest.fn();
const mockShutdown = jest.fn();

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation(() => ({})),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation(() => ({
    forceFlush: mockForceFlush,
    getTracer: jest.fn(() => ({
      startActiveSpan: mockStartActiveSpan,
      startSpan: mockStartSpan,
    })),
    shutdown: mockShutdown,
  })),
}));

describe('Langfuse callback composition', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('runs explicit per-agent tracing when callbacks is a CallbackManager', async () => {
    const manager = CallbackManager.fromHandlers({
      handleCustomEvent: async (): Promise<void> => undefined,
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-callback-manager',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-test',
              secretKey: 'sk-test',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);

    const config = {
      callbacks: manager,
      configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };

    await run.processStream({ messages: [new HumanMessage('hello')] }, config);

    expect(mockStartActiveSpan).toHaveBeenCalled();
  });

  it('attaches Langfuse callbacks for direct graph invocations', async () => {
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-direct-graph',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-test',
              secretKey: 'sk-test',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);
    const workflow = run.Graph?.createWorkflow();
    await workflow?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: [],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(mockStartActiveSpan).toHaveBeenCalled();
  });

  it('preserves per-agent Langfuse config when a stream callback already exists', async () => {
    const { LangfuseSpanProcessor } = await import('@langfuse/otel');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { createLangfuseHandler } = await import('@/langfuse');
    initializeLangfuseTracing({
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
    });
    const streamHandler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.run',
      },
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-agent-callback-override',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
            langfuse: {
              enabled: true,
              publicKey: 'pk-agent',
              secretKey: 'sk-agent',
              baseUrl: 'https://langfuse.agent',
            },
          },
        ],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);
    const workflow = run.Graph?.createWorkflow();
    await workflow?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: streamHandler != null ? [streamHandler] : [],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(LangfuseSpanProcessor).toHaveBeenCalledWith(
      expect.objectContaining({
        publicKey: 'pk-agent',
        secretKey: 'sk-agent',
        baseUrl: 'https://langfuse.agent',
      })
    );
  });
});
