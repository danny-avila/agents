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
    delete process.env.LANGFUSE_FORCE_FLUSH_ON_DISPOSE;
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
    expect(mockForceFlush).not.toHaveBeenCalled();
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

  it('adds current agent metadata when a stream Langfuse callback already exists', async () => {
    const metadataSpy = jest.fn();
    const { createLangfuseHandler } = await import('@/langfuse');
    const streamHandler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.run',
      },
    });
    const run = await Run.create<t.IState>({
      runId: 'test-langfuse-agent-metadata-with-stream-callback',
      graphConfig: {
        type: 'multi-agent',
        agents: [
          {
            agentId: 'agent_default',
            name: 'Default Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
          {
            agentId: 'agent_specialist',
            name: 'Specialist Agent',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
        ],
        edges: [],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello from specialist']);
    const agentNode = run.Graph?.createAgentNode('agent_specialist');
    await agentNode?.invoke(
      { messages: [new HumanMessage('hello')] },
      {
        callbacks: [
          ...(streamHandler != null ? [streamHandler] : []),
          {
            handleChatModelStart: async (
              _llm: unknown,
              _messages: unknown,
              _runId: string,
              _parentRunId?: string,
              _extraParams?: unknown,
              _tags?: string[],
              metadata?: Record<string, unknown>
            ): Promise<void> => {
              metadataSpy(metadata);
            },
          },
        ],
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
      }
    );

    expect(metadataSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        agentId: 'agent_specialist',
        agentName: 'Specialist Agent',
      })
    );
  });
});
