import { CallbackManager } from '@langchain/core/callbacks/manager';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import { CallbackHandler as LangfuseCallbackHandler } from '@langfuse/langchain';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
} from '@langchain/core/messages';
import type { ChatGeneration, LLMResult } from '@langchain/core/outputs';
import type * as t from '@/types';
import { PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY } from '@/messages/cache';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import { Providers } from '@/common';
import { Run } from '@/run';

const mockProcessorStarts: Array<{
  params: unknown;
  traceId: string;
}> = [];
const mockSpanAttributeSets: Array<Record<string, unknown>> = [];
let mockProviderInput:
  | {
      spanProcessors?: Array<{
        onStart?: (span: unknown, parentContext: unknown) => void;
        onEnd?: (span: unknown) => void;
      }>;
      idGenerator?: {
        generateTraceId: () => string;
        generateSpanId: () => string;
      };
    }
  | undefined;

const createMockSpan = (traceIdOverride?: string) => {
  const traceId =
    traceIdOverride ??
    mockProviderInput?.idGenerator?.generateTraceId() ??
    'trace-id';
  const spanId = mockProviderInput?.idGenerator?.generateSpanId() ?? 'span-id';
  const span = {
    end: jest.fn(() => {
      for (const processor of mockProviderInput?.spanProcessors ?? []) {
        processor.onEnd?.(span);
      }
    }),
    spanContext: jest.fn(() => ({
      traceId,
      spanId,
      traceFlags: 1,
    })),
    setAttributes: jest.fn((attributes: Record<string, unknown>) => {
      mockSpanAttributeSets.push(attributes);
    }),
    setStatus: jest.fn(),
    attributes: {},
  };
  for (const processor of mockProviderInput?.spanProcessors ?? []) {
    processor.onStart?.(span, otelContext.active());
  }
  return span;
};

const mockStartSpan = jest.fn(() => createMockSpan());
const mockStartActiveSpan = jest.fn(
  (
    _name: string,
    _options: unknown,
    activeContext: Parameters<typeof otelTrace.getSpanContext>[0],
    callback: (span: ReturnType<typeof createMockSpan>) => unknown
  ) =>
    callback(createMockSpan(otelTrace.getSpanContext(activeContext)?.traceId))
);
const mockForceFlush = jest.fn();
const mockShutdown = jest.fn();

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation((params) => ({
    forceFlush: jest.fn(),
    onEnd: jest.fn(),
    onStart: jest.fn((span) => {
      mockProcessorStarts.push({
        params,
        traceId: span.spanContext().traceId,
      });
    }),
    shutdown: jest.fn(),
  })),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation((input) => {
    mockProviderInput = input;
    return {
      forceFlush: mockForceFlush,
      getTracer: jest.fn(() => ({
        startActiveSpan: mockStartActiveSpan,
        startSpan: mockStartSpan,
      })),
      shutdown: mockShutdown,
    };
  }),
}));

describe('Langfuse callback composition', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockProcessorStarts.length = 0;
    mockSpanAttributeSets.length = 0;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
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

  it('binds handler callback spans to their own Langfuse config and trace seed', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const tenantA = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.proxy',
      deterministicTraceId: true,
    };
    const tenantB = {
      publicKey: 'pk-tenant-b',
      secretKey: 'sk-tenant-b',
      baseUrl: 'https://langfuse.proxy',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(tenantA);
    initializeLangfuseTracing(tenantB);

    const handlerA = createLangfuseHandler({
      langfuse: tenantA,
      traceIdSeed: 'run-tenant-a',
    });
    const handlerB = createLangfuseHandler({
      langfuse: tenantB,
      traceIdSeed: 'run-tenant-b',
    });

    await Promise.all([
      handlerA?.handleChainStart(
        { lc: 1, type: 'not_implemented', id: ['TenantAChain'] },
        { input: 'tenant a' },
        'lc-run-a'
      ),
      handlerB?.handleChainStart(
        { lc: 1, type: 'not_implemented', id: ['TenantBChain'] },
        { input: 'tenant b' },
        'lc-run-b'
      ),
    ]);

    expect(mockProcessorStarts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-tenant-a',
            secretKey: 'sk-tenant-a',
            baseUrl: 'https://langfuse.proxy',
          }),
          traceId: traceIdFromSeed('run-tenant-a'),
        }),
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-tenant-b',
            secretKey: 'sk-tenant-b',
            baseUrl: 'https://langfuse.proxy',
          }),
          traceId: traceIdFromSeed('run-tenant-b'),
        }),
      ])
    );
  });

  it('preserves an active agent Langfuse runtime scope for callback-created spans', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const { withLangfuseRuntimeScope } = await import('@/langfuseRuntimeScope');
    const runLangfuse = {
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.run',
      deterministicTraceId: true,
    };
    const agentLangfuse = {
      publicKey: 'pk-agent',
      secretKey: 'sk-agent',
      baseUrl: 'https://langfuse.agent',
      deterministicTraceId: true,
    };
    initializeLangfuseTracing(runLangfuse);
    initializeLangfuseTracing(agentLangfuse);
    const streamHandler = createLangfuseHandler({
      langfuse: runLangfuse,
      traceIdSeed: 'run-seed',
    });

    await withLangfuseRuntimeScope(
      { langfuse: agentLangfuse, traceIdSeed: 'agent-seed' },
      () =>
        streamHandler?.handleChainStart(
          { lc: 1, type: 'not_implemented', id: ['AgentScopedChain'] },
          { input: 'agent scoped' },
          'lc-agent-run'
        )
    );

    expect(mockProcessorStarts).toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.agent',
        }),
        traceId: traceIdFromSeed('agent-seed'),
      })
    );
    expect(mockProcessorStarts).not.toContainEqual(
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: 'pk-run',
          secretKey: 'sk-run',
          baseUrl: 'https://langfuse.run',
        }),
        traceId: traceIdFromSeed('run-seed'),
      })
    );
  });

  it('attaches configured trace attributes to Langfuse callback spans', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const { initializeLangfuseTracing } = await import('@/instrumentation');
    const langfuse = {
      publicKey: 'pk-tenant-a',
      secretKey: 'sk-tenant-a',
      baseUrl: 'https://langfuse.proxy',
      librechatTraceAttributes: {
        'librechat.langfuse.tenant_export.enabled': 'true',
        'librechat.langfuse.destination': 'eu',
        ignored: '',
      },
    };
    initializeLangfuseTracing(langfuse);
    const handler = createLangfuseHandler({ langfuse });

    await handler?.handleChainStart(
      { lc: 1, type: 'not_implemented', id: ['TenantAChain'] },
      { input: 'tenant a' },
      'lc-run-a'
    );

    expect(mockSpanAttributeSets).toContainEqual({
      'librechat.langfuse.tenant_export.enabled': 'true',
      'librechat.langfuse.destination': 'eu',
    });
  });

  it('uses deterministic trace ids when tracing is configured from env only', async () => {
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-env';
    process.env.LANGFUSE_SECRET_KEY = 'sk-env';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.env';

    const runId = 'test-langfuse-env-deterministic-run';
    const run = await Run.create<t.IState>({
      runId,
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent_abc123',
            name: 'DWAINE',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4' },
            tools: [],
          },
        ],
      },
      langfuse: {
        deterministicTraceId: true,
        metadata: { 'librechat.tenant.id': 'tenant-env' },
        tags: ['tenant:tenant-env'],
      },
      skipCleanup: true,
    });

    run.Graph?.overrideTestModel(['hello']);

    await run.processStream(
      { messages: [new HumanMessage('hello')] },
      {
        configurable: { thread_id: 'thread-1', user_id: 'user-1' },
        version: 'v2' as const,
      }
    );

    expect(mockProcessorStarts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          params: expect.objectContaining({
            publicKey: 'pk-env',
            secretKey: 'sk-env',
            baseUrl: 'https://langfuse.env',
          }),
          traceId: traceIdFromSeed(runId),
        }),
      ])
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

  it('normalizes Bedrock cache write tokens into Langfuse TTL usage details', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessage({
      content: 'done',
      response_metadata: {
        [PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY]: '1h',
      },
      usage_metadata: {
        input_tokens: 8283,
        output_tokens: 58,
        total_tokens: 8341,
        input_token_details: {
          cache_read: 7618,
          cache_creation: 604,
        },
      },
    });
    const output: LLMResult = {
      generations: [[{ text: 'done', message } as ChatGeneration]],
    };

    const normalized = normalizePromptCacheUsageForLangfuse(output);
    const normalizedMessage = (normalized.generations[0][0] as ChatGeneration)
      .message as AIMessage;

    expect(normalizedMessage.usage_metadata?.input_token_details).toEqual({
      cache_read: 7618,
      cache_creation: 0,
      cache_creation_1h: 604,
    });
    expect(message.usage_metadata?.input_token_details).toEqual({
      cache_read: 7618,
      cache_creation: 604,
    });
  });

  it('normalizes Bedrock 5m cache write tokens for Langfuse', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessage({
      content: 'done',
      response_metadata: {
        [PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY]: '5m',
      },
      usage_metadata: {
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        input_token_details: {
          cache_creation: 25,
        },
      },
    });

    const normalized = normalizePromptCacheUsageForLangfuse({
      generations: [[{ text: 'done', message } as ChatGeneration]],
    });
    const normalizedMessage = (normalized.generations[0][0] as ChatGeneration)
      .message as AIMessage;

    expect(normalizedMessage.usage_metadata?.input_token_details).toEqual({
      cache_creation: 0,
      cache_creation_5m: 25,
    });
  });

  it('leaves generic cache creation untouched when Bedrock TTL is missing', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessage({
      content: 'done',
      usage_metadata: {
        input_tokens: 100,
        output_tokens: 10,
        total_tokens: 110,
        input_token_details: {
          cache_creation: 25,
        },
      },
    });

    const output: LLMResult = {
      generations: [[{ text: 'done', message } as ChatGeneration]],
    };

    expect(normalizePromptCacheUsageForLangfuse(output)).toBe(output);
  });

  it('normalizes Anthropic explicit cache creation buckets for Langfuse', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessage({
      content: 'done',
      response_metadata: {
        model_provider: 'anthropic',
        usage: {
          cache_creation: {
            ephemeral_5m_input_tokens: 10,
            ephemeral_1h_input_tokens: 20,
          },
        },
      },
      usage_metadata: {
        input_tokens: 40,
        output_tokens: 5,
        total_tokens: 45,
        input_token_details: {
          cache_read: 3,
          cache_creation: 30,
        },
      },
    });
    const output: LLMResult = {
      generations: [[{ text: 'done', message } as ChatGeneration]],
    };

    const normalized = normalizePromptCacheUsageForLangfuse(output);
    const normalizedMessage = (normalized.generations[0][0] as ChatGeneration)
      .message as AIMessage;

    expect(normalizedMessage.usage_metadata?.input_token_details).toEqual({
      cache_read: 3,
      cache_creation: 0,
      cache_creation_5m: 10,
      cache_creation_1h: 20,
    });
    expect(message.usage_metadata?.input_token_details).toEqual({
      cache_read: 3,
      cache_creation: 30,
    });
  });

  it('keeps any unbucketed cache creation remainder on the generic key', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessage({
      content: 'done',
      response_metadata: {
        model_provider: 'anthropic',
        usage: {
          cache_creation: {
            ephemeral_5m_input_tokens: 10,
            ephemeral_1h_input_tokens: 20,
          },
        },
      },
      usage_metadata: {
        input_tokens: 50,
        output_tokens: 5,
        total_tokens: 55,
        input_token_details: {
          cache_creation: 35,
        },
      },
    });

    const normalized = normalizePromptCacheUsageForLangfuse({
      generations: [[{ text: 'done', message } as ChatGeneration]],
    });
    const normalizedMessage = (normalized.generations[0][0] as ChatGeneration)
      .message as AIMessage;

    expect(normalizedMessage.usage_metadata?.input_token_details).toEqual({
      cache_creation: 5,
      cache_creation_5m: 10,
      cache_creation_1h: 20,
    });
  });

  it('normalizes AIMessageChunk usage metadata', async () => {
    const { normalizePromptCacheUsageForLangfuse } = await import(
      '@/langfuse/promptCacheUsage'
    );
    const message = new AIMessageChunk({
      content: 'done',
      response_metadata: {
        [PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY]: '1h',
      },
      usage_metadata: {
        input_tokens: 50,
        output_tokens: 5,
        total_tokens: 55,
        input_token_details: {
          cache_creation: 12,
        },
      },
    });

    const normalized = normalizePromptCacheUsageForLangfuse({
      generations: [[{ text: 'done', message } as ChatGeneration]],
    });
    const normalizedMessage = (normalized.generations[0][0] as ChatGeneration)
      .message as AIMessageChunk;

    expect(normalizedMessage).toBeInstanceOf(AIMessageChunk);
    expect(normalizedMessage.usage_metadata?.input_token_details).toEqual({
      cache_creation: 0,
      cache_creation_1h: 12,
    });
  });

  it('normalizes prompt cache usage before delegating to Langfuse handleLLMEnd', async () => {
    const { createLangfuseHandler } = await import('@/langfuse');
    const handleLLMEndSpy = jest
      .spyOn(LangfuseCallbackHandler.prototype, 'handleLLMEnd')
      .mockResolvedValue(undefined);
    const handler = createLangfuseHandler({
      langfuse: {
        publicKey: 'pk-test',
        secretKey: 'sk-test',
      },
    });
    const message = new AIMessage({
      content: 'done',
      response_metadata: {
        [PROMPT_CACHE_TTL_RESPONSE_METADATA_KEY]: '1h',
      },
      usage_metadata: {
        input_tokens: 50,
        output_tokens: 5,
        total_tokens: 55,
        input_token_details: {
          cache_creation: 12,
        },
      },
    });

    await handler?.handleLLMEnd(
      { generations: [[{ text: 'done', message } as ChatGeneration]] },
      'llm-run'
    );

    const delegatedOutput = handleLLMEndSpy.mock.calls[0]?.[0] as LLMResult;
    handleLLMEndSpy.mockRestore();
    const delegatedMessage = (
      delegatedOutput.generations[0][0] as ChatGeneration
    ).message as AIMessage;
    expect(delegatedMessage.usage_metadata?.input_token_details).toEqual({
      cache_creation: 0,
      cache_creation_1h: 12,
    });
    expect(message.usage_metadata?.input_token_details).toEqual({
      cache_creation: 12,
    });
  });
});
